"""STAGE 5 — OCR: scanned-page prose via PyMuPDF's bundled Tesseract.

No system install, no container: PyMuPDF wheels ship with libtesseract
compiled in. The only external artifact is the language data file
(eng.traineddata, ~4 MB), auto-downloaded to .tessdata/ on first use —
the same lazy-fetch pattern as the YOLO weights.

Why this beats the alternatives for this repo:
  * a system Tesseract (winget/apt) + pytesseract shells out per call and
    pollutes the host;
  * a docker-compose OCR service (e.g. hertzg/tesseract-server) is the
    right shape for PRODUCTION — OCR gets its own CPU pool and scales
    independently — but hands back raw hOCR/TSV over HTTP, so the
    block/line/span mapping we get for free here would need rebuilding.
    Documented as the scaling path in docs/edge_cases.md, not built.

The key property: get_textpage_ocr() returns a TextPage with the SAME
block/line/span structure as a native text layer, so stage 2's entire
extraction walk (heading rules, paragraph merging) runs on scanned pages
unchanged — a scanned page becomes "text-native after OCR". Only the
Unit.source tag differs, so downstream consumers can always tell exact
text from OCR text.

Known limitation (ledger #16): the bundled integration does not expose
per-word confidence, so stamps/signatures/handwriting come through as
low-quality text instead of being confidence-filtered. What we DO have
(ledger #28) is a page-level quality gate: ocr_quality_score() measures
whether OCR output looks like language at all, flags garbage pages
needs_review, and drives orientation recovery for sideways scans.
"""

from __future__ import annotations

import logging
import re
import urllib.request
from pathlib import Path
from typing import cast

import numpy as np
import pymupdf

from .config import (
    OCR_DPI,
    OCR_MIN_QUALITY,
    ORIENTATION_DPI,
    ORIENTATION_MIN_GAIN,
    TESSDATA_DIR,
    TESSDATA_URL,
)
from .local_extract import extract_page, page_body_font_size
from .models import Source, Unit

log = logging.getLogger(__name__)

# A token is "language-like" when it is a word (letters incl. a vowel —
# OCR of sideways/degraded input produces consonant shrapnel like "dd",
# real prose almost never does beyond 3 letters) or a number. Punctuation
# is stripped before judging; pure-symbol tokens score zero.
_WORDISH = re.compile(r"[A-Za-z]{2,}")
_NUMBERISH = re.compile(r"[0-9][0-9.,/x%-]*")
_VOWELS = re.compile(r"[aeiouAEIOU]")


def ocr_quality_score(text: str) -> float:
    """Fraction of tokens that look like language rather than symbol soup.

    Judge AGGREGATE text (a whole page, a whole table) — single short
    strings are legitimately noisy ("338", "e.g."). Calibrated against
    real pipeline output; the threshold lives in config.OCR_MIN_QUALITY.
    Empty text scores 1.0: no output is no *evidence* of garbage (a blank
    page must not fail the gate).
    """
    tokens = text.split()
    if not tokens:
        return 1.0
    ok = 0
    for t in tokens:
        core = t.strip(".,;:()[]{}\"'!?%")
        if not core:
            continue
        if _WORDISH.fullmatch(core):
            if _VOWELS.search(core) or len(core) <= 3:
                ok += 1
        elif _NUMBERISH.fullmatch(core) or core in ("a", "A", "I"):
            ok += 1
    return ok / len(tokens)


def _ocr_text_of_array(rgb: np.ndarray, tessdata: str) -> str:
    """OCR a HxWx3 RGB array via the pdfocr wrapper; return plain text."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, rgb.shape[1], rgb.shape[0], rgb.tobytes(), False)
    doc = pymupdf.open("pdf", pix.pdfocr_tobytes(language="eng", tessdata=tessdata))
    try:
        # get_text's stub type is a union over its mode argument; "text" is str.
        return cast(str, doc.load_page(0).get_text("text"))
    finally:
        doc.close()


def detect_orientation(page: pymupdf.Page) -> tuple[int, float, float]:
    """Probe a scanned page's orientation. Returns (rotation_delta_degrees,
    score_at_current, score_at_best).

    A landscape/rotated scan OCRs into symbol soup, and without per-word
    confidences it would sail through as confident garbage — the exact
    failure class this pipeline promises not to have (ledger #28). The
    probe is cheap: one low-DPI OCR when the page is healthy; three more
    (90/180/270 via np.rot90) only when it is not. The winner must clear
    OCR_MIN_QUALITY *and* beat the current orientation by
    ORIENTATION_MIN_GAIN, so a genuinely bad scan is flagged rather than
    randomly rotated.

    The delta is in PDF /Rotate convention (clockwise): applying
    ``page.set_rotation((page.rotation + delta) % 360)`` makes future
    renders come out upright for every downstream consumer (YOLO, OCR,
    table crops, stored figures).
    """
    tessdata = ensure_tessdata()
    pix = page.get_pixmap(dpi=ORIENTATION_DPI, alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).copy()

    score0 = ocr_quality_score(_ocr_text_of_array(rgb, tessdata))
    if score0 >= OCR_MIN_QUALITY:
        return 0, score0, score0

    best_k, best_score = 0, score0
    for k in (1, 2, 3):  # np.rot90 turns counterclockwise
        score = ocr_quality_score(_ocr_text_of_array(np.rot90(rgb, k).copy(), tessdata))
        if score > best_score:
            best_k, best_score = k, score
    # k CCW turns of the rendered image == rendering with /Rotate reduced
    # by 90k (clockwise convention), i.e. a delta of -90k mod 360.
    if best_k and best_score >= OCR_MIN_QUALITY and best_score - score0 >= ORIENTATION_MIN_GAIN:
        return (-90 * best_k) % 360, score0, best_score
    return 0, score0, score0


def ensure_tessdata() -> str:
    """Download eng.traineddata on first use; return the tessdata dir."""
    td = Path(TESSDATA_DIR)
    target = td / "eng.traineddata"
    if not target.exists():
        td.mkdir(parents=True, exist_ok=True)
        log.info("downloading tessdata -> %s", target)
        urllib.request.urlretrieve(TESSDATA_URL, target)  # noqa: S310 - pinned https URL
    return str(td)


def get_ocr_textpage(page: pymupdf.Page) -> pymupdf.TextPage:
    """One OCR pass per scanned page, shared by stages 5 AND 6: the same
    textpage that yields prose units also supplies the words that fill
    table cells (tables.extract_scanned_table). OCR is the most expensive
    local operation — never run it twice on one page."""
    return page.get_textpage_ocr(dpi=OCR_DPI, full=True, tessdata=ensure_tessdata())


def ocr_page_units(
    page: pymupdf.Page,
    page_index: int,
    figures_dir: Path,
    textpage: pymupdf.TextPage | None = None,
) -> list[Unit]:
    """OCR one scanned page and run the standard extraction walk on it.

    full=True OCRs the entire page as one image (right for scanned pages,
    where nothing has a text layer); OCR_DPI=300 is Tesseract's sweet
    spot. Slow (~1-3 s/page on CPU) — by far the heaviest local stage,
    which is why it runs only on pages triage marked SCANNED.

    Heading detection is judged against THIS page's own OCR size
    distribution, not the document-wide native body size — OCR-synthesized
    sizes and native sizes are different measurement systems (ledger #17).
    """
    if textpage is None:
        textpage = get_ocr_textpage(page)
    body_size = page_body_font_size(page, textpage)
    units = extract_page(
        page,
        page_index,
        body_size,
        figures_dir,
        textpage=textpage,
        source=Source.TESSERACT_OCR,
        include_figures=False,
    )
    # Quality gate (ledger #28): if the page's OCR output as a whole does
    # not look like language, every unit from it is flagged for review —
    # garbage must never enter the corpus as confident text.
    quality = ocr_quality_score(page.get_text(textpage=textpage))
    if quality < OCR_MIN_QUALITY:
        for u in units:
            u.needs_review = True
        log.warning(
            "p%04d: OCR quality %.2f < %.2f — %d unit(s) flagged needs_review",
            page_index,
            quality,
            OCR_MIN_QUALITY,
            len(units),
        )
    return units
