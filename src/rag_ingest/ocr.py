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
import unicodedata
import urllib.request
from pathlib import Path
from typing import cast

import numpy as np
import pymupdf

from .config import (
    OCR_DPI,
    OCR_LANGUAGES,
    OCR_MIN_QUALITY,
    ORIENTATION_APPLY_GAIN,
    ORIENTATION_APPLY_SCORE,
    ORIENTATION_APPLY_WORDS,
    ORIENTATION_DPI,
    ORIENTATION_EXIT_WORDS,
    TESSDATA_BASE_URL,
    TESSDATA_DIR,
)
from .local_extract import extract_page, page_body_font_size
from .models import Source, Unit

log = logging.getLogger(__name__)

# A token is "language-like" when it is a word or a number, in ANY
# script. A "word" is letters plus combining marks — the marks matter:
# Indic scripts spell vowels as combining matras (बंद carries U+0902),
# so a bare isalpha() rejects most real Hindi. ASCII words additionally
# need a vowel (OCR of sideways/degraded input produces consonant
# shrapnel like "dd"; real prose almost never does beyond 3 letters).
# Punctuation is stripped before judging; bilingual label tokens
# ("विवरण/Bid") are judged per '/'-separated part; pure-symbol tokens
# and anything carrying control chars score zero.
_NUMBERISH = re.compile(r"[0-9][0-9.,/x%-]*")
_VOWELS = re.compile(r"[aeiouAEIOU]")


def _is_word(core: str) -> bool:
    has_letter = False
    for ch in core:
        if ch.isalpha():
            has_letter = True
        elif unicodedata.category(ch) not in ("Mn", "Mc"):
            return False
    return has_letter


def _token_ok(core: str) -> bool:
    if not core:
        return False
    if core in ("a", "A", "I"):
        return True
    if _NUMBERISH.fullmatch(core) or core.isdigit():
        return True
    if _is_word(core):
        if core.isascii():
            return len(core) >= 2 and bool(_VOWELS.search(core) or len(core) <= 3)
        return True  # non-Latin scripts: no vowel heuristic, letters suffice
    if "/" in core:
        parts = [p for p in core.split("/") if p]
        return len(parts) > 1 and all(_token_ok(p) for p in parts)
    return False


def _real_words(text: str) -> int:
    """Count tokens that read as genuine words — the only orientation-
    sensitive signal OCR gives. Measured on real pages: aggregate quality
    scores are nearly rotation-INVARIANT (digits and 2-letter shrapnel
    pass in any orientation; a sideways page scored 0.64), but sideways
    OCR produces almost no >= 3-letter vowel-bearing words while upright
    OCR produces dozens. ASCII: >= 3 letters incl. a vowel; other
    scripts: >= 2 letters (no vowel heuristic)."""
    count = 0
    for t in text.split():
        core = t.strip(".,;:()[]{}\"'!?%")
        if not _is_word(core):
            continue
        if core.isascii():
            if len(core) >= 3 and _VOWELS.search(core):
                count += 1
        elif len(core) >= 2:
            count += 1
    return count


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
    ok = sum(1 for t in tokens if _token_ok(t.strip(".,;:()[]{}\"'!?%")))
    return ok / len(tokens)


def _ocr_text_of_array(rgb: np.ndarray, tessdata: str) -> str:
    """OCR a HxWx3 RGB array via the pdfocr wrapper; return plain text."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, rgb.shape[1], rgb.shape[0], rgb.tobytes(), False)
    doc = pymupdf.open("pdf", pix.pdfocr_tobytes(language=OCR_LANGUAGES, tessdata=tessdata))
    try:
        # get_text's stub type is a union over its mode argument; "text" is str.
        return cast(str, doc.load_page(0).get_text("text"))
    finally:
        doc.close()


def detect_orientation(page: pymupdf.Page) -> tuple[int, int, int]:
    """Probe a scanned page's orientation. Returns (rotation_delta_degrees,
    real_words_at_current, real_words_at_best).

    A landscape/rotated scan OCRs into symbol soup, and without per-word
    confidences it would sail through as confident garbage — the exact
    failure class this pipeline promises not to have (ledger #28). The
    probe is cheap: one low-DPI OCR when the page is healthy; three more
    (90/180/270 via np.rot90) only when it is not. The decision rides on
    _real_words, not quality scores — see config's ORIENTATION block for
    the measurements behind that.

    The delta is in PDF /Rotate convention (clockwise): applying
    ``page.set_rotation((page.rotation + delta) % 360)`` makes future
    renders come out upright for every downstream consumer (YOLO, OCR,
    table crops, stored figures).
    """
    tessdata = ensure_tessdata()
    pix = page.get_pixmap(dpi=ORIENTATION_DPI, alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).copy()

    text0 = _ocr_text_of_array(rgb, tessdata)
    words0, score0 = _real_words(text0), ocr_quality_score(text0)
    if words0 >= ORIENTATION_EXIT_WORDS:
        # Enough genuine words at the current orientation — upright.
        # One cheap probe and out; upright-but-noisy pages never pay
        # the 4-rotation search (was ~8s/page on real scans).
        return 0, words0, words0
    if score0 >= OCR_MIN_QUALITY and words0 < ORIENTATION_APPLY_WORDS:
        # High score with almost no words = number-dominated page (a
        # scanned drawing full of dimension labels). Any rotation would
        # be REFUSED by the word floor below, so searching is pure
        # waste — found live: 15 such pages cost ~10s each for nothing.
        return 0, words0, words0

    best_k, best_words, best_score = 0, words0, score0
    for k in (1, 2, 3):  # np.rot90 turns counterclockwise
        text = _ocr_text_of_array(np.rot90(rgb, k).copy(), tessdata)
        words, score = _real_words(text), ocr_quality_score(text)
        if (score, words) > (best_score, best_words):
            best_k, best_words, best_score = k, words, score
    # k CCW turns of the rendered image == rendering with /Rotate reduced
    # by 90k (clockwise convention), i.e. a delta of -90k mod 360.
    # BOTH signals must agree (see config's ORIENTATION block for the
    # measurements): a clear score win over the current orientation AND
    # enough real words that the win isn't digit noise. Anything less
    # is left alone for the quality gate to flag.
    if (
        best_k
        and best_score >= ORIENTATION_APPLY_SCORE
        and best_score - score0 >= ORIENTATION_APPLY_GAIN
        and best_words >= ORIENTATION_APPLY_WORDS
    ):
        return (-90 * best_k) % 360, words0, best_words
    return 0, words0, best_words


def ensure_tessdata() -> str:
    """Download the traineddata for every OCR_LANGUAGES language on first
    use; return the tessdata dir. '+'-separated per Tesseract convention
    ("eng+hin" needs eng.traineddata AND hin.traineddata)."""
    td = Path(TESSDATA_DIR)
    for lang in OCR_LANGUAGES.split("+"):
        target = td / f"{lang}.traineddata"
        if not target.exists():
            td.mkdir(parents=True, exist_ok=True)
            url = TESSDATA_BASE_URL.format(lang=lang)
            log.info("downloading tessdata -> %s", target)
            urllib.request.urlretrieve(url, target)  # noqa: S310 - pinned https URL
    return str(td)


def get_ocr_textpage(page: pymupdf.Page) -> pymupdf.TextPage:
    """One OCR pass per scanned page, shared by stages 5 AND 6: the same
    textpage that yields prose units also supplies the words that fill
    table cells (tables.extract_scanned_table). OCR is the most expensive
    local operation — never run it twice on one page."""
    return page.get_textpage_ocr(
        dpi=OCR_DPI, full=True, language=OCR_LANGUAGES, tessdata=ensure_tessdata()
    )


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
    quality = ocr_quality_score(cast(str, page.get_text("text", textpage=textpage)))
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
