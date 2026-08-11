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
low-quality text instead of being confidence-filtered. The pytesseract
TSV route or an OCR service would provide confidences; accepted for now.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import pymupdf

from .config import OCR_DPI, TESSDATA_DIR, TESSDATA_URL
from .local_extract import extract_page, page_body_font_size
from .models import Source, Unit

log = logging.getLogger(__name__)


def ensure_tessdata() -> str:
    """Download eng.traineddata on first use; return the tessdata dir."""
    td = Path(TESSDATA_DIR)
    target = td / "eng.traineddata"
    if not target.exists():
        td.mkdir(parents=True, exist_ok=True)
        log.info("downloading tessdata -> %s", target)
        urllib.request.urlretrieve(TESSDATA_URL, target)  # noqa: S310 - pinned https URL
    return str(td)


def ocr_page_units(page: pymupdf.Page, page_index: int, figures_dir: Path) -> list[Unit]:
    """OCR one scanned page and run the standard extraction walk on it.

    full=True OCRs the entire page as one image (right for scanned pages,
    where nothing has a text layer); OCR_DPI=300 is Tesseract's sweet
    spot. Slow (~1-3 s/page on CPU) — by far the heaviest local stage,
    which is why it runs only on pages triage marked SCANNED.

    Heading detection is judged against THIS page's own OCR size
    distribution, not the document-wide native body size — OCR-synthesized
    sizes and native sizes are different measurement systems (ledger #17).
    """
    textpage = page.get_textpage_ocr(dpi=OCR_DPI, full=True, tessdata=ensure_tessdata())
    body_size = page_body_font_size(page, textpage)
    return extract_page(
        page,
        page_index,
        body_size,
        figures_dir,
        textpage=textpage,
        source=Source.TESSERACT_OCR,
        include_figures=False,
    )
