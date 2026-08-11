"""Stage 5 OCR against the synthetic sample PDF.

The sample's scanned pages are raster images OF printed text (rendered
then embedded, so there is genuinely no text layer). OCR must recover
that body text through the same extraction walk stage 2 uses, tagged
with source=tesseract_ocr so downstream consumers can always tell exact
text from OCR text.

Needs .tessdata/eng.traineddata — auto-downloaded on first use.
"""

import pymupdf
import pytest

from rag_ingest.models import Source, UnitType
from rag_ingest.ocr import ocr_page_units
from rag_ingest.sample_pdf import build_sample


@pytest.fixture(scope="module")
def scanned_units(tmp_path_factory):
    doc = pymupdf.open(build_sample(tmp_path_factory.mktemp("pdf") / "sample_doc.pdf"))
    units = ocr_page_units(doc.load_page(2), 2, tmp_path_factory.mktemp("out") / "figures")
    yield units
    doc.close()


def test_ocr_recovers_scanned_body_text(scanned_units):
    text = " ".join(u.content for u in scanned_units if u.type == UnitType.TEXT)
    # OCR of clean 200-DPI print should be near-perfect on these phrases.
    assert "supplier shall complete" in text
    assert "liquidated damages" in text.lower()


def test_ocr_headings_judged_against_ocr_sizes(scanned_units):
    # OCR sizes and native sizes are different measurement systems: judged
    # against the native body size (10pt), this page's 11.9pt body lines
    # would all become headings. Judged against its own distribution, only
    # the genuinely larger heading survives.
    titles = [u.content for u in scanned_units if u.type == UnitType.TITLE]
    assert titles == ["4. Delivery Conditions"]


def test_ocr_units_are_tagged_as_ocr_source(scanned_units):
    assert scanned_units, "expected OCR to produce units"
    assert all(u.source == Source.TESSERACT_OCR for u in scanned_units)


def test_ocr_emits_no_figure_units(scanned_units):
    # The page-sized scan image itself must NOT come back as a figure;
    # real figure regions on scanned pages arrive via YOLO instead.
    assert all(u.type != UnitType.FIGURE for u in scanned_units)


def test_ocr_units_carry_page_and_bboxes(scanned_units):
    for u in scanned_units:
        assert u.page == 2
        x0, y0, x1, y1 = u.bbox
        assert 0 <= x0 < x1 and 0 <= y0 < y1
