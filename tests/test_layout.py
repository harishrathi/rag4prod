"""Stage 4 tests.

The coordinate conversion is pure math and runs in every test session —
it's the spec's "most likely bug in this pipeline", so it gets the
paranoid tests. The model integration test costs a ~40 MB weights
download plus a torch import, so it only runs when RUN_YOLO_TESTS=1:

    $env:RUN_YOLO_TESTS = "1"; pytest tests/test_layout.py -q
"""

import os

import pymupdf
import pytest

from rag_ingest.layout import pixel_rect_to_pdf

A4 = pymupdf.Rect(0, 0, 595, 842)


def test_pixel_to_pdf_uses_actual_dimensions():
    # 200 DPI render of A4 -> 1654 x 2339 px. Bottom-right quadrant.
    bbox = pixel_rect_to_pdf(827, 1169.5, 1654, 2339, A4, 1654, 2339)
    assert bbox == pytest.approx((297.5, 421.0, 595.0, 842.0), abs=0.5)


def test_pixel_to_pdf_respects_cropbox_offset():
    # A page whose rect does not start at the origin: the offset must be
    # added, which the naive 72/DPI constant-scale conversion gets wrong.
    page_rect = pymupdf.Rect(10, 20, 605, 862)
    bbox = pixel_rect_to_pdf(0, 0, 1654, 2339, page_rect, 1654, 2339)
    assert bbox == pytest.approx((10.0, 20.0, 605.0, 862.0), abs=0.5)


def test_pixel_to_pdf_clamps_padded_boxes_into_page():
    # Padding pushes boxes past the image edge; the conversion must clamp,
    # because a get_pixmap clip outside page.rect is a silently empty crop.
    bbox = pixel_rect_to_pdf(-15, -15, 1700, 2400, A4, 1654, 2339)
    assert bbox == (0.0, 0.0, 595.0, 842.0)


def test_pixel_to_pdf_rejects_degenerate_boxes():
    with pytest.raises(ValueError):
        # Entirely outside the page: clamping collapses it -> raises.
        # A real exception, not an assert: must survive `python -O`.
        pixel_rect_to_pdf(-100, -100, -50, -50, A4, 1654, 2339)


@pytest.mark.skipif(os.environ.get("RUN_YOLO_TESTS") != "1", reason="set RUN_YOLO_TESTS=1")
def test_model_finds_table_on_table_page(tmp_path):
    from rag_ingest.layout import LayoutDetector
    from rag_ingest.render import pixmap_to_rgb_array, render_page
    from rag_ingest.sample_pdf import build_sample

    doc = pymupdf.open(build_sample(tmp_path / "sample_doc.pdf"))
    page = doc.load_page(6)  # ruled table + prose
    regions = LayoutDetector().detect(pixmap_to_rgb_array(render_page(page)), page.rect, 6)
    doc.close()

    tables = [g for g in regions if g.label == "table"]
    assert tables, "expected at least one table detection on the table page"
    x0, y0, x1, y1 = tables[0].bbox_pdf
    # Ground truth grid is (72, 220, 522, 332); allow padding + model slack.
    assert x0 < 100 and y0 < 260 and x1 > 480 and y1 > 300
