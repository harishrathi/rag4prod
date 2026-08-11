"""Triage against the synthetic sample PDF, where every page's correct
classification is known ground truth (sample_pdf.EXPECTED_KINDS).

The interesting assertions are the trap pages:
  * page 3 has >50 chars of real text but must still be SCANNED
    (header-over-scan guard);
  * page 4 has no text and no raster image but must be DRAWING, not SCANNED
    (vector segment count).
"""

import pymupdf
import pytest

from rag_ingest.sample_pdf import EXPECTED_KINDS, build_sample
from rag_ingest.triage import triage


@pytest.fixture(scope="module")
def sample_doc(tmp_path_factory):
    path = build_sample(tmp_path_factory.mktemp("pdf") / "sample_doc.pdf")
    doc = pymupdf.open(path)
    yield doc
    doc.close()


def test_every_page_classified_correctly(sample_doc):
    records = triage(sample_doc)
    got = {r.page: r.kind for r in records}
    assert got == EXPECTED_KINDS


def test_header_over_scan_trap_caught_by_coverage_not_text(sample_doc):
    record = triage(sample_doc)[3]
    # The trap is only meaningful if the text layer alone WOULD have passed:
    assert record.text_chars >= 50
    assert record.max_image_coverage > 0.7


def test_drawing_page_detected_by_segments(sample_doc):
    record = triage(sample_doc)[4]
    assert record.drawing_segments is not None
    assert record.drawing_segments >= 100
