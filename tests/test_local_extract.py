"""Local extraction (stage 2) against the synthetic sample PDF.

What matters here:
  * body font size comes out as the CHARACTER-weighted mode (10pt), even
    though heading spans exist at 13/14/18pt;
  * both heading rules fire — size ratio ("7. Payment Terms", 18pt) and
    bold+numbered at body size ("7.3.1 Delay Notices", 10pt bold);
  * paragraphs are merged from wrapped lines, not fragmented per span;
  * the embedded figure is cropped, stored, and referenced by storage_key;
  * the ruled grid is found on exactly the table page, with the expected
    segment counts (5 horizontal, 4 vertical for a 4x3 ruled table).
"""

import pymupdf
import pytest

from rag_ingest.local_extract import extract
from rag_ingest.models import UnitType
from rag_ingest.sample_pdf import build_sample
from rag_ingest.triage import triage


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    doc = pymupdf.open(build_sample(tmp_path_factory.mktemp("pdf") / "sample_doc.pdf"))
    kinds = {r.page: r.kind for r in triage(doc)}
    body_size, units, grids = extract(doc, kinds, out)
    yield {"out": out, "body_size": body_size, "units": units, "grids": grids}
    doc.close()


def test_body_font_size_is_character_weighted_mode(extracted):
    assert extracted["body_size"] == 10.0


def test_size_based_heading_detected(extracted):
    titles = [u.content for u in extracted["units"] if u.type == UnitType.TITLE]
    assert "7. Payment Terms" in titles


def test_body_size_bold_numbered_heading_detected(extracted):
    # 10pt — invisible to the size rule; only bold + clause numbering catches it.
    titles = [u.content for u in extracted["units"] if u.type == UnitType.TITLE]
    assert "7.3.1 Delay Notices" in titles


def test_title_units_carry_font_size_for_stage6_clustering(extracted):
    sizes = {u.content: u.font_size for u in extracted["units"] if u.type == UnitType.TITLE}
    assert sizes["7. Payment Terms"] == 18.0
    assert sizes["7.3.1 Delay Notices"] == 10.0


def test_paragraphs_merged_not_fragmented(extracted):
    page0_text = [u for u in extracted["units"] if u.type == UnitType.TEXT and u.page == 0]
    # LOREM*6 wraps over many visual lines but is ONE textbox paragraph block.
    assert any("supplier shall complete all services" in u.content for u in page0_text)
    assert len(page0_text) <= 3


def test_embedded_figure_stored_with_storage_key(extracted):
    figs = [u for u in extracted["units"] if u.type == UnitType.FIGURE]
    assert [f.page for f in figs] == [0]
    stored = extracted["out"] / figs[0].storage_key
    assert stored.exists() and stored.stat().st_size > 0


def test_ruled_grids_on_exactly_the_table_pages(extracted):
    # Pages 6, 8, 9 draw ruled tables; the drawing page (4) is not
    # TEXT_NATIVE so stage 2 never sees its line-work.
    assert [g.page for g in extracted["grids"]] == [6, 8, 9]
    grid6 = extracted["grids"][0]
    assert grid6.h_segments == 5  # 4 rows -> 5 horizontal rules
    assert grid6.v_segments == 4  # 3 cols -> 4 vertical rules
