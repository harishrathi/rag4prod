"""Complex-table tests against sample_data/complex_doc.pdf (ledgers #27-28).

Every assertion is against EXPECTED output, declared next to the
generator that draws the pages (complex_pdf.GT_*): the unmerged logical
content — a merged value repeated into every grid position it covers.
The old tier-2 (pixel grid + Tesseract) tests are gone with their
engine; scanned-page tables now come through the VLM lane, whose parser
and verification are covered in test_vlm_extract.py.
"""

import pymupdf
import pytest

from rag_ingest.assemble import build_walk
from rag_ingest.chunking import chunk_document
from rag_ingest.complex_pdf import (
    GT_CONTINUATION,
    GT_LONG,
    GT_ROWSPAN,
    GT_TWO_TIER,
    build_complex,
)
from rag_ingest.layout import Region
from rag_ingest.models import Source, Unit, UnitType
from rag_ingest.tables import (
    TableResult,
    cells_to_grid,
    extract_native_tables,
    finalize,
)


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    d = pymupdf.open(build_complex(tmp_path_factory.mktemp("pdf") / "complex_doc.pdf"))
    yield d
    d.close()


# --- Tier 1: merged cells on native pages ------------------------------------


def test_tier1_two_tier_header_unmerges_exactly(doc):
    t = extract_native_tables(doc.load_page(0), 0)[0]
    assert t.cells == GT_TWO_TIER
    assert t.header_rows == 2
    # Item/Description row-span the header; Rate Breakdown col-spans 2.
    assert [0, 0, 2, 1] in t.merges
    assert [0, 2, 1, 2] in t.merges


def test_tier1_rowspan_category_fills_every_row(doc):
    t = extract_native_tables(doc.load_page(1), 1)[0]
    assert t.cells == GT_ROWSPAN
    assert t.header_rows == 1
    assert [1, 0, 3, 1] in t.merges and [4, 0, 2, 1] in t.merges


def test_tier1_combined_merges(doc):
    t = extract_native_tables(doc.load_page(2), 2)[0]
    # Row-span Remarks fills down; col-span Subtotal fills across.
    assert t.cells[2][3] == "Rates incl. haulage to 5 km"
    assert t.cells[3][0] == t.cells[3][1] == "Subtotal - earthworks"
    # The genuinely empty cell stays empty — fill must not invent content.
    assert t.cells[3][3] == ""


# --- Stitching: row-span crossing the page break -----------------------------


@pytest.fixture(scope="module")
def stitched(doc, tmp_path_factory):
    raw = []
    for p in (6, 7):
        raw.extend(extract_native_tables(doc.load_page(p), p))
    heights = {p: doc.load_page(p).rect.height for p in (6, 7)}
    return finalize(raw, heights, doc, tmp_path_factory.mktemp("out"))


def test_continuation_with_rowspan_merges_to_expected_table(stitched):
    table = next(t for t in stitched if len(t.pages) > 1)
    assert table.pages == [6, 7]
    # Repeated header dropped, page-break-crossing Finishes span filled.
    assert table.cells == GT_CONTINUATION
    assert table.markdown.count("| Category |") == 1


# --- Chunking: merged labels + multi-row headers survive row-grouping --------


def test_row_groups_never_stranded_from_category(doc, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    raw = extract_native_tables(doc.load_page(9), 9)
    tables = finalize(raw, {9: doc.load_page(9).rect.height}, doc, out)
    assert tables[0].cells == GT_LONG
    chunks, _ = chunk_document("t", build_walk([], tables), rows_per_chunk=10)
    table_chunks = [c for c in chunks if c.table_id]
    assert len(table_chunks) == 3  # 30 data rows at 10 per group
    # Expected categories per group: rows 1-10 Civil, 11-20 mixed, 21-30
    # Electrical — every row must carry its label inside its own chunk.
    groups = (GT_LONG[1:11], GT_LONG[11:21], GT_LONG[21:31])
    for c, expected_rows in zip(table_chunks, groups, strict=True):
        for row in expected_rows:
            assert f"| {row[0]} | {row[1]} | {row[2]} |" in c.content


def test_two_tier_header_repeats_fully_in_every_group(doc, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    raw = extract_native_tables(doc.load_page(0), 0)
    tables = finalize(raw, {0: doc.load_page(0).rect.height}, doc, out)
    chunks, _ = chunk_document("t", build_walk([], tables), rows_per_chunk=2)
    table_chunks = [c for c in chunks if c.table_id]
    assert len(table_chunks) == 3  # 5 data rows at 2 per group
    for c in table_chunks:
        # Both header rows — tier labels AND sub-columns — open every group.
        lines = c.content.splitlines()
        assert lines[0] == "| Item | Description | Rate Breakdown | Rate Breakdown |"
        assert lines[2] == "| Item | Description | Material | Labour |"


# --- ASCII grid rendering ----------------------------------------------------


def test_grid_reconstructs_the_printed_layout():
    cells = [["A", "A", "B"], ["A", "A", "C"]]
    merges = [[0, 0, 2, 2]]
    expected = "\n".join(
        [
            "+-----+-----+-----+",
            "| A         | B   |",
            "|           +-----+",
            "|           | C   |",
            "+-----+-----+-----+",
        ]
    )
    assert cells_to_grid(cells, merges) == expected


def test_grid_two_tier_header_looks_like_the_pdf(doc):
    t = extract_native_tables(doc.load_page(0), 0)[0]
    grid = cells_to_grid(t.cells, t.merges)
    lines = grid.splitlines()
    # Column widths follow content: "Description" (11), "1300.00" (7).
    # Row-spanned header cells draw once, open below (no rule under them)…
    assert "| Item | Description         | Rate Breakdown     |" in lines
    assert "|      |                     +----------+---------+" in lines
    assert "|      |                     | Material | Labour  |" in lines
    # …and the header closes with a full rule before the data rows.
    assert "+------+---------------------+----------+---------+" in lines


def test_finalized_table_carries_merges_and_grid(stitched):
    table = next(t for t in stitched if len(t.pages) > 1)
    # The Civil Works span (4 rows from row 1) survives stitching intact.
    assert [1, 0, 4, 1] in table.merges
    # One label, one box: "Civil Works" appears once in the grid drawing.
    assert table.grid.count("Civil Works") == 1


# --- Artifact round-trips (--from-stage resume) ------------------------------


def test_unit_region_table_roundtrip_through_artifacts(stitched):
    u = Unit(
        page=3, bbox=(1.0, 2.0, 3.0, 4.0), type=UnitType.TEXT, content="x", source=Source.PYMUPDF
    )
    assert Unit.from_dict(u.to_dict()) == u
    r = Region(page=1, label="table", conf=0.9, bbox_px=(1, 2, 3, 4), bbox_pdf=(1.0, 2.0, 3.0, 4.0))
    assert Region.from_dict(r.to_dict()) == r
    t = stitched[0]
    assert TableResult.from_dict(t.to_dict()) == t
