"""Stage 6 tests: the table ladder + multi-page stitching.

Ground truth comes from sample_pdf's cell constants: page 6 is a bordered
native table (tier 1), pages 8-9 are one table split across a page
boundary with a repeated header (stitching). Scanned-page tables now
arrive via the VLM lane (tested in test_vlm_extract.py), so the old
tier-2 pixel-grid tests are gone with their engine.
"""

import pymupdf
import pytest

from rag_ingest.sample_pdf import CONT_ROWS_P8, CONT_ROWS_P9, HEADER_ROW, build_sample
from rag_ingest.tables import (
    extract_native_tables,
    finalize,
    stitch,
    validate_cells,
)


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    d = pymupdf.open(build_sample(tmp_path_factory.mktemp("pdf") / "sample_doc.pdf"))
    yield d
    d.close()


# --- Tier 1: bordered, text-native ------------------------------------------


def test_tier1_extracts_exact_cells(doc):
    found = extract_native_tables(doc.load_page(6), 6)
    assert len(found) == 1
    cells = found[0].cells
    assert cells[0] == HEADER_ROW
    assert cells[1] == ["1", "Excavation", "120.00"]
    assert validate_cells(cells) is None


# --- Validation gate ---------------------------------------------------------


def test_validation_rejects_ragged_and_empty():
    assert validate_cells([]) is not None
    assert validate_cells([["a", "b"]]) is not None  # header only
    assert validate_cells([["a", "b"], ["1"]]) is not None  # ragged
    assert validate_cells([["", ""], ["1", "2"]]) is not None  # empty header
    assert validate_cells([["a", "b"], ["1", "2"]]) is None


# --- Multi-page stitching ----------------------------------------------------


@pytest.fixture(scope="module")
def stitched(doc, tmp_path_factory):
    raw = []
    for p in (6, 8, 9):
        raw.extend(extract_native_tables(doc.load_page(p), p))
    heights = {p: doc.load_page(p).rect.height for p in (6, 8, 9)}
    return finalize(raw, heights, doc, tmp_path_factory.mktemp("out"))


def test_continuation_merges_into_one_table(stitched):
    multi = [t for t in stitched if len(t.pages) > 1]
    assert len(multi) == 1
    assert multi[0].pages == [8, 9]


def test_repeated_header_dropped_not_duplicated(stitched):
    table = next(t for t in stitched if t.pages == [8, 9])
    # header + 6 rows (p8) + 4 rows (p9, its repeated header dropped)
    assert table.row_count == 1 + len(CONT_ROWS_P8) + len(CONT_ROWS_P9)
    assert table.markdown.count("| Item |") == 1


def test_standalone_table_not_swallowed_by_stitching(stitched):
    assert any(t.pages == [6] for t in stitched)


def test_column_mismatch_refuses_to_merge(doc):
    from rag_ingest.tables import RawTable

    a = RawTable(page=0, bbox=(72, 700, 500, 800), cells=[["a", "b"], ["1", "2"]], source="x")
    b = RawTable(
        page=1, bbox=(72, 30, 500, 100), cells=[["a", "b", "c"], ["1", "2", "3"]], source="x"
    )
    chains = stitch([a, b], {0: 842.0, 1: 842.0})
    assert len(chains) == 2  # refused: guessing at structure is corruption
