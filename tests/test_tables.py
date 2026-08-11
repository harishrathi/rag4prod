"""Stage 6 tests: the tiered table ladder + multi-page stitching.

Ground truth comes from sample_pdf's cell constants: page 6 is a bordered
native table (tier 1), page 7 is the SAME table but existing only as
pixels (tier 2), pages 8-9 are one table split across a page boundary
with a repeated header (stitching).
"""

import pymupdf
import pytest

from rag_ingest.sample_pdf import CONT_ROWS_P8, CONT_ROWS_P9, HEADER_ROW, build_sample
from rag_ingest.tables import (
    extract_native_tables,
    extract_scanned_table,
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


# --- Tier 2: bordered, scanned -----------------------------------------------


def test_tier2_recovers_grid_and_cells_from_pixels(doc):
    page = doc.load_page(7)
    # Region as YOLO would report it: the table area plus padding.
    raw = extract_scanned_table(page, 7, (60.0, 140.0, 535.0, 290.0))
    assert raw.source == "grid_ocr"
    assert len(raw.cells) == 4 and len(raw.cells[0]) == 3
    assert "Item" in raw.cells[0][0]
    assert "Dewatering" in raw.cells[2][1]
    assert "75" in raw.cells[1][2]  # OCR may fuzz the decimals; digits must survive
    assert validate_cells(raw.cells) is None


def test_tier2_without_grid_returns_empty_cells_not_crash(doc):
    page = doc.load_page(2)  # scanned prose page: no grid anywhere
    raw = extract_scanned_table(page, 2, (72.0, 400.0, 300.0, 500.0))
    assert raw.cells == []
    assert validate_cells(raw.cells) is not None  # -> needs_review downstream


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
