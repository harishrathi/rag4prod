"""Stage 7 tests: dedup, heading levels, the walk, and chunking.

These run the local (non-YOLO) half of the real pipeline on the sample
PDF: stage-2 units + tier-1 tables, assembled and chunked. Scanned pages
are exercised at the unit level elsewhere; here the focus is the merge
semantics — the part interviewers probe with "so what happens when...".
"""

import pymupdf
import pytest

from rag_ingest.assemble import assign_heading_levels, build_walk, dedup_units
from rag_ingest.chunking import chunk_document, split_text
from rag_ingest.local_extract import extract
from rag_ingest.models import UnitType
from rag_ingest.sample_pdf import build_sample
from rag_ingest.tables import extract_native_tables, finalize
from rag_ingest.triage import triage


@pytest.fixture(scope="module")
def assembled(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    doc = pymupdf.open(build_sample(tmp_path_factory.mktemp("pdf") / "sample_doc.pdf"))
    kinds = {r.page: r.kind for r in triage(doc)}
    _, units, _ = extract(doc, kinds, out)
    raw = []
    for p in (6, 8, 9):
        raw.extend(extract_native_tables(doc.load_page(p), p))
    tables = finalize(raw, {p: doc.load_page(p).rect.height for p in kinds}, doc, out)
    walk = build_walk(units, tables)
    chunks, merged_md = chunk_document("sample", walk, rows_per_chunk=5)
    doc.close()
    return {"units": units, "tables": tables, "walk": walk, "chunks": chunks, "md": merged_md}


# --- Dedup -------------------------------------------------------------------


def test_table_cell_text_deduped_but_survives_in_table_chunk(assembled):
    text_chunks = [c for c in assembled["chunks"] if c.type == UnitType.TEXT]
    table_chunks = [c for c in assembled["chunks"] if c.type == UnitType.TABLE]
    # "Excavation" lives inside a table region: must NOT appear as prose...
    assert not any("Excavation" in c.content for c in text_chunks)
    # ...but must survive inside a table chunk.
    assert any("Excavation" in c.content for c in table_chunks)


def test_dedup_uses_center_not_intersection(assembled):
    # The prose paragraph above the page-6 table merely NEIGHBORS the
    # table bbox; its center is outside, so it must survive dedup.
    kept = dedup_units(assembled["units"], assembled["tables"])
    page6_text = [u for u in kept if u.page == 6 and u.type == UnitType.TEXT]
    assert any("supplier shall" in u.content for u in page6_text)


# --- Heading levels ----------------------------------------------------------


def test_numbered_headings_are_authoritative(assembled):
    units = assembled["units"]
    levels = assign_heading_levels(units)
    by_content = {units[i].content: lvl for i, lvl in levels.items()}
    assert by_content["7. Payment Terms"] == 1
    assert by_content["7.3 Liquidated Damages"] == 2
    assert by_content["7.3.1 Delay Notices"] == 3  # 10pt bold — size says body!


def test_unnumbered_heading_gets_size_rank(assembled):
    units = assembled["units"]
    levels = assign_heading_levels(units)
    by_content = {units[i].content: lvl for i, lvl in levels.items()}
    # Native title sizes rank 18 > 14 > 13, so 13pt "Background" is 3rd.
    assert by_content["Background"] == 3


# --- Chunks ------------------------------------------------------------------


def test_pages_are_one_based_in_chunks(assembled):
    for c in assembled["chunks"]:
        assert c.pages and min(c.pages) >= 1
    # Page-0 prose must be cited as page 1.
    first_text = next(c for c in assembled["chunks"] if c.type == UnitType.TEXT)
    assert first_text.pages == [1]


def test_breadcrumb_only_in_embedding_text(assembled):
    c = next(
        c
        for c in assembled["chunks"]
        if c.type == UnitType.TEXT and "7.3 Liquidated Damages" in c.headings
    )
    assert c.embedding_text.startswith("[7. Payment Terms (contd.) > 7.3 Liquidated Damages]")
    assert not c.content.startswith("[")  # display text stays clean


def test_big_table_split_into_row_groups_with_header(assembled):
    groups = [c for c in assembled["chunks"] if c.table_id == "t0008_00"]
    # 10 data rows at 5 per group -> 2 chunks, each carrying the header.
    assert len(groups) == 2
    assert all(c.content.splitlines()[0].startswith("| Item |") for c in groups)
    assert "Formwork" in groups[0].content and "Drainage" in groups[1].content
    assert all(c.pages == [9, 10] for c in groups)  # 1-based citation


def test_figure_chunk_carries_storage_key(assembled):
    figs = [c for c in assembled["chunks"] if c.type == UnitType.FIGURE]
    assert figs and figs[0].storage_key and figs[0].pages == [1]


def test_merged_md_contains_tables_and_headings(assembled):
    assert "# 7. Payment Terms" in assembled["md"]
    assert "| Item |" in assembled["md"]


# --- Splitter ----------------------------------------------------------------


def test_split_text_respects_sentences_and_token_limit():
    text = "One sentence here. " * 300  # ~1200 gpt2 tokens
    pieces = split_text(text)
    assert len(pieces) >= 2
    assert all(t.rstrip().endswith(".") for t, _ in pieces)  # never mid-sentence
    assert all(tokens <= 512 for _, tokens in pieces)  # REAL limit, not estimate
