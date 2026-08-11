"""Paid-lane table extraction: the VLM already returned each table as a
GFM pipe table inside the page markdown; here that markdown becomes an
unmerged cell fragment on the same contract as every other extractor.

The VLM prompt requires merged cells to repeat their value in every
row/column they cover, so the fragment arrives ALREADY unmerged (#27's
contract, for free); the printed merge layout is not recoverable from
pipe markdown, so ``merges`` stays empty and merged.md renders the plain
pipe table."""

from __future__ import annotations

from rag_ingest.models import Unit
from rag_ingest.vlm_extract import markdown_table_cells

from . import RawTable


def vlm_table_fragment(unit: Unit) -> RawTable:
    """One paid-lane TABLE unit -> a raw fragment for the ladder."""
    return RawTable(
        page=unit.page,
        bbox=unit.bbox,
        cells=markdown_table_cells(unit.content),
        source="gemini",
    )
