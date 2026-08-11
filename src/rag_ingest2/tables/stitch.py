"""Cross-page table stitching: continuation detection (equal-column rule,
#21), repeated-header dedup, and print-convention span fill (#27). Logic
identical to v1; a column-count mismatch REFUSES to merge — guessing at
structure is how silent corruption happens."""

from __future__ import annotations

from difflib import SequenceMatcher

from ..config import TableConfig
from . import RawTable


def _norm_row(row: list[str]) -> str:
    return " ".join(" ".join(row).lower().split())


def _full_page_bbox(t: RawTable, page_heights: dict[int, float]) -> bool:
    """Paid-lane tables that YOLO could not box carry the full-page rect
    (page-level provenance) — such a bbox proves nothing about geometry,
    and real tables never start at literal y=0."""
    h = page_heights.get(t.page, 842.0)
    return t.bbox[1] <= 1.0 and t.bbox[3] >= h - 1.0


def _is_continuation(
    prev: RawTable, nxt: RawTable, page_heights: dict[int, float], cfg: TableConfig
) -> bool:
    if nxt.page != prev.page + 1:
        return False
    if not prev.cells or not nxt.cells:
        return False
    if len(prev.cells[0]) != len(nxt.cells[0]):
        return False  # column mismatch: refuse — never guess at structure
    if _full_page_bbox(prev, page_heights) or _full_page_bbox(nxt, page_heights):
        return False  # no real geometry to test: refuse rather than guess
    prev_h = page_heights.get(prev.page, 842.0)
    next_h = page_heights.get(nxt.page, 842.0)
    exits_bottom = prev.bbox[3] >= prev_h * cfg.cont_bottom_frac
    enters_top = nxt.bbox[1] <= next_h * cfg.cont_top_frac
    return exits_bottom and enters_top


def stitch(
    raw: list[RawTable], page_heights: dict[int, float], cfg: TableConfig
) -> list[list[RawTable]]:
    """Group fragments into chains, greedily over the page-sorted list."""
    chains: list[list[RawTable]] = []
    for frag in sorted(raw, key=lambda t: (t.page, t.bbox[1])):
        tail = chains[-1][-1] if chains else None
        if tail is not None and _is_continuation(tail, frag, page_heights, cfg):
            chains[-1].append(frag)
        else:
            chains.append([frag])
    return chains


def _fill_continued_spans(merged: list[list[str]], start: int) -> None:
    """Print convention: a row-span crossing a page break leaves its cells
    BLANK on the continuation page. Evidence required before filling: the
    merged rows must END with a run (>= 2) of identical non-empty values
    in that column — a column of unique values never qualifies (#27
    records the residual risk)."""
    if start < 2:
        return
    n_cols = min(len(r) for r in merged)
    for c in range(n_cols):
        val = merged[start - 1][c]
        if not val or merged[start - 2][c] != val:
            continue
        i = start
        while i < len(merged) and merged[i][c] == "":
            merged[i][c] = val
            i += 1


def merge_chain(
    chain: list[RawTable], cfg: TableConfig
) -> tuple[list[list[str]], list[list[int]]]:
    """Chain of fragments -> (merged cells, merges with re-based rows).
    Continuation pages often repeat the header (all of it — two-tier
    headers repeat both rows); matched fuzzily because paid-lane
    fragments can carry transcription noise."""
    head = chain[0]
    merged = [row[:] for row in head.cells]
    merges = [m[:] for m in head.merges]
    header_norms = [_norm_row(r) for r in head.cells[: head.header_rows]] if head.cells else []
    for frag in chain[1:]:
        rows = frag.cells
        dropped = 0
        for h in header_norms:
            if rows and (
                SequenceMatcher(None, _norm_row(rows[0]), h).ratio() >= cfg.header_match_ratio
            ):
                rows = rows[1:]
                dropped += 1
        if not rows:
            continue
        start = len(merged)
        merged.extend(row[:] for row in rows)
        for r0, c0, rs, cs in frag.merges:
            end = r0 + rs - 1
            if end < dropped:
                continue
            new_r0 = max(r0, dropped)
            merges.append([new_r0 - dropped + start, c0, end - new_r0 + 1, cs])
        _fill_continued_spans(merged, start)
    return merged, merges
