"""Native table extraction: find_tables + span geometry -> unmerged cell
grid (#27). Logic identical to v1's tier 1 — it cannot hallucinate
(vector grid lines + exact native text, zero OCR, zero inference)."""

from __future__ import annotations

import logging

import pymupdf

from . import RawTable

log = logging.getLogger(__name__)


def _unmerge_native(tab) -> tuple[list[list[str]], int, list[list[int]]]:
    """find_tables output -> unmerged cells + header depth (#27).

    find_tables represents a merged cell as its anchor (top-left grid
    position, bbox spanning the whole merge) plus ``None`` for every
    covered position — distinct from ``""``, a genuinely empty cell.
    Each covered position is filled with its anchor's text, so a 15-row
    category span reads on every row it governs. Header depth falls out
    of the same geometry. Returns (cells, header_rows, merges)."""
    texts = tab.extract()
    boxes = [row.cells for row in tab.rows]
    n_rows, n_cols = len(boxes), max((len(r) for r in boxes), default=0)
    out = [[("" if t is None else str(t)).strip() for t in row] for row in texts]

    xs = sorted({round(b[0], 1) for row in boxes for b in row if b})
    ys = sorted({round(b[1], 1) for row in boxes for b in row if b})
    if len(xs) != n_cols or len(ys) != n_rows:
        return out, 1, []  # ambiguous geometry: keep print-convention blanks

    xs.append(max(b[2] for row in boxes for b in row if b))
    ys.append(max(b[3] for row in boxes for b in row if b))
    anchors = [
        (r, c, boxes[r][c])
        for r in range(n_rows)
        for c in range(len(boxes[r]))
        if boxes[r][c] is not None
    ]
    extents: dict[tuple[int, int], list[int]] = {}  # (ar, ac) -> [max_r, max_c]
    header_rows = 1
    for r in range(n_rows):
        for c in range(len(boxes[r])):
            if boxes[r][c] is not None:
                continue
            cx, cy = (xs[c] + xs[c + 1]) / 2, (ys[r] + ys[r + 1]) / 2
            for ar, ac, b in anchors:
                if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                    out[r][c] = out[ar][ac]
                    ext = extents.setdefault((ar, ac), [ar, ac])
                    ext[0], ext[1] = max(ext[0], r), max(ext[1], c)
                    if ar == 0:
                        header_rows = max(header_rows, r + 1)
                    break
    merges = [
        [ar, ac, mr - ar + 1, mc - ac + 1] for (ar, ac), (mr, mc) in sorted(extents.items())
    ]
    return out, header_rows, merges


def extract_native_tables(page: pymupdf.Page, page_index: int) -> list[RawTable]:
    """Every native-routed page runs page-wide find_tables — the vector
    lines are authoritative where they exist."""
    out: list[RawTable] = []
    finder = page.find_tables()
    for tab in finder.tables if finder is not None else []:
        cells, header_rows, merges = _unmerge_native(tab)
        out.append(
            RawTable(
                page=page_index,
                bbox=tuple(tab.bbox),
                cells=cells,
                source="find_tables",
                header_rows=header_rows,
                merges=merges,
            )
        )
    if out:
        log.debug("p%04d: find_tables -> %d table(s)", page_index, len(out))
    return out
