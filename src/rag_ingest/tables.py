"""STAGE 6 — Tables: the tiered ladder + multi-page stitching.

Tiering principle (design spec §7): never run an expensive extractor on
content a cheaper one already handled, and never let any tier fail
silently.

  Tier 1  bordered table, TEXT_NATIVE page
          PyMuPDF find_tables(): vector grid lines + exact native text.
          Zero OCR, zero inference — cannot hallucinate.

  Paid lane  any table on a SCANNED/rerouted page
          Arrives from stage 5 as GFM markdown inside the page's VLM
          response; pipeline.py parses it to cells and feeds it through
          the SAME stitching/validation path (source="gemini").

  Fallback  anything failing validation
          needs_review=true + a stored crop PNG. The reviewer is a HUMAN.

Multi-page stitching: a table that runs into the bottom margin of page N
and resumes at the top of page N+1 with the same column count is ONE
table. Repeated headers on the continuation are dropped (fuzzy match —
paid-lane rows can carry transcription noise). A column-count mismatch
REFUSES to merge and flags both fragments: guessing at structure is how
silent corruption happens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import pymupdf

from .config import (
    FIGURE_DPI,
    FURNITURE_MIN_REPEATS,
    HEADER_MATCH_RATIO,
    TABLE_CONT_BOTTOM_FRAC,
    TABLE_CONT_TOP_FRAC,
)
from .models import BBox

log = logging.getLogger(__name__)


@dataclass
class RawTable:
    """One table fragment on one page, before stitching/validation.

    ``cells`` are UNMERGED (ledger #27): a merged cell's value is
    repeated into every grid position it covers, so no downstream
    consumer ever has to guess whether a blank means "merged" or
    "empty". The original merge layout is preserved separately in
    ``merges`` — [row, col, rowspan, colspan] per merged cell — so a
    consumer can reconstruct the exact printed table (cells_to_html does
    exactly that). ``header_rows`` is how many leading rows form the
    header — 2+ when header cells span rows (two-tier headers)."""

    page: int
    bbox: BBox
    cells: list[list[str]]  # [] when a region was detected but not parsed
    source: str  # "find_tables" | "gemini" | "yolo_only"
    header_rows: int = 1
    merges: list[list[int]] = field(default_factory=list)  # [row, col, rowspan, colspan]


@dataclass
class TableResult:
    """A finished (possibly multi-page) table.

    Three renderings of the same content, each for a different consumer:
    ``cells`` (unmerged matrix — the machine-readable truth, plus
    ``merges`` to reconstruct the printed layout), ``markdown`` (pipe
    table for retrieval chunks), and ``grid`` (ASCII box drawing that
    looks like the PDF — what merged.md embeds when merges exist)."""

    table_id: str
    pages: list[int]  # 0-based, ascending
    page_spans: list[tuple[int, BBox]]  # per-fragment location, for citation
    col_count: int
    row_count: int
    markdown: str
    source: str
    needs_review: bool = False
    review_reason: str | None = None
    crop_key: str | None = None  # stored crop PNG when needs_review
    cells: list[list[str]] = field(default_factory=list)
    header_rows: int = 1
    merges: list[list[int]] = field(default_factory=list)  # [row, col, rowspan, colspan]
    grid: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["page_spans"] = [[p, list(b)] for p, b in self.page_spans]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TableResult:
        """Rehydrate from a stage-6 artifact row (--from-stage resume)."""
        d = dict(d)
        d.pop("html", None)  # pre-grid artifacts carried an HTML rendering
        d["page_spans"] = [(p, tuple(b)) for p, b in d["page_spans"]]
        return cls(**d)


# ---------------------------------------------------------------------------
# Tier 1 — bordered, text-native: vector lines + exact text
# ---------------------------------------------------------------------------


def _unmerge_native(tab) -> tuple[list[list[str]], int, list[list[int]]]:
    """find_tables output -> unmerged cells + header depth (ledger #27).

    find_tables represents a merged cell as its anchor (top-left grid
    position, bbox spanning the whole merge) plus ``None`` for every
    covered position — distinct from ``""``, which is a genuinely empty
    cell. The naive ``(c or "").strip()`` erases that distinction; here
    each covered position is filled with its anchor's text instead, so a
    15-row category span reads on every row it governs. Header depth
    falls out of the same geometry: a span anchored in row 0 that covers
    row 1 means the header is (at least) two rows deep. Returns
    (cells, header_rows, merges) — merges as [row, col, rowspan, colspan].
    """
    texts = tab.extract()
    boxes = [row.cells for row in tab.rows]
    n_rows, n_cols = len(boxes), max((len(r) for r in boxes), default=0)
    out = [[("" if t is None else str(t)).strip() for t in row] for row in texts]

    # Grid edges, recovered from the geometry that IS there: every column
    # contributes at least one unmerged x0, every row one y0.
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
    # Extent of each anchor's merge, grown as covered positions resolve.
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


# ---------------------------------------------------------------------------
# (Tier 2 — the pixel-grid + Tesseract path — was deleted with its engine:
# scanned-page tables now arrive from the VLM lane as markdown. See
# docs/gemini_extractor_spec.md §6.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + markdown
# ---------------------------------------------------------------------------


def validate_cells(cells: list[list[str]]) -> str | None:
    """Reason the table is NOT trustworthy, or None if it passes. Cheap
    structural checks — the gate between 'extracted' and 'reviewed'."""
    if len(cells) < 2:
        return "fewer than 2 rows (no grid found, or header-only)"
    widths = {len(r) for r in cells}
    if len(widths) != 1:
        return f"ragged rows: column counts {sorted(widths)}"
    if not any(c for c in cells[0]):
        return "empty header row"
    filled = sum(1 for r in cells for c in r if c)
    if filled / (len(cells) * len(cells[0])) < 0.4:
        return "mostly empty cells (grid without content?)"
    return None


def cells_to_markdown(cells: list[list[str]]) -> str:
    esc = [[c.replace("|", "\\|") for c in row] for row in cells]
    lines = ["| " + " | ".join(esc[0]) + " |", "|" + "---|" * len(esc[0])]
    lines += ["| " + " | ".join(row) + " |" for row in esc[1:]]
    return "\n".join(lines)


def cells_to_grid(cells: list[list[str]], merges: list[list[int]]) -> str:
    """Visually faithful rendering: an ASCII box grid reconstructed from
    the unmerged matrix + merge list. A merged cell draws as ONE box —
    no interior rules, value shown once — so the text looks like the
    printed table. Pipe markdown cannot express spans; merged.md embeds
    this (inside a code fence, for monospace) whenever a table has them.
    """
    rows, cols = len(cells), len(cells[0]) if cells else 0
    if not rows or not cols:
        return ""
    anchor = {(m[0], m[1]): (m[2], m[3]) for m in merges}
    owner: dict[tuple[int, int], tuple[int, int]] = {}
    for r0, c0, rs, cs in merges:
        for r in range(r0, r0 + rs):
            for c in range(c0, c0 + cs):
                owner[(r, c)] = (r0, c0)

    def own(r: int, c: int) -> tuple[int, int]:
        return owner.get((r, c), (r, c))

    # Column widths from single-column content; a col-span wider than the
    # columns it covers widens the last one.
    width = [3] * cols
    for r in range(rows):
        for c in range(cols):
            if own(r, c) == (r, c) and anchor.get((r, c), (1, 1))[1] == 1:
                width[c] = max(width[c], len(cells[r][c]))
    for (r0, c0), (_rs, cs) in anchor.items():
        if cs > 1:
            span = sum(width[c0 : c0 + cs]) + 3 * (cs - 1)
            if len(cells[r0][c0]) > span:
                width[c0 + cs - 1] += len(cells[r0][c0]) - span

    def v_border(r: int, b: int) -> bool:
        """Vertical rule at column boundary b (0..cols) crossing row r."""
        return b in (0, cols) or own(r, b - 1) != own(r, b)

    def h_border(line: int, c: int) -> bool:
        """Horizontal rule above row `line` (0..rows) across column c."""
        return line in (0, rows) or own(line - 1, c) != own(line, c)

    out: list[str] = []
    for line in range(rows + 1):
        s = ""
        for b in range(cols + 1):
            left = b > 0 and h_border(line, b - 1)
            right = b < cols and h_border(line, b)
            up = line > 0 and v_border(line - 1, b)
            down = line < rows and v_border(line, b)
            s += "+" if (left or right) else ("|" if (up or down) else " ")
            if b < cols:
                s += ("-" if h_border(line, b) else " ") * (width[b] + 2)
        out.append(s.rstrip())
        if line == rows:
            break
        s, c = "", 0
        while c < cols:
            s += "|" if v_border(line, c) else " "
            r0, c0 = own(line, c)
            cs = anchor.get((r0, c0), (1, 1))[1]
            span = sum(width[c : c + cs]) + 3 * (cs - 1)
            text = cells[line][c] if (r0, c0) == (line, c) else ""
            s += " " + text.ljust(span) + " "
            c += cs
        out.append(s + "|")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Multi-page stitching
# ---------------------------------------------------------------------------


def _norm_row(row: list[str]) -> str:
    return " ".join(" ".join(row).lower().split())


def _full_page_bbox(t: RawTable, page_heights: dict[int, float]) -> bool:
    """Paid-lane tables that YOLO could not box carry the full-page rect
    (page-level provenance). Such a bbox says nothing about where the
    table actually sits, so the geometry tests below would vacuously
    pass — real tables never start at literal y=0."""
    h = page_heights.get(t.page, 842.0)
    return t.bbox[1] <= 1.0 and t.bbox[3] >= h - 1.0


def _is_continuation(prev: RawTable, nxt: RawTable, page_heights: dict[int, float]) -> bool:
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
    exits_bottom = prev.bbox[3] >= prev_h * TABLE_CONT_BOTTOM_FRAC
    enters_top = nxt.bbox[1] <= next_h * TABLE_CONT_TOP_FRAC
    return exits_bottom and enters_top


def stitch(raw: list[RawTable], page_heights: dict[int, float]) -> list[list[RawTable]]:
    """Group fragments into chains. Chains are built greedily over the
    page-sorted list; each fragment either continues the chain whose tail
    it follows, or starts its own."""
    chains: list[list[RawTable]] = []
    for frag in sorted(raw, key=lambda t: (t.page, t.bbox[1])):
        tail = chains[-1][-1] if chains else None
        if tail is not None and _is_continuation(tail, frag, page_heights):
            chains[-1].append(frag)
        else:
            chains.append([frag])
    return chains


def _fill_continued_spans(merged: list[list[str]], start: int) -> None:
    """Print convention: a row-span crossing a page break leaves its cells
    BLANK on the continuation page — the label exists only on the page
    where the span started. Evidence required before filling: the merged
    rows must END with a run (>= 2) of identical non-empty values in that
    column, i.e. an unmerged span, not a coincidental value. A column of
    unique values (item numbers, rates) never qualifies, so genuinely
    empty leading cells stay empty. Ledger #27 records the residual risk
    (two identical adjacent data values enable a wrong fill)."""
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


def _merge_chain(chain: list[RawTable]) -> tuple[list[list[str]], list[list[int]]]:
    """Chain of fragments -> (merged cells, merges with re-based rows)."""
    head = chain[0]
    merged = [row[:] for row in head.cells]
    merges = [m[:] for m in head.merges]
    header_norms = [_norm_row(r) for r in head.cells[: head.header_rows]] if head.cells else []
    for frag in chain[1:]:
        rows = frag.cells
        dropped = 0
        # Continuation pages often repeat the header (all of it — two-tier
        # headers repeat both rows) — fuzzy match because tier-2 fragments
        # carry OCR noise ("Descripton").
        for h in header_norms:
            if rows and SequenceMatcher(None, _norm_row(rows[0]), h).ratio() >= HEADER_MATCH_RATIO:
                rows = rows[1:]
                dropped += 1
        if not rows:
            continue
        start = len(merged)
        merged.extend(row[:] for row in rows)
        for r0, c0, rs, cs in frag.merges:
            # Re-base onto the merged table; clip away the part of a merge
            # that lived in the dropped repeated-header rows.
            end = r0 + rs - 1
            if end < dropped:
                continue
            new_r0 = max(r0, dropped)
            merges.append([new_r0 - dropped + start, c0, end - new_r0 + 1, cs])
        _fill_continued_spans(merged, start)
    return merged, merges


# ---------------------------------------------------------------------------
# Finalize: stitch -> validate -> markdown -> review crops
# ---------------------------------------------------------------------------


def _suppress_repeated_suspects(raw: list[RawTable]) -> list[RawTable]:
    """Drop yolo_only suspects that recur at the same position on many
    pages (ledger #30). A real borderless table lives at one place in a
    document; a bordered page-title box lives at the same spot on EVERY
    page — YOLO calls it a table each time, and each one would open a
    needs_review item for a human. Repetition across
    FURNITURE_MIN_REPEATS distinct pages is the tell."""
    groups: dict[tuple[int, ...], list[RawTable]] = {}
    for t in raw:
        if t.source == "yolo_only":
            groups.setdefault(tuple(round(v / 10) for v in t.bbox), []).append(t)
    drop = {
        id(t)
        for g in groups.values()
        if len({t.page for t in g}) >= FURNITURE_MIN_REPEATS
        for t in g
    }
    if drop:
        log.info("suppressed %d repeated page-furniture table suspect(s)", len(drop))
    return [t for t in raw if id(t) not in drop]


def finalize(
    raw: list[RawTable],
    page_heights: dict[int, float],
    doc: pymupdf.Document,
    doc_out: Path,
) -> list[TableResult]:
    raw = _suppress_repeated_suspects(raw)
    results: list[TableResult] = []
    for chain in stitch(raw, page_heights):
        first = chain[0]
        table_id = f"t{first.page:04d}_{sum(1 for t in results if t.pages[0] == first.page):02d}"
        cells, merges = _merge_chain(chain)
        reason = validate_cells(cells)
        if reason is None and first.source == "find_tables":
            # Native cells can carry mojibake on pages whose text layer is
            # only mildly broken (below triage's reroute threshold,
            # ledger #29) — the same junk-char tell flags them for review.
            from .triage import JUNK_CHARS_RE

            n_junk = sum(1 for row in cells for c in row if JUNK_CHARS_RE.search(c))
            if n_junk:
                reason = f"text layer junk in {n_junk} cell(s) — broken font encoding"

        result = TableResult(
            table_id=table_id,
            pages=[f.page for f in chain],
            page_spans=[(f.page, f.bbox) for f in chain],
            col_count=len(cells[0]) if cells else 0,
            row_count=len(cells),
            markdown=cells_to_markdown(cells) if reason is None else "",
            source=first.source,
            needs_review=reason is not None,
            review_reason=reason,
            cells=cells,
            header_rows=first.header_rows,
            merges=merges,
            grid=cells_to_grid(cells, merges) if reason is None else "",
        )
        if result.needs_review:
            # The fallback tier is a human: store what a reviewer needs to
            # fix it — the exact region, full quality.
            key = f"figures/{table_id}_review.png"
            pix = doc.load_page(first.page).get_pixmap(
                clip=pymupdf.Rect(first.bbox), dpi=FIGURE_DPI
            )
            (doc_out / key).parent.mkdir(parents=True, exist_ok=True)
            (doc_out / key).write_bytes(pix.tobytes("png"))
            result.crop_key = key
        results.append(result)

    n_review = sum(1 for t in results if t.needs_review)
    log.info(
        "tables: %d fragment(s) -> %d table(s), %d multi-page, %d need review",
        len(raw),
        len(results),
        sum(1 for t in results if len(t.pages) > 1),
        n_review,
    )
    return results


def overlap_frac(a: BBox, b: BBox) -> float:
    """Intersection area over the smaller box's area — for matching YOLO
    regions against find_tables results."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller = min(area_a, area_b)
    return (ix * iy) / smaller if smaller > 0 else 0.0
