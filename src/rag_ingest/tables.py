"""STAGE 6 — Tables: the tiered ladder + multi-page stitching.

Tiering principle (design spec §7): never run an expensive extractor on
content a cheaper one already handled, and never let any tier fail
silently.

  Tier 1  bordered table, TEXT_NATIVE page
          PyMuPDF find_tables(): vector grid lines + exact native text.
          Zero OCR, zero inference — cannot hallucinate.

  Tier 2  bordered table, SCANNED page
          The grid exists only as pixels: detect line rows/columns by ink
          coverage in the rendered crop, intersect into cells, then fill
          cells with the words the stage-5 OCR textpage already produced
          (no re-OCR — words are assigned to cells by center containment).

  Fallback  anything failing validation
          needs_review=true + a stored crop PNG. The reviewer is a HUMAN.
          This slot is where a VLM tier would plug in if the corpus ever
          grows borderless tables (spec §7 records why it was dropped).

Multi-page stitching: a table that runs into the bottom margin of page N
and resumes at the top of page N+1 with the same column count is ONE
table. Repeated headers on the continuation are dropped (fuzzy match —
tier-2 rows carry OCR noise). A column-count mismatch REFUSES to merge
and flags both fragments: guessing at structure is how silent corruption
happens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import cast

import numpy as np
import pymupdf

from .config import (
    FIGURE_DPI,
    FURNITURE_MIN_REPEATS,
    GRID_DARK_THRESHOLD,
    GRID_LINE_MIN_COVERAGE,
    HEADER_MATCH_RATIO,
    OCR_LANGUAGES,
    OCR_MIN_QUALITY,
    RENDER_DPI,
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
# Tier 2 — bordered, scanned: pixel grid + OCR word assignment
# ---------------------------------------------------------------------------


def _line_centers(mask: np.ndarray) -> list[float]:
    """Centers of consecutive True runs — a 2-3 px thick grid line becomes
    one center instead of 2-3 separate 'lines'."""
    centers: list[float] = []
    start: int | None = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            centers.append((start + i - 1) / 2)
            start = None
    if start is not None:
        centers.append((start + len(mask) - 1) / 2)
    return centers


def drop_empty_lines(
    cells: list[list[str]], merges: list[list[int]], header_rows: int
) -> tuple[list[list[str]], list[list[int]], int]:
    """Remove all-empty rows/columns from a grid-OCR matrix (ledger #29).

    Vector-crisp pages rerouted to the OCR path (broken text layers)
    have page borders and table borders millimeters apart — the pixel
    grid reads the space between them as extra rows/columns of nothing.
    An all-empty line carries no information, so dropping it is safe;
    merges are remapped (and clipped) onto the kept indices."""
    if not cells:
        return cells, merges, header_rows
    keep_r = [i for i, row in enumerate(cells) if any(c.strip() for c in row)]
    keep_c = [j for j in range(len(cells[0])) if any(row[j].strip() for row in cells)]
    if len(keep_r) == len(cells) and len(keep_c) == len(cells[0]):
        return cells, merges, header_rows
    rmap = {old: new for new, old in enumerate(keep_r)}
    cmap = {old: new for new, old in enumerate(keep_c)}
    new_cells = [[cells[r][c] for c in keep_c] for r in keep_r]
    new_merges: list[list[int]] = []
    for r0, c0, rs, cs in merges:
        rows = [rmap[r] for r in range(r0, r0 + rs) if r in rmap]
        cols = [cmap[c] for c in range(c0, c0 + cs) if c in cmap]
        if rows and cols and (len(rows) > 1 or len(cols) > 1):
            new_merges.append([rows[0], cols[0], len(rows), len(cols)])
    new_header = max(1, sum(1 for r in range(header_rows) if r in rmap))
    return new_cells, new_merges, new_header


def extract_scanned_table(page: pymupdf.Page, page_index: int, region: BBox) -> RawTable:
    """Grid-from-pixels + line-removal + region OCR for one YOLO table
    region on a scanned page.

    Why not reuse the stage-5 full-page OCR words: Tesseract's layout
    analysis treats tightly ruled regions as non-text and SILENTLY DROPS
    the cell contents (found live — a full-page OCR of the sample's
    scanned table returned only the heading above it). The standard fix
    is line-removal preprocessing, and tier 2 is perfectly positioned for
    it: the grid detection has already located every line, so erasing
    them costs nothing — then a clean OCR pass reads the naked cell text,
    and the words are re-anchored into cells using the grid geometry we
    kept.

    Returns cells=[] (source intact) when no grid is found — validation
    downstream turns that into needs_review + crop, never a crash.
    """
    from .ocr import ensure_tessdata  # local import: avoid cycle at module load

    clip = pymupdf.Rect(region)
    pix = page.get_pixmap(clip=clip, dpi=RENDER_DPI)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).copy()
    dark = arr.mean(axis=2) < GRID_DARK_THRESHOLD

    # A pixel row/column is a grid line when enough of it is ink. Dense
    # text rows peak around ~40% coverage; solid rules sit near the
    # table-width/crop-width ratio, well above it. Merged cells only dent
    # this: a boundary interrupted by a row-span still clears the bar as
    # long as the span covers < ~half the table width (ledger #27 records
    # the giant-merge case where it wouldn't).
    h_centers = _line_centers(dark.mean(axis=1) > GRID_LINE_MIN_COVERAGE)
    v_centers = _line_centers(dark.mean(axis=0) > GRID_LINE_MIN_COVERAGE)
    if len(h_centers) < 2 or len(v_centers) < 2:
        log.debug("p%04d: no pixel grid in region %s", page_index, region)
        return RawTable(page=page_index, bbox=region, cells=[], source="grid_ocr")

    rows, cols = len(h_centers) - 1, len(v_centers) - 1

    # --- Per-segment border presence (ledger #27) -----------------------
    # The uniform grid says where boundaries CAN be; merged cells are the
    # boundaries that aren't there. Each candidate segment is checked for
    # ink individually (trimmed a few px at the ends so crossing rules
    # don't vote). Everything downstream — line erasure, word assignment,
    # merge structure — keys off these presence maps.
    def _h_border(line: int, col: int) -> bool:
        lo, hi = max(0, int(h_centers[line]) - 2), int(h_centers[line]) + 3
        x0, x1 = int(v_centers[col]) + 3, int(v_centers[col + 1]) - 2
        if x1 <= x0:
            return True
        return bool(dark[lo:hi, x0:x1].any(axis=0).mean() > GRID_LINE_MIN_COVERAGE)

    def _v_border(line: int, row: int) -> bool:
        lo, hi = max(0, int(v_centers[line]) - 2), int(v_centers[line]) + 3
        y0, y1 = int(h_centers[row]) + 3, int(h_centers[row + 1]) - 2
        if y1 <= y0:
            return True
        return bool(dark[y0:y1, lo:hi].any(axis=1).mean() > GRID_LINE_MIN_COVERAGE)

    # Union-find over grid cells: neighbors with no border between them
    # are one logical (merged) cell.
    parent = list(range(rows * cols))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(a: int, b: int) -> None:
        parent[_find(a)] = _find(b)

    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows and not _h_border(r + 1, c):
                _union(r * cols + c, (r + 1) * cols + c)
            if c + 1 < cols and not _v_border(c + 1, r):
                _union(r * cols + c, r * cols + c + 1)

    # Erase only borders that exist. Blanket erasure along a candidate
    # line would cut through the middle of a merged cell — exactly where
    # its vertically-centered label sits.
    clean = arr.copy()
    for line in range(len(h_centers)):
        lo, hi = max(0, int(h_centers[line]) - 2), int(h_centers[line]) + 3
        for c in range(cols):
            if line in (0, rows) or _h_border(line, c):
                clean[lo:hi, int(v_centers[c]) - 2 : int(v_centers[c + 1]) + 3] = 255
    for line in range(len(v_centers)):
        lo, hi = max(0, int(v_centers[line]) - 2), int(v_centers[line]) + 3
        for r in range(rows):
            if line in (0, cols) or _v_border(line, r):
                clean[int(h_centers[r]) - 2 : int(h_centers[r + 1]) + 3, lo:hi] = 255

    clean_pix = pymupdf.Pixmap(pymupdf.csRGB, pix.width, pix.height, clean.tobytes(), False)
    ocr_doc = pymupdf.open(
        "pdf", clean_pix.pdfocr_tobytes(language=OCR_LANGUAGES, tessdata=ensure_tessdata())
    )

    # The wrapper PDF's page is NOT guaranteed to be 1 px = 1 pt (pdfocr
    # picks its own page scale) — same trap as stage 4, same medicine:
    # derive the word->pixel scale from actual dimensions, never assume.
    opage = ocr_doc.load_page(0)
    wx = pix.width / opage.rect.width
    wy = pix.height / opage.rect.height

    # Words bucket into GRID cells first (with their position, so merged
    # regions can be re-joined in reading order afterwards).
    buckets: list[list[list[tuple[float, float, str]]]] = [
        [[] for _ in range(cols)] for _ in range(rows)
    ]
    # get_text("words") returns (x0, y0, x1, y1, word, ...) tuples; the
    # stubs type the mode union loosely, so narrow it explicitly.
    words = cast(list[tuple[float, float, float, float, str]], opage.get_text("words"))
    for x0, y0, x1, y1, word, *_ in words:
        cx, cy = (x0 + x1) / 2 * wx, (y0 + y1) / 2 * wy
        row = next((i for i in range(rows) if h_centers[i] <= cy < h_centers[i + 1]), None)
        col = next((j for j in range(cols) if v_centers[j] <= cx < v_centers[j + 1]), None)
        if row is not None and col is not None:
            buckets[row][col].append((cy, cx, word))
    ocr_doc.close()

    # Each merged region's words join once (reading order) and the value
    # lands in EVERY grid position the region covers — the same unmerged
    # contract as tier 1, so stitching and chunking never see the
    # difference between tiers.
    regions: dict[int, list[tuple[float, float, str]]] = {}
    members: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            root = _find(r * cols + c)
            regions.setdefault(root, []).extend(buckets[r][c])
            members.setdefault(root, []).append((r, c))
    cells = [["" for _ in range(cols)] for _ in range(rows)]
    header_rows = 1
    for r in range(rows):
        for c in range(cols):
            root = _find(r * cols + c)
            cells[r][c] = " ".join(w for _, _, w in sorted(regions[root]))
    merges: list[list[int]] = []
    for _root, cells_of in sorted(members.items()):
        if len(cells_of) < 2:
            continue
        r0 = min(r for r, _ in cells_of)
        c0 = min(c for _, c in cells_of)
        r1 = max(r for r, _ in cells_of)
        c1 = max(c for _, c in cells_of)
        merges.append([r0, c0, r1 - r0 + 1, c1 - c0 + 1])
        if r0 == 0:
            header_rows = max(header_rows, r1 + 1)

    # Grid extent -> PDF points for the output bbox: same actual-dimensions
    # rule as stage 4's pixel_rect_to_pdf, with the crop origin as offset.
    sx = clip.width / pix.width
    sy = clip.height / pix.height
    bbox: BBox = (
        clip.x0 + v_centers[0] * sx,
        clip.y0 + h_centers[0] * sy,
        clip.x0 + v_centers[-1] * sx,
        clip.y0 + h_centers[-1] * sy,
    )
    cells, merges, header_rows = drop_empty_lines(cells, merges, header_rows)
    return RawTable(
        page=page_index,
        bbox=bbox,
        cells=cells,
        source="grid_ocr",
        header_rows=header_rows,
        merges=merges,
    )


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
        if reason is None and first.source == "grid_ocr":
            # OCR garbage gate (ledger #28): a structurally valid grid full
            # of symbol soup (sideways scan, dirt) must not ship confident.
            from .ocr import ocr_quality_score  # local import, matches ensure_tessdata

            quality = ocr_quality_score(" ".join(c for row in cells for c in row))
            if quality < OCR_MIN_QUALITY:
                reason = f"OCR quality {quality:.2f} below {OCR_MIN_QUALITY} (garbage text?)"
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
