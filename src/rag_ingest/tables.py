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

import numpy as np
import pymupdf

from .config import (
    FIGURE_DPI,
    GRID_DARK_THRESHOLD,
    GRID_LINE_MIN_COVERAGE,
    HEADER_MATCH_RATIO,
    RENDER_DPI,
    TABLE_CONT_BOTTOM_FRAC,
    TABLE_CONT_TOP_FRAC,
)
from .models import BBox

log = logging.getLogger(__name__)


@dataclass
class RawTable:
    """One table fragment on one page, before stitching/validation."""

    page: int
    bbox: BBox
    cells: list[list[str]]  # [] when a region was detected but not parsed
    source: str  # "find_tables" | "grid_ocr" | "yolo_only"


@dataclass
class TableResult:
    """A finished (possibly multi-page) table."""

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

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["page_spans"] = [[p, list(b)] for p, b in self.page_spans]
        return d


# ---------------------------------------------------------------------------
# Tier 1 — bordered, text-native: vector lines + exact text
# ---------------------------------------------------------------------------


def extract_native_tables(page: pymupdf.Page, page_index: int) -> list[RawTable]:
    out: list[RawTable] = []
    for tab in page.find_tables().tables:
        cells = [[(c or "").strip() for c in row] for row in tab.extract()]
        out.append(
            RawTable(page=page_index, bbox=tuple(tab.bbox), cells=cells, source="find_tables")
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
    # table-width/crop-width ratio, well above it.
    h_centers = _line_centers(dark.mean(axis=1) > GRID_LINE_MIN_COVERAGE)
    v_centers = _line_centers(dark.mean(axis=0) > GRID_LINE_MIN_COVERAGE)
    if len(h_centers) < 2 or len(v_centers) < 2:
        log.debug("p%04d: no pixel grid in region %s", page_index, region)
        return RawTable(page=page_index, bbox=region, cells=[], source="grid_ocr")

    # Erase the grid lines (±2 px band around each detected line), then
    # OCR the cleaned crop. Pixmap.pdfocr_tobytes wraps the image in a
    # 1-page PDF with a Tesseract text layer; its page coordinates are
    # crop PIXELS (1 px = 1 pt), so cell assignment happens in pixel
    # space against the raw line centers.
    clean = arr.copy()
    for c in h_centers:
        clean[max(0, int(c) - 2) : int(c) + 3, :] = 255
    for c in v_centers:
        clean[:, max(0, int(c) - 2) : int(c) + 3] = 255
    clean_pix = pymupdf.Pixmap(pymupdf.csRGB, pix.width, pix.height, clean.tobytes(), False)
    ocr_doc = pymupdf.open(
        "pdf", clean_pix.pdfocr_tobytes(language="eng", tessdata=ensure_tessdata())
    )

    # The wrapper PDF's page is NOT guaranteed to be 1 px = 1 pt (pdfocr
    # picks its own page scale) — same trap as stage 4, same medicine:
    # derive the word->pixel scale from actual dimensions, never assume.
    opage = ocr_doc[0]
    wx = pix.width / opage.rect.width
    wy = pix.height / opage.rect.height

    rows, cols = len(h_centers) - 1, len(v_centers) - 1
    buckets: list[list[list[str]]] = [[[] for _ in range(cols)] for _ in range(rows)]
    for x0, y0, x1, y1, word, *_ in opage.get_text("words"):
        cx, cy = (x0 + x1) / 2 * wx, (y0 + y1) / 2 * wy
        row = next((i for i in range(rows) if h_centers[i] <= cy < h_centers[i + 1]), None)
        col = next((j for j in range(cols) if v_centers[j] <= cx < v_centers[j + 1]), None)
        if row is not None and col is not None:
            buckets[row][col].append(word)
    ocr_doc.close()

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
    cells = [[" ".join(c) for c in r] for r in buckets]
    return RawTable(page=page_index, bbox=bbox, cells=cells, source="grid_ocr")


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


# ---------------------------------------------------------------------------
# Multi-page stitching
# ---------------------------------------------------------------------------


def _norm_row(row: list[str]) -> str:
    return " ".join(" ".join(row).lower().split())


def _is_continuation(prev: RawTable, nxt: RawTable, page_heights: dict[int, float]) -> bool:
    if nxt.page != prev.page + 1:
        return False
    if not prev.cells or not nxt.cells:
        return False
    if len(prev.cells[0]) != len(nxt.cells[0]):
        return False  # column mismatch: refuse — never guess at structure
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


def _merge_chain(chain: list[RawTable]) -> list[list[str]]:
    merged = [row[:] for row in chain[0].cells]
    header = _norm_row(chain[0].cells[0]) if chain[0].cells else ""
    for frag in chain[1:]:
        rows = frag.cells
        # Continuation pages often repeat the header — fuzzy match because
        # tier-2 fragments carry OCR noise ("Descripton").
        if (
            rows
            and header
            and SequenceMatcher(None, _norm_row(rows[0]), header).ratio() >= HEADER_MATCH_RATIO
        ):
            rows = rows[1:]
        merged.extend(row[:] for row in rows)
    return merged


# ---------------------------------------------------------------------------
# Finalize: stitch -> validate -> markdown -> review crops
# ---------------------------------------------------------------------------


def finalize(
    raw: list[RawTable],
    page_heights: dict[int, float],
    doc: pymupdf.Document,
    doc_out: Path,
) -> list[TableResult]:
    results: list[TableResult] = []
    for chain in stitch(raw, page_heights):
        first = chain[0]
        table_id = f"t{first.page:04d}_{sum(1 for t in results if t.pages[0] == first.page):02d}"
        cells = _merge_chain(chain)
        reason = validate_cells(cells)

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
