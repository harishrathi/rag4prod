"""STAGE 2 — Local extraction: everything a TEXT_NATIVE page gives us for free.

This stage runs only on pages triage classified TEXT_NATIVE. It produces:

  * TEXT units   — paragraphs, with exact bboxes
  * TITLE units  — heading lines, carrying their raw font size for the
                   document-wide level clustering in stage 6
  * FIGURE units — embedded raster images, cropped to PNG and stored;
                   the PNG is the artifact, nothing is sent to a vision API
  * RuledGrid records — pages whose vector drawings form a table-like grid;
                   used later as a cross-check on YOLO's table boxes

Granularity decision (interview favourite): PyMuPDF offers spans (same-font
runs), lines, and blocks. Emitting per *span* fragments a sentence with one
bold word into three units; emitting per *block* merges a heading with its
following paragraph when they share a block. We classify per LINE (a line
is either all heading or all body in practice) and then merge consecutive
body lines of a block into one paragraph unit — so units line up with how
a human would segment the page.

Heading rule, per line:
    dominant-span size >= body_size * HEADING_SIZE_RATIO
    OR (dominant span is bold AND the line matches HEADING_NUMBERED_RE)

Body font size is computed once per document: sample text pages spread
across the whole file, histogram characters per rounded span size, take the
mode by CHARACTER count. Span count would be wrong: a page has many small
spans of header/footer/caption text, but body text dominates by characters.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pymupdf

from .config import (
    BODY_FONT_SAMPLE_PAGES,
    FIGURE_DPI,
    FIGURE_MIN_AREA_FRAC,
    HEADING_NUMBERED_RE,
    HEADING_SIZE_RATIO,
    RULED_MIN_H_SEGMENTS,
    RULED_MIN_V_SEGMENTS,
)
from .models import BBox, PageKind, Source, Unit, UnitType

log = logging.getLogger(__name__)

_NUMBERED = re.compile(HEADING_NUMBERED_RE)
_BOLD_FLAG = 16  # bit 4 of span["flags"] in PyMuPDF's dict output


def _bbox(rect: pymupdf.Rect) -> BBox:
    """Rect -> plain tuple. The stubs don't declare Rect.__iter__, so
    tuple(rect) type-checks as Unknown; explicit fields keep it typed."""
    return (rect.x0, rect.y0, rect.x1, rect.y1)


@dataclass
class RuledGrid:
    """Evidence that a page's vector drawings form a table-like grid.
    Cross-check data for stage 4 — never a router (page borders and
    letterhead rules look exactly like this to a segment counter)."""

    page: int
    h_segments: int
    v_segments: int
    bbox: BBox  # union of the counted segments, PDF points

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Body font size
# ---------------------------------------------------------------------------


def estimate_body_font_size(doc: pymupdf.Document, text_pages: list[int]) -> float:
    """Mode of span font sizes weighted by character count, sampled from
    text-native pages spread evenly across the document."""
    if not text_pages:
        return 10.0  # nothing to measure; a sane default for the logs

    step = max(1, len(text_pages) // BODY_FONT_SAMPLE_PAGES)
    chars_per_size: Counter[int] = Counter()
    for pno in text_pages[::step][:BODY_FONT_SAMPLE_PAGES]:
        d = cast(dict, doc.load_page(pno).get_text("dict"))
        for block in d["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    chars_per_size[round(span["size"])] += len(span["text"].strip())

    if not chars_per_size:
        return 10.0
    body = float(chars_per_size.most_common(1)[0][0])
    log.info("body font size: %.0fpt (histogram: %s)", body, dict(chars_per_size))
    return body


# ---------------------------------------------------------------------------
# Per-page extraction
# ---------------------------------------------------------------------------


def _dominant_span(spans: list[dict]) -> dict:
    """The span carrying the most characters decides the line's identity —
    a single bold word inside a body line must not make the line a heading."""
    return max(spans, key=lambda s: len(s["text"]))


def _extract_figure(
    page: pymupdf.Page, bbox: pymupdf.Rect, page_index: int, fig_index: int, figures_dir: Path
) -> Unit | None:
    page_area = abs(page.rect)
    if page_area <= 0 or abs(bbox) / page_area < FIGURE_MIN_AREA_FRAC:
        return None  # logo/watermark/bullet-glyph noise, skip

    figures_dir.mkdir(parents=True, exist_ok=True)
    key = f"figures/p{page_index:04d}_f{fig_index:02d}.png"
    pix = page.get_pixmap(clip=bbox, dpi=FIGURE_DPI)
    (figures_dir.parent / key).write_bytes(pix.tobytes("png"))
    return Unit(
        page=page_index,
        bbox=_bbox(bbox),
        type=UnitType.FIGURE,
        storage_key=key,
        source=Source.PYMUPDF,
    )


def extract_page(
    page: pymupdf.Page, page_index: int, body_size: float, figures_dir: Path
) -> list[Unit]:
    """One TEXT_NATIVE page -> TITLE/TEXT/FIGURE units in top-to-bottom order."""
    units: list[Unit] = []
    fig_index = 0
    d = cast(dict, page.get_text("dict"))

    for block in d["blocks"]:
        if block["type"] == 1:  # image block
            fig = _extract_figure(
                page, pymupdf.Rect(block["bbox"]), page_index, fig_index, figures_dir
            )
            if fig is not None:
                units.append(fig)
                fig_index += 1
            continue

        # Text block: classify per line, merge consecutive body lines into
        # one paragraph unit (lines inside a block are visual wraps).
        para_lines: list[str] = []
        para_bbox: pymupdf.Rect | None = None

        def flush_paragraph() -> None:
            nonlocal para_lines, para_bbox
            if para_lines and para_bbox is not None:
                units.append(
                    Unit(
                        page=page_index,
                        bbox=_bbox(para_bbox),
                        type=UnitType.TEXT,
                        content=" ".join(para_lines),
                        source=Source.PYMUPDF,
                    )
                )
            para_lines, para_bbox = [], None

        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            main = _dominant_span(spans)
            is_bold = bool(main["flags"] & _BOLD_FLAG)
            is_heading = main["size"] >= body_size * HEADING_SIZE_RATIO or (
                is_bold and _NUMBERED.match(text) is not None
            )

            if is_heading:
                flush_paragraph()
                units.append(
                    Unit(
                        page=page_index,
                        bbox=_bbox(pymupdf.Rect(line["bbox"])),
                        type=UnitType.TITLE,
                        content=text,
                        font_size=round(main["size"], 1),
                        source=Source.PYMUPDF,
                    )
                )
            else:
                para_lines.append(text)
                line_rect = pymupdf.Rect(line["bbox"])
                para_bbox = line_rect if para_bbox is None else para_bbox | line_rect

        flush_paragraph()

    return units


# ---------------------------------------------------------------------------
# Ruled-grid detection
# ---------------------------------------------------------------------------


Segment = tuple[float, float, float, float]  # (x0, y0, x1, y1), normalized


def detect_ruled_grid(page: pymupdf.Page, page_index: int) -> RuledGrid | None:
    """Count axis-aligned segments in the page's vector drawings. A ruled
    table draws its grid as 'l' (line) and 're' (rectangle) path items —
    see get_drawings() vs get_image_info(): paths are exact vector data,
    so this costs nothing and has perfect coordinates.

    Segments are DEDUPLICATED by rounded position before counting: real
    PDF writers routinely emit the same rule twice (path-closing segments,
    overlapping cell borders, re-stroked lines — MuPDF happily reports
    each one), so a raw count would inflate. Sets of normalized coordinate
    tuples make duplicates free to drop.

    The grid bbox is computed with explicit min/max, NOT Rect union: a
    zero-height line is an "empty" rect in PyMuPDF and empty rects are
    ignored by the union operator — unioning line rects silently produces
    garbage."""
    h_segs: set[Segment] = set()
    v_segs: set[Segment] = set()

    def add_segment(ax: float, ay: float, bx: float, by: float) -> None:
        x0, x1 = sorted((ax, bx))
        y0, y1 = sorted((ay, by))
        seg = (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))
        if y1 - y0 < 1.0 and x1 - x0 > 5.0:
            h_segs.add(seg)
        elif x1 - x0 < 1.0 and y1 - y0 > 5.0:
            v_segs.add(seg)

    for path in page.get_drawings():
        for item in path["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                add_segment(p1.x, p1.y, p2.x, p2.y)
            elif item[0] == "re":
                r = pymupdf.Rect(item[1]).normalize()
                # A drawn rectangle contributes its four edges to the grid.
                add_segment(r.x0, r.y0, r.x1, r.y0)
                add_segment(r.x0, r.y1, r.x1, r.y1)
                add_segment(r.x0, r.y0, r.x0, r.y1)
                add_segment(r.x1, r.y0, r.x1, r.y1)

    if len(h_segs) < RULED_MIN_H_SEGMENTS or len(v_segs) < RULED_MIN_V_SEGMENTS:
        return None

    all_segs = h_segs | v_segs
    bbox = (
        min(s[0] for s in all_segs),
        min(s[1] for s in all_segs),
        max(s[2] for s in all_segs),
        max(s[3] for s in all_segs),
    )
    return RuledGrid(page=page_index, h_segments=len(h_segs), v_segments=len(v_segs), bbox=bbox)


# ---------------------------------------------------------------------------
# Stage driver
# ---------------------------------------------------------------------------


def extract(
    doc: pymupdf.Document, page_kinds: dict[int, PageKind], doc_out: Path
) -> tuple[float, list[Unit], list[RuledGrid]]:
    """Run local extraction over every TEXT_NATIVE page.

    Returns (body_font_size, units, ruled_grids). Figure PNGs are written
    to <doc_out>/figures/ as a side effect; their storage_key is relative
    to doc_out so the whole output directory stays relocatable.
    """
    text_pages = [p for p, k in page_kinds.items() if k == PageKind.TEXT_NATIVE]
    body_size = estimate_body_font_size(doc, text_pages)

    figures_dir = doc_out / "figures"
    units: list[Unit] = []
    grids: list[RuledGrid] = []
    for pno in text_pages:
        # load_page over doc[pno]: __getitem__ is typed as a union (it also
        # accepts slices), load_page is typed -> Page.
        page = doc.load_page(pno)
        units.extend(extract_page(page, pno, body_size, figures_dir))
        grid = detect_ruled_grid(page, pno)
        if grid is not None:
            grids.append(grid)

    counts = Counter(u.type.value for u in units)
    log.info(
        "local extract: %d text-native pages -> %s, %d ruled grid(s)",
        len(text_pages),
        dict(counts),
        len(grids),
    )
    return body_size, units, grids
