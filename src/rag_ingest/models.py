"""Data contracts shared by every pipeline stage.

Two shapes matter:

  * ``Unit``  — one region of one page (a paragraph, a heading, a table,
    a figure). Every extraction path (PyMuPDF, Gemini full-page, Gemini
    crop) must emit Units of this exact shape *before* reassembly, so the
    stage-6 merge walk never cares where a Unit came from.

  * ``Chunk`` — the retrieval-ready output. This is what gets embedded,
    stored, and cited. Chunks are produced only in stage 6.

Page-number convention (decide once, write it down, never mix):
  * INTERNALLY everything is 0-based, because PyMuPDF is 0-based
    (``doc[0]`` is the first page).
  * The final ``Chunk.pages`` list is 1-based, because citations are for
    humans ("see page 14" must mean the page a PDF viewer labels 14).
  The conversion happens in exactly one place: chunk assembly (stage 6).
  Mixing these silently is a classic off-by-one bug in citation systems.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum


class PageKind(StrEnum):
    """Triage verdict for a page — decides its entire downstream route."""

    TEXT_NATIVE = "text_native"  # usable text layer -> extract locally, free
    SCANNED = "scanned"  # no usable text    -> image -> Gemini
    DRAWING = "drawing"  # vector graphics   -> render to PNG, store as figure


class UnitType(StrEnum):
    TITLE = "title"
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class Source(StrEnum):
    """Which extraction path produced a Unit. Carried through to the final
    chunk so that quality problems can be traced back to their origin
    ("all the broken tables came from gemini_page? then full-page prompts
    are the problem, not crops")."""

    PYMUPDF = "pymupdf"
    GEMINI_PAGE = "gemini_page"
    GEMINI_CROP = "gemini_crop"


# A bbox is always (x0, y0, x1, y1) in PDF points (72 per inch), in the
# page's coordinate space — NEVER in rendered-image pixels. Pixel-space
# boxes (from YOLO) must be converted at the layout-stage boundary; no
# pixel coordinate is allowed to escape stage 4. See layout.py.
# PEP 695 `type` statement (3.12+): a bare `BBox = tuple[...]` assignment
# is only an *implicit* alias, which type checkers sometimes resolve as a
# plain variable ("BBox is not iterable" on unpacking); `type` makes the
# alias explicit and unambiguous.
type BBox = tuple[float, float, float, float]


@dataclass
class Unit:
    """One region of one page, normalized to the common contract (§7.1)."""

    page: int  # 0-based page index
    bbox: BBox  # PDF points, page coordinate space
    type: UnitType
    content: str = ""  # markdown/text; "" for figures (image is the artifact)
    level: int | None = None  # heading depth, TITLE units only
    # Raw font size in points, TITLE units only. Levels can't be assigned
    # per page (a 14pt heading might be level 1 in one chapter and level 2
    # in another) — stage 6 clusters sizes document-wide, so TITLE units
    # must carry the evidence until then.
    font_size: float | None = None
    storage_key: str | None = None  # relative path of stored PNG, FIGURE units only
    source: Source = Source.PYMUPDF
    needs_review: bool = False  # extraction was lossy/failed; keep, but flag

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    """Retrieval-ready output (§7.5). ``content`` is what a human sees when
    the chunk is cited; ``embedding_text`` is what the vector index sees.
    They differ because retrieval needs context the display text shouldn't
    repeat: the heading breadcrumb ("[7. Payment Terms > 7.3 Liquidated
    Damages]") is prepended only to embedding_text, which is what makes
    "0.5% per week" findable for the query "what is the LD rate?"."""

    chunk_id: str
    type: UnitType
    content: str
    embedding_text: str
    headings: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)  # 1-based, for citation
    bbox: BBox | None = None
    storage_key: str | None = None
    source: Source = Source.PYMUPDF
    needs_review: bool = False
    token_count: int = 0
    # Set only for multi-page tables: all row-group chunks split from one
    # logical table share this id, so a consumer can reassemble the whole
    # table when a single row-group hit isn't enough context.
    table_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
