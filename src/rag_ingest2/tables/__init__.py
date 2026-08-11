"""LAYER 4 — the table ladder, split into its real parts (rewrite §Layer 4).

v1's tables.py was four modules in one; the split follows the seams the
patches revealed:

    grids.py        vector-line grid evidence (#7) — cross-check, never a router
    cells_native.py find_tables + span geometry -> unmerged cell grid (#27)
    cells_vlm.py    GFM markdown -> unmerged cell grid (paid-lane pages)
    validate.py     structural checks + junk cells + the three renderings
    stitch.py       cross-page continuation, equal-column rule (#21), span fill (#27)

The unmerged-cell contract (#27) is kept exactly: a merged cell's value
is repeated into every grid position it covers, the printed layout is
preserved in ``merges`` — no downstream consumer ever guesses whether a
blank means "merged" or "empty".

Trust note: extractors here produce fragments and evidence; the quality
gate (quality.py) is the ONLY place ``needs_review`` is derived — v1
sprinkled that across four call sites (theme B).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_ingest.models import BBox


@dataclass
class RawTable:
    """One table fragment on one page, before stitching/validation.
    Cells are UNMERGED (#27); ``merges`` is [row, col, rowspan, colspan]
    per merged cell; ``header_rows`` counts leading header rows."""

    page: int
    bbox: BBox
    cells: list[list[str]]  # [] when a region was detected but not parsed
    source: str  # "find_tables" | "gemini" | "yolo_only"
    header_rows: int = 1
    merges: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["bbox"] = list(self.bbox)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RawTable:
        d = dict(d)
        d["bbox"] = tuple(d["bbox"])
        return cls(**d)


@dataclass
class TableResult:
    """A finished (possibly multi-page) table. ``needs_review`` and
    ``crop_key`` are filled by the quality gate, nowhere else."""

    table_id: str
    pages: list[int]  # 0-based, ascending
    page_spans: list[tuple[int, BBox]]  # per-fragment location, for citation
    col_count: int
    row_count: int
    markdown: str
    source: str
    needs_review: bool = False
    review_reason: str | None = None
    crop_key: str | None = None
    cells: list[list[str]] = field(default_factory=list)
    header_rows: int = 1
    merges: list[list[int]] = field(default_factory=list)
    grid: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["page_spans"] = [[p, list(b)] for p, b in self.page_spans]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TableResult:
        d = dict(d)
        d["page_spans"] = [(p, tuple(b)) for p, b in d["page_spans"]]
        return cls(**d)


def overlap_frac(a: BBox, b: BBox) -> float:
    """Intersection area over the smaller box's area — for matching YOLO
    regions against find_tables results (#23)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller = min(area_a, area_b)
    return (ix * iy) / smaller if smaller > 0 else 0.0
