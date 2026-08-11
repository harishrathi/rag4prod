"""Vector-line grid evidence (#7): a page whose drawings form a
table-like grid. Cross-check data for the ladder — never a router (page
borders and letterhead rules look exactly like a grid to a segment
counter). Logic identical to v1's detect_ruled_grid; only the threshold
plumbing changed (config values in, module globals out)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pymupdf

from rag_ingest.models import BBox

Segment = tuple[float, float, float, float]  # (x0, y0, x1, y1), normalized


@dataclass
class RuledGrid:
    page: int
    h_segments: int
    v_segments: int
    bbox: BBox  # union of the counted segments, PDF points

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RuledGrid:
        d = dict(d)
        d["bbox"] = tuple(d["bbox"])
        return cls(**d)


def detect_ruled_grid(
    page: pymupdf.Page, page_index: int, min_h: int, min_v: int
) -> RuledGrid | None:
    """Count deduplicated axis-aligned segments in the page's vector
    drawings. Segments are deduplicated by rounded position (PDF writers
    routinely emit the same rule twice); the grid bbox uses explicit
    min/max, NOT Rect union — zero-height line rects are "empty" and the
    union operator silently ignores them (v1 learned this live)."""
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
                add_segment(r.x0, r.y0, r.x1, r.y0)
                add_segment(r.x0, r.y1, r.x1, r.y1)
                add_segment(r.x0, r.y0, r.x0, r.y1)
                add_segment(r.x1, r.y0, r.x1, r.y1)

    if len(h_segs) < min_h or len(v_segs) < min_v:
        return None

    all_segs = h_segs | v_segs
    bbox = (
        min(s[0] for s in all_segs),
        min(s[1] for s in all_segs),
        max(s[2] for s in all_segs),
        max(s[3] for s in all_segs),
    )
    return RuledGrid(page=page_index, h_segments=len(h_segs), v_segments=len(v_segs), bbox=bbox)
