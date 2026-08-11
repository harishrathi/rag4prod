"""Coordinate types — make the pixel-vs-point bug class unrepresentable.

The same bug shipped twice in v1 (ledger #11 at the YOLO boundary, #20 at
the pdfocr boundary), which proves vigilance fails once per coordinate
boundary. So v2 makes the confusion a TYPE error: a ``PixelBox`` carries
the raster it was measured in, a ``PdfBox`` is always PDF points, and the
only path between them is ``to_pdf()``, whose scale is derived from
ACTUAL dimensions on both sides — measured, never assumed. ``Unit.bbox``
remains a plain PDF-points tuple at the models.py contract boundary
(``PdfBox.as_tuple()`` is the single exit).
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_ingest.models import BBox


@dataclass(frozen=True)
class PdfBox:
    """A rectangle in PDF points (72/inch), page coordinate space."""

    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> BBox:
        """The models.py contract boundary — the only place a PdfBox
        degrades to an untyped tuple."""
        return (self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_tuple(cls, b: BBox) -> PdfBox:
        return cls(*b)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class PixelBox:
    """A rectangle in rendered-image pixels, bound to its raster: the
    raster dimensions travel WITH the box, so the conversion can never
    silently use the wrong render."""

    x0: float
    y0: float
    x1: float
    y1: float
    raster_w: int
    raster_h: int


def to_pdf(box: PixelBox, page_rect: PdfBox) -> PdfBox:
    """The one conversion path. Scale comes from actual dimensions
    (page rect vs the box's own raster), which absorbs rotation, cropbox
    offsets, and render rounding — `72/DPI` absorbs none of those. The
    result is clamped into the page rect (padded boxes near an edge may
    poke outside; a crop clip outside the page is a silent empty image).

    Raises ValueError on a degenerate result — a real exception, not an
    assert, because the guard is load-bearing and must survive
    ``python -O`` (v1 learned this at the same boundary).
    """
    sx = page_rect.width / box.raster_w
    sy = page_rect.height / box.raster_h
    out = PdfBox(
        x0=max(page_rect.x0, page_rect.x0 + box.x0 * sx),
        y0=max(page_rect.y0, page_rect.y0 + box.y0 * sy),
        x1=min(page_rect.x1, page_rect.x0 + box.x1 * sx),
        y1=min(page_rect.y1, page_rect.y0 + box.y1 * sy),
    )
    if out.x0 > out.x1 or out.y0 > out.y1:
        raise ValueError(f"degenerate box after conversion: {out} from {box}")
    return out
