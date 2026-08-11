"""STAGE 4 — Layout detection: DocLayout-YOLO finds table/figure regions.

YOLO runs on every SCANNED and TEXT_NATIVE page (DRAWING pages are stored
wholesale and skip it). Design decision recap (design spec §5): YOLO is
not gated on ruled-line detection — the ruled heuristic false-positives on
borders and misses the second, borderless table on a mixed page. Local
inference is cheap; missed tables are silent data loss.

COORDINATE CONVERSION — the most likely bug in this pipeline (spec §5):

YOLO boxes arrive in RENDERED-IMAGE PIXELS. PyMuPDF operates in PDF
POINTS (72/inch). Every crop and every stored bbox must be converted, and
the naive constant `72/DPI` breaks silently on rotated pages and
non-origin cropboxes. The rule enforced here:

  * scale is derived from ACTUAL dimensions — page.rect vs the rendered
    pixmap's width/height — never from the DPI constant;
  * every conversion goes through pixel_rect_to_pdf(), which clamps into
    page.rect and asserts sanity;
  * no pixel coordinate escapes this stage: Region carries both boxes,
    and only bbox_pdf is allowed downstream. bbox_px exists for the
    stage artifact, where seeing both side by side makes the conversion
    auditable.

The torch import lives inside LayoutDetector so that pipelines running
only stages 1-2 (and the test suite's fast path) never pay the ~2 s
torch import or require the model download.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pymupdf

from .config import (
    YOLO_BOX_PAD_PX,
    YOLO_CONF_THRESHOLD,
    YOLO_DEVICE,
    YOLO_HF_FILENAME,
    YOLO_HF_REPO,
    YOLO_IMG_SIZE,
    YOLO_KEEP_LABELS,
)
from .models import BBox

log = logging.getLogger(__name__)


@dataclass
class Region:
    """One detected layout region on one page, in both coordinate spaces."""

    page: int
    label: str  # "table" | "figure" (others filtered by YOLO_KEEP_LABELS)
    conf: float
    bbox_px: tuple[int, int, int, int]  # rendered-image pixels, padded
    bbox_pdf: BBox  # PDF points — the only box downstream code may use

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Region:
        """Rehydrate from the stage-4 artifact (--from-stage resume)."""
        d = dict(d)
        d["bbox_px"] = tuple(d["bbox_px"])
        d["bbox_pdf"] = tuple(d["bbox_pdf"])
        return cls(**d)


def pixel_rect_to_pdf(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_rect: pymupdf.Rect,
    pix_width: int,
    pix_height: int,
) -> BBox:
    """Convert a rendered-image pixel box to PDF points.

    Scale comes from actual dimensions (page_rect vs pixmap size), which
    absorbs rotation, cropbox offsets, and any rounding PyMuPDF applied
    during rendering — `72/DPI` absorbs none of those. The result is
    clamped into the page rect: padded boxes near an edge may poke
    outside, and a crop clip outside the page is a silent empty image.
    """
    sx = page_rect.width / pix_width
    sy = page_rect.height / pix_height
    bbox = (
        max(page_rect.x0, page_rect.x0 + x0 * sx),
        max(page_rect.y0, page_rect.y0 + y0 * sy),
        min(page_rect.x1, page_rect.x0 + x1 * sx),
        min(page_rect.y1, page_rect.y0 + y1 * sy),
    )
    # A real exception, not an assert: this guard is load-bearing (a
    # degenerate box means a silent empty crop downstream) and must not
    # vanish under `python -O`.
    if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
        raise ValueError(f"degenerate bbox after conversion: {bbox} from px ({x0},{y0},{x1},{y1})")
    return bbox


class LayoutDetector:
    """Lazy wrapper around DocLayout-YOLO: the model loads on first use
    and is reused for every page (loading is ~seconds, inference ~tens of
    ms on GPU / ~1 s on CPU per page)."""

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            from doclayout_yolo import YOLOv10  # heavy import, deferred
            from huggingface_hub import hf_hub_download

            weights = hf_hub_download(YOLO_HF_REPO, YOLO_HF_FILENAME)
            self._model = YOLOv10(weights)
            log.info("layout model loaded from %s", weights)
        return self._model

    def detect(self, rgb: np.ndarray, page_rect: pymupdf.Rect, page_index: int) -> list[Region]:
        """Run the model on one rendered page (HxWx3 RGB uint8)."""
        model = self._load()
        # The wrapper follows ultralytics conventions: ndarray input is
        # assumed BGR (an OpenCV heritage). Feed RGB unflipped and red/blue
        # swap — mostly harmless for layout, but "mostly" is not a word to
        # build on. Flip explicitly.
        result = model.predict(
            rgb[:, :, ::-1],
            imgsz=YOLO_IMG_SIZE,
            conf=YOLO_CONF_THRESHOLD,
            device=YOLO_DEVICE,
            verbose=False,
        )[0]

        height, width = rgb.shape[:2]
        regions: list[Region] = []
        names: dict[int, str] = result.names
        for box in result.boxes:
            label = names[int(box.cls.item())].lower()
            if label not in YOLO_KEEP_LABELS:
                continue
            x0, y0, x1, y1 = (float(v) for v in box.xyxy[0].tolist())
            # Pad in pixel space so tight boxes don't clip captions or the
            # last table row; clamp to the image before converting.
            x0 = max(0.0, x0 - YOLO_BOX_PAD_PX)
            y0 = max(0.0, y0 - YOLO_BOX_PAD_PX)
            x1 = min(float(width), x1 + YOLO_BOX_PAD_PX)
            y1 = min(float(height), y1 + YOLO_BOX_PAD_PX)
            regions.append(
                Region(
                    page=page_index,
                    label=label,
                    conf=round(float(box.conf.item()), 3),
                    bbox_px=(int(x0), int(y0), int(x1), int(y1)),
                    bbox_pdf=pixel_rect_to_pdf(x0, y0, x1, y1, page_rect, width, height),
                )
            )

        log.debug("p%04d: %d region(s) kept", page_index, len(regions))
        return regions
