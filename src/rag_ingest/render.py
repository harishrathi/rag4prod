"""STAGE 3 — Rendering: page images for everything downstream that *sees*.

Consumers: YOLO (stage 4) needs every SCANNED and TEXT_NATIVE page as an
image; Gemini (stage 5) needs SCANNED pages and table crops; DRAWING pages
are rendered once, stored as figure PNGs, and never looked at again.

Memory rule (the one that matters at 3000 pages): NEVER accumulate page
images. A 200-DPI A4 pixmap is ~11 MB raw; 3000 of them is 30+ GB. The
pipeline renders inside the per-page loop and lets each pixmap die before
the next page renders. This module therefore exposes per-page helpers, not
a "render everything" function — the loop lives in pipeline.py where the
lifetime is visible.

Debug copies are a separate concern from pipeline images: the pipeline
works on full-quality pixmaps in memory; what lands in debug/ is a
downscaled JPEG for a human to eyeball (config.DEBUG_IMAGE_MAX_DIM /
DEBUG_JPEG_QUALITY). MBs on disk are fine; GBs of PNGs are not.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pymupdf

from .config import DEBUG_IMAGE_MAX_DIM, DEBUG_JPEG_QUALITY, FIGURE_DPI, RENDER_DPI

log = logging.getLogger(__name__)


def render_page(page: pymupdf.Page, dpi: int = RENDER_DPI) -> pymupdf.Pixmap:
    """Full-page render, no alpha (alpha triples downstream conversions
    for nothing — no vision consumer uses transparency)."""
    return page.get_pixmap(dpi=dpi, alpha=False)


def pixmap_to_rgb_array(pix: pymupdf.Pixmap) -> np.ndarray:
    """Pixmap -> HxWx3 uint8 RGB ndarray for model input.

    The .copy() is load-bearing: np.frombuffer is a VIEW into the pixmap's
    buffer, and we mutate/discard pixmaps aggressively (see
    save_debug_jpeg) — a view would dangle.
    """
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    return arr.reshape(pix.height, pix.width, pix.n).copy()


def save_debug_jpeg(page: pymupdf.Page, path: Path) -> None:
    """Write a small JPEG copy for humans — as a FRESH low-res render,
    deliberately not derived from the pipeline pixmap.

    First version shrank the pipeline pixmap in place. That collides with
    pixmap_to_rgb_array: touching .samples makes PyMuPDF cache a
    memoryview into the pixel buffer, shrink() reallocates that buffer,
    and the destructor then warns 'operation forbidden on released
    memoryview' on every page. A fresh render at debug size is a few ms
    and has no sharing to reason about.
    """
    zoom = DEBUG_IMAGE_MAX_DIM / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pix.tobytes("jpg", jpg_quality=DEBUG_JPEG_QUALITY))


def store_drawing_page(page: pymupdf.Page, page_index: int, doc_out: Path) -> str:
    """DRAWING pages (CAD plans etc.) become one full-page figure PNG.
    The PNG is the artifact — these pages are never text-extracted.
    Returns the storage key, relative to doc_out."""
    key = f"figures/p{page_index:04d}_page.png"
    out = doc_out / key
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(page.get_pixmap(dpi=FIGURE_DPI, alpha=False).tobytes("png"))
    return key
