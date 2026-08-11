"""LAYER 3 — Extraction workers: per (page, route), process-parallel.

Each shard task takes ``(pdf_path, page_numbers, ...)`` — a PATH, not a
live Document — opens its own document, and emits Units, layout Regions,
grid evidence, and VLM page records through the existing contracts. That
is what resolves ledger #4 (PyMuPDF is not thread-safe) architecturally:
workers share nothing, so ``ProcessPoolExecutor`` with page-range
sharding is safe by construction. YOLO loads once per worker process;
the VLM response cache is a shared directory of hash-keyed files, so
concurrent workers need no coordination.

``workers=1`` (the default) runs inline — no executor, no spawn cost —
because the paid lane is API-bound and small documents don't repay
process startup. Determinism either way: results are keyed by page and
reassembled in page order.

Three extractors, one per route:
  * native  — the v1 stage-2 line walk, unchanged in logic (#6, #17);
              plus embedded-figure storage and ruled-grid evidence
  * vlm     — render -> shared VLM seam (client, cache, parser, checks)
  * drawing — render to PNG, store as the page's figure (#2)

Workers do NOT set quality flags for local extraction — units leave here
raw and the quality gate (quality.py) derives ``needs_review`` in one
place. (Paid-lane units arrive flagged by the shared seam; the gate
treats the recorded reasons as its evidence and re-derives identically.)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pymupdf

from rag_ingest.models import Source, Unit, UnitType
from rag_ingest.vlm_extract import (
    CachedVlmClient,
    GeminiClient,
    VlmPageRecord,
    vlm_page_units,
)

from .boxes import PdfBox, PixelBox, to_pdf
from .config import ExtractionConfig, IngestConfig
from .routing import Extractor, PageRoute

log = logging.getLogger(__name__)

_BOLD_FLAG = 16  # bit 4 of span["flags"] in PyMuPDF's dict output


@dataclass
class Region:
    """One detected layout region, in both coordinate spaces — typed, so
    pixel coordinates cannot leak downstream (boxes.py)."""

    page: int
    label: str  # "table" | "figure"
    conf: float
    box_px: PixelBox
    box_pdf: PdfBox

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "label": self.label,
            "conf": self.conf,
            "box_px": asdict(self.box_px),
            "box_pdf": asdict(self.box_pdf),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Region:
        return cls(
            page=d["page"],
            label=d["label"],
            conf=d["conf"],
            box_px=PixelBox(**d["box_px"]),
            box_pdf=PdfBox(**d["box_pdf"]),
        )


# ---------------------------------------------------------------------------
# Per-process singletons (loaded once per worker process, reused per page)
# ---------------------------------------------------------------------------

_detector = None
_vlm_client: CachedVlmClient | None = None


def _get_detector(cfg: ExtractionConfig):
    global _detector
    if _detector is None:
        from doclayout_yolo import YOLOv10  # heavy import, deferred
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download(cfg.yolo_hf_repo, cfg.yolo_hf_filename)
        _detector = YOLOv10(weights)
        log.info("layout model loaded from %s", weights)
    return _detector


def _get_vlm_client(cache_dir: str) -> CachedVlmClient:
    global _vlm_client
    if _vlm_client is None or str(_vlm_client.cache_dir) != cache_dir:
        _vlm_client = CachedVlmClient(GeminiClient(), Path(cache_dir))
    return _vlm_client


# ---------------------------------------------------------------------------
# YOLO with typed boxes
# ---------------------------------------------------------------------------


def detect_regions(
    rgb: np.ndarray, page_rect: PdfBox, page_index: int, cfg: ExtractionConfig
) -> list[Region]:
    """Run DocLayout-YOLO on one rendered page (HxWx3 RGB uint8). The
    wrapper assumes BGR ndarray input (OpenCV heritage) — flip explicitly."""
    model = _get_detector(cfg)
    result = model.predict(
        rgb[:, :, ::-1],
        imgsz=cfg.yolo_img_size,
        conf=cfg.yolo_conf_threshold,
        device=cfg.yolo_device,
        verbose=False,
    )[0]

    height, width = rgb.shape[:2]
    regions: list[Region] = []
    names: dict[int, str] = result.names
    for box in result.boxes:
        label = names[int(box.cls.item())].lower()
        if label not in cfg.yolo_keep_labels:
            continue
        x0, y0, x1, y1 = (float(v) for v in box.xyxy[0].tolist())
        px = PixelBox(
            x0=max(0.0, x0 - cfg.yolo_box_pad_px),
            y0=max(0.0, y0 - cfg.yolo_box_pad_px),
            x1=min(float(width), x1 + cfg.yolo_box_pad_px),
            y1=min(float(height), y1 + cfg.yolo_box_pad_px),
            raster_w=width,
            raster_h=height,
        )
        regions.append(
            Region(
                page=page_index,
                label=label,
                conf=round(float(box.conf.item()), 3),
                box_px=px,
                box_pdf=to_pdf(px, page_rect),
            )
        )
    return regions


# ---------------------------------------------------------------------------
# The native line walk (v1 stage-2 logic, quality flags removed)
# ---------------------------------------------------------------------------


def estimate_body_font_size(
    doc: pymupdf.Document, native_pages: list[int], cfg: ExtractionConfig
) -> float:
    """Mode of span font sizes weighted by character count, sampled from
    native pages spread evenly across the document (front matter is
    typographically unrepresentative)."""
    if not native_pages:
        return 10.0
    step = max(1, len(native_pages) // cfg.body_font_sample_pages)
    chars_per_size: Counter[int] = Counter()
    for pno in native_pages[::step][: cfg.body_font_sample_pages]:
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
    log.info("body font size: %.0fpt", body)
    return body


def _dominant_span(spans: list[dict]) -> dict:
    """The span carrying the most characters decides the line's identity."""
    return max(spans, key=lambda s: len(s["text"]))


def native_page_units(
    page: pymupdf.Page,
    page_index: int,
    body_size: float,
    doc_out: Path,
    cfg: ExtractionConfig,
) -> list[Unit]:
    """One native page -> TITLE/TEXT/FIGURE units in top-to-bottom order.
    Classification per LINE, consecutive body lines of a block merged
    into one paragraph — identical to v1's walk (see its docstring for
    the granularity rationale)."""
    numbered = re.compile(cfg.heading_numbered_re)
    units: list[Unit] = []
    fig_index = 0
    page_area = abs(page.rect)
    d = cast(dict, page.get_text("dict"))

    for block in d["blocks"]:
        if block["type"] == 1:  # image block -> stored figure (or skipped noise)
            bbox = pymupdf.Rect(block["bbox"])
            if page_area <= 0 or abs(bbox) / page_area < cfg.figure_min_area_frac:
                continue  # logo/watermark/bullet-glyph noise
            key = f"figures/p{page_index:04d}_f{fig_index:02d}.png"
            out = doc_out / key
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(page.get_pixmap(clip=bbox, dpi=cfg.figure_dpi).tobytes("png"))
            units.append(
                Unit(
                    page=page_index,
                    bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                    type=UnitType.FIGURE,
                    storage_key=key,
                    source=Source.PYMUPDF,
                )
            )
            fig_index += 1
            continue

        para_lines: list[str] = []
        para_bbox: pymupdf.Rect | None = None

        def flush_paragraph() -> None:
            nonlocal para_lines, para_bbox
            if para_lines and para_bbox is not None:
                units.append(
                    Unit(
                        page=page_index,
                        bbox=(para_bbox.x0, para_bbox.y0, para_bbox.x1, para_bbox.y1),
                        type=UnitType.TEXT,
                        content=" ".join(para_lines),
                        source=Source.PYMUPDF,
                    )
                )
            para_lines, para_bbox = [], None

        for line in block["lines"]:
            # Join ALL spans including whitespace-only ones (ledger #17).
            text = re.sub(r"\s+", " ", "".join(s["text"] for s in line["spans"])).strip()
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans or not text:
                continue
            main = _dominant_span(spans)
            is_bold = bool(main["flags"] & _BOLD_FLAG)
            is_heading = main["size"] >= body_size * cfg.heading_size_ratio or (
                is_bold and numbered.match(text) is not None
            )
            if is_heading:
                flush_paragraph()
                rect = pymupdf.Rect(line["bbox"])
                units.append(
                    Unit(
                        page=page_index,
                        bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
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
# The shard task
# ---------------------------------------------------------------------------


def _save_debug_jpeg(page: pymupdf.Page, path: Path, cfg: ExtractionConfig) -> None:
    """Small fresh low-res render for humans — deliberately not derived
    from the pipeline pixmap (v1 learned the shared-buffer lesson)."""
    zoom = cfg.debug_image_max_dim / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pix.tobytes("jpg", jpg_quality=cfg.debug_jpeg_quality))


def run_shard(
    pdf_path: str,
    pages: list[int],
    routes: dict[int, str],  # page -> Extractor value
    text_layer_chars: dict[int, int],  # only pages with countable lying layers
    body_size: float,
    cfg: IngestConfig,
    doc_out_s: str,
    cache_dir_s: str,
) -> list[dict]:
    """Process one page-range shard with one open Document. Runs in a
    worker process (or inline for workers=1); everything in and out is
    picklable, results are per-page dicts keyed by page."""
    from .tables.grids import detect_ruled_grid  # local: keep module import light

    doc_out = Path(doc_out_s)
    ext = cfg.extraction
    out: list[dict] = []
    doc = pymupdf.open(pdf_path)
    try:
        for pno in pages:
            page = doc.load_page(pno)
            route = Extractor(routes[pno])
            rect = PdfBox(page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1)
            result: dict = {
                "page": pno,
                "units": [],
                "regions": [],
                "grid": None,
                "vlm": None,
                "render": {},
            }

            if route == Extractor.DRAWING:
                key = f"figures/p{pno:04d}_page.png"
                (doc_out / key).parent.mkdir(parents=True, exist_ok=True)
                (doc_out / key).write_bytes(
                    page.get_pixmap(dpi=ext.figure_dpi, alpha=False).tobytes("png")
                )
                result["render"] = {"drawing_figure_key": key}
                result["units"] = [
                    Unit(
                        page=pno,
                        bbox=rect.as_tuple(),
                        type=UnitType.FIGURE,
                        storage_key=key,
                        source=Source.PYMUPDF,
                    ).to_dict()
                ]
                out.append(result)
                continue

            # Render once; YOLO consumes it, the VLM lane reuses the PNG
            # (#13's interleaving, kept per page).
            pix = page.get_pixmap(dpi=ext.render_dpi, alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            regions = detect_regions(rgb, rect, pno, ext)
            result["regions"] = [g.to_dict() for g in regions]
            result["render"] = {"px_width": pix.width, "px_height": pix.height}
            if cfg.debug_images:
                _save_debug_jpeg(page, doc_out / "debug" / "renders" / f"p{pno:04d}.jpg", ext)

            if route == Extractor.NATIVE:
                units = native_page_units(page, pno, body_size, doc_out, ext)
                grid = detect_ruled_grid(
                    page, pno, ext.ruled_min_h_segments, ext.ruled_min_v_segments
                )
                result["units"] = [u.to_dict() for u in units]
                result["grid"] = grid.to_dict() if grid else None
            else:  # VLM lane
                client = _get_vlm_client(cache_dir_s)
                yolo_tables = [
                    g.box_pdf.as_tuple() for g in regions if g.label == "table"
                ]
                units, record = vlm_page_units(
                    pix.tobytes("png"),
                    pno,
                    rect.as_tuple(),
                    yolo_tables,
                    client,
                    text_layer_chars=text_layer_chars.get(pno),
                )
                # Figure regions YOLO found become stored crops, exactly
                # as v1 stage 5 did.
                for i, g in enumerate(regions):
                    if g.label != "figure":
                        continue
                    key = f"figures/p{pno:04d}_r{i:02d}.png"
                    crop = page.get_pixmap(
                        clip=pymupdf.Rect(g.box_pdf.as_tuple()), dpi=ext.figure_dpi
                    )
                    (doc_out / key).parent.mkdir(parents=True, exist_ok=True)
                    (doc_out / key).write_bytes(crop.tobytes("png"))
                    units.append(
                        Unit(
                            page=pno,
                            bbox=g.box_pdf.as_tuple(),
                            type=UnitType.FIGURE,
                            storage_key=key,
                            source=Source.PYMUPDF,
                        )
                    )
                result["units"] = [u.to_dict() for u in units]
                result["vlm"] = record.to_dict()

            out.append(result)
    finally:
        doc.close()
    return out


# ---------------------------------------------------------------------------
# Layer driver
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Everything Layer 3 produced, reassembled in page order."""

    body_font_size: float
    units: list[Unit]
    regions: list[Region]
    grids: list  # list[RuledGrid] — typed at the tables layer
    vlm_pages: list[VlmPageRecord]
    render_meta: list[dict]


def extract_document(
    pdf_path: Path,
    routes: list[PageRoute],
    profiles_text_chars: dict[int, int],
    cfg: IngestConfig,
    doc_out: Path,
) -> ExtractionResult:
    """Shard the routed pages and run them — inline for workers=1, else
    under ProcessPoolExecutor. ``profiles_text_chars`` maps VLM-routed
    pages with a countable lying text layer to that count (verification
    evidence, VLM spec §5)."""
    from .tables.grids import RuledGrid

    route_map = {r.page: r.extractor.value for r in routes}
    all_pages = [r.page for r in routes]

    # Body size comes from the routed-native pages, measured once here
    # (document-wide evidence a per-page worker cannot compute).
    doc = pymupdf.open(pdf_path)
    try:
        native_pages = [r.page for r in routes if r.extractor == Extractor.NATIVE]
        body_size = estimate_body_font_size(doc, native_pages, cfg.extraction)
    finally:
        doc.close()

    cache_dir = (
        Path(cfg.vlm_cache_dir) if cfg.vlm_cache_dir else doc_out / "cache" / "vlm"
    )

    n_workers = max(1, cfg.workers)
    if n_workers == 1:
        shards = [
            run_shard(
                str(pdf_path),
                all_pages,
                route_map,
                profiles_text_chars,
                body_size,
                cfg,
                str(doc_out),
                str(cache_dir),
            )
        ]
    else:
        # Contiguous page-range sharding: amortizes the per-process doc
        # open and YOLO load over many pages.
        chunk = -(-len(all_pages) // n_workers)  # ceil division
        ranges = [all_pages[i : i + chunk] for i in range(0, len(all_pages), chunk)]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            shards = list(
                pool.map(
                    run_shard,
                    [str(pdf_path)] * len(ranges),
                    ranges,
                    [route_map] * len(ranges),
                    [profiles_text_chars] * len(ranges),
                    [body_size] * len(ranges),
                    [cfg] * len(ranges),
                    [str(doc_out)] * len(ranges),
                    [str(cache_dir)] * len(ranges),
                )
            )

    by_page = {r["page"]: r for shard in shards for r in shard}
    units: list[Unit] = []
    regions: list[Region] = []
    grids: list[RuledGrid] = []
    vlm_pages: list[VlmPageRecord] = []
    render_meta: list[dict] = []
    for pno in sorted(by_page):
        r = by_page[pno]
        units.extend(Unit.from_dict(d) for d in r["units"])
        regions.extend(Region.from_dict(d) for d in r["regions"])
        if r["grid"]:
            grids.append(RuledGrid.from_dict(r["grid"]))
        if r["vlm"]:
            vlm_pages.append(VlmPageRecord(**r["vlm"]))
        render_meta.append({"page": pno, **r["render"]})

    fresh = [p for p in vlm_pages if not p.cached]
    log.info(
        "extract: %d unit(s), %d region(s), %d grid(s); vlm %d page(s) "
        "(%d cached), tokens in=%d out=%d",
        len(units),
        len(regions),
        len(grids),
        len(vlm_pages),
        len(vlm_pages) - len(fresh),
        sum(p.input_tokens for p in fresh),
        sum(p.output_tokens for p in fresh),
    )
    return ExtractionResult(
        body_font_size=body_size,
        units=units,
        regions=regions,
        grids=grids,
        vlm_pages=vlm_pages,
        render_meta=render_meta,
    )
