"""v2 configuration: one frozen dataclass, grouped by the layer that
reads it, passed into ``run()`` and recorded wholesale in the manifest.

v1 kept 267 lines of module globals and patched per-stage snapshots into
artifacts; here the snapshot is automatic because the config is a value,
not ambient state. Every number is still a tuning knob with its
calibration story — the stories live with the v1 constants they were
measured for (rag_ingest/config.py) and are not repeated here; values
must stay in sync with v1 until cutover (diff-parity, rewrite §5.4).

The VLM engine knobs (model id, retries, verification thresholds) are
NOT duplicated here: vlm_extract.py is a shared module and reads them
from rag_ingest.config directly.

Intentional-improvement toggles default OFF: v2 must reach diff-parity
with v1 first, then each improvement is switched on with its own
before/after diff (rewrite §5.4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RoutingRules:
    """Layer 2 thresholds — every reroute decision in one place."""

    min_text_chars: int = 50
    scan_image_coverage: float = 0.70
    drawing_min_segments: int = 100
    junk_min: int = 20
    junk_ratio: float = 0.005
    mojibake_min: int = 8
    mojibake_ratio: float = 0.01


@dataclass(frozen=True)
class ExtractionConfig:
    """Layer 3 — rendering, the native line walk, YOLO, figures."""

    render_dpi: int = 200
    figure_dpi: int = 200
    figure_min_area_frac: float = 0.005
    body_font_sample_pages: int = 60
    heading_size_ratio: float = 1.15
    heading_numbered_re: str = r"^\d+(\.\d+)*\.?\s+\S"
    ruled_min_h_segments: int = 4
    ruled_min_v_segments: int = 4
    yolo_hf_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    yolo_hf_filename: str = "doclayout_yolo_docstructbench_imgsz1024.pt"
    yolo_img_size: int = 1024
    yolo_conf_threshold: float = 0.2
    yolo_keep_labels: tuple[str, ...] = ("table", "figure")
    yolo_box_pad_px: int = 10
    yolo_device: str = "cpu"
    debug_image_max_dim: int = 1200
    debug_jpeg_quality: int = 70


@dataclass(frozen=True)
class TableConfig:
    """Layer 4/5 — stitching geometry and header dedup."""

    cont_bottom_frac: float = 0.90
    cont_top_frac: float = 0.12
    header_match_ratio: float = 0.8


@dataclass(frozen=True)
class NormalizeConfig:
    """Layer 5 — document-wide passes."""

    furniture_min_repeats: int = 3
    furniture_band_frac: float = 0.2
    max_heading_level: int = 6
    # Intentional improvement over v1 (#8's production note): OFF until
    # diff-parity is reached, then toggled with its own diff.
    figure_dedup: bool = False


@dataclass(frozen=True)
class ChunkConfig:
    """Layer 7 — sizing behind the split_text() seam."""

    size_tokens: int = 512
    overlap_tokens: int = 0
    tokenizer: str = "gpt2"
    table_rows_per_chunk: int = 20


@dataclass(frozen=True)
class IngestConfig:
    routing: RoutingRules = field(default_factory=RoutingRules)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    tables: TableConfig = field(default_factory=TableConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    # Layer 3 parallelism. 1 = inline (no executor), the honest default:
    # the paid lane is API-bound and small documents don't repay process
    # spawn + model reload costs. Raise for large local-heavy documents.
    workers: int = 1
    debug_images: bool = True
    # Override to point v2 at v1's response cache (same key scheme), so
    # one paid run covers both pipelines during migration. None = the
    # document's own output dir (cache/vlm/).
    vlm_cache_dir: str | None = None

    def to_dict(self) -> dict:
        """Wholesale manifest snapshot — the v1 per-stage threshold
        snapshots, generalized for free."""
        return asdict(self)
