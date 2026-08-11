"""LAYER 1 — Page profiling: evidence per page, no decisions.

v1's triage both measured and decided, so every new signal was a patch
to triage AND to whatever consumed its verdicts. Profiles are pure data:
a new field breaks nothing downstream until a routing rule reads it
(rewrite_design.md Layer 1). All decisions live in routing.py.

One deliberate cost leak, stated honestly: ``vector_segments`` is only
counted on near-textless pages (``get_drawings`` is the one expensive
probe, and a page with a rich text layer never routes DRAWING). The
threshold for "near-textless" comes from the routing rules, so the
budget hint and the decision rule can never drift apart. The field is
``None`` when not measured — "not measured" is itself evidence.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import cast

import pymupdf

from rag_ingest.text_quality import mojibake_score, text_layer_junk

from .config import RoutingRules

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageProfile:
    """Measurements for one page. Frozen: evidence never mutates."""

    page: int  # 0-based
    text_chars: int  # stripped text-layer length
    text_compact_chars: int  # non-whitespace length (ratio denominators)
    junk_chars: int  # C0 / U+FFFD / PUA count (ledger #29)
    mojibake_chars: int  # orphan marks + interleaved symbols (VLM spec §3)
    max_image_coverage: float  # largest single raster image / page area (#1)
    text_bbox_area_frac: float  # text block area / page area (#1 note; no rule yet)
    vector_segments: int | None  # deduped path items; None = not measured (#2, #7)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PageProfile:
        return cls(**d)


def profile_page(page: pymupdf.Page, page_index: int, rules: RoutingRules) -> PageProfile:
    """Measure one page. Cheapest signals first; the expensive one
    (get_drawings) only runs when the text layer is too thin to ever
    route NATIVE anyway."""
    text = cast(str, page.get_text("text")).strip()
    compact = len("".join(text.split()))
    page_area = abs(page.rect)  # abs(Rect) is its area in pt^2

    max_coverage = 0.0
    if page_area > 0:
        for img in page.get_image_info():
            max_coverage = max(max_coverage, abs(pymupdf.Rect(img["bbox"])) / page_area)

    text_area = 0.0
    if page_area > 0:
        # get_text("blocks") rows: (x0, y0, x1, y1, text, block_no, type);
        # type 0 = text. Overlaps are not subtracted — this is a coarse
        # coverage signal, not geometry.
        for b in cast(list, page.get_text("blocks")):
            if b[6] == 0:
                text_area += max(0.0, (b[2] - b[0]) * (b[3] - b[1]))

    junk, _ = text_layer_junk(text)
    moji, _ = mojibake_score(text)

    segments: int | None = None
    if len(text) < rules.min_text_chars:
        segments = sum(len(path["items"]) for path in page.get_drawings())

    return PageProfile(
        page=page_index,
        text_chars=len(text),
        text_compact_chars=compact,
        junk_chars=junk,
        mojibake_chars=moji,
        max_image_coverage=round(max_coverage, 3),
        text_bbox_area_frac=round(min(1.0, text_area / page_area), 3) if page_area else 0.0,
        vector_segments=segments,
    )


def profile_document(doc: pymupdf.Document, rules: RoutingRules) -> list[PageProfile]:
    """Profile every page. Single-threaded on purpose: profiling is
    get_text-bound (~10-20 s for 3000 pages) and shares the orchestrator's
    open Document — the parallel layer is extraction (Layer 3), where the
    wall-clock actually lives."""
    profiles = [profile_page(doc.load_page(i), i, rules) for i in range(doc.page_count)]
    log.info("profiled %d page(s)", len(profiles))
    return profiles
