"""STAGE 1 — Triage: one decision per page.

Question answered here: *does this page have a usable text layer?*
The answer routes the page's entire downstream life:

    TEXT_NATIVE -> stage 2 extracts text locally (free, exact)
    SCANNED     -> stage 3 renders it, stage 5 sends it to Gemini (costs money)
    DRAWING     -> stage 3 renders it, stored as a figure PNG (never Gemini)

Getting triage wrong is expensive in both directions:
  * scanned page misread as TEXT_NATIVE -> garbage text silently enters the
    corpus (worst failure: nobody notices until retrieval returns junk);
  * text page misread as SCANNED -> a needless Gemini call (mild: costs
    cents and the output is usually still correct).
So the heuristics below are deliberately biased toward SCANNED when in doubt.

Every verdict is emitted as a TriageRecord carrying the *evidence* it was
based on (text length, image coverage, segment count) and a one-line reason.
The stage artifact (stages/01_triage.json) is therefore a decision log you
can audit page by page — when a page is misrouted, the record shows which
heuristic fired and how close the numbers were to the threshold.

Concurrency note (interview favourite): the spec suggests a thread pool
across cores. PyMuPDF Document objects are NOT thread-safe — concurrent
access from multiple threads segfaults or corrupts state. Real options are
(a) one Document opened per *process* over a page-range shard, or
(b) single-threaded. For 3000 pages, `get_text` triage runs in ~10-20 s
single-threaded, so the multiprocessing machinery isn't worth its
complexity here. We take (b) on purpose.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import cast

import pymupdf

from .config import DRAWING_MIN_SEGMENTS, MIN_TEXT_CHARS, SCAN_IMAGE_COVERAGE
from .models import PageKind

log = logging.getLogger(__name__)


@dataclass
class TriageRecord:
    """Verdict for one page, plus the evidence behind it."""

    page: int  # 0-based
    kind: PageKind
    text_chars: int
    max_image_coverage: float  # largest single raster image / page area
    drawing_segments: int | None  # only counted for near-textless pages
    reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


def triage_page(page: pymupdf.Page, page_index: int) -> TriageRecord:
    """Classify a single page. Checks are ordered cheapest-first; the
    expensive one (``get_drawings``) only runs on near-textless pages."""

    # get_text's return type depends on its mode argument (str for "text",
    # dict/list for others); the stubs expose the whole union, so narrow it.
    text = cast(str, page.get_text("text")).strip()
    page_area = abs(page.rect)  # abs(Rect) is its area in pt^2

    max_coverage = 0.0
    if page_area > 0:
        for img in page.get_image_info():
            max_coverage = max(max_coverage, abs(pymupdf.Rect(img["bbox"])) / page_area)

    # --- Guard: header/footer text layer over a scanned body ---------------
    # A page can pass the text-length test with a real-looking text layer
    # that is only a scanner-stamped header. If one raster image blankets
    # the page, trust the image, not the text.
    if max_coverage > SCAN_IMAGE_COVERAGE:
        return TriageRecord(
            page=page_index,
            kind=PageKind.SCANNED,
            text_chars=len(text),
            max_image_coverage=round(max_coverage, 3),
            drawing_segments=None,
            reason=f"raster image covers {max_coverage:.0%} of page "
            f"(> {SCAN_IMAGE_COVERAGE:.0%}); any text layer is likely "
            f"a scanner-stamped header",
        )

    if len(text) >= MIN_TEXT_CHARS:
        return TriageRecord(
            page=page_index,
            kind=PageKind.TEXT_NATIVE,
            text_chars=len(text),
            max_image_coverage=round(max_coverage, 3),
            drawing_segments=None,
            reason=f"{len(text)} chars of text (>= {MIN_TEXT_CHARS}), no dominating raster image",
        )

    # --- Near-textless page: scanned raster, vector drawing, or blank -----
    # CAD plans / site drawings are pure vector paths: no text layer, no
    # raster image. Sending one to Gemini as a "scanned page" returns junk
    # at real cost, so they get their own class and are stored as figures.
    segments = sum(len(path["items"]) for path in page.get_drawings())
    if segments >= DRAWING_MIN_SEGMENTS:
        return TriageRecord(
            page=page_index,
            kind=PageKind.DRAWING,
            text_chars=len(text),
            max_image_coverage=round(max_coverage, 3),
            drawing_segments=segments,
            reason=f"only {len(text)} chars of text but {segments} vector "
            f"segments (>= {DRAWING_MIN_SEGMENTS}): CAD plan / drawing",
        )

    return TriageRecord(
        page=page_index,
        kind=PageKind.SCANNED,
        text_chars=len(text),
        max_image_coverage=round(max_coverage, 3),
        drawing_segments=segments,
        reason=f"only {len(text)} chars of text (< {MIN_TEXT_CHARS}), "
        f"not a drawing: presumed scan (or near-blank page — "
        f"accepted cost of one extra Gemini call)",
    )


def triage(doc: pymupdf.Document) -> list[TriageRecord]:
    """Classify every page of an open document.

    Deliberately single-threaded; see the module docstring for why a
    thread pool would be a bug here.
    """
    # load_page rather than iterating the Document: iteration and
    # __getitem__ are loosely typed in the stubs (getitem also accepts
    # slices), load_page is typed -> Page — and the index is needed for
    # the record anyway.
    records = [triage_page(doc.load_page(i), i) for i in range(doc.page_count)]

    counts: dict[str, int] = {}
    for r in records:
        counts[r.kind.value] = counts.get(r.kind.value, 0) + 1
    log.info("triage: %d pages -> %s", len(records), counts)
    for r in records:
        log.debug("  p%04d %-12s %s", r.page, r.kind.value, r.reason)
    return records
