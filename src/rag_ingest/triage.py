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
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import cast

import pymupdf

from .config import (
    DRAWING_MIN_SEGMENTS,
    MIN_TEXT_CHARS,
    MOJIBAKE_MIN,
    MOJIBAKE_RATIO,
    SCAN_IMAGE_COVERAGE,
    TEXT_LAYER_JUNK_MIN,
    TEXT_LAYER_JUNK_RATIO,
)
from .models import PageKind

log = logging.getLogger(__name__)

# Junk characters that healthy text layers NEVER contain: C0 controls
# (except tab/newline/CR, which the whitespace strip removes anyway),
# U+FFFD replacement chars, and Private Use Area codepoints. Their
# presence means the font's ToUnicode CMap is broken — the page renders
# fine but its text layer is lying (ledger #29).
JUNK_CHARS_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f�-]")


def text_layer_junk(text: str) -> tuple[int, float]:
    """(junk char count, junk ratio) over non-whitespace text."""
    compact = "".join(text.split())
    if not compact:
        return 0, 0.0
    n = len(JUNK_CHARS_RE.findall(compact))
    return n, n / len(compact)


# --- Script-agnostic mojibake detection (gemini_extractor_spec.md §3) ------
# The junk-char test above catches text layers that map to NON-printable
# garbage. A second CMap-failure mode maps glyphs to PRINTABLE garbage:
# `लाभाथ\ के प] म= होनी चा<हए` extracts as valid Unicode with zero C0/PUA
# hits and sails through as a healthy native page. Two Unicode-general
# signals catch it without per-language tables (per-language tables defeat
# the purpose — the product goal is every script):
#
#   * orphan combining marks — an Mn/Mc char whose base is not a letter
#     (or is a letter from a different script block). Healthy text ~0.
#   * ASCII symbols interleaved inside non-Latin words — `प]`, `म=`.
#     Only symbols that never legitimately appear mid-word count;
#     ./,-%() occur in real text (dates, abbreviations) and are excluded.

# Marks whose Unicode script is Inherited attach to letters of ANY script
# (NFD `é` carries U+0301) — exempt from the cross-script test.
_INHERITED_MARK_RANGES = (
    (0x0300, 0x036F),  # Combining Diacritical Marks
    (0x1AB0, 0x1AFF),  # Combining Diacritical Marks Extended
    (0x1DC0, 0x1DFF),  # Combining Diacritical Marks Supplement
    (0x20D0, 0x20FF),  # Combining Diacritical Marks for Symbols
    (0xFE20, 0xFE2F),  # Combining Half Marks
)

_MOJIBAKE_SYMBOLS = frozenset("\\][=<>^_@#|~")


def orphan_combining_marks(text: str) -> int:
    """Count combining marks (Mn/Mc) not attached to a same-script letter.

    The base of a mark is the most recent non-mark, non-format char: marks
    stack (र + ि + ं) and ZWJ/ZWNJ are transparent, so simply looking at
    the preceding char would miscount healthy Indic text. Script identity
    is approximated by the 128-codepoint block (exact for the Indic blocks
    this was built against; coarse elsewhere, which only softens the
    signal, never inflates it for healthy text)."""
    orphans = 0
    base: str | None = None
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Mn", "Mc"):
            cp = ord(ch)
            if base is None or not base.isalpha():
                orphans += 1
            elif not any(lo <= cp <= hi for lo, hi in _INHERITED_MARK_RANGES) and (
                cp // 0x80 != ord(base) // 0x80
            ):
                orphans += 1
        elif cat != "Cf":  # format chars (ZWJ/ZWNJ) are transparent
            base = ch
    return orphans


def interleaved_ascii_symbols(text: str) -> int:
    """Count mojibake-symbol chars inside tokens that contain non-Latin
    letters — `प]` and `म=` are CMap shrapnel; a bare `x=y` is not."""
    count = 0
    for token in text.split():
        if any(ch.isalpha() and not ch.isascii() for ch in token):
            count += sum(1 for ch in token if ch in _MOJIBAKE_SYMBOLS)
    return count


def mojibake_score(text: str) -> tuple[int, float]:
    """(mojibake char count, ratio) over non-whitespace text — the
    printable-garbage counterpart of text_layer_junk()."""
    compact = "".join(text.split())
    if not compact:
        return 0, 0.0
    n = orphan_combining_marks(text) + interleaved_ascii_symbols(text)
    return n, n / len(compact)


@dataclass
class TriageRecord:
    """Verdict for one page, plus the evidence behind it."""

    page: int  # 0-based
    kind: PageKind
    text_chars: int
    max_image_coverage: float  # largest single raster image / page area
    drawing_segments: int | None  # only counted for near-textless pages
    reason: str
    rotation_applied: int = 0

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
        # --- Guard: broken ToUnicode CMap (ledger #29) -------------------
        # The text layer can be PRESENT but lying: glyphs render fine while
        # mapping to garbage codepoints (typical for custom-encoded Indic
        # fonts). Local extraction would emit mojibake; OCR reads the
        # rendered glyphs correctly, so the page reroutes to the OCR path.
        junk, junk_ratio = text_layer_junk(text)
        if junk >= TEXT_LAYER_JUNK_MIN or junk_ratio >= TEXT_LAYER_JUNK_RATIO:
            return TriageRecord(
                page=page_index,
                kind=PageKind.SCANNED,
                text_chars=len(text),
                max_image_coverage=round(max_coverage, 3),
                drawing_segments=None,
                reason=f"text layer corrupt: {junk} junk chars ({junk_ratio:.1%}) — "
                f"broken font-to-Unicode map; rerouted to the VLM lane",
            )
        # --- Guard: printable mojibake (gemini_extractor_spec.md §3) -----
        # Same disease, different symptom: a broken CMap that maps to
        # PRINTABLE garbage (orphan matras, ASCII symbols inside Indic
        # words) has zero junk chars. The page renders fine — re-read it
        # from pixels.
        moji, moji_ratio = mojibake_score(text)
        if moji >= MOJIBAKE_MIN or moji_ratio >= MOJIBAKE_RATIO:
            return TriageRecord(
                page=page_index,
                kind=PageKind.SCANNED,
                text_chars=len(text),
                max_image_coverage=round(max_coverage, 3),
                drawing_segments=None,
                reason=f"text layer mojibake: {moji} suspect chars ({moji_ratio:.1%}) — "
                f"broken font-to-Unicode map; rerouted to the VLM lane",
            )
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


def triage(doc: pymupdf.Document, fix_orientation: bool = True) -> list[TriageRecord]:
    """Classify every page of an open document.

    SCANNED pages additionally get an orientation probe (ledger #28): a
    landscape/rotated scan OCRs into garbage in every later stage, and
    triage is the last point where fixing it is cheap — one in-memory
    ``set_rotation`` here and rendering, YOLO, OCR, and table crops all
    see the page upright. The probe costs one low-DPI OCR per healthy
    scanned page; ``fix_orientation=False`` skips it for callers that
    only need the classification.

    Deliberately single-threaded; see the module docstring for why a
    thread pool would be a bug here.
    """
    # load_page rather than iterating the Document: iteration and
    # __getitem__ are loosely typed in the stubs (getitem also accepts
    # slices), load_page is typed -> Page — and the index is needed for
    # the record anyway.
    records = [triage_page(doc.load_page(i), i) for i in range(doc.page_count)]

    if fix_orientation:
        from .ocr import detect_orientation  # deferred: pulls numpy + tessdata

        for r in records:
            if r.kind != PageKind.SCANNED:
                continue
            page = doc.load_page(r.page)
            delta, before, after = detect_orientation(page)
            if delta:
                page.set_rotation((page.rotation + delta) % 360)
                r.rotation_applied = delta
                r.reason += f"; rotated {delta} deg (real words {before} -> {after})"
                log.info(
                    "p%04d: orientation fixed by %d deg (real words %d -> %d)",
                    r.page,
                    delta,
                    before,
                    after,
                )

    counts: dict[str, int] = {}
    for r in records:
        counts[r.kind.value] = counts.get(r.kind.value, 0) + 1
    log.info("triage: %d pages -> %s", len(records), counts)
    for r in records:
        log.debug("  p%04d %-12s %s", r.page, r.kind.value, r.reason)
    return records
