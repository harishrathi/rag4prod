"""STAGE 5 — VLM extraction: the paid lane, one code path for every script.

Replaces the Tesseract OCR stage (docs/gemini_extractor_spec.md). Pages
triage routed here — true scans AND text-native pages with lying CMaps —
are re-read from pixels by a vision-language model that returns the page
as GitHub-flavored markdown. A deterministic parser (no LLM) turns that
markdown into the same ``Unit`` contract every other extraction path
emits, so downstream stages never care which lane a page took.

Why a VLM and not per-language OCR: the product goal is truly
multilingual, and per-language traineddata plus per-script quality
heuristics do not scale. On real printed Devanagari scans, Gemini-class
VLMs lead at ~86 chrF++ where Tesseract-class engines sit near 58
(arXiv 2606.29213). The engine itself is config-pluggable behind the
``VlmClient`` protocol — engines are perishable; the seam is what is
built once.

Failure philosophy (same as the OCR gate it replaces, different checks):
Tesseract failed as symbol soup; VLMs fail as *fluent lies* — omission,
repetition loops, confident hallucination. verify_page_markdown() runs
per-page plausibility checks and anything suspicious ships flagged
``needs_review``, never silently. A page that fails the API entirely
becomes ONE empty flagged unit — a page must never vanish from the
corpus.

Cost shape: most pages of most PDFs are honest text-natives extracted
locally for free; only triage failures reach this module. Re-runs are
free: raw responses are cached on disk keyed by (PNG bytes, model id,
prompt version), which is what makes --from-stage iteration and
threshold tuning affordable. Sequential per-document requests are fine
at current volumes; the Batch API (-50% cost) is the scaling lever if
volume ever demands it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pymupdf

from .config import (
    GRID_DARK_THRESHOLD,
    VLM_DENSE_INK_FRAC,
    VLM_LEN_HI,
    VLM_LEN_LO,
    VLM_MAX_REPEATS,
    VLM_MAX_RETRIES,
    VLM_MIN_CHARS_DENSE,
    VLM_MODEL,
)
from .models import BBox, Source, Unit, UnitType
from .text_quality import JUNK_CHARS_RE, orphan_combining_marks

log = logging.getLogger(__name__)

ILLEGIBLE_TOKEN = "[ILLEGIBLE]"

# The prompt is part of the cache key (via PROMPT_VERSION): editing the
# wording invalidates cached responses ONLY if the version is bumped —
# bump it on any change that could alter model output.
PROMPT_VERSION = "v1"
PROMPT = """Transcribe this document page into GitHub-flavored Markdown.

Rules:
- Transcribe EXACTLY what is printed. Never translate, never summarize, \
never "fix" spelling or grammar. Preserve every script as written; \
bilingual lines (e.g. "विवरण/Bid Number") stay verbatim on one line.
- Follow visual reading order, top to bottom.
- Render headings as # to ###### according to their visual hierarchy. \
Render body text as paragraphs separated by blank lines.
- Render tables as GitHub-flavored pipe tables. When a cell visually \
spans several rows or columns, repeat its value in every row/column it \
covers. Keep the row and column structure faithful.
- Numbers, dates, and codes must be character-exact.
- If a region is genuinely illegible, write the literal token \
[ILLEGIBLE] instead of guessing.
- Output ONLY the markdown. No code fences, no commentary, no preamble."""


# ---------------------------------------------------------------------------
# Client seam
# ---------------------------------------------------------------------------


class VlmError(Exception):
    """The engine failed permanently for one page (config, auth, or
    retries exhausted). Callers turn this into a flagged empty unit —
    never into a lost page."""


@dataclass(frozen=True)
class VlmResponse:
    """One raw engine response. Token counts come from the API's usage
    metadata and are recorded per page — tokens, not currency, because
    prices change."""

    text: str
    input_tokens: int
    output_tokens: int
    cached: bool = False


class VlmClient(Protocol):
    """PNG bytes + prompt -> markdown. The seam exists for tests (a fake
    client, no network) and for engine swaps (Sarvam Vision is the
    standing challenger — see gemini_extractor_spec.md §8)."""

    model_id: str

    def generate(self, png: bytes, prompt: str) -> VlmResponse: ...


def _load_dotenv_key() -> None:
    """Fallback: read GEMINI_API_KEY from a ``.env`` file in the working
    directory (KEY=value lines, # comments). Deliberately minimal — one
    key, no dependency — and .env is gitignored: this is a public repo,
    the key must never be committable, and it is never written to config
    or artifacts."""
    env_file = Path(".env")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("GEMINI_API_KEY="):
            os.environ["GEMINI_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
            return


class GeminiClient:
    """Gemini via the google-genai SDK (the current one — ``from google
    import genai`` — NOT the deprecated google-generativeai).

    Auth is the GEMINI_API_KEY env var (or a gitignored ``.env`` file),
    read lazily on first call so a fully cached run never needs it.
    Never written to config or artifacts. Retries: exponential backoff
    with full jitter on 429/5xx/network errors, VLM_MAX_RETRIES attempts
    after the first.
    """

    def __init__(self, model: str = VLM_MODEL) -> None:
        self.model_id = model
        self._client = None

    def _load(self):
        if self._client is None:
            if not os.environ.get("GEMINI_API_KEY"):
                _load_dotenv_key()
            if not os.environ.get("GEMINI_API_KEY"):
                raise VlmError(
                    "GEMINI_API_KEY is not set (env var or .env file) — "
                    "the paid lane needs it for uncached pages"
                )
            from google import genai  # deferred: keep import cost off free-lane runs

            self._client = genai.Client()
        return self._client

    def generate(self, png: bytes, prompt: str) -> VlmResponse:
        import httpx  # google-genai's transport; its timeouts are retryable
        from google.genai import errors, types

        client = self._load()
        delay = 1.0
        for attempt in range(VLM_MAX_RETRIES + 1):
            try:
                resp = client.models.generate_content(
                    model=self.model_id,
                    contents=[
                        types.Part.from_bytes(data=png, mime_type="image/png"),
                        prompt,
                    ],
                    # Deterministic transcription — this is dictation, not
                    # generation.
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                usage = resp.usage_metadata
                out = VlmResponse(
                    text=resp.text or "",
                    input_tokens=(usage.prompt_token_count or 0) if usage else 0,
                    output_tokens=(usage.candidates_token_count or 0) if usage else 0,
                )
                # An HTTP-200 response with NO text is a real failure mode
                # (observed live: 0 candidate tokens on a normal page).
                # Retry it like a 5xx; if it stays empty, return it and
                # let verification flag the page.
                if out.text.strip() or attempt == VLM_MAX_RETRIES:
                    return out
                log.warning(
                    "empty response (attempt %d/%d) — retrying", attempt + 1, VLM_MAX_RETRIES
                )
            except errors.APIError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == VLM_MAX_RETRIES:
                    raise VlmError(f"Gemini API error {e.code}: {e.message}") from e
                log.warning("Gemini %s (attempt %d/%d)", e.code, attempt + 1, VLM_MAX_RETRIES)
            except (httpx.HTTPError, TimeoutError, ConnectionError) as e:
                if attempt == VLM_MAX_RETRIES:
                    raise VlmError(f"network failure after retries: {e}") from e
                log.warning("network error (attempt %d/%d): %s", attempt + 1, VLM_MAX_RETRIES, e)
            time.sleep(delay * (1 + random.random()))  # noqa: S311 - jitter, not crypto
            delay *= 2
        raise VlmError("unreachable")  # loop always returns or raises


class CachedVlmClient:
    """Disk cache in front of any engine (§4.4): one JSON file per page
    under <doc_out>/cache/vlm/, keyed by SHA-256 over (PNG bytes, model
    id, prompt version). On hit the API is skipped entirely."""

    def __init__(self, inner: VlmClient, cache_dir: Path) -> None:
        self.inner = inner
        self.cache_dir = cache_dir

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def generate(self, png: bytes, prompt: str) -> VlmResponse:
        h = hashlib.sha256()
        h.update(png)
        h.update(self.inner.model_id.encode())
        h.update(PROMPT_VERSION.encode())
        path = self.cache_dir / f"{h.hexdigest()}.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return VlmResponse(
                text=d["text"],
                input_tokens=d["input_tokens"],
                output_tokens=d["output_tokens"],
                cached=True,
            )
        resp = self.inner.generate(png, prompt)
        if not resp.text.strip():
            # Never cache an empty response: caching it would make a
            # transient engine failure permanent across every re-run.
            return resp
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model": self.inner.model_id,
                    "prompt_version": PROMPT_VERSION,
                    "text": resp.text,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return resp


# ---------------------------------------------------------------------------
# Markdown -> Units (deterministic, no LLM)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
# GFM delimiter row: | --- | :---: | ... (one cell minimum)
_TABLE_SEP_RE = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$")


def strip_outer_fence(md: str) -> str:
    """The prompt forbids code fences; models sometimes add one anyway.
    Unwrap a single fence enclosing the WHOLE response — never fences
    inside it (those are content)."""
    lines = md.strip().splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return md.strip()


def _parse_blocks(md: str) -> list[tuple[str, str | int, str]]:
    """Markdown -> [(kind, meta, content)]: ("title", level, text),
    ("table", "", markdown), ("text", "", paragraph). Pure function —
    the whole paid-lane parse is testable without a network."""
    blocks: list[tuple[str, str | int, str]] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(("text", "", "\n".join(para)))
            para.clear()

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
            i += 1
            continue
        m = _HEADING_RE.match(line)
        if m:
            flush()
            blocks.append(("title", len(m.group(1)), m.group(2)))
            i += 1
            continue
        # A table starts at a pipe row whose NEXT line is the delimiter
        # row; a lone pipe-looking line is prose.
        if (
            _TABLE_ROW_RE.match(line)
            and i + 1 < len(lines)
            and _TABLE_SEP_RE.match(lines[i + 1])
        ):
            flush()
            start = i
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                i += 1
            blocks.append(("table", "", "\n".join(lines[start:i])))
            continue
        para.append(line)
        i += 1
    flush()
    return blocks


def markdown_table_cells(table_md: str) -> list[list[str]]:
    """GFM pipe table -> cell matrix (delimiter row dropped, ``\\|``
    unescaped). The inverse of the format the prompt requests; stage 6
    feeds the result through the same validation/stitching path as every
    other table source."""
    rows: list[list[str]] = []
    for line in table_md.splitlines():
        if not line.strip() or _TABLE_SEP_RE.match(line):
            continue
        inner = line.strip().strip("|")
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", inner)]
        rows.append(cells)
    return rows


def parse_page_markdown(
    md: str, page_index: int, page_rect: BBox, yolo_tables: list[BBox]
) -> list[Unit]:
    """One page's markdown -> Units under the common contract.

    Bboxes (§4.3): TEXT/TITLE units carry the full-page rect — paid-lane
    provenance is page-level, an accepted and recorded trade. TABLE
    units get YOLO's region boxes when the page's YOLO table count
    matches the markdown's table count (assigned in reading order) —
    restoring region provenance exactly where citations care most.
    Ordering note: full-page bboxes make the stage-7 (page, y0) sort
    degenerate to emission order for paid-lane prose, which IS reading
    order; a YOLO-boxed table sorts by its real position.
    """
    blocks = _parse_blocks(md)
    n_tables = sum(1 for kind, _, _ in blocks if kind == "table")
    table_boxes: list[BBox] = []
    if n_tables and n_tables == len(yolo_tables):
        table_boxes = sorted(yolo_tables, key=lambda b: (b[1], b[0]))

    units: list[Unit] = []
    t_i = 0
    for kind, meta, content in blocks:
        if kind == "title":
            units.append(
                Unit(
                    page=page_index,
                    bbox=page_rect,
                    type=UnitType.TITLE,
                    content=content,
                    level=int(meta),
                    font_size=None,  # explicit level, no size evidence
                    source=Source.GEMINI,
                )
            )
        elif kind == "table":
            bbox = table_boxes[t_i] if table_boxes else page_rect
            t_i += 1
            units.append(
                Unit(
                    page=page_index,
                    bbox=bbox,
                    type=UnitType.TABLE,
                    content=content,
                    source=Source.GEMINI,
                )
            )
        else:
            units.append(
                Unit(
                    page=page_index,
                    bbox=page_rect,
                    type=UnitType.TEXT,
                    content=content,
                    source=Source.GEMINI,
                )
            )
    return units


# ---------------------------------------------------------------------------
# Verification (replaces ocr_quality_score — §5)
# ---------------------------------------------------------------------------


def ink_fraction(png: bytes) -> float:
    """Fraction of render pixels darker than GRID_DARK_THRESHOLD — the
    length-sanity proxy for true scans, where no text-layer char count
    exists to compare against."""
    pix = pymupdf.Pixmap(png)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return float((arr[:, :, :3].mean(axis=2) < GRID_DARK_THRESHOLD).mean())


def verify_page_markdown(
    md: str,
    *,
    text_layer_chars: int | None = None,
    ink_frac: float | None = None,
    yolo_table_count: int = 0,
) -> list[str]:
    """Plausibility checks on one page's raw markdown; returns the list
    of failure reasons (empty = trustworthy). Every check errs toward a
    review flag, never toward silent garbage.

    ``text_layer_chars`` is triage's char count for pages rerouted with a
    lying-but-countable text layer; ``ink_frac`` is the render's ink
    coverage for true scans. Callers pass whichever exists.
    """
    reasons: list[str] = []
    norm = " ".join(md.split())

    # 1. Repetition loops — the documented VLM runaway failure mode
    # (outputs up to 71x the reference length). Any 20-char window
    # recurring more than VLM_MAX_REPEATS times is not natural prose.
    win = 20
    if len(norm) > win:
        counts: dict[str, int] = {}
        worst = 0
        for i in range(len(norm) - win + 1):
            w = norm[i : i + win]
            counts[w] = counts.get(w, 0) + 1
            worst = max(worst, counts[w])
        if worst > VLM_MAX_REPEATS:
            reasons.append(f"repetition loop: a 20-char sequence recurs {worst}x")

    # 2. Length sanity, two-sided.
    if text_layer_chars is not None and text_layer_chars > 0:
        lo, hi = text_layer_chars * VLM_LEN_LO, text_layer_chars * VLM_LEN_HI
        if not (lo <= len(norm) <= hi):
            reasons.append(
                f"length {len(norm)} outside [{lo:.0f}, {hi:.0f}] "
                f"(text layer had {text_layer_chars} chars)"
            )
    elif ink_frac is not None:
        if ink_frac >= VLM_DENSE_INK_FRAC and len(norm) < VLM_MIN_CHARS_DENSE:
            reasons.append(
                f"page has {ink_frac:.1%} ink but only {len(norm)} chars came back "
                f"(silent omission?)"
            )

    # 3. Mojibake echo: a healthy VLM response contains zero junk chars
    # and zero orphan marks — any hit means the model echoed garbage.
    orphans = orphan_combining_marks(md)
    if JUNK_CHARS_RE.search(md) or orphans:
        reasons.append(f"mojibake in output ({orphans} orphan mark(s))")

    # 4. YOLO cross-check, one-directional: YOLO saw a table the markdown
    # lacks -> silent omission suspect. The inverse is NOT flagged —
    # YOLO misses are expected (its threshold is deliberately low).
    if yolo_table_count >= 1:
        has_table = any(kind == "table" for kind, _, _ in _parse_blocks(md))
        if not has_table:
            reasons.append(f"YOLO found {yolo_table_count} table region(s), markdown has none")

    return reasons


# ---------------------------------------------------------------------------
# Stage seam
# ---------------------------------------------------------------------------


@dataclass
class VlmPageRecord:
    """Per-page operational record for the stage artifact: cost
    accounting (tokens, cache hits) and which verification checks fired.
    This is the auditable trail of every API call the document cost."""

    page: int
    cached: bool
    input_tokens: int
    output_tokens: int
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def vlm_page_units(
    page_png: bytes,
    page_index: int,
    page_rect: BBox,
    yolo_tables: list[BBox],
    client: VlmClient,
    *,
    text_layer_chars: int | None = None,
) -> tuple[list[Unit], VlmPageRecord]:
    """One paid-lane page: engine call -> parse -> verify -> Units.

    ``text_layer_chars`` is passed for pages rerouted with a countable
    (garbled) text layer; true scans leave it None and the ink-coverage
    proxy takes over. A page that fails the engine permanently returns
    ONE empty flagged TEXT unit — pages never vanish.
    """
    try:
        resp = client.generate(page_png, PROMPT)
    except VlmError as e:
        log.error("p%04d: VLM failed permanently: %s", page_index, e)
        record = VlmPageRecord(
            page=page_index,
            cached=False,
            input_tokens=0,
            output_tokens=0,
            review_reasons=[f"vlm call failed: {e}"],
        )
        unit = Unit(
            page=page_index,
            bbox=page_rect,
            type=UnitType.TEXT,
            content="",
            source=Source.GEMINI,
            needs_review=True,
        )
        return [unit], record

    md = strip_outer_fence(resp.text)
    reasons = verify_page_markdown(
        md,
        text_layer_chars=text_layer_chars,
        ink_frac=ink_fraction(page_png) if text_layer_chars is None else None,
        yolo_table_count=len(yolo_tables),
    )
    units = parse_page_markdown(md, page_index, page_rect, yolo_tables)
    if not units:
        # Model returned nothing parseable — keep the page visible.
        reasons.append("empty response")
        units = [
            Unit(
                page=page_index,
                bbox=page_rect,
                type=UnitType.TEXT,
                content="",
                source=Source.GEMINI,
                needs_review=True,
            )
        ]

    if reasons:
        for u in units:
            u.needs_review = True
        log.warning("p%04d: flagged needs_review: %s", page_index, "; ".join(reasons))
    # [ILLEGIBLE] flags only the unit carrying it (§5 check 5), not the
    # whole page — the model being honest about one region is not doubt
    # about the rest.
    for u in units:
        if ILLEGIBLE_TOKEN in u.content:
            u.needs_review = True

    record = VlmPageRecord(
        page=page_index,
        cached=resp.cached,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        review_reasons=reasons,
    )
    return units, record
