# Design spec — Gemini VLM extractor (paid lane)

Status: **approved for implementation** (2026-08-11).
Supersedes: design_spec.md §7's "no LLM tier" decision, and all of stage 5
(Tesseract OCR). The rest of design_spec.md stands.

This document is self-contained: it carries the diagnosis, the decisions
already made, the module contract, and the migration order. An implementer
should be able to work from this file plus the existing code.

---

## 1. Why (diagnosis, from real output)

Hindi extraction fails today in two distinct ways, verified on
`output/GeM-Bidding-9419898/merged.md` (bilingual GeM bid document):

1. **Text-native pages with lying text layers.** Custom-encoded Devanagari
   fonts with broken/partial ToUnicode CMaps render correctly but extract
   as printable-ASCII mojibake (`लाभाथ\ के प] म= होनी चा<हए`) and
   visual-order matras (`रािश` for `राशि`). `JUNK_CHARS_RE` in triage.py
   only counts C0 controls / U+FFFD / PUA, so these pages pass as healthy
   natives and ship garbage unflagged. PyMuPDF is not at fault — no
   extractor can recover a lying CMap from the text layer; the page must
   be re-read from pixels.

2. **Scanned pages through Tesseract `eng+hin`.** Devanagari words glued
   together (`लगानेकीसमयसीमास्वतःनहीं`), Devanagari misread as Latin
   shrapnel (`fas` for बोली, `vefa`, `Gs`, `faaxor`). The quality gate in
   ocr.py is Latin-biased (any non-Latin letter run passes; short ASCII
   words with vowels pass), so none of this got `needs_review`.

Independent evidence (arXiv 2606.29213, real printed Devanagari scans):
Gemini-class VLMs lead at ~86 chrF++; Tesseract-class engines sit near 58.
The product goal is truly multilingual; per-language traineddata and
per-script quality heuristics do not scale. Decision: **one VLM code path
for every script**, engine = Gemini (config-pluggable).

## 2. Decision summary

| Component | Verdict | Role after this change |
|---|---|---|
| Triage + health checks | keep, extend | per-page router: free lane vs paid lane |
| local_extract.py (PyMuPDF) | keep unchanged | free lane for honest text-native pages, exact bboxes |
| render.py | keep unchanged | page PNGs feed YOLO and Gemini |
| layout.py (YOLO) | keep | figure crops; region bboxes for the paid lane; missed-table cross-check |
| ocr.py (Tesseract) | **delete** | replaced by vlm_extract.py |
| tables.py tier-2 (pixel grid + OCR words) | **delete** | paid-lane pages come back with tables inline |
| tables.py tier-1 (native) + multi-page stitching + furniture | keep | unchanged; stitching is format logic, not extraction logic |
| assemble.py + chunking.py | keep, one small change | honor explicit `level` on TITLE units (see §5.3) |
| vlm_extract.py | **build** | this spec |

Economic shape (unchanged in spirit): most pages of most PDFs are honest →
extracted locally for free. Only pages that fail triage pay the API cost.
On the sample corpus that is a minority of pages; cost per dense page on a
Flash-class model is well under one US cent.

## 3. Triage changes (the router gets script-aware)

`PageKind.SCANNED` already means "image → Gemini" (models.py wrote this
down before the engine existed). The reroute mechanism for lying text
layers (ledger #29) also already exists. What changes is **detection**:

Add a script-agnostic mojibake score alongside `text_layer_junk()`.
Signals, computed over extracted page text (all are Unicode-general — no
per-language tables, or they defeat the purpose):

* **Orphan combining marks**: a char of category Mn/Mc whose preceding
  base char is not a letter (or is a letter of a different script block).
  Healthy Indic text ~0; CMap-broken pages are full of them.
* **ASCII symbols interleaved inside non-Latin words**: within a
  whitespace-token containing non-Latin letters, count chars from
  `\\ ] [ = < > ^ _ @ # | ~` (NOT ./,-%() which occur legitimately).
* Existing C0/PUA junk count (keep as-is).

Page reroutes to the paid lane when the combined mojibake count crosses
thresholds analogous to `TEXT_LAYER_JUNK_MIN` / `_RATIO` (new constants,
calibrate on the GeM corpus the same way #29's were — measure healthy vs
broken pages, put the numbers and the measurement in the config comment).

Keep the unit-level safety net in local_extract.py (`needs_review` on
mojibake units that survive on mostly-clean pages) and extend its regex
check with the same orphan-combining-mark test.

DRAWING classification stays — CAD pages must still never reach the API.

## 4. The extractor module: `src/rag_ingest/vlm_extract.py`

Replaces ocr.py as stage 5. One public function mirroring the old seam:

```
def vlm_page_units(page_png: bytes, page_index: int, page_rect: BBox,
                   yolo_tables: list[BBox], client: VlmClient) -> list[Unit]
```

plus a `VlmClient` protocol (call: PNG bytes + prompt → text) with a
`GeminiClient` implementation. The protocol seam exists for tests (fake
client, no network) and for future engine swaps (Sarvam Vision is the
standing challenger; engines are perishable — Gemini 2.5 Flash retires
2026-10-16 — the seam is what is built "once for all").

### 4.1 API

* SDK: `google-genai` (the current SDK — `from google import genai`), NOT
  the deprecated `google-generativeai`.
* Auth: `GEMINI_API_KEY` env var. Never written to config or artifacts.
* Model: `config.VLM_MODEL`, default `"gemini-3-flash"` — **verify the
  exact current model id at implementation time** (`client.models.list()`
  or ai.google.dev); Flash tier, not Pro (benchmarks show no OCR gain
  from Pro; cost is ~10x).
* Input: the stage-3 render at `RENDER_DPI` (reuse it; do not re-render).
* Deterministic settings: temperature 0.

### 4.2 Prompt contract (v1 — version it, see caching)

The prompt requests GitHub-flavored markdown with these rules, stated
explicitly to the model:

* Transcribe EXACTLY what is printed. Never translate, never summarize,
  never "fix" spelling. Preserve every script as written; bilingual lines
  (`विवरण/Bid Number`) stay verbatim on one line.
* Reading order, top to bottom. Headings as `#`–`######` by visual
  hierarchy; body text as paragraphs separated by blank lines.
* Tables as GFM pipe tables; a visually merged cell repeats its value in
  each row it spans; keep row/column structure faithful.
* Numbers, dates, codes: character-exact.
* A region that is genuinely illegible becomes the literal token
  `[ILLEGIBLE]` — never a guess.
* Output ONLY the markdown. No code fences, no commentary, no preamble.

### 4.3 Markdown → Units parser (deterministic, no LLM)

* `#{n} text` → TITLE unit, `level=n`, `font_size=None`.
* GFM table block → TABLE unit, `content` = the markdown table.
* Everything else → TEXT units per blank-line-separated paragraph.
* `source=Source.GEMINI` on every unit (new enum value; keep
  `TESSERACT_OCR` in the enum so `from_dict` can rehydrate old artifacts —
  mark it deprecated in the docstring).
* Bboxes: TEXT/TITLE units get the full-page rect (paid-lane provenance
  is page-level — an accepted, recorded trade). TABLE units: when the
  page's YOLO table detections and the markdown's tables agree in count,
  assign YOLO bboxes to tables in reading order; otherwise full-page rect.
  This restores region-level provenance on exactly the units where
  citations care most.

### 4.4 Caching (re-runs must be free)

Key = SHA-256 over (PNG bytes, model id, prompt version). Value = raw
model text + usage metadata, stored as one JSON file per page under
`output/<doc_id>/cache/vlm/`. On hit, skip the API entirely. This is what
makes `--from-stage` and threshold-tuning iterations affordable.

### 4.5 Operational plumbing

* Retries: exponential backoff with jitter on 429/5xx/timeouts,
  `VLM_MAX_RETRIES` (default 4). A page that exhausts retries yields ONE
  full-page TEXT unit with empty content and `needs_review=True` — a page
  must never silently vanish from the corpus.
* Cost accounting: write per-page input/output token counts (from the API
  response's usage metadata) into the stage artifact, and a per-document
  total to the log. Record tokens, not currency — prices change.
* Sequential per-document requests are fine at current volumes; note the
  Batch API (−50%) as the scaling lever in a comment, do not build it.

## 5. Verification (replaces `ocr_quality_score`)

Tesseract failed as symbol soup; VLMs fail as *fluent lies* — omission,
repetition loops, confident hallucination. Different gate, same
philosophy: **garbage never enters the corpus as confident text; failures
are flagged `needs_review`, never dropped.**

Checks on the raw markdown, per page:

1. **Repetition**: if any normalized substring of ≥ 20 chars repeats more
   than `VLM_MAX_REPEATS` (default 10) times, flag the page (the
   documented VLM failure mode is runaway loops emitting up to 71x the
   reference length).
2. **Length sanity, two-sided**:
   * Pages rerouted for lying CMaps have a text-layer char count C from
     triage (garbled but *countable*). Output outside
     `[C x VLM_LEN_LO, C x VLM_LEN_HI]` (defaults 0.3 / 3.0) → flag.
   * True scans have no C. Proxy: ink coverage of the render (fraction of
     pixels darker than `GRID_DARK_THRESHOLD`, reusing the tables.py
     constant). Coverage above `VLM_DENSE_INK_FRAC` with output under
     `VLM_MIN_CHARS_DENSE` chars → flag. Calibrate both on the corpus.
3. **Mojibake echo**: `JUNK_CHARS_RE` plus the §3 orphan-mark check on the
   output — a healthy VLM response contains zero; any hit → flag.
4. **YOLO cross-check**: YOLO found ≥ 1 table region on the page but the
   markdown contains no table → flag (silent-omission catch). The inverse
   (markdown table, no YOLO box) is NOT flagged — YOLO misses are why
   `YOLO_CONF_THRESHOLD` is already low.
5. **`[ILLEGIBLE]`**: any unit containing the token → that unit (not the
   whole page) gets `needs_review=True`.

### 5.3 Assembly change (small)

Gemini TITLE units carry explicit `level` and no `font_size`. The stage-7
heading clustering must prefer an explicit `level` when present and only
cluster by `font_size` for units that lack it (the PyMuPDF path is
unchanged). Levels still cap at `MAX_HEADING_LEVEL`.

## 6. Deletions

* `ocr.py` entirely: Tesseract calls, `ensure_tessdata`,
  `ocr_quality_score`, `_token_ok`/vowel heuristics, orientation probe
  (Gemini reads rotated pages; if corpus evidence later disagrees, rotate
  via a cheap render-time check, not via OCR probing).
* config: `TESSDATA_*`, `OCR_LANGUAGES`, `OCR_DPI`, `OCR_MIN_QUALITY`,
  `ORIENTATION_*`.
* tables.py: the tier-2 scanned-table path (pixel-grid + OCR word
  filling). Tier-1 native extraction, multi-page stitching, header
  dedup, and furniture stripping all stay.
* `.tessdata/` from the working tree and .gitignore.
* Tests covering the above (they assert Tesseract-specific behavior);
  port any that encode format-level logic (stitching, furniture) — those
  assertions are engine-independent.

## 7. Implementation order (each step leaves the repo green)

1. models.py: add `Source.GEMINI`; deprecation note on `TESSERACT_OCR`.
2. Triage: script-aware mojibake score + thresholds; tests with synthetic
   healthy/orphan-mark/interleaved-ASCII strings (assert expected counts,
   not observed — repo convention).
3. vlm_extract.py: parser + verification first (pure functions, fully
   testable offline with expected-value tests), then `GeminiClient`,
   caching, retries. All tests use a fake `VlmClient`.
4. pipeline.py: wire stage 5 to vlm_extract; route SCANNED + rerouted
   pages through it; stage artifact `stages/05_vlm.jsonl` (+ cache dir).
5. assemble.py: explicit-level handling (§5.3).
6. Deletions (§6) in one commit AFTER the new path passes on the sample
   corpus — diff `merged.md` before/after on GeM-Bidding-9419898; the
   Hindi table region (bid parameters) and the CMap-broken prose
   (`लाभाथ\`-class lines) are the acceptance cases.
7. docs: design_spec.md §7 gets a pointer to this file; edge_cases.md
   entries about tessdata/orientation get updated.

## 8. Out of scope (recorded so the next session doesn't re-litigate)

* Sarvam Vision / other engines: standing challenger via the `VlmClient`
  seam; bake-off on cached corpus pages when desired. Not v1.
* Docling: benchmark baseline and possible future DOCX/PPTX front-door.
  Not adopted as pipeline — its pipeline choice is per-document, not
  per-page, which breaks the free/paid routing this design is built on.
* Batch API, word-level bboxes on the paid lane, language tags on chunks
  (do add language tags when the retrieval side lands — one Unicode-block
  histogram per chunk).
