# Edge-case ledger

Every edge case this pipeline meets, with the symptom, what we chose to do,
and what a production system at scale would do differently. Code comments
reference this file; this file references code. Grows with each phase.

**Status legend:**

- `handled` — implemented in code
- `flagged` — detected and marked `needs_review`, not auto-fixed
- `accepted` — consciously not handled; cost/benefit documented

---

## Phase 1 — Triage

### 1. Header/footer text layer over a scanned body

- **Symptom:** scanners often stamp page numbers/headers as real text on
  scanned pages. Text-length triage says TEXT_NATIVE; local extraction then
  emits a header and *silently drops the entire page body*. Worst class of
  failure: nothing errors, retrieval just can't find the page's content.
- **Handling:** `handled` — if any single raster image covers >70% of the
  page, classify SCANNED regardless of text length
  ([triage.py](../src/rag_ingest/triage.py), `SCAN_IMAGE_COVERAGE`).
- **Production note:** same approach; some systems additionally compare
  text-layer bbox area vs page area (text covering <5% of a page that has
  an image is suspicious even below 70% coverage).

### 2. Vector-drawing pages (CAD plans, engineering drawings)

- **Symptom:** no text layer, no raster image — plain triage calls it
  SCANNED and pays a vision model to hallucinate prose over a piping
  diagram.
- **Handling:** `handled` — near-textless pages with ≥100 vector segments
  become `DRAWING`: rendered to PNG, stored as a figure, never sent for
  text extraction (`DRAWING_MIN_SEGMENTS`).
- **Production note:** a real system might caption these with a vision
  model ("Site layout plan, sheet 4 of 12") to make them retrievable.
  We do this for embedded figures in Phase 4; drawing pages reuse that path.

### 3. Near-blank pages (title pages, separators)

- **Symptom:** "DOCUMENT No. 42" is ~15 chars → classified SCANNED → one
  needless vision-API call that returns 15 correct characters.
- **Handling:** `accepted` — the misroute costs cents and the output is
  still correct. The alternative (lowering `MIN_TEXT_CHARS`) risks the far
  worse failure of case 1. Thresholds are biased so that errors fall
  toward "wasted API call", never toward "silent garbage".

### 4. PyMuPDF is not thread-safe

- **Symptom:** the intuitive "thread pool over cores" for triage would
  share one `pymupdf.Document` across threads → segfaults/corruption.
- **Handling:** `handled` — triage is deliberately single-threaded
  (~10–20 s for 3000 pages; see [triage.py](../src/rag_ingest/triage.py)
  module docstring).
- **Production note:** for true parallelism, shard page ranges across
  *processes*, one `Document` open per process. Worth it for rendering
  (stage 3, CPU-heavy), not for triage.

### 5. Encrypted / corrupt / zero-page PDFs

- **Symptom:** `pymupdf.open()` raises, or `page_count == 0`.
- **Handling:** `accepted` for now — the pipeline fails fast with the
  library's own error before any stage runs. Phase 6 wraps this in a
  clean rejection recorded in the manifest.

---

## Phase 2 — Local extraction

### 6. Span vs line vs block granularity

- **Symptom:** emitting a unit per *span* fragments "the **supplier** shall"
  into three units; per *block* merges a heading into its following
  paragraph. Both wreck chunk quality downstream.
- **Handling:** `handled` — classify per LINE (a line is heading or body,
  never both), merge consecutive body lines of a block into one paragraph
  unit ([local_extract.py](../src/rag_ingest/local_extract.py)). The
  line's identity comes from its *dominant span by character count*, so a
  single bold word can't turn a sentence into a heading.

### 7. Duplicate / degenerate vector segments

- **Symptom:** PDF writers emit the same rule twice (path-closing
  segments, re-stroked borders) — MuPDF reports each one, inflating raw
  segment counts. Found live: our own sample's 9-line grid came back as
  10 items, the extra being a reversed duplicate of the last vertical.
  Related trap: a zero-height line is an "empty" rect in PyMuPDF, and
  empty rects are *ignored* by rect union — so unioning line rects
  silently produces a garbage bbox.
- **Handling:** `handled` — segments dedupe into sets of rounded
  coordinate tuples; the grid bbox is explicit min/max, never Rect union.

### 8. Tiny embedded images (logos, watermarks, bullet glyphs)

- **Symptom:** a letterhead logo on every page becomes hundreds of
  near-identical figure chunks that retrieval can hit instead of content.
- **Handling:** `handled` — images under 0.5% of page area are skipped
  (`FIGURE_MIN_AREA_FRAC`).
- **Production note:** a real system might also dedupe by image hash —
  the same logo has the same xref/bytes on every page.

### 9. Headings at body size, not bold, not numbered

- **Symptom:** a document that styles headings only via spacing or color
  defeats both heading rules; those headings become plain TEXT units.
- **Handling:** `accepted` — the breadcrumb for affected sections
  attaches to the nearest detected ancestor, which degrades retrieval
  precision but never correctness.

### 10. Text inside table regions is (currently) duplicated

- **Symptom:** stage 2 extracts ALL text on a page — including text
  inside tables. When stage 5 extracts the same table via the vision
  path, the content would appear twice after assembly.
- **Handling:** `flagged (forward)` — final table bboxes only exist after
  stage 4 (YOLO); the dedup rule "drop text units whose bbox falls inside
  a table region" is applied there, not here. Stage 2 deliberately emits
  everything with exact bboxes so stage 4 *can* filter. See design spec §3.

---

## Phase 3 — Rendering + layout detection

### 11. Pixel-vs-point coordinate confusion

- **Symptom:** YOLO boxes are in rendered-image pixels; PyMuPDF crops in
  PDF points. The naive `72/DPI` constant breaks silently on rotated
  pages and non-origin cropboxes — crops shift, nobody errors.
- **Handling:** `handled` — one helper
  ([layout.py](../src/rag_ingest/layout.py) `pixel_rect_to_pdf`) derives
  scale from *actual* dimensions (`page.rect` vs pixmap size), clamps
  into the page, and asserts non-degeneracy. Pixel coordinates never
  leave stage 4; the stage artifact records both boxes side by side so
  every conversion is auditable.

### 12. RGB vs BGR channel order

- **Symptom:** the YOLO wrapper follows OpenCV conventions — ndarray
  input is assumed BGR. Feeding RGB swaps red/blue; for layout detection
  the accuracy loss is small, which makes it the worst kind of bug:
  quietly present, never crashing.
- **Handling:** `handled` — explicit `[:, :, ::-1]` flip at the model
  boundary, with a comment. "Mostly harmless" is not a contract.

### 13. Memory at 3000 pages

- **Symptom:** a 200-DPI A4 pixmap is ~11 MB raw; render-then-detect as
  separate full passes holds gigabytes.
- **Handling:** `handled` — stages 3 and 4 interleave in one per-page
  loop; each pixmap dies before the next renders. The debug JPEG is a
  separate low-res render, NOT a shrink of the pipeline pixmap: touching
  `.samples` caches a memoryview, `shrink()` reallocates the buffer
  under it, and PyMuPDF's destructor then warns "operation forbidden on
  released memoryview" on every page (found live on the first full run).
  Sharing a mutable buffer between consumers wasn't worth the few ms a
  fresh render costs.

### 14. Torch as a deployment dependency

- **Symptom:** the model itself is ~40 MB but rides on a ~2.5 GB PyTorch
  stack — container bloat, slow cold starts.
- **Handling:** `accepted` for this repo — the framework wrapper owns
  pre/post-processing (letterboxing, NMS, coordinate mapping), which is
  exactly where hand-rolled bugs live.
- **Production note:** export to ONNX and run under `onnxruntime`
  (~50 MB) once behavior is pinned by tests; that also unlocks
  quantization. Standard sequencing: correctness on the framework first,
  ONNX as a deployment optimization second.

---

## Phase 4 — OCR (scanned pages)

### 15. Tesseract deployment: bundled vs system vs Docker

- **Symptom:** OCR engines are system binaries; "pip install" doesn't
  cover them, and every deployment option changes the code shape.
- **Handling:** `handled` — PyMuPDF wheels BUNDLE libtesseract; only the
  language data file is fetched (auto-download to .tessdata/, same
  pattern as YOLO weights). Zero system installs, zero containers, and
  `get_textpage_ocr` returns the same textpage structure as native text,
  so the stage-2 walk runs on scanned pages unchanged.
- **Production note:** at scale, OCR-as-a-service in docker-compose
  (e.g. hertzg/tesseract-server) gives OCR its own CPU pool and
  independent scaling — at the cost of an HTTP hop and rebuilding the
  block/line mapping from hOCR/TSV yourself.

### 16. OCR confidence and scan-resolution limits

- **Symptom:** stamps, signatures, and handwriting OCR into junk words;
  and low-resolution scans lose word boundaries — a 150-DPI 11pt test
  scan came back with words glued together because inter-word gaps fell
  below the space-synthesis threshold.
- **Handling:** `accepted` — the bundled integration does not expose
  per-word confidence, so junk words pass through untagged for now.
  OCR_DPI upsampling cannot restore detail the scan never captured;
  quality is capped at scan time.
- **Production note:** the pytesseract TSV route or an OCR service
  exposes per-word confidence — filter below ~60 and flag the page
  `needs_review`. That is the right tool against stamps/handwriting;
  a "better engine" is not.

### 17. OCR textpages emit words as separate spans

- **Symptom:** on native textpages, spaces live inside word spans; on
  OCR textpages every word is its own span with whitespace-only spans
  between. Our extraction walk filtered whitespace spans before joining
  — silently gluing all OCR text into `Thesuppliershallcomplete...`.
- **Handling:** `handled` — join all spans (then collapse whitespace)
  for the line text; filter whitespace spans only for classification.
  The general lesson: an invariant that holds for one producer of a
  shared structure will silently break for the next producer.

### 18. Gemini dropped — the fallback tier is a human, not a model

- **Symptom:** the original design used a vision LLM for scanned tables,
  borderless tables, and figure captions.
- **Handling:** `accepted` (decision, 2026-08-11) — the target corpus
  has bordered tables and machine-typeset scans, so the paid tier was
  dropped entirely. Tables: tier 1 `find_tables()` (text-native),
  tier 2 image-line grid + per-cell OCR (scanned); anything failing
  validation gets `needs_review=true` plus a stored crop PNG for human
  review instead of an API fallback. Figures get no captions; their
  retrievability relies on breadcrumbs + nearby caption text (Phase 6).
- **Production note:** if borderless tables ever appear in the corpus,
  the tier-3 slot is where a VLM plugs back in — the validation gate
  that would route to it already exists.

---

## Phase 5 — Tables

### 19. Tesseract silently drops text inside ruled cells

- **Symptom:** full-page OCR of a scanned page containing a ruled table
  returned only the heading above the table — every cell was skipped.
  Tesseract's layout analysis treats tightly ruled regions as non-text.
  No error, no warning: the words simply don't exist in the output.
- **Handling:** `handled` — tier 2 erases the grid lines before OCR
  (line-removal preprocessing, the standard OCR-pipeline fix). We get it
  nearly free: grid detection has already located every line, so tier 2
  erases a ±2 px band per line, OCRs the cleaned crop, and re-anchors
  the words into cells using the kept grid geometry
  ([tables.py](../src/rag_ingest/tables.py) `extract_scanned_table`).

### 20. The wrapper PDF's coordinate scale — the stage-4 lesson, again

- **Symptom:** words OCR'd from the cleaned crop landed in the wrong
  cells: the code assumed `pdfocr_tobytes`'s page uses 1 px = 1 pt. It
  doesn't — pdfocr picks its own page scale. Every trusted-constant
  assumption about coordinate systems eventually breaks.
- **Handling:** `handled` — scale derived from actual dimensions
  (pixmap size vs OCR page rect), exactly like stage 4's
  `pixel_rect_to_pdf`. Second occurrence of the same bug class in one
  codebase; the rule generalizes: *at every boundary between coordinate
  systems, measure — never assume.*

### 21. Column mismatch across a page boundary

- **Symptom:** a table fragment on page N+1 whose column count differs
  from page N's may be a continuation with a merged/split column — or a
  different table entirely. Merging on a guess corrupts both.
- **Handling:** `handled` (by refusing) — stitching requires equal column
  counts; mismatches stay separate fragments and fail validation into
  `needs_review` with stored crops. Guessing at structure is how silent
  corruption happens; a human resolves the ambiguity.

---

## Phase 6 — Assembly + chunking

### 22. `insert_textbox` silently renders NOTHING on overflow

- **Symptom:** the sample's page-6 prose never existed: PyMuPDF's
  `insert_textbox` does not truncate text that overflows its rect — it
  refuses to render *anything* and only signals via a negative return
  value nobody checks. Invisible until a Phase-6 test asserted on that
  paragraph.
- **Handling:** `handled` in the generator (text sized to fit, loud
  comment). The transferable lesson: APIs that signal failure through
  return values instead of exceptions produce bugs that surface far from
  their cause — a test that asserts on *content*, not just on "no
  crash", is what catches them.

### 23. Dedup by center containment, not intersection

- **Symptom:** table bboxes are padded (stage 4) or slightly generous
  (find_tables); any-intersection dedup would delete the prose line that
  merely touches a table's padded edge — silent text loss, the failure
  class this pipeline is designed to never have.
- **Handling:** `handled` — a text unit dies only when its bbox CENTER
  lies inside a table span. A unit genuinely inside a table has its
  center there; a neighbor never does.

### 24. Hand-rolled splitter and estimated token counts

- **Symptom:** chunk sizing needs a tokenizer and a splitting library;
  both are dependencies with API drift.
- **Handling:** `accepted` — a ~20-line paragraph/sentence splitter
  (never splits mid-sentence) and ~4-chars/token estimation. This repo
  exists to show internals; the seams are one function each.
- **Production note:** swap `split_text()` for a chunking library
  (Chonkie et al.) and `_estimate_tokens()` for the embedding model's
  real tokenizer. Nothing else changes — that isolation is the design.

### 25. Sparse-OCR pages: heading detection degrades gracefully

- **Symptom:** on a scanned page whose only OCR-visible text is its
  heading (e.g. a table-only page, where Tesseract skips ruled cells —
  case 19), the per-page body font size (case 17's fix) *equals* the
  heading size, so the size rule cannot fire and the heading lands as a
  TEXT chunk under the previous section's breadcrumb.
- **Handling:** `accepted` — two correct fixes composing into a small
  gap. The content is preserved and retrievable; only its depth in the
  heading tree is lost. Visible live in the sample: the scanned annex's
  heading chunks as text. A fix (numbered-pattern rule without the bold
  requirement for OCR sources) is noted but not worth its false-positive
  risk yet.

### 26. Scanner-stamped headers/footers pollute content

- **Symptom:** a repeating page header ("DOCUMENT NO 42/2026 - PAGE 17
  OF 300") lands inside prose chunks: on scanned pages the OCR render
  includes the stamped text layer, and nothing distinguishes it from
  body text. Visible live in the sample's merged.md.
- **Handling:** `accepted` for now — it costs retrieval a little noise,
  never correctness.
- **Production note:** the standard fix is repetition analysis: lines
  occurring on many pages at the same y-position are headers/footers —
  strip them before assembly. Needs a real multi-page corpus to tune;
  pointless to fake on a 10-page synthetic.

---

The pipeline is complete; future phases (retrieval, evaluation) extend
from the chunk store, not this file.
