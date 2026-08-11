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

Phases 4–6 append their sections here as they land.
