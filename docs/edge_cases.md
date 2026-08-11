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
- **Handling:** `handled` (Phase 7) — `pipeline._open_checked` rejects
  encrypted, unreadable, and zero-page PDFs before any stage runs; the
  manifest records `status="rejected"` plus a human-readable reason and
  the CLI exits 2. No raw library traceback, no partial artifact tree
  for a later `--from-stage` run to trust.

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
- **Handling:** `accepted` at word level, `handled` at page level
  (Phase 7, case 28) — per-word confidence is still unavailable, but
  `ocr_quality_score` now judges each page's aggregate OCR output and
  flags garbage pages `needs_review`, so junk can no longer enter the
  corpus *silently*. OCR_DPI upsampling cannot restore detail the scan
  never captured; quality is capped at scan time.
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

### 24. Text splitting: library behind a seam (revised)

- **Symptom:** chunk sizing needs a tokenizer and sentence splitting;
  a ~4-chars/token estimate can overshoot an embedding model's real
  limit by 20-30% on number-dense text.
- **Handling:** `handled` — started hand-rolled (to learn the shape),
  then swapped Chonkie's SentenceChunker in behind the `split_text()`
  seam: chunks are now sized by a real tokenizer (gpt2 as proxy until
  the retrieval side picks an embedding model). The API-drift warning
  proved true live: the constructor kwarg is `tokenizer` in 1.7.0 and
  was named differently in other releases — hence the exact pin.
- **Honest limits:** the library's sentence boundaries are no smarter
  than a naive regex (splits after "e.g. ", "No. 42"); its
  structure-aware recipes are deliberately unused because structure is
  exploded into typed units *before* chunking. The seam is the design:
  swapping implementations touched one function and one config block.

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
- **Handling:** `handled` (Phase 8, case #30) — exactly the repetition
  analysis the production note called for, tuned on the real corpus it
  needed: units in the top/bottom page band whose digit-normalized text
  repeats at the same y on enough pages are stripped before assembly
  (`assemble.strip_repeated_furniture`).

---

## Phase 7 — Complex tables + hardening

Torture-tested against `sample_data/complex_doc.pdf`
(`python -m rag_ingest.complex_pdf`): merged cells in both tiers,
two-tier headers, a row-span crossing a page break, a borderless table,
a rotated scan, and a 31-row schedule vs row-group chunking.

### 27. Merged cells (row-span / col-span), both tiers

- **Symptom:** merged cells break every uniform-grid assumption. Tier 1
  flattened find_tables' span info into `""`, so a category label
  existed on one row and vanished from the rest; tier 2 was worse — its
  imposed uniform grid put a vertically-centered span label into the
  *middle* row of its span, and blanket line-erasure sliced through
  span labels before OCR ("Item" → "tein"). Row-group chunking then
  stranded rows from their labels entirely: a chunk of rows 21-30 had
  no idea it was "Electrical".
- **Handling:** `handled` — the cell contract is now UNMERGED: a merged
  value is repeated into every grid position it covers, in both tiers.
  Tier 1 reads span geometry from find_tables (`None` = covered, with
  the anchor's bbox spanning the merge); tier 2 checks each candidate
  cell boundary for ink individually, union-finds unbordered neighbors
  into regions, erases only borders that exist, and assigns each
  region's words to all its cells. The printed layout is preserved
  alongside as `merges` ([row, col, rowspan, colspan]) in
  06_tables.jsonl, and `cells_to_grid` redraws it as an ASCII box grid
  (merged cells = one box) that merged.md embeds — visually the printed
  table. Multi-row headers (`header_rows`)
  repeat in full in every row-group chunk; a row-span crossing a page
  break is filled onto the continuation rows when the previous fragment
  ends with a run (>= 2) of identical values in that column.
- **Honest limits:** a merge covering more than ~half the table's width
  can drop a whole grid line below the detection threshold (two rows
  collapse); the cross-page fill heuristic can mis-fill when two
  *identical adjacent data values* happen to end a page. Both fall
  toward `needs_review` or a visible artifact, never silent loss.

### 28. Rotated / landscape scans OCR into confident garbage

- **Symptom:** a scanned page with /Rotate 90 (or a landscape scan)
  fed Tesseract sideways text. The output — symbol soup — passed every
  structural check and shipped as confident chunks: the exact "silent
  garbage" failure class this pipeline promises not to have. Found live
  on the complex sample before the fix (quality 0.50).
- **Handling:** `handled` — two layers. (1) Triage probes every SCANNED
  page with a cheap low-DPI OCR and, when warranted, applies a rotation
  in-memory (`set_rotation`), so rendering, YOLO, OCR, and table crops
  all see the page upright (the fix is recorded in the triage artifact
  and reapplied on `--from-stage` resume). (2) What still scores below
  the threshold — stage-5 prose or a tier-2 table — is flagged
  `needs_review`, propagated through to chunks.
- **Revised on real data (Phase 8):** the original score-threshold rule
  was wrong twice on a real corpus. Upright-but-noisy scans score below
  the full-DPI bar at probe DPI, so every healthy scanned page paid the
  full 4-rotation search (~8s/page); and a sideways scanned DRAWING got
  rotated the WRONG way — its text is dimension numbers, and digits
  carry no chirality (a sideways "6000" reads fine), so the two
  landscape orientations were a coin flip. Neither signal alone
  separates the cases (measured: scores 0.57-0.61 in all four
  orientations on the drawing; word counts highest at the wrong
  rotation). The rule now uses BOTH: >= 40 real words at the current
  orientation = upright, exit; a rotation is applied only on a clear
  score win (>= 0.8, gap >= 0.25) AND >= 15 real words. Number-heavy
  pages fail the word floor in every orientation and are left alone for
  the quality gate to flag. Known limit: a 180-degree flip of dense
  text can read enough junk "words" to take the early exit; Tesseract's
  real OSD (not exposed by the bundled integration) is the production
  answer.

---

## Phase 8 — Real-corpus findings

The first three documents of a 10-document real evaluation corpus
(bilingual procurement-style PDFs, not committed) surfaced two failure
classes no synthetic sample had predicted. Both are the valuable kind:
invisible in logs, obvious in output.

### 29. The text layer is present but LYING (broken ToUnicode CMap)

- **Symptom:** bilingual pages whose embedded Indic font renders
  perfectly but maps glyphs to garbage codepoints — extraction emits
  `\x01`-riddled mojibake for every Hindi string while English extracts
  cleanly. ~33% of chunks on affected documents carried mojibake, and
  nothing flagged them: the quality gate only judged OCR text, and this
  is "trusted" native text. Endemic in certain government/enterprise
  PDF toolchains.
- **Handling:** `handled` — three layers. (1) Triage counts junk chars
  (C0 controls, U+FFFD, Private Use Area — healthy text layers contain
  ZERO, measured; broken pages carry 20-451): past either threshold the
  page reroutes to the OCR path, where Tesseract reads the rendered
  glyphs the text layer couldn't express. (2) OCR runs multilingual
  (`OCR_LANGUAGES = "eng+hin"`, per-language tessdata auto-download) —
  OCRing these pages in English only would re-lose the same text.
  (3) Mostly-clean pages that stay native get a unit-level net: any
  unit still containing junk chars is flagged `needs_review`, and a
  tier-1 table with junk in its cells flags the same way. (4) Rerouted
  pages are vector-crisp, so page borders sit millimeters from table
  borders and the pixel grid reads the gap as empty rows/columns —
  `drop_empty_lines` removes all-empty lines (merges remapped) before
  validation, restoring the true table shape.
- **Scoring lesson (found live by a test):** `str.isalpha()` rejects
  most real Hindi — Indic vowels are COMBINING MARKS (बंद carries
  U+0902, category Mn), not letters. The quality scorer counts
  letters-plus-marks as words, or legitimate Devanagari OCR output
  would have been flagged as garbage by the very gate meant to protect
  it.

### 30. Page furniture masquerading as tables

- **Symptom:** a bordered page-title box repeats on every page; YOLO
  calls it a table each time (conf ~0.5 vs ~0.97 for real tables), and
  find_tables rightly disagrees — so the borderless-suspect cross-check
  opened 11 needs_review items on one 33-page document, all the same
  box. Review noise is a real production failure: a drowning reviewer
  stops reading flags.
- **Handling:** `handled` — yolo_only suspects recurring at the same
  position (bbox rounded to 10pt) on >= FURNITURE_MIN_REPEATS distinct
  pages are page decoration and are suppressed; a suspect appearing
  once stays a review item. Case #26's unit-level strip removes the
  same furniture from prose chunks.
- **Residual risk:** a genuine borderless table repeated at the same
  position on 3+ pages would be suppressed with the furniture —
  accepted; such a table is itself boilerplate.

---

The pipeline is complete; future phases (retrieval, evaluation) extend
from the chunk store, not this file.
