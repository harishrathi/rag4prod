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

Phases 2–6 append their sections here as they land.
