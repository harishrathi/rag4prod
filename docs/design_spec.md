# Design spec — PDF ingestion pipeline for RAG

**Input:** large mixed-content PDFs — text-native pages, scanned pages, and
vector-drawing pages in one document (contracts, technical manuals, scanned
archives).
**Output:** retrieval-ready chunks that carry page numbers and bounding
boxes, so every answer can cite its source location.
**Stack:** PyMuPDF (local) + DocLayout-YOLO (local) + Gemini (vision API).
**Targets:** 3000-page PDF end-to-end in < 20 minutes; table accuracy is
prioritized over API cost.

This document is the *current* design: it started from an initial spec and
was revised in review. Where a revision overruled the original, the
reasoning is kept inline — those trade-off notes are the point of this repo.

---

## 0. Division of responsibility

| Component | Does | Does NOT |
| --- | --- | --- |
| **PyMuPDF** | open PDF, triage pages, extract text on text-native pages, render pixmaps, crop bboxes, extract embedded images | classify regions semantically, read scanned pages |
| **DocLayout-YOLO** | return `{label, bbox, conf}` for a page image | open PDFs, render, extract content |
| **Gemini** | convert page/crop images to markdown; caption figures | anything a local tool already does for free |

**Cost rule:** never send the API something a local tool already produced.
(Captioning figures is allowed — that's *description*, which no local tool
can do, not re-extraction.)

---

## 1. Stages and artifacts

| Stage | Module | Work | Artifact |
| --- | --- | --- | --- |
| 1 Triage | `triage.py` | page kind per page | `stages/01_triage.json` |
| 2 Local extract | `local_extract.py` | text/heading/figure units, free | `stages/02_units_local.jsonl` |
| 3 Render | `render.py` | page images for vision paths | `stages/03_render.json` |
| 4 Layout | `layout.py` | YOLO boxes, coord conversion | `stages/04_layout.jsonl` |
| 5 Gemini | `gemini_client.py` | tables + scanned pages -> markdown | `stages/05_gemini.jsonl` |
| 6 Assemble | `stitch.py`, `assemble.py`, `chunking.py` | merge, stitch, chunk | `stages/06_chunks.jsonl` |

Every stage checkpoints its full output to disk before the next stage reads
it. `--from-stage N` reloads stage N-1's artifact instead of recomputing —
re-tuning a prompt never re-runs (or re-pays for) earlier stages. Stages
1–4 are local, totalling ~3–4 minutes on 3000 pages; stage 5 owns the rest
of the time budget and runs concurrently.

*Trade-off, stated honestly:* production pipelines usually keep
intermediates in memory/queues and dump artifacts only on failure. Here the
artifacts are the learning and debugging instrument, so they are always
written; images saved for humans are downscaled JPEGs (`debug/`), while the
pipeline works on full-quality images in memory.

---

## 2. STAGE 1 — Triage

One decision per page: does it have a usable text layer?

```text
TEXT_NATIVE  >= 50 chars of text and no dominating raster image
SCANNED      raster image covers > 70% of the page (regardless of text!)
             OR near-textless and not a drawing
DRAWING      near-textless with >= 100 vector segments (CAD plans etc.)
```

The >70%-image guard catches the nastiest triage trap: scanned pages where
a scanner stamped real text headers — text length alone would call them
TEXT_NATIVE and local extraction would silently drop the page body.
Thresholds are biased so errors fall toward "one wasted API call", never
toward "silent garbage in the corpus". Every verdict is logged with its
evidence and a one-line reason (see the stage artifact).

Triage is single-threaded on purpose: PyMuPDF `Document` objects are not
thread-safe, and 3000 `get_text` calls take ~10–20 s anyway.

---

## 3. STAGE 2 — Local extraction (TEXT_NATIVE pages)

* **Body font size**, computed once document-wide: sample pages spread
  across the document (not just the front — front matter is
  unrepresentative), count characters per rounded span size, take the mode
  by *character* count.
* **Per page:** walk `get_text("dict")` blocks; emit TEXT units for prose,
  TITLE units where `span.size >= body_size * 1.15` or bold text matching
  `^\d+(\.\d+)*\s+\S` (numbered clauses), FIGURE units for embedded images
  (cropped to PNG at 200 DPI and stored — the PNG is the artifact).
* **Ruled-line detection:** count axis-aligned h/v segments from
  `get_drawings()`. Revised role (see §5): this is a *cross-check* on YOLO
  table boxes, **not** a router that decides whether YOLO runs.

**Dedup rule (added in review):** once table bboxes are known, all local
text units whose bbox falls inside a table region are dropped. Without
this, table content appears twice in assembly — once as garbled prose
spans, once as extracted markdown. This is the most important correctness
rule in the local path.

---

## 4. STAGE 3 — Rendering

`page.get_pixmap(dpi=200)` for every page a vision path needs. 150 DPI
nearly halves render time and memory; benchmark table accuracy at both
before committing (§11). Never hold thousands of page images in memory —
render lazily in the consuming worker or behind a bounded queue. For
parallel rendering, shard page ranges across *processes* (one `Document`
per process — thread pools are unsafe here).

---

## 5. STAGE 4 — Layout detection

**Revised decision: YOLO runs on every page.** The original design skipped
YOLO on text-native pages with ruled tables, using the ruled-line grid as
the box source. Review overruled it: the ruled heuristic false-positives on
page borders and letterheads, merges adjacent tables into one box, and —
worst — a page with one ruled table and one borderless table would skip
YOLO and silently miss the second table. YOLO on 3000 pages costs a few
GPU-minutes inside a budget dominated by stage 5; the simpler pipeline wins.
Ruled-line grids survive as a cross-check that refines/validates YOLO boxes.

Keep `table` and `figure` boxes; pad by ~10 px to avoid clipping captions
and footnote rows.

**Coordinate conversion — the most likely bug in this pipeline.** YOLO
returns rendered-image *pixels*; PyMuPDF operates in PDF *points*. One
helper does every conversion, and it derives scale from actual dimensions —
`scale_x = page.rect.width / pix.width` — rather than assuming `72/DPI`,
which silently breaks on rotated pages and non-origin cropboxes. The stage
artifact records both pixel and point boxes so the conversion is auditable.
No pixel coordinate escapes stage 4.

---

## 6. STAGE 5 — Gemini

### 6.1 Routing

| Page kind | Region | Route |
| --- | --- | --- |
| TEXT_NATIVE | prose | local (stage 2) — never sent |
| TEXT_NATIVE | table | crop -> Gemini |
| TEXT_NATIVE | figure | local PNG + one captioning call |
| SCANNED | whole page | full-page image -> Gemini |
| SCANNED | table | crop -> Gemini separately; spliced over the page result |
| SCANNED | figure | crop -> PNG + one captioning call |
| DRAWING | whole page | rendered PNG + one captioning call; never text-extracted |

Scanned pages are sent twice (whole page + isolated table crops): isolated
crops extract tables measurably better than tables embedded in a dense
page, and accuracy outranks cost here.

**Splice mechanics (added in review):** the full-page prompt instructs the
model to emit a `[TABLE]` placeholder where each table sits instead of
attempting the table inline. Table crops then replace placeholders in
reading order. Without a placeholder protocol, "crop overrides page result"
requires locating a table inside free-form markdown — unspecified and
fragile. Figure markers work the same way (`![figure](FIG)`), matched to
YOLO figure boxes by reading order; a count mismatch sets `needs_review`.

**Captioning (added in review):** figures and drawing pages get one cheap
vision call for a 1–2 sentence caption. A figure chunk with empty text is
unfindable by vector search; the caption is what makes it retrievable.

### 6.2 Model tiering (revised from a single fixed model)

`gemini-2.5-flash` for everything; escalate to `gemini-2.5-pro` only for
table crops that fail validation. Validation has two independent triggers:

* the model's own sanctioned failure channel — prompts instruct it to
  output `<!-- BROKEN -->` when structure can't be recovered (without a
  sanctioned way to fail, a vision model invents a plausible table);
* a local check — column-count consistency across rows, non-empty header —
  which catches failures the model does *not* admit to.

Escalation failing too -> `needs_review = true`, keep the best attempt,
never block the run.

### 6.3 Concurrency

Async semaphore sized from actual rate limits; 3 retries with exponential
backoff and jitter; a page that fails all retries is flagged, not fatal.
Partial results with review flags beat all-or-nothing batches.

### 6.4 What the API is never asked for

Bounding boxes, region types, page numbers — the local pipeline already
knows them. Ask only for content.

---

## 7. Multi-page table stitching

Large tables (price schedules, item lists) routinely span 10+ pages, so
this is in scope, not deferred.

**Detection** — table on page N continues onto page N+1 when all hold:

1. table N's bbox bottom reaches the page's bottom margin zone;
2. page N+1's first content region (by y-order) is a table starting in the
   top margin zone;
3. column signatures are compatible (same column count after extraction;
   for ruled tables, column x-boundaries match before extraction — a
   stronger signal).

**Merge** — fragments are extracted independently (crops stay small and
accurate; images are never merged, markdown is):

* first row of fragment N+1 fuzzy-matches fragment N's header -> repeated
  header, drop it, concatenate rows;
* column counts match, no repeated header -> plain concatenation;
* column counts disagree -> **refuse to guess**: keep separate chunks, flag
  both `needs_review`, record the failed merge in the manifest.
* 3+ page chains: apply pairwise.

**Chunking a merged table** — a 10-page table exceeds any chunk size. It
stays *logically* atomic but is split into row groups of ~N rows, each
chunk repeating the header row, all sharing a `table_id` and the full page
range. Retrieval hits a row group with headers intact; a consumer can
reassemble the full table via `table_id`.

---

## 8. STAGE 6 — Assembly and chunking

### 8.1 Unit contract

Every extraction path emits the same shape before merging (see
`models.py::Unit`): page, bbox (PDF points), type
(title/text/table/figure), content, optional heading level and storage
key, source tag, review flag. Scanned-page markdown from stage 5 is parsed
into units first (headings from `#` lines, placeholders from §6.1) — the
merge walk never special-cases by source.

### 8.2 Heading level normalization — before the walk

The vision model assigns `#` depths per page with no document context; a
`##` on page 412 need not match a `##` on page 8. Reconcile globally:

1. numbered headings (`^\d+(\.\d+)*`): depth = number of segments —
   authoritative when present;
2. otherwise cluster local font sizes document-wide, sorted descending ->
   levels 1..N;
3. the model's `#` count is a last resort.

### 8.3 Walk

Sort units by `(page, bbox.y0)` (known limitation: fails on multi-column
layouts). Maintain a heading stack; TITLE units push/pop it. TABLE and
FIGURE units become atomic chunks immediately — they bypass the text
chunker entirely so no boundary can split a table mid-row. TEXT units
accumulate per heading section.

### 8.4 Text chunking — per section, not per document

Each heading section's text is chunked separately (Chonkie recursive
markdown recipe, ~512 tokens). Revised from a global chunk-then-map-back
design: per-section chunking means every chunk inherits its section's
heading breadcrumb and page range directly — no offset arithmetic to
drift when the chunker normalizes whitespace.

### 8.5 Chunk contract

See `models.py::Chunk`. Key fields: `content` (displayed) vs
`embedding_text` (vectorized — the heading breadcrumb
`[7. Payment Terms > 7.3 Liquidated Damages]` is prepended here only,
which is what makes clause values retrievable from bare-value queries),
`pages` (1-based, for citation), `bbox`, `source`, `needs_review`,
`table_id`.

---

## 9. Outputs

| Artifact | Purpose | Source of truth? |
| --- | --- | --- |
| `chunks.jsonl` | retrieval, citation, Q&A | **yes** |
| `merged.md` | human review, debugging, diffing versions | no |
| `manifest.json` | page kinds, timings, review flags | run record |

The `.md` loses page numbers and bboxes — the things citation needs — so
it must never become authoritative. Tables are inlined into the `.md` for
readability even though chunking treats them atomically.

Storage is local disk (`output/<doc_id>/`); `storage_key` values are
relative paths. Swapping to object storage + a vector DB is a storage-
adapter change, not a pipeline change.

---

## 10. Edge-case policy

Every edge case is either **handled**, **flagged** (`needs_review`), or
**accepted** with its cost documented — see
[edge_cases.md](edge_cases.md). Nothing is silently ignored. Known open
items: multi-column reading order (untested), unnumbered headings at body
size (accepted miss).

---

## 11. Validate before building

In order — each can invalidate work below it:

1. **DPI test:** ~20 hard tables at 150 vs 200 DPI — decides render budget.
2. **Table bake-off:** same tables via PyMuPDF `find_tables()` vs
   Gemini-on-crop vs Gemini-on-full-page — decides whether YOLO and the
   crop path earn their place.
3. **Triage thresholds:** run stage 1 on real documents, audit the
   decision log for misclassification.
4. **Timing spike:** ~200 representative pages end-to-end, extrapolate,
   confirm the 20-minute budget before writing stages 5–6.
5. **YOLO box tightness:** tables with captions/footnotes directly
   beneath — tune the padding.

---

## 12. Dependencies

```text
pymupdf          (pinned exactly)
doclayout-yolo   (pinned — API moves between minor releases)
google-genai
chonkie          (pinned — same reason)
pillow
```

No agent framework: routing here is deterministic booleans over
validators. An LLM decides nothing about control flow.
