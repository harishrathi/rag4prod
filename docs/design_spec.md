# Design spec — PDF ingestion pipeline for RAG

**Input:** large mixed-content PDFs — text-native pages, scanned pages, and
vector-drawing pages in one document (contracts, technical manuals, scanned
business documents).
**Output:** retrieval-ready chunks that carry page numbers and bounding
boxes, so every answer can cite its source location.
**Stack:** PyMuPDF + DocLayout-YOLO + Tesseract (bundled in PyMuPDF) —
**fully local, no API dependencies**.
**Targets:** 3000-page PDF end-to-end in well under an hour on CPU; table
accuracy prioritized; zero silent data loss (errors must surface as
`needs_review`, never as quietly wrong content).

This document is the *current* design. It started from an initial spec
that used a vision LLM (Gemini) for scanned pages and tables; review and
corpus analysis revised it repeatedly, and the biggest revision — dropping
the LLM entirely — is documented in §7 with its reasoning. The revision
trail is deliberate: the trade-offs are the point of this repo.

---

## 0. Division of responsibility

| Component | Does | Does NOT |
| --- | --- | --- |
| **PyMuPDF** | open PDF, triage pages, extract text on text-native pages, render pixmaps, crop bboxes, extract embedded images, host the OCR engine | classify regions semantically, find borderless tables |
| **DocLayout-YOLO** | return `{label, bbox, conf}` per page image | open PDFs, render, extract any content |
| **Tesseract** (via PyMuPDF) | recover text from scanned pages | table *structure*, image description |

**Cost rule (generalized from "never pay the API twice"):** never run an
expensive extractor on content a cheaper one already handled. Exact native
text beats OCR; OCR beats nothing; every escalation must be justified by
the cheaper tier's failure.

---

## 1. Stages and artifacts

| Stage | Module | Work | Artifact |
| --- | --- | --- | --- |
| 1 Triage | `triage.py` | page kind per page | `stages/01_triage.json` |
| 2 Local extract | `local_extract.py` | text/heading/figure units, free | `stages/02_units_local.jsonl`, `02_ruled_grids.json` |
| 3 Render | `render.py` | page images, debug JPEGs, drawing-page PNGs | `stages/03_render.json` |
| 4 Layout | `layout.py` | YOLO boxes, px→pt conversion | `stages/04_layout.jsonl` |
| 5 OCR | `ocr.py` | scanned-page prose via Tesseract | `stages/05_ocr_units.jsonl` |
| 6 Tables | `tables.py` | tiered table extraction + stitching | `stages/06_tables.jsonl` |
| 7 Assemble | `assemble.py`, `chunking.py` | dedup, merge walk, chunk | `stages/07_chunks.jsonl` |

Every stage checkpoints its full output before the next stage reads it;
`--from-stage N` reloads instead of recomputing. All stages are local, so
checkpointing here buys iteration speed and crash recovery rather than
API-cost savings.

---

## 2. STAGE 1 — Triage

One decision per page: does it have a usable text layer?

```text
TEXT_NATIVE  >= 50 chars of text and no dominating raster image
SCANNED      raster image covers > 70% of the page (regardless of text!)
             OR near-textless and not a drawing
DRAWING      near-textless with >= 100 deduplicated vector segments
```

The >70%-image guard catches the nastiest trap: scanned pages with a
scanner-stamped text header, which text-length alone would misroute into
local extraction — silently dropping the page body. Thresholds are biased
so errors fall toward "one wasted OCR pass", never toward silent garbage.
Every verdict is logged with its evidence in the stage artifact.

Single-threaded on purpose: PyMuPDF `Document` objects are not
thread-safe; parallelism, when needed, shards page ranges across
*processes*.

---

## 3. STAGE 2 — Local extraction (TEXT_NATIVE pages)

* **Body font size**: character-weighted mode of span sizes, sampled
  across the whole document (front matter is unrepresentative).
* **Per line**: heading if `size >= body * 1.15` OR (bold AND matches
  `^\d+(\.\d+)*\s`); consecutive body lines of a block merge into one
  paragraph unit. Classification granularity is the LINE — spans fragment,
  blocks over-merge.
* **TITLE units carry their raw font size** — heading *levels* can only be
  assigned document-wide (stage 7 clusters sizes).
* **Embedded figures** cropped to `figures/*.png`; images under 0.5% of
  page area are skipped as logo/watermark noise.
* **Ruled grids**: deduplicated axis-aligned segment counts + tight bbox —
  *evidence for the table stage, never a router*.

The extraction walk accepts any textpage (`textpage=` seam), which is what
lets stage 5 reuse it verbatim for OCR output.

---

## 4. STAGE 3 — Rendering

200 DPI default (150 halves cost — benchmark before committing, §10).
Rendering interleaves with detection in one per-page loop; page images are
never accumulated (a 200-DPI A4 pixmap is ~11 MB; 3000 of them is 30+ GB).
DRAWING pages are stored wholesale as figure PNGs and skip all further
processing. Debug JPEGs are separate low-res renders — never derived from
the pipeline pixmap (mutating a pixmap after exporting its buffer breaks
PyMuPDF's cached memoryview).

---

## 5. STAGE 4 — Layout detection

DocLayout-YOLO on every SCANNED and TEXT_NATIVE page; keep `table` and
`figure` regions; pad boxes ~10 px so captions and final rows aren't
clipped. YOLO is not gated on ruled-line detection: the ruled heuristic
false-positives on borders and misses the borderless table on a mixed
page. (Corpus says borderless is rare — but "rare" is found by looking,
and YOLO is how we look.)

**Coordinate conversion — the most likely bug in this pipeline.** YOLO
returns rendered-image pixels; PyMuPDF crops in PDF points. One helper
(`pixel_rect_to_pdf`) does every conversion: scale derived from *actual*
`page.rect` vs pixmap dimensions (never `72/DPI`, which breaks silently on
rotation and cropbox offsets), clamped into the page, asserted
non-degenerate. Pixel coordinates never escape stage 4; the artifact
records both boxes side by side for auditability.

---

## 6. STAGE 5 — OCR (SCANNED pages)

PyMuPDF wheels bundle libtesseract — no system install, no container; the
language file auto-downloads to `.tessdata/` (same lazy-fetch pattern as
the YOLO weights). `get_textpage_ocr(dpi=300, full=True)` returns a
textpage with the same block/line/span shape as native text, so scanned
pages flow through the stage-2 walk unchanged, tagged
`source=tesseract_ocr`.

Known limits, accepted and ledgered: no per-word confidence through the
bundled integration (stamps/signatures pass as junk words); OCR quality is
capped by the scan's native resolution — upsampling cannot restore detail.
Production variants: OCR-as-a-service in docker-compose for independent
scaling; pytesseract TSV for confidence filtering.

---

## 7. STAGE 6 — Tables: the tiered ladder (and why there is no LLM tier)

| Tier | Case | Method |
| --- | --- | --- |
| 1 | Bordered table, text-native page | PyMuPDF `find_tables()` — vector lines + exact native text; zero OCR, zero hallucination |
| 2 | Bordered table, scanned page | image line detection → cell grid → per-cell OCR |
| — | Anything failing validation | `needs_review=true` + stored crop PNG for **human** review |

**Validation gate** (decides tier success): column-count consistency
across rows, non-empty header, cell coverage of the region bbox.

**The dropped tier.** The original design used a vision LLM as tier 3
(borderless tables, low-quality scans, figure captions). Dropped after
corpus analysis: bordered tables dominate and scans are machine-typeset.
The determining arguments, kept for the record:

* geometric+OCR table extraction **cannot hallucinate** — a misread is
  visibly garbled, a VLM error is fluent and plausible (`120.00` →
  `210.00`), which is the worst property for numeric tables;
* the fallback tier is now a human (`needs_review` + crop) — honest,
  auditable, and free;
* the cost: borderless tables and figure captions are unhandled. YOLO
  still *detects* borderless tables, so they surface as review items, not
  silent misses. If the corpus changes, the validation gate is exactly
  where a VLM tier plugs back in.

**Multi-page tables are in scope** (price schedules span many pages):
continuation detected via bottom-margin exit + top-margin table entry +
compatible column signature; merged by row concatenation (repeated-header
fuzzy match dropped); column-count mismatch → refuse to merge, flag both.
Merged tables chunk as row groups with the header repeated per chunk,
sharing a `table_id`.

---

## 8. STAGE 7 — Assembly and chunking

1. **Dedup rule (most important correctness rule):** drop stage-2/5 TEXT
   units whose bbox falls inside a table region — otherwise table content
   appears twice (garbled prose + extracted table). Containment by unit
   *center*, not any-intersection, so padded table boxes don't eat
   neighboring prose lines.
2. **Heading levels**, document-wide: numbered headings (`7.3.1` → depth
   3) are authoritative; otherwise cluster TITLE font sizes descending →
   levels. (Dropping the LLM simplified this: no per-page `#` depths to
   reconcile.)
3. **Walk** units sorted by `(page, y0)`: heading stack, tables/figures as
   atomic chunks, text accumulated per heading section. Known limit:
   multi-column reading order (untested; column detection via x-gap
   clustering is the production fix).
4. **Chunking per heading section** (~512 tokens): each chunk inherits its
   section's breadcrumb and page range directly — no offset arithmetic.
   `content` is displayed; `embedding_text` carries the breadcrumb prefix
   that makes bare values retrievable ("what is the LD rate?").

---

## 9. Outputs

| Artifact | Purpose | Source of truth? |
| --- | --- | --- |
| `chunks.jsonl` (stage 7) | retrieval, citation, Q&A | **yes** |
| `merged.md` | human review, debugging, diffing | no |
| `manifest.json` | page kinds, timings, review flags | run record |

Local disk; `storage_key` values are relative paths. Object storage + a
vector DB is a storage-adapter swap, not a pipeline change.

---

## 10. Validate before building

1. **DPI test** — 20 hard tables at 150 vs 200 DPI render.
2. **Table bake-off** — same tables: `find_tables()` vs cell-grid+OCR;
   decides how much tier 2 must carry.
3. **Triage thresholds** — stage-1 decision log audited on real docs.
4. **OCR quality sample** — real scanned pages through stage 5; decides
   whether confidence filtering must be pulled forward.
5. **Timing spike** — 200 representative pages end-to-end, extrapolated.

---

## 11. Dependencies

```text
pymupdf          (pinned; bundles the OCR engine)
doclayout-yolo   (pinned; pulls torch — see ledger #14 for the ONNX path)
huggingface-hub  (model weights fetch)
numpy
chonkie          (stage 7, pinned when added)
```

Fully local: no API keys, no network calls after model/tessdata download.
No agent framework: routing is deterministic booleans over validators.
