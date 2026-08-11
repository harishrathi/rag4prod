# Rewrite design — v2 architecture from the lessons of v1

**Status:** proposal. Nothing here is implemented; the current pipeline
(`src/rag_ingest/`, [design_spec.md](design_spec.md)) is complete and
working. This document answers a different question: *knowing everything
the [edge-case ledger](edge_cases.md) now records, what would the
architecture be if designed from that knowledge on day 1?*

The premise matters. v1's fixes landed as patches because the edge cases
were discovered in phase order, and each fix was placed wherever the
symptom surfaced. That was the correct way to *learn* the domain. It is
not the shape you would choose once the domain is known. The ledger's 30
cases, read as a whole, are the requirements document v1 never had — and
they cluster into a small number of recurring themes that each got
patched locally three or four times because the architecture had no home
for them.

---

## 1. What the patches actually revealed

### Theme A — Page routing is dynamic; v1 treated it as static

v1's model: triage assigns a `PageKind` once, every downstream stage
filters by kind. Reality kept breaking it:

- a text layer over a scanned body forces a reroute (ledger #1)
- a *lying* text layer (broken ToUnicode CMap) forces a native→OCR
  reroute (#29)
- orientation fixes mutate pages inside triage (#28), which only exists
  in memory — so every `--from-stage` resume must re-apply rotations via
  a special-case rehydration path (`pipeline._load_triage`)
- OCR language selection depends on what triage saw (#29)

The routing logic ended up smeared across `triage.py`, `pipeline.py`,
and the resume path. Each new signal was a patch to two or three files.

### Theme B — Trust is one concern; v1 enforced it in four places

The pipeline's single real invariant — *silent garbage never enters the
corpus* — is enforced by four independently patched mechanisms:

- the OCR page-quality score (#16, #28)
- junk-char flags on native units (#29)
- table validation gates → `needs_review` + stored crop (#21, #27)
- furniture suppression, protecting reviewer attention (#30)

Each lives in a different module, each sets `needs_review` by its own
code path. Adding a fifth source of doubt today means finding all the
places flags are set and threading a new one through.

### Theme C — Coordinate-system confusion is a type error, not a coding error

The same bug class shipped twice: pixel-vs-point at the YOLO boundary
(#11) and again at the pdfocr wrapper boundary (#20). The ledger's own
generalization — *at every boundary between coordinate systems, measure,
never assume* — is a statement about types, not about vigilance.
Vigilance failed once per coordinate boundary; that is its expected
failure rate.

### Theme D — Document-wide passes were bolted onto a per-page design

Furniture stripping (#26, #30), body-font-size and heading-level
clustering (#6, #25), cross-page table stitching (#21, #27), and the
image-hash dedup production note (#8) all need the *whole document* in
view. v1's stages think per-page, so these passes were wedged into
whichever stage was closest — which is why `assemble.py` and the
710-line `tables.py` carry the most scar tissue.

### Theme E — Checkpointing is hand-rolled per stage

Every stage in `pipeline.run()` has a bespoke run-vs-load branch with
its own rehydration code; roughly 150 of its 500 lines are serialization
plumbing. The rotation-reapply hack (Theme A) exists precisely because
page normalization is not itself an artifact. One stage (3+4) shares a
checkpoint and needs a documented clamp; nothing enforces that the
run-branch and load-branch produce equivalent state.

### Theme F — Parallelism was blocked by an avoidable design choice

PyMuPDF is not thread-safe (#4), so v1 is single-threaded end to end.
The ledger's own production note has the answer — shard page ranges
across *processes*, one open `Document` each — but v1's stages pass a
shared live `Document` object around, so the fix can't be retrofitted
without restructuring. Rendering + OCR dominate wall-clock time and are
embarrassingly parallel per page.

---

## 2. Target architecture

Eight layers. The count is similar to v1's seven stages, but the
boundaries move: decisions separate from evidence, per-page work
separates from document-wide work, and quality judgment becomes one
layer instead of four patches.

```text
0  ingest gate         reject encrypted / corrupt / zero-page      (kept from v1)
1  page profiling      EVIDENCE per page, no decisions
2  routing             PageProfile -> PageRoute, pure function
3  extraction workers  per (page, route); process-parallel; emit Units
4  table ladder        ordered extractors, each behind a validation gate
5  doc normalization   whole-document passes over the unit pool
6  quality gate        one pass derives needs_review; emits review report
7  chunking            unchanged behind the split_text() seam
```

### Layer 0 — Ingest gate

`_open_checked` promoted to the front door, unchanged (#5): encrypted,
unreadable, and zero-page PDFs are rejected with a manifest reason
before any stage exists.

### Layer 1 — Page profiling: evidence, no decisions

One pass per page collecting *measurements only*:

```python
@dataclass(frozen=True)
class PageProfile:
    page: int
    text_chars: int
    junk_chars: int              # C0 / U+FFFD / PUA count (#29)
    max_image_coverage: float    # (#1)
    text_bbox_area_frac: float   # (#1 production note — now free to include)
    vector_segments: int         # deduped (#7), (#2)
    orientation_probe: OrientationProbe | None   # low-DPI OCR evidence (#28)
```

The critical property: **profiling decides nothing.** v1's triage both
measured and decided, which is why every new signal (junk chars,
orientation) had to be patched into it *and* into whatever consumed its
verdicts. Profiles are pure data, serialized as the layer's artifact, and
cheap to extend — a new field breaks nothing downstream until a routing
rule reads it.

### Layer 2 — Routing: decisions, no I/O

A pure function per page:

```python
def route(profile: PageProfile) -> PageRoute

@dataclass(frozen=True)
class PageRoute:
    extractor: Extractor         # NATIVE | OCR | DRAWING
    rotation: int                # 0 | 90 | 180 | 270  (#28)
    languages: str               # e.g. "eng+hin"      (#29)
    reasons: list[str]           # every rule that fired, for the artifact
```

All threshold logic from thirty ledger cases lives in this one
table-driven, trivially testable module: the image-coverage override
(#1), the drawing rule (#2), the near-blank bias (#3), the junk-char
reroute (#29), the rotation decision with its word-floor and score-gap
guards (#28). Three patches in three files become three rules in one
file, each unit-tested against a recorded profile.

**Rotation becomes data in the route**, applied by whoever opens the
page. The v1 resume hack — re-applying `set_rotation` from the triage
artifact on every reopen — is deleted, not fixed: a route is immutable
data, so *every* consumer (render, OCR, table crops, a resumed run)
applies it identically by construction.

### Layer 3 — Extraction workers: per-page, process-parallel

Each worker takes `(pdf_path, page_number, route)` — a path, not a live
`Document` — opens its own document, extracts, and emits `Unit`s,
`RuledGrid`s, and layout `Region`s through the existing contracts.
Three workers, one per extractor route:

- **native**: the v1 stage-2 line walk (#6, #17), unchanged in logic
- **ocr**: render → Tesseract textpage → the *same* line walk (the
  textpage seam that already worked in v1)
- **drawing**: render to PNG, store as figure (#2)

YOLO layout detection runs inside the worker on the same pixmap the
page renders once (#13's interleaving, kept) — but it is demoted from
"stage" to *evidence provider*: its regions are just more per-page
output consumed later by the table ladder and figure cropping.

Because workers share nothing, this layer runs under
`ProcessPoolExecutor` with page-range sharding — resolving #4
architecturally instead of by "triage is deliberately single-threaded."
The model load cost is amortized by loading YOLO once per worker
process, not per page. Sequential execution remains a `--workers 1`
flag, and determinism is preserved because workers emit keyed-by-page
results that the orchestrator reassembles in page order.

### Layer 4 — The table ladder, split into its real parts

v1's `tables.py` is four modules in one; the rewrite splits along the
seams the patches revealed:

```text
tables/
    grids.py        vector-line grids (#7) + pixel-line grids, ink checks (#27)
    cells_native.py find_tables + span geometry -> unmerged cell grid (#27)
    cells_ocr.py    selective line-erasure (#19, #27) + word re-anchoring (#20)
    validate.py     structural checks; junk-char cell checks (#29); empty-line drop
    stitch.py       cross-page continuation, equal-column rule (#21), span fill (#27)
```

The tier ladder becomes an explicit ordered list of extractors, each
behind a validation gate; failing the last gate yields `needs_review`
plus a stored crop. That makes #18's observation structural: the VLM
tier-3 slot is literally an empty list entry, and plugging one in later
touches the list, not the ladder.

The **unmerged-cell contract** (#27 — merged values repeated into every
covered position, printed layout preserved in `merges`) is kept exactly:
it survived the hardest torture tests and downstream chunking depends
on it.

### Layer 5 — Document normalization: the layer v1 never had

Runs once, with the full unit pool from all pages. Everything here is a
pure function over lists — the most testable code in the system:

- **furniture stripping** (#26, #30): repeated-position analysis over
  text units *and* YOLO table suspects in the same pass — they are the
  same phenomenon and v1 handled them in two different modules
- **table-region dedup** by bbox-center containment (#23)
- **cross-page table stitching** (calls `tables/stitch.py`)
- **heading-size clustering** into document-wide levels (#6), including
  the sparse-OCR degradation (#25)
- **figure dedup by image hash** (#8's production note — free to
  include now that a document-wide pass exists)

### Layer 6 — Quality gate: one place, per-source validators

A single pass that *derives* `needs_review` instead of six call sites
sprinkling it:

```python
VALIDATORS = {
    Source.TESSERACT_OCR: ocr_quality_score,        # (#16, #28) — counts marks (#29)
    Source.PYMUPDF:       junk_char_check,          # (#29)
    UnitType.TABLE:       structural_validation,    # (#21, #27)
}
```

Every flag records *which validator fired and why* — v1's flags are
bare booleans, and #30 proved that reviewer experience is itself a
production concern. So this layer's second output is a designed
artifact, not a byproduct: `review_report.md`, grouping flagged items
with their stored crops and firing reasons, ordered by page. A drowning
reviewer stops reading flags; the report exists to keep them afloat.

### Layer 7 — Chunking

Unchanged behind the `split_text()` seam (#24) — the seam already
proved itself when the hand-rolled splitter was swapped for Chonkie in
one function and one config block. Row-group table chunking with
repeated headers (#27) carries over as-is.

---

## 3. Cross-cutting mechanics

### Coordinate types: make the bug class unrepresentable

Two distinct frozen types replace the "be careful at boundaries" rule
that failed twice (#11, #20):

```python
@dataclass(frozen=True)
class PixelBox:
    x0: float; y0: float; x1: float; y1: float
    raster_w: int; raster_h: int      # the raster this box is measured in

@dataclass(frozen=True)
class PdfBox:
    x0: float; y0: float; x1: float; y1: float

def to_pdf(box: PixelBox, page_rect: Rect) -> PdfBox:
    # scale derived from ACTUAL dimensions on both sides — measured, never assumed
```

The only conversion path is a constructor that holds both references.
Passing a `PixelBox` where a `PdfBox` is expected is a type error, not a
silent crop shift. `Unit.bbox` remains PDF points only, as in v1.

### A generic Stage abstraction replaces hand-rolled checkpointing

```python
class Stage(Protocol):
    name: str
    artifact: str
    def run(self, ctx: PipelineContext) -> object: ...
    def serialize(self, result) -> dict | list[dict]: ...
    def deserialize(self, raw) -> object: ...
```

The orchestrator owns run-vs-load, timing, artifact I/O, and config
snapshotting *uniformly*. This deletes most of `pipeline.run()`'s bulk,
makes `--from-stage` impossible to get wrong per-stage, and generalizes
v1's `_triage_thresholds` snapshot patch to every stage automatically.
The v1 principle is kept in full: every layer writes its complete
artifact before the next layer reads it — the files remain the learning
instrument.

### Config: a frozen dataclass, not module globals

v1's `config.py` is 267 lines of module constants; snapshotting them
into artifacts was patched in per-stage. v2 passes one frozen
`IngestConfig` into `run()` and records it wholesale in the manifest.
Thresholds get grouped by the layer that reads them (profile thresholds,
routing rules, table geometry, chunk sizing), which also makes the
routing table (Layer 2) data-driven for free.

### The error-direction principle, stated once

v1's bias — *errors fall toward wasted work, never toward silent
garbage* (#3, #21, #27, #28) — is the explicit design rule of Layers 2
and 6. Every routing threshold and every validator must answer: "when
this is wrong, which direction does it fail?" The acceptable answers are
"wasted compute" and "a review flag." "Quietly wrong content" is never
acceptable; that is the invariant the whole rewrite exists to serve.

---

## 4. What is deliberately kept from v1

These survived eight phases and two torture corpora without structural
change — which is the signature of a correct design, and the rewrite
converges on them rather than reinventing them:

- **`Unit` and `Chunk` contracts** ([models.py](../src/rag_ingest/models.py)),
  including the 0-based-internal / 1-based-citation page rule with its
  single conversion point
- **artifact-per-layer JSONL** checkpoints and the manifest
- **the edge-case ledger discipline** — v2 gets its own ledger from day 1
- **the tiered-cost rule** — never run an expensive extractor on content
  a cheaper one already handled
- **merged.md** as the human-review render (never authoritative)
- **exact dependency pins** where API drift was observed live (#24)

---

## 5. Migration and validation strategy

The rewrite is validated by *diffing against v1*, not by re-reasoning:

1. **Golden baseline first.** Run v1 on the full evaluation corpus (the
   two synthetic samples plus the real 10-document set) and freeze its
   `07_chunks.jsonl`, `06_tables.jsonl`, and manifests as golden files.
2. **Port the content-asserting tests** before writing v2 code. The
   existing tests assert on expected content, not on "no crash" (#22's
   lesson), so they transfer almost directly to the new module
   boundaries and become the acceptance suite.
3. **Build bottom-up along the layers** — coordinate types and the
   Stage abstraction first (pure, fast to test), then profiling +
   routing (testable against recorded v1 triage evidence), then
   workers, then the document layers.
4. **Diff chunks, not code.** Where v2's chunks differ from the golden
   baseline (modulo chunk IDs and ordering), one of the two pipelines is
   wrong — and the ledger usually says which. Every intentional
   improvement (e.g. figure dedup, the text-area triage signal) must be
   toggleable off to reach diff-parity first, then toggled on with its
   own before/after diff.
5. **Cut over per-corpus, keep v1 runnable** until the diff on the real
   corpus is empty-or-explained. The two pipelines share no code except
   `models.py`, so they can coexist as `rag_ingest/` and `rag_ingest2/`
   (or a branch) during the overlap.

This turns the rewrite from a leap into a refactor with a safety net:
the internals are new, but every output is checked against a pipeline
that thirty documented edge cases already hardened.

---

## 6. Honest limits of the rewrite

- **It does not add capability.** Borderless tables still route to
  review (#18), Tesseract still lacks per-word confidence through the
  bundled integration (#16), and true orientation detection still wants
  real OSD (#28). The rewrite changes where the *next* fix lands, not
  what the pipeline can extract today.
- **Process-parallelism adds real complexity** (worker lifecycle, model
  loads per process, result reassembly). It is justified by render+OCR
  wall-clock on 3000-page documents; for small documents `--workers 1`
  is the honest default.
- **A generic Stage abstraction can over-abstract.** The rule: the
  orchestrator may know about artifacts and ordering, never about page
  kinds, units, or tables. If a layer's logic starts leaking into the
  orchestrator, that is v1's disease returning under a new name.
