# rag4prod

A production-*shaped* PDF ingestion pipeline for RAG, built as a learning
project: large mixed-content PDFs (text-native, scanned, and drawing pages)
in; retrieval-ready, citation-carrying chunks out. Every stage writes an
inspectable artifact, every threshold is documented, and every edge case is
either handled or consciously accepted in
[docs/edge_cases.md](docs/edge_cases.md).

**Stack:** [PyMuPDF](https://pymupdf.readthedocs.io/) (triage, local
extraction, rendering) + DocLayout-YOLO (layout detection) + a
Gemini-class VLM for the **paid lane** — only pages the free lane cannot
read honestly (true scans, broken font encodings) ever cost an API call.
Full architecture and the reasoning behind every design decision:
[docs/design_spec.md](docs/design_spec.md); why Tesseract was replaced
by one VLM code path for every script:
[docs/gemini_extractor_spec.md](docs/gemini_extractor_spec.md).
The clean-room v2 built from all 30 ledgered edge cases — the patches
distilled back into an architecture — is implemented as `rag_ingest2`:
[docs/rewrite_design.md](docs/rewrite_design.md).

## Pipeline

```text
PDF ──1 triage──▶ page kinds ──2 local extract──▶ units (free, exact)
        │
        └─▶ 3 render ──▶ 4 layout (YOLO) ──▶ 5 VLM lane (paid) ──▶ units
                              │                                      │
                              └──▶ 6 tables (tiered ladder) ──▶ units│
                                                                     │
                    7 assemble + chunk ◀─────────────────────────────┘
                        │
                        └─▶ chunks.jsonl (retrieval-ready, cited by page + bbox)
```

Every stage checkpoints its output under `output/<doc_id>/stages/NN_*.json[l]`,
so a document can be traced through the pipeline file by file, and any stage
can be re-run from its predecessor's artifact (`--from-stage N`) without
recomputing — or re-paying for — earlier stages.

## Status

| Phase | Stage | Status |
| --- | --- | --- |
| 1 | Project setup, contracts, triage | ✅ |
| 2 | Local extraction (text, headings, figures, ruled tables) | ✅ |
| 3 | Rendering + YOLO layout detection | ✅ |
| 4 | OCR for scanned pages (bundled Tesseract; later replaced by the VLM lane) | ✅ |
| 5 | Tiered table extraction + multi-page stitching | ✅ |
| 6 | Assembly, dedup, chunking | ✅ |
| 7 | Complex tables (merged cells, spans → JSON + ASCII grid), OCR quality gate + orientation recovery, real `--from-stage` resume, clean rejection | ✅ |
| 8 | Real-corpus hardening: broken text layers → reroute, repeating header/footer suppression | ✅ |
| 9 | Gemini VLM lane: one paid code path for every script, mojibake triage, response cache, verification gate | ✅ |
| 10 | v2 rewrite (`rag_ingest2`): eight layers, typed coordinates, process-parallel workers, one quality gate | ✅ |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

## Run

```bash
# generate the synthetic test PDFs: sample_doc (every triage branch) and
# complex_doc (merged cells, rotated scans, borderless + multi-page tables)
python -m rag_ingest.sample_pdf
python -m rag_ingest.complex_pdf

# the paid lane needs a key (env var, or a gitignored .env at the repo
# root with GEMINI_API_KEY=...); the free lane runs without one
rag-ingest sample_data/sample_doc.pdf
rag-ingest sample_data/complex_doc.pdf

# the v2 pipeline (same documents, eight-layer architecture; --vlm-cache
# can point at a v1 cache so re-runs cost nothing)
rag-ingest2 sample_data/sample_doc.pdf

# inspect what happened, stage by stage
cat output/sample_doc/stages/01_triage.json

# run tests
pytest
```
