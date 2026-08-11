# rag4prod

A production-*shaped* PDF ingestion pipeline for RAG, built as a learning
project: large mixed-content PDFs (text-native, scanned, and drawing pages)
in; retrieval-ready, citation-carrying chunks out. Every stage writes an
inspectable artifact, every threshold is documented, and every edge case is
either handled or consciously accepted in
[docs/edge_cases.md](docs/edge_cases.md).

**Stack:** [PyMuPDF](https://pymupdf.readthedocs.io/) (triage, local
extraction, rendering) + DocLayout-YOLO (layout detection) + Gemini
(vision extraction of tables and scanned pages). Full architecture and
the reasoning behind every design decision:
[docs/design_spec.md](docs/design_spec.md).

## Pipeline

```text
PDF ──1 triage──▶ page kinds ──2 local extract──▶ units (free, exact)
        │
        └─▶ 3 render ──▶ 4 layout (YOLO) ──▶ 5 Gemini ──▶ units
                                                            │
                    6 assemble + chunk ◀────────────────────┘
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
| 2 | Local extraction (text, headings, figures, ruled tables) | ⬜ |
| 3 | Rendering + YOLO layout detection | ⬜ |
| 4 | Gemini extraction (tables, scanned pages) | ⬜ |
| 5 | Multi-page table stitching, assembly, chunking | ⬜ |
| 6 | End-to-end hardening | ⬜ |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

## Run

```bash
# generate the synthetic 7-page test PDF (every triage branch represented)
python -m rag_ingest.sample_pdf

# run the pipeline on it
rag-ingest sample_data/sample_doc.pdf

# inspect what happened, stage by stage
cat output/sample_doc/stages/01_triage.json

# run tests
pytest
```
