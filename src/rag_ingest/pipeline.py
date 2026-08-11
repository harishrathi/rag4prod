"""Pipeline orchestrator and CLI entry point.

Usage:
    python -m rag_ingest.pipeline sample_data/sample_doc.pdf
    rag-ingest <pdf> [--out output] [--from-stage N] [--no-debug]

Design: every stage writes its full output to a numbered artifact under
``output/<doc_id>/stages/`` *before* the next stage reads it. This buys:

  * Traceability — one page's journey through the pipeline is readable by
    opening the stage files in order and following its records.
  * Cheap iteration — ``--from-stage N`` reloads stage N-1's artifact
    instead of recomputing stages 1..N-1. Re-tuning a Gemini prompt never
    re-runs triage/render/YOLO, and never re-spends API calls.
    (Production name for this pattern: checkpointing.)
  * Debuggability — when output looks wrong, binary-search the stage files
    to find the stage that broke, instead of re-running the world.

Trade-off, stated honestly: production pipelines usually keep intermediates
in memory or a queue and dump artifacts only on failure/sampling, because
per-stage disk round-trips cost throughput. Here the files ARE the learning
instrument, so they are always written (they are small JSON; the heavier
image copies under debug/ are downscaled JPEGs, see config.py).

Output layout (local disk; production would swap the figure/chunk writes
behind object storage + a vector DB — a storage-adapter change, not a
pipeline change):

    output/<doc_id>/
        stages/01_triage.json     # decision log: kind + evidence per page
        stages/02_units_local.jsonl   # TITLE/TEXT/FIGURE units, one per line
        stages/02_ruled_grids.json    # body font size + table-grid evidence
        stages/03_render.json                                    [Phase 3]
        stages/04_layout.jsonl                                   [Phase 3]
        stages/05_gemini.jsonl                                   [Phase 4]
        stages/06_chunks.jsonl                                   [Phase 5]
        debug/                    # downscaled image copies      [Phase 3+]
        figures/                  # full-quality stored PNGs
        merged.md                                                [Phase 5]
        manifest.json             # summary: counts, timings, review flags
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pymupdf

from . import local_extract
from .models import PageKind
from .triage import TriageRecord, triage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage artifact helpers
# ---------------------------------------------------------------------------


def _write_stage(doc_out: Path, name: str, payload: object) -> Path:
    path = doc_out / "stages" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("stage artifact -> %s", path)
    return path


def _write_stage_jsonl(doc_out: Path, name: str, rows: list[dict]) -> Path:
    """JSONL = one JSON object per line: greppable, diffable, streamable,
    and 'go through it line by line' is meant literally."""
    path = doc_out / "stages" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("stage artifact -> %s (%d records)", path, len(rows))
    return path


def _load_triage(doc_out: Path) -> list[TriageRecord]:
    """Rehydrate stage 1's artifact for --from-stage >= 2 runs."""
    raw = json.loads((doc_out / "stages" / "01_triage.json").read_text(encoding="utf-8"))
    return [
        TriageRecord(
            page=r["page"],
            kind=PageKind(r["kind"]),
            text_chars=r["text_chars"],
            max_image_coverage=r["max_image_coverage"],
            drawing_segments=r["drawing_segments"],
            reason=r["reason"],
        )
        for r in raw["pages"]
    ]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(pdf_path: Path, out_dir: Path, from_stage: int = 1, debug: bool = True) -> dict:
    """Run the pipeline on one PDF. Returns the manifest dict.

    ``from_stage`` skips completed stages by reloading their artifacts —
    e.g. from_stage=2 trusts stages/01_triage.json instead of re-triaging.
    """
    doc_id = pdf_path.stem
    doc_out = out_dir / doc_id
    doc_out.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    doc = pymupdf.open(pdf_path)
    try:
        # ---- STAGE 1: triage -------------------------------------------
        if from_stage <= 1:
            t0 = time.perf_counter()
            records = triage(doc)
            timings["triage"] = round(time.perf_counter() - t0, 3)
            _write_stage(
                doc_out,
                "01_triage.json",
                {
                    "pdf": str(pdf_path),
                    "thresholds_used": _triage_thresholds(),
                    "pages": [r.to_dict() for r in records],
                },
            )
        else:
            records = _load_triage(doc_out)
            log.info("stage 1 skipped, loaded %d records from artifact", len(records))

        # ---- STAGE 2: local extraction (TEXT_NATIVE pages) -------------
        page_kinds = {r.page: r.kind for r in records}
        t0 = time.perf_counter()
        body_size, units, grids = local_extract.extract(doc, page_kinds, doc_out)
        timings["local_extract"] = round(time.perf_counter() - t0, 3)
        _write_stage_jsonl(doc_out, "02_units_local.jsonl", [u.to_dict() for u in units])
        _write_stage(
            doc_out,
            "02_ruled_grids.json",
            {"body_font_size": body_size, "grids": [g.to_dict() for g in grids]},
        )

        # ---- STAGE 3+: added phase by phase ----------------------------
        # (debug flag will gate the downscaled image copies from Phase 3 on)
        _ = debug

        counts: dict[str, int] = {}
        for r in records:
            counts[r.kind.value] = counts.get(r.kind.value, 0) + 1
        unit_counts: dict[str, int] = {}
        for u in units:
            unit_counts[u.type.value] = unit_counts.get(u.type.value, 0) + 1

        manifest = {
            "doc_id": doc_id,
            "source_pdf": str(pdf_path),
            "page_count": doc.page_count,
            "counts": counts,
            "body_font_size": body_size,
            "unit_counts": unit_counts,
            "ruled_grid_pages": [g.page for g in grids],
            "timings_secs": timings,
            # 0-based internally (see models.py); only Chunk.pages is 1-based.
            "page_kinds": {str(r.page): r.kind.value for r in records},
        }
    finally:
        doc.close()

    (doc_out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _triage_thresholds() -> dict:
    """Snapshot the config the stage ran with, INTO the artifact. When a
    threshold gets tuned later, old artifacts still say what produced them."""
    from . import config

    return {
        "MIN_TEXT_CHARS": config.MIN_TEXT_CHARS,
        "SCAN_IMAGE_COVERAGE": config.SCAN_IMAGE_COVERAGE,
        "DRAWING_MIN_SEGMENTS": config.DRAWING_MIN_SEGMENTS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF ingestion pipeline for RAG")
    parser.add_argument("pdf", type=Path, help="path to the PDF to ingest")
    parser.add_argument(
        "--out", type=Path, default=Path("output"), help="output root directory (default: ./output)"
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        default=1,
        metavar="N",
        help="resume from stage N, reloading earlier stage artifacts",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="skip writing downscaled debug images (full-size runs)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="per-page DEBUG logging (default: per-stage summaries)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    manifest = run(args.pdf, args.out, from_stage=args.from_stage, debug=not args.no_debug)

    print(f"\n{manifest['doc_id']}: {manifest['page_count']} pages")
    for kind, n in sorted(manifest["counts"].items()):
        print(f"  {kind:12s} {n}")
    print(f"\nartifacts -> {args.out / manifest['doc_id']}")


if __name__ == "__main__":
    main()
