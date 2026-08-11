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
        stages/03_render.json         # per-page render metadata + drawing keys
        stages/04_layout.jsonl        # YOLO regions, px AND pdf boxes
        stages/05_ocr_units.jsonl     # scanned-page units via Tesseract
        stages/06_tables.jsonl        # tiered ladder results + stitching
        stages/07_chunks.jsonl        # THE output: retrieval-ready chunks
        debug/renders/            # downscaled page JPEGs (what YOLO saw)
        figures/                  # full-quality stored PNGs
        merged.md                 # human-review markdown (never authoritative)
        manifest.json             # summary: counts, timings, review flags
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pymupdf

from . import assemble, chunking, local_extract, ocr, render, tables
from .config import FIGURE_DPI, RENDER_DPI
from .layout import LayoutDetector
from .models import PageKind, Source, Unit, UnitType
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

        # ---- STAGES 3+4: render + layout, one pass ---------------------
        # Interleaved on purpose: rendering all pages first would hold
        # hundreds of ~11 MB pixmaps alive. Each page renders, YOLO
        # consumes it, the debug JPEG is cut from it, and it dies before
        # the next page renders (see render.py module docstring).
        detector = LayoutDetector()
        render_meta: list[dict] = []
        regions = []
        drawing_units: list[Unit] = []
        t_render = t_layout = 0.0
        for r in records:
            page = doc.load_page(r.page)
            if r.kind == PageKind.DRAWING:
                key = render.store_drawing_page(page, r.page, doc_out)
                render_meta.append({"page": r.page, "drawing_figure_key": key})
                # The whole page is the figure — enters assembly as a unit.
                drawing_units.append(
                    Unit(
                        page=r.page,
                        bbox=(page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1),
                        type=UnitType.FIGURE,
                        storage_key=key,
                        source=Source.PYMUPDF,
                    )
                )
                continue

            t0 = time.perf_counter()
            pix = render.render_page(page)
            rgb = render.pixmap_to_rgb_array(pix)
            t_render += time.perf_counter() - t0

            t0 = time.perf_counter()
            regions.extend(detector.detect(rgb, page.rect, r.page))
            t_layout += time.perf_counter() - t0

            render_meta.append({"page": r.page, "px_width": pix.width, "px_height": pix.height})
            if debug:
                render.save_debug_jpeg(page, doc_out / "debug" / "renders" / f"p{r.page:04d}.jpg")

        timings["render"] = round(t_render, 3)
        timings["layout"] = round(t_layout, 3)
        _write_stage(doc_out, "03_render.json", {"dpi": RENDER_DPI, "pages": render_meta})
        _write_stage_jsonl(doc_out, "04_layout.jsonl", [g.to_dict() for g in regions])
        log.info(
            "layout: %d region(s) across %d page(s)",
            len(regions),
            len({g.page for g in regions}),
        )

        # ---- STAGE 5: OCR scanned prose + store scanned figure crops ---
        # Scanned pages flow through the SAME extraction walk as native
        # pages (see local_extract.extract_page's textpage seam); figure
        # regions YOLO found on scanned pages become stored PNG crops —
        # cropped via the converted bbox_pdf, i.e. the stage-4 coordinate
        # helper is what makes these crops land on the right pixels.
        t0 = time.perf_counter()
        ocr_units: list[Unit] = []
        textpages: dict[int, pymupdf.TextPage] = {}  # shared with stage 6 — OCR once
        for r in records:
            if r.kind != PageKind.SCANNED:
                continue
            page = doc.load_page(r.page)
            textpages[r.page] = ocr.get_ocr_textpage(page)
            ocr_units.extend(
                ocr.ocr_page_units(page, r.page, doc_out / "figures", textpage=textpages[r.page])
            )
        for g in regions:
            if g.label != "figure" or page_kinds.get(g.page) != PageKind.SCANNED:
                continue
            key = f"figures/p{g.page:04d}_r{len(ocr_units):02d}.png"
            pix = doc.load_page(g.page).get_pixmap(clip=pymupdf.Rect(g.bbox_pdf), dpi=FIGURE_DPI)
            (doc_out / key).parent.mkdir(parents=True, exist_ok=True)
            (doc_out / key).write_bytes(pix.tobytes("png"))
            ocr_units.append(
                Unit(
                    page=g.page,
                    bbox=g.bbox_pdf,
                    type=UnitType.FIGURE,
                    storage_key=key,
                    source=Source.PYMUPDF,
                )
            )
        timings["ocr"] = round(time.perf_counter() - t0, 3)
        _write_stage_jsonl(doc_out, "05_ocr_units.jsonl", [u.to_dict() for u in ocr_units])

        # ---- STAGE 6: tables — tiered ladder + stitching ---------------
        t0 = time.perf_counter()
        raw_tables: list[tables.RawTable] = []
        # Tier 1: every text-native page through find_tables (vector grid
        # + exact text). Runs page-wide, not per YOLO region — the lines
        # are authoritative where they exist.
        for r in records:
            if r.kind == PageKind.TEXT_NATIVE:
                raw_tables.extend(tables.extract_native_tables(doc.load_page(r.page), r.page))
        # Tier 2: scanned-page YOLO table regions -> pixel grid, line
        # removal, then a dedicated OCR pass on the cleaned crop (the
        # stage-5 full-page OCR is unusable here: Tesseract drops text
        # inside ruled cells — see tables.extract_scanned_table).
        for g in regions:
            if g.label == "table" and page_kinds.get(g.page) == PageKind.SCANNED:
                raw_tables.append(
                    tables.extract_scanned_table(doc.load_page(g.page), g.page, g.bbox_pdf)
                )
        # Cross-check: a YOLO table on a NATIVE page that find_tables did
        # not see is a borderless-table suspect -> review item, not a miss.
        for g in regions:
            if g.label != "table" or page_kinds.get(g.page) != PageKind.TEXT_NATIVE:
                continue
            if not any(
                t.page == g.page and tables.overlap_frac(t.bbox, g.bbox_pdf) > 0.3
                for t in raw_tables
            ):
                raw_tables.append(
                    tables.RawTable(page=g.page, bbox=g.bbox_pdf, cells=[], source="yolo_only")
                )
        page_heights = {r.page: doc.load_page(r.page).rect.height for r in records}
        table_results = tables.finalize(raw_tables, page_heights, doc, doc_out)
        timings["tables"] = round(time.perf_counter() - t0, 3)
        _write_stage_jsonl(doc_out, "06_tables.jsonl", [t.to_dict() for t in table_results])

        # ---- STAGE 7: assembly + chunking ------------------------------
        # Everything converges: all units from all sources, deduplicated
        # against table regions, walked in reading order, chunked per
        # heading section. This is where 0-based pages become 1-based.
        t0 = time.perf_counter()
        all_units = units + ocr_units + drawing_units
        walk = assemble.build_walk(all_units, table_results)
        chunks, merged_md = chunking.chunk_document(doc_id, walk)
        timings["assemble"] = round(time.perf_counter() - t0, 3)
        _write_stage_jsonl(doc_out, "07_chunks.jsonl", [c.to_dict() for c in chunks])
        (doc_out / "merged.md").write_text(merged_md, encoding="utf-8")
        log.info("merged.md -> %s", doc_out / "merged.md")

        counts: dict[str, int] = {}
        for r in records:
            counts[r.kind.value] = counts.get(r.kind.value, 0) + 1
        unit_counts: dict[str, int] = {}
        for u in units + ocr_units:
            unit_counts[u.type.value] = unit_counts.get(u.type.value, 0) + 1

        manifest = {
            "doc_id": doc_id,
            "source_pdf": str(pdf_path),
            "page_count": doc.page_count,
            "counts": counts,
            "body_font_size": body_size,
            "unit_counts": unit_counts,
            "ruled_grid_pages": [g.page for g in grids],
            "tables": {
                "count": len(table_results),
                "multi_page": [t.table_id for t in table_results if len(t.pages) > 1],
                "needs_review": [t.table_id for t in table_results if t.needs_review],
            },
            "chunks": {
                "count": len(chunks),
                "by_type": {
                    kind: sum(1 for c in chunks if c.type.value == kind)
                    for kind in {c.type.value for c in chunks}
                },
                "needs_review": [c.chunk_id for c in chunks if c.needs_review],
            },
            "layout_regions": {
                label: sum(1 for g in regions if g.label == label)
                for label in {g.label for g in regions}
            },
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
