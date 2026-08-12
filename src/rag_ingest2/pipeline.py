"""v2 orchestrator + CLI: the eight layers as StageSpecs.

This is the DEFAULT pipeline since the cutover (the `rag-ingest` console
script): validated by a chunk-identical diff against v1 on the full
runnable corpus (scripts/diff_v1_v2.py).

Usage:
    python -m rag_ingest2.pipeline <pdf> [--out output] [--from-stage N]
                                   [--workers N] [--no-debug] [--vlm-cache DIR]

Layer 0 (ingest gate) runs before any stage exists: encrypted, corrupt,
and zero-page PDFs are rejected with a manifest reason, never a
traceback. Stages 1-7 then run under the uniform orchestrator
(stages.py); each writes its full artifact before the next reads it.

Artifacts under output/<doc_id>/stages/:
    01_profiles.json    evidence per page (no decisions)
    02_routes.json      the decision log
    03_extract.json     units + regions + grids + vlm records + body size
    04_fragments.jsonl  raw table fragments from every extractor
    05_normalized.json  furniture-stripped, deduped, leveled pool + tables
    06_reviewed.json    the same pool after the quality gate (+ review_report.md)
    07_chunks.jsonl     retrieval-ready chunks (+ merged.md)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pymupdf

from rag_ingest.models import Unit, UnitType
from rag_ingest.vlm_extract import VlmPageRecord

from . import chunking, normalize, quality, workers
from .config import IngestConfig
from .profiles import PageProfile, profile_document
from .routing import Extractor, PageRoute, route_document
from .stages import PipelineContext, StageSpec, run_stages
from .tables import RawTable, TableResult, overlap_frac
from .tables.cells_native import extract_native_tables
from .tables.cells_vlm import vlm_table_fragment
from .tables.grids import RuledGrid

log = logging.getLogger(__name__)


class IngestRejected(Exception):
    """Layer 0: the document cannot be ingested at all."""


def _open_checked(pdf_path: Path) -> pymupdf.Document:
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        raise IngestRejected(f"unreadable PDF: {e}") from e
    if doc.needs_pass:
        doc.close()
        raise IngestRejected("encrypted PDF: password required")
    if doc.page_count == 0:
        doc.close()
        raise IngestRejected("PDF has zero pages")
    return doc


# ---------------------------------------------------------------------------
# Stage run functions (each reads prior results from ctx.results)
# ---------------------------------------------------------------------------


def _page_heights(ctx: PipelineContext) -> dict[int, float]:
    doc = _open_checked(ctx.pdf_path)
    try:
        return {i: doc.load_page(i).rect.height for i in range(doc.page_count)}
    finally:
        doc.close()


def _run_profiles(ctx: PipelineContext) -> list[PageProfile]:
    doc = _open_checked(ctx.pdf_path)
    try:
        return profile_document(doc, ctx.cfg.routing)
    finally:
        doc.close()


def _run_routes(ctx: PipelineContext) -> list[PageRoute]:
    profiles: list[PageProfile] = ctx.results["profiles"]  # type: ignore[assignment]
    return route_document(profiles, ctx.cfg.routing)


def _run_extract(ctx: PipelineContext) -> workers.ExtractionResult:
    profiles: list[PageProfile] = ctx.results["profiles"]  # type: ignore[assignment]
    routes: list[PageRoute] = ctx.results["routes"]  # type: ignore[assignment]
    rules = ctx.cfg.routing
    # VLM pages rerouted with a countable-but-lying text layer keep the
    # char count as verification evidence (the conditions mirror the
    # routing guards); true scans rely on the ink proxy instead.
    text_chars = {
        p.page: p.text_chars
        for p, r in zip(profiles, routes, strict=True)
        if r.extractor == Extractor.VLM
        and p.text_chars >= rules.min_text_chars
        and p.max_image_coverage <= rules.scan_image_coverage
    }
    return workers.extract_document(ctx.pdf_path, routes, text_chars, ctx.cfg, ctx.doc_out)


def _run_fragments(ctx: PipelineContext) -> list[RawTable]:
    """LAYER 4 — the ladder: every extractor contributes fragments."""
    routes: list[PageRoute] = ctx.results["routes"]  # type: ignore[assignment]
    ext: workers.ExtractionResult = ctx.results["extract"]  # type: ignore[assignment]
    fragments: list[RawTable] = []

    # Native rung: page-wide find_tables on every NATIVE page — the
    # vector lines are authoritative where they exist.
    native_pages = [r.page for r in routes if r.extractor == Extractor.NATIVE]
    doc = _open_checked(ctx.pdf_path)
    try:
        for pno in native_pages:
            fragments.extend(extract_native_tables(doc.load_page(pno), pno))
    finally:
        doc.close()

    # Paid rung: the VLM returned tables inline; parse to cells.
    for u in ext.units:
        if u.type == UnitType.TABLE:
            fragments.append(vlm_table_fragment(u))

    # Cross-check rung (#23): a YOLO table on a NATIVE page that
    # find_tables did not see is a borderless-table suspect -> review
    # item, never a silent miss.
    native_set = set(native_pages)
    for g in ext.regions:
        if g.label != "table" or g.page not in native_set:
            continue
        gb = g.box_pdf.as_tuple()
        if not any(
            t.page == g.page and overlap_frac(t.bbox, gb) > 0.3 for t in fragments
        ):
            fragments.append(RawTable(page=g.page, bbox=gb, cells=[], source="yolo_only"))

    log.info("ladder: %d fragment(s)", len(fragments))
    return fragments


def _run_normalize(ctx: PipelineContext) -> tuple[list[Unit], list[TableResult]]:
    ext: workers.ExtractionResult = ctx.results["extract"]  # type: ignore[assignment]
    fragments: list[RawTable] = ctx.results["fragments"]  # type: ignore[assignment]
    # TABLE units became fragments in Layer 4 — the pool keeps the rest.
    pool = [u for u in ext.units if u.type != UnitType.TABLE]
    return normalize.normalize_document(
        pool,
        fragments,
        _page_heights(ctx),
        ctx.cfg.normalize,
        ctx.cfg.tables,
        ctx.doc_out,
    )


def _run_quality(ctx: PipelineContext) -> quality.QualityResult:
    units, tables = ctx.results["normalize"]  # type: ignore[misc]
    ext: workers.ExtractionResult = ctx.results["extract"]  # type: ignore[assignment]
    result = quality.apply_quality_gate(
        units, tables, ext.vlm_pages, ctx.pdf_path, ctx.doc_out, ctx.cfg.extraction
    )
    report = quality.render_review_report(ctx.pdf_path.stem, result)
    (ctx.doc_out / "review_report.md").write_text(report, encoding="utf-8")
    log.info("review report -> %s", ctx.doc_out / "review_report.md")
    return result


def _run_chunks(ctx: PipelineContext) -> list:
    reviewed: quality.QualityResult = ctx.results["quality"]  # type: ignore[assignment]
    chunks, merged_md = chunking.chunk_document(
        ctx.pdf_path.stem, reviewed.units, reviewed.tables, ctx.cfg.chunking
    )
    (ctx.doc_out / "merged.md").write_text(merged_md, encoding="utf-8")
    log.info("merged.md -> %s", ctx.doc_out / "merged.md")
    return chunks


# ---------------------------------------------------------------------------
# Serialization (dumb dict plumbing, one place per stage)
# ---------------------------------------------------------------------------


def _ser_extract(r: workers.ExtractionResult) -> dict:
    return {
        "body_font_size": r.body_font_size,
        "units": [u.to_dict() for u in r.units],
        "regions": [g.to_dict() for g in r.regions],
        "grids": [g.to_dict() for g in r.grids],
        "vlm_pages": [p.to_dict() for p in r.vlm_pages],
        "render_meta": r.render_meta,
    }


def _de_extract(raw: object, _ctx: PipelineContext) -> workers.ExtractionResult:
    d = raw  # type: ignore[assignment]
    return workers.ExtractionResult(
        body_font_size=d["body_font_size"],  # type: ignore[index]
        units=[Unit.from_dict(u) for u in d["units"]],  # type: ignore[index]
        regions=[workers.Region.from_dict(g) for g in d["regions"]],  # type: ignore[index]
        grids=[RuledGrid.from_dict(g) for g in d["grids"]],  # type: ignore[index]
        vlm_pages=[VlmPageRecord(**p) for p in d["vlm_pages"]],  # type: ignore[index]
        render_meta=d["render_meta"],  # type: ignore[index]
    )


def _ser_pool(r: tuple[list[Unit], list[TableResult]]) -> dict:
    units, tables = r
    return {
        "units": [u.to_dict() for u in units],
        "tables": [t.to_dict() for t in tables],
    }


def _de_pool(raw: object, _ctx: PipelineContext) -> tuple[list[Unit], list[TableResult]]:
    return (
        [Unit.from_dict(u) for u in raw["units"]],  # type: ignore[index]
        [TableResult.from_dict(t) for t in raw["tables"]],  # type: ignore[index]
    )


def _ser_quality(r: quality.QualityResult) -> dict:
    d = _ser_pool((r.units, r.tables))
    d["review_items"] = [item.__dict__ for item in r.items]
    return d


def _de_quality(raw: object, _ctx: PipelineContext) -> quality.QualityResult:
    units, tables = _de_pool(raw, _ctx)
    items = [quality.ReviewItem(**i) for i in raw["review_items"]]  # type: ignore[index]
    return quality.QualityResult(units=units, tables=tables, items=items)


STAGES: list[StageSpec] = [
    StageSpec(
        name="profiles",
        artifact="01_profiles.json",
        run=_run_profiles,
        serialize=lambda r: {"pages": [p.to_dict() for p in r]},  # type: ignore[union-attr]
        deserialize=lambda raw, _ctx: [PageProfile.from_dict(p) for p in raw["pages"]],  # type: ignore[index]
    ),
    StageSpec(
        name="routes",
        artifact="02_routes.json",
        run=_run_routes,
        serialize=lambda r: {"pages": [x.to_dict() for x in r]},  # type: ignore[union-attr]
        deserialize=lambda raw, _ctx: [PageRoute.from_dict(p) for p in raw["pages"]],  # type: ignore[index]
    ),
    StageSpec(
        name="extract",
        artifact="03_extract.json",
        run=_run_extract,
        serialize=_ser_extract,  # type: ignore[arg-type]
        deserialize=_de_extract,
    ),
    StageSpec(
        name="fragments",
        artifact="04_fragments.jsonl",
        run=_run_fragments,
        serialize=lambda r: [t.to_dict() for t in r],  # type: ignore[union-attr]
        deserialize=lambda raw, _ctx: [RawTable.from_dict(t) for t in raw],  # type: ignore[union-attr]
    ),
    StageSpec(
        name="normalize",
        artifact="05_normalized.json",
        run=_run_normalize,
        serialize=_ser_pool,  # type: ignore[arg-type]
        deserialize=_de_pool,
    ),
    StageSpec(
        name="quality",
        artifact="06_reviewed.json",
        run=_run_quality,
        serialize=_ser_quality,  # type: ignore[arg-type]
        deserialize=_de_quality,
    ),
    StageSpec(
        name="chunks",
        artifact="07_chunks.jsonl",
        run=_run_chunks,
        serialize=lambda r: [c.to_dict() for c in r],  # type: ignore[union-attr]
        deserialize=lambda raw, _ctx: raw,
    ),
]


# ---------------------------------------------------------------------------
# run() + manifest
# ---------------------------------------------------------------------------


def run(
    pdf_path: Path,
    out_dir: Path,
    cfg: IngestConfig | None = None,
    from_stage: int = 1,
) -> dict:
    """Run the v2 pipeline on one PDF; returns the manifest dict."""
    cfg = cfg or IngestConfig()
    doc_id = pdf_path.stem
    doc_out = out_dir / doc_id
    doc_out.mkdir(parents=True, exist_ok=True)

    # LAYER 0 — ingest gate.
    try:
        _open_checked(pdf_path).close()
    except IngestRejected as e:
        manifest = {
            "doc_id": doc_id,
            "source_pdf": str(pdf_path),
            "status": "rejected",
            "reason": str(e),
        }
        (doc_out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log.error("rejected %s: %s", pdf_path, e)
        return manifest

    ctx = PipelineContext(pdf_path=pdf_path, doc_out=doc_out, cfg=cfg)
    run_stages(STAGES, ctx, from_stage=from_stage)

    routes: list[PageRoute] = ctx.results["routes"]  # type: ignore[assignment]
    ext: workers.ExtractionResult = ctx.results["extract"]  # type: ignore[assignment]
    reviewed: quality.QualityResult = ctx.results["quality"]  # type: ignore[assignment]
    chunks = ctx.results["chunks"]

    route_counts: dict[str, int] = {}
    for r in routes:
        route_counts[r.extractor.value] = route_counts.get(r.extractor.value, 0) + 1
    unit_counts: dict[str, int] = {}
    for u in ext.units:
        unit_counts[u.type.value] = unit_counts.get(u.type.value, 0) + 1
    fresh = [p for p in ext.vlm_pages if not p.cached]

    manifest = {
        "doc_id": doc_id,
        "source_pdf": str(pdf_path),
        "pipeline": "rag_ingest2",
        "status": "ok",
        "page_count": len(routes),
        "routes": route_counts,
        "body_font_size": ext.body_font_size,
        "unit_counts": unit_counts,
        "vlm": {
            "pages": len(ext.vlm_pages),
            "cached": len(ext.vlm_pages) - len(fresh),
            "input_tokens": sum(p.input_tokens for p in fresh),
            "output_tokens": sum(p.output_tokens for p in fresh),
        },
        "tables": {
            "count": len(reviewed.tables),
            "multi_page": [t.table_id for t in reviewed.tables if len(t.pages) > 1],
            "needs_review": [t.table_id for t in reviewed.tables if t.needs_review],
        },
        "chunks": {
            "count": len(chunks),  # type: ignore[arg-type]
            "needs_review": [c.chunk_id for c in chunks if c.needs_review],  # type: ignore[union-attr]
        },
        "review_items": len(reviewed.items),
        "timings_secs": ctx.timings,
        # The whole config, wholesale — v1's per-stage threshold
        # snapshots, generalized (rewrite §3).
        "config": cfg.to_dict(),
        "page_routes": {str(r.page): r.extractor.value for r in routes},
    }
    (doc_out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF ingestion pipeline v2 (rewrite)")
    parser.add_argument("pdf", type=Path, help="path to the PDF to ingest")
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument(
        "--from-stage",
        type=int,
        default=1,
        choices=range(1, 8),
        metavar="N",
        help="resume from stage N (1-7), reloading earlier stage artifacts",
    )
    parser.add_argument("--workers", type=int, default=1, help="extraction worker processes")
    parser.add_argument("--no-debug", action="store_true", help="skip debug images")
    parser.add_argument(
        "--vlm-cache",
        type=Path,
        default=None,
        help="shared VLM response cache dir (e.g. v1's output/<doc>/cache/vlm)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    cfg = IngestConfig(
        workers=args.workers,
        debug_images=not args.no_debug,
        vlm_cache_dir=str(args.vlm_cache) if args.vlm_cache else None,
    )
    manifest = run(args.pdf, args.out, cfg=cfg, from_stage=args.from_stage)
    if manifest.get("status") == "rejected":
        print(f"\nREJECTED {manifest['doc_id']}: {manifest['reason']}")
        sys.exit(2)

    print(f"\n{manifest['doc_id']}: {manifest['page_count']} pages")
    for kind, n in sorted(manifest["routes"].items()):
        print(f"  {kind:12s} {n}")
    print(f"\nartifacts -> {args.out / manifest['doc_id']}")


if __name__ == "__main__":
    main()
