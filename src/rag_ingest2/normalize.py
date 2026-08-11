"""LAYER 5 — Document normalization: the layer v1 never had.

Whole-document passes over the unit pool and table fragments — pure
functions over lists, the most testable code in the system:

  * furniture stripping (#26, #30): repeated-position analysis over text
    units AND YOLO table suspects in the same pass — v1 handled the same
    phenomenon in two different modules
  * table-region dedup / suspect creation (#23)
  * cross-page table stitching (tables/stitch.py)
  * heading levels resolved document-wide (#6, #25) — written INTO the
    units, so downstream layers read data, not a side table
  * figure dedup by content hash (#8's note; toggleable, default off)

No quality flags here — evidence in, normalized pool out; the quality
gate is next door.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from rag_ingest.models import Source, Unit, UnitType

from .config import NormalizeConfig, TableConfig
from .tables import RawTable, TableResult
from .tables.stitch import merge_chain, stitch
from .tables.validate import cells_to_grid, cells_to_markdown

log = logging.getLogger(__name__)

_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")


def _furniture_norm(text: str) -> str:
    """Digits normalized out so 'Page 5 of 33' matches 'Page 6 of 33'."""
    return re.sub(r"\d+", "#", " ".join(text.lower().split()))


def strip_repeated_furniture(
    units: list[Unit], page_heights: dict[int, float], cfg: NormalizeConfig
) -> list[Unit]:
    """Drop repeating page headers/footers (#26/#30): a text/title unit
    in the top/bottom band whose (normalized text, y-band) recurs on
    enough distinct pages. Body text never matches — it doesn't recur at
    one fixed y across pages."""
    candidates: dict[tuple[int, str], set[int]] = {}
    for u in units:
        if u.type not in (UnitType.TEXT, UnitType.TITLE) or not u.content.strip():
            # Empty units never match as furniture: a failed paid-lane
            # page leaves one empty flagged placeholder per page, and
            # "same emptiness on N pages" must not delete the pages.
            continue
        h = page_heights.get(u.page, 842.0)
        if u.bbox[1] <= h * cfg.furniture_band_frac or u.bbox[3] >= h * (
            1 - cfg.furniture_band_frac
        ):
            key = (round(u.bbox[1] / 8), _furniture_norm(u.content))
            candidates.setdefault(key, set()).add(u.page)

    furniture = {k for k, pages in candidates.items() if len(pages) >= cfg.furniture_min_repeats}
    if not furniture:
        return units
    kept: list[Unit] = []
    dropped = 0
    for u in units:
        if (
            u.type in (UnitType.TEXT, UnitType.TITLE)
            and u.content.strip()
            and (round(u.bbox[1] / 8), _furniture_norm(u.content)) in furniture
        ):
            dropped += 1
            continue
        kept.append(u)
    log.info("furniture: dropped %d repeating unit(s) (%d distinct)", dropped, len(furniture))
    return kept


def suppress_repeated_suspects(
    fragments: list[RawTable], cfg: NormalizeConfig
) -> list[RawTable]:
    """The suspect half of the same furniture phenomenon (#30): a
    yolo_only suspect recurring at the same position on many pages is a
    bordered page-title box, not a table — each one would open a
    needs_review item for a human."""
    groups: dict[tuple[int, ...], list[RawTable]] = {}
    for t in fragments:
        if t.source == "yolo_only":
            groups.setdefault(tuple(round(v / 10) for v in t.bbox), []).append(t)
    drop = {
        id(t)
        for g in groups.values()
        if len({t.page for t in g}) >= cfg.furniture_min_repeats
        for t in g
    }
    if drop:
        log.info("suppressed %d repeated page-furniture table suspect(s)", len(drop))
    return [t for t in fragments if id(t) not in drop]


def dedup_units_in_tables(units: list[Unit], tables: list[TableResult]) -> list[Unit]:
    """Drop text/title units whose center sits inside any table span
    (#23) — stages extracted that text once as prose and once as table
    cells. Paid-lane (GEMINI) units are exempt: page-level bboxes put
    every center at midpage, and the VLM emits prose and tables
    disjointly anyway."""
    spans: dict[int, list[tuple[float, float, float, float]]] = {}
    for t in tables:
        for page, bbox in t.page_spans:
            spans.setdefault(page, []).append(bbox)

    kept: list[Unit] = []
    dropped = 0
    for u in units:
        if (
            u.type in (UnitType.TEXT, UnitType.TITLE)
            and u.source != Source.GEMINI
            and u.page in spans
        ):
            cx = (u.bbox[0] + u.bbox[2]) / 2
            cy = (u.bbox[1] + u.bbox[3]) / 2
            if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in spans[u.page]):
                dropped += 1
                continue
        kept.append(u)
    if dropped:
        log.info("dedup: dropped %d text unit(s) inside table regions", dropped)
    return kept


def resolve_heading_levels(units: list[Unit], cfg: NormalizeConfig) -> None:
    """Write a document-wide level into every TITLE unit, in place.
    Priority: explicit level (paid lane saw the hierarchy) > numbering
    ("7.3.1" is depth 3 — the author said so) > font-size rank, ranked
    SEPARATELY per source (#17: different measurement systems)."""
    sizes_by_source: dict[str, set[float]] = {}
    for u in units:
        if u.type == UnitType.TITLE and u.font_size:
            sizes_by_source.setdefault(u.source.value, set()).add(round(u.font_size))
    rank: dict[tuple[str, float], int] = {}
    for source, sizes in sizes_by_source.items():
        for i, size in enumerate(sorted(sizes, reverse=True), start=1):
            rank[(source, size)] = i

    for u in units:
        if u.type != UnitType.TITLE:
            continue
        m = _NUMBERED.match(u.content)
        if u.level is not None:
            level = u.level
        elif m:
            level = m.group(1).count(".") + 1
        elif u.font_size:
            level = rank.get((u.source.value, round(u.font_size)), 1)
        else:
            level = 1
        u.level = min(level, cfg.max_heading_level)


def dedup_figures_by_hash(units: list[Unit], doc_out: Path) -> list[Unit]:
    """Intentional improvement over v1 (#8), behind cfg.figure_dedup:
    identical stored images (repeated logos that cleared the area
    threshold) keep only their first occurrence."""
    seen: dict[str, int] = {}
    kept: list[Unit] = []
    dropped = 0
    for u in units:
        if u.type == UnitType.FIGURE and u.storage_key:
            path = doc_out / u.storage_key
            if path.exists():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen:
                    dropped += 1
                    continue
                seen[digest] = u.page
        kept.append(u)
    if dropped:
        log.info("figure dedup: dropped %d duplicate stored image(s)", dropped)
    return kept


def build_tables(
    fragments: list[RawTable],
    page_heights: dict[int, float],
    norm_cfg: NormalizeConfig,
    table_cfg: TableConfig,
) -> list[TableResult]:
    """Suspect suppression -> stitching -> TableResults with renderings.
    No flags here — the quality gate validates and may blank the
    renderings of anything it rejects."""
    fragments = suppress_repeated_suspects(fragments, norm_cfg)
    results: list[TableResult] = []
    for chain in stitch(fragments, page_heights, table_cfg):
        first = chain[0]
        table_id = f"t{first.page:04d}_{sum(1 for t in results if t.pages[0] == first.page):02d}"
        cells, merges = merge_chain(chain, table_cfg)
        results.append(
            TableResult(
                table_id=table_id,
                pages=[f.page for f in chain],
                page_spans=[(f.page, f.bbox) for f in chain],
                col_count=len(cells[0]) if cells else 0,
                row_count=len(cells),
                markdown=cells_to_markdown(cells) if cells else "",
                source=first.source,
                cells=cells,
                header_rows=first.header_rows,
                merges=merges,
                grid=cells_to_grid(cells, merges) if cells else "",
            )
        )
    log.info(
        "tables: %d fragment(s) -> %d table(s), %d multi-page",
        len(fragments),
        len(results),
        sum(1 for t in results if len(t.pages) > 1),
    )
    return results


def normalize_document(
    units: list[Unit],
    fragments: list[RawTable],
    page_heights: dict[int, float],
    norm_cfg: NormalizeConfig,
    table_cfg: TableConfig,
    doc_out: Path,
) -> tuple[list[Unit], list[TableResult]]:
    """The whole layer, in its fixed order: furniture -> tables ->
    unit-vs-table dedup -> heading levels -> (optional) figure dedup."""
    units = strip_repeated_furniture(units, page_heights, norm_cfg)
    tables = build_tables(fragments, page_heights, norm_cfg, table_cfg)
    units = dedup_units_in_tables(units, tables)
    resolve_heading_levels(units, norm_cfg)
    if norm_cfg.figure_dedup:
        units = dedup_figures_by_hash(units, doc_out)
    return units, tables
