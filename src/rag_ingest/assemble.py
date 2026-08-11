"""STAGE 7a — Assembly: dedup, heading levels, and the ordered walk.

Three jobs, in order:

1. **Dedup** — the most important correctness rule in the pipeline
   (design spec §8.1): stages 2 and 5 extracted ALL text, including the
   text inside table regions; stage 6 extracted those same regions as
   structured tables. Text units whose CENTER falls inside a table span
   are dropped — center containment, not any-intersection, so a padded
   table box cannot swallow the prose line above it.

2. **Heading levels**, assigned document-wide:
     * numbered headings are authoritative: "7.3.1 Delay Notices" is
       depth 3 because it has three number segments — no font analysis
       can beat the author telling us the depth;
     * otherwise, TITLE font sizes cluster by rank: biggest distinct
       size -> level 1, next -> 2, ... Ranked SEPARATELY per source:
       native sizes and OCR-synthesized sizes are different measurement
       systems (ledger #17) and must never share a ranking.

3. **The walk** — everything (text, titles, tables, figures) merged into
   one stream sorted by (page, y0), tables anchored at their first
   fragment's position. The walk maintains the heading stack that gives
   every downstream chunk its breadcrumb. Known limit: multi-column
   reading order (spec §8.3, accepted).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .config import MAX_HEADING_LEVEL
from .models import Unit, UnitType
from .tables import TableResult

log = logging.getLogger(__name__)

_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")


@dataclass
class WalkItem:
    """One element of the document stream, in reading order."""

    kind: str  # "title" | "text" | "figure" | "table"
    page: int  # anchor page (0-based)
    y0: float  # anchor position for ordering
    unit: Unit | None = None  # title/text/figure items
    table: TableResult | None = None  # table items
    level: int | None = None  # resolved heading level, titles only


def dedup_units(units: list[Unit], tables: list[TableResult]) -> list[Unit]:
    """Drop text/title units whose center sits inside any table span."""
    spans: dict[int, list[tuple[float, float, float, float]]] = {}
    for t in tables:
        for page, bbox in t.page_spans:
            spans.setdefault(page, []).append(bbox)

    kept: list[Unit] = []
    dropped = 0
    for u in units:
        if u.type in (UnitType.TEXT, UnitType.TITLE) and u.page in spans:
            cx = (u.bbox[0] + u.bbox[2]) / 2
            cy = (u.bbox[1] + u.bbox[3]) / 2
            if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in spans[u.page]):
                dropped += 1
                continue
        kept.append(u)
    if dropped:
        log.info("dedup: dropped %d text unit(s) inside table regions", dropped)
    return kept


def assign_heading_levels(units: list[Unit]) -> dict[int, int]:
    """Resolve a level for every TITLE unit; keyed by index into `units`.

    Numbered headings win outright. The size fallback ranks each source's
    distinct font sizes separately (see module docstring)."""
    # Rank distinct title sizes per source, descending: 18pt -> 1, 13pt -> 2 ...
    sizes_by_source: dict[str, set[float]] = {}
    for u in units:
        if u.type == UnitType.TITLE and u.font_size:
            sizes_by_source.setdefault(u.source.value, set()).add(round(u.font_size))
    rank: dict[tuple[str, float], int] = {}
    for source, sizes in sizes_by_source.items():
        for i, size in enumerate(sorted(sizes, reverse=True), start=1):
            rank[(source, size)] = i

    levels: dict[int, int] = {}
    for i, u in enumerate(units):
        if u.type != UnitType.TITLE:
            continue
        m = _NUMBERED.match(u.content)
        if m:
            level = m.group(1).count(".") + 1
        elif u.font_size:
            level = rank.get((u.source.value, round(u.font_size)), 1)
        else:
            level = 1
        levels[i] = min(level, MAX_HEADING_LEVEL)
    return levels


def build_walk(units: list[Unit], tables: list[TableResult]) -> list[WalkItem]:
    """Dedup'd units + tables -> one (page, y0)-ordered stream."""
    units = dedup_units(units, tables)
    levels = assign_heading_levels(units)

    items: list[WalkItem] = []
    for i, u in enumerate(units):
        kind = {UnitType.TITLE: "title", UnitType.TEXT: "text", UnitType.FIGURE: "figure"}.get(
            u.type
        )
        if kind is None:
            continue  # TABLE units never come through this path
        items.append(WalkItem(kind=kind, page=u.page, y0=u.bbox[1], unit=u, level=levels.get(i)))
    for t in tables:
        first_page, first_bbox = t.page_spans[0]
        items.append(WalkItem(kind="table", page=first_page, y0=first_bbox[1], table=t))

    items.sort(key=lambda w: (w.page, w.y0))
    return items
