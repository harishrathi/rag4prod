"""LAYER 2 — Routing: decisions, no I/O.

A pure function from evidence to verdict. All threshold logic from the
ledger lives here, in one trivially testable module: the image-coverage
override (#1), the drawing rule (#2), the near-blank bias (#3), and the
junk/mojibake reroutes (#29 + VLM spec §3). Each route records EVERY
rule that fired, so the stage artifact is a per-page decision log.

Error direction (rewrite §3): every rule here errs toward wasted compute
(an extra VLM call), never toward silent garbage — the same bias v1's
triage had, now stated once.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from .config import RoutingRules
from .profiles import PageProfile

log = logging.getLogger(__name__)


class Extractor(StrEnum):
    NATIVE = "native"  # honest text layer -> free local extraction
    VLM = "vlm"  # scan or lying text layer -> paid lane
    DRAWING = "drawing"  # CAD/vector page -> stored figure, never the API


@dataclass(frozen=True)
class PageRoute:
    """Immutable verdict for one page. Every consumer — worker, resumed
    run, table crop — reads the same data, so there is nothing to
    re-apply on resume (the v1 rotation-rehydration hack has no v2
    equivalent by construction)."""

    page: int
    extractor: Extractor
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["extractor"] = self.extractor.value
        d["reasons"] = list(self.reasons)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PageRoute:
        return cls(
            page=d["page"],
            extractor=Extractor(d["extractor"]),
            reasons=tuple(d["reasons"]),
        )


def route(profile: PageProfile, rules: RoutingRules) -> PageRoute:
    """Evidence -> verdict for one page. Pure: same profile, same route."""
    p = profile
    compact = max(1, p.text_compact_chars)

    # Rule 1 — a raster blanketing the page outranks any text layer
    # (scanner-stamped headers pass the length test; #1).
    if p.max_image_coverage > rules.scan_image_coverage:
        return PageRoute(
            page=p.page,
            extractor=Extractor.VLM,
            reasons=(
                f"raster image covers {p.max_image_coverage:.0%} of page "
                f"(> {rules.scan_image_coverage:.0%})",
            ),
        )

    if p.text_chars >= rules.min_text_chars:
        # Rules 2a/2b — the text layer is present but may be lying
        # (broken ToUnicode CMap, #29 + VLM spec §3). Both symptoms are
        # checked and BOTH are recorded when both fire.
        fired: list[str] = []
        junk_ratio = p.junk_chars / compact
        moji_ratio = p.mojibake_chars / compact
        if p.junk_chars >= rules.junk_min or junk_ratio >= rules.junk_ratio:
            fired.append(f"text layer corrupt: {p.junk_chars} junk chars ({junk_ratio:.1%})")
        if p.mojibake_chars >= rules.mojibake_min or moji_ratio >= rules.mojibake_ratio:
            fired.append(
                f"text layer mojibake: {p.mojibake_chars} suspect chars ({moji_ratio:.1%})"
            )
        if fired:
            return PageRoute(page=p.page, extractor=Extractor.VLM, reasons=tuple(fired))
        return PageRoute(
            page=p.page,
            extractor=Extractor.NATIVE,
            reasons=(f"{p.text_chars} chars of clean text (>= {rules.min_text_chars})",),
        )

    # Rule 3 — near-textless: vector drawings get their own class; the
    # VLM would return junk for a CAD plan at real cost (#2).
    if p.vector_segments is not None and p.vector_segments >= rules.drawing_min_segments:
        return PageRoute(
            page=p.page,
            extractor=Extractor.DRAWING,
            reasons=(
                f"only {p.text_chars} chars but {p.vector_segments} vector "
                f"segments (>= {rules.drawing_min_segments}): CAD plan / drawing",
            ),
        )

    # Rule 4 — near-blank bias (#3): presumed scan; a blank page costs
    # one cheap VLM call, the accepted error direction.
    return PageRoute(
        page=p.page,
        extractor=Extractor.VLM,
        reasons=(
            f"only {p.text_chars} chars of text (< {rules.min_text_chars}), "
            f"not a drawing: presumed scan",
        ),
    )


def route_document(profiles: list[PageProfile], rules: RoutingRules) -> list[PageRoute]:
    routes = [route(p, rules) for p in profiles]
    counts: dict[str, int] = {}
    for r in routes:
        counts[r.extractor.value] = counts.get(r.extractor.value, 0) + 1
    log.info("routing: %d page(s) -> %s", len(routes), counts)
    return routes
