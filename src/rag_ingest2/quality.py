"""LAYER 6 — Quality gate: ONE pass derives ``needs_review``.

v1 enforced its single real invariant — silent garbage never enters the
corpus — in four independently patched places (theme B). Here every flag
is derived by a per-source validator in one pass, and every flag records
WHICH validator fired and why:

    Source.PYMUPDF  -> junk/orphan-mark check per unit (#29, VLM spec §3)
    Source.GEMINI   -> the page's recorded VLM verification evidence
                       (repetition/length/echo/omission) + [ILLEGIBLE]
    tables          -> structural validation + junk cells; rejects get
                       their renderings blanked and a stored crop

Second output, designed not incidental (#30): ``review_report.md`` —
flagged items grouped by page with reasons and crop links. A drowning
reviewer stops reading flags; the report keeps them afloat.

Error direction (rewrite §3): every validator errs toward a review flag,
never toward silent garbage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from rag_ingest.models import Source, Unit, UnitType
from rag_ingest.text_quality import JUNK_CHARS_RE, orphan_combining_marks
from rag_ingest.vlm_extract import ILLEGIBLE_TOKEN, VlmPageRecord

from .config import ExtractionConfig
from .tables import TableResult
from .tables.validate import junk_cell_count, validate_cells

log = logging.getLogger(__name__)


@dataclass
class ReviewItem:
    """One flagged item, as the report and manifest see it."""

    page: int  # 0-based
    kind: str  # "text" | "title" | "table" | "figure"
    reason: str
    excerpt: str = ""
    crop_key: str | None = None


@dataclass
class QualityResult:
    units: list[Unit]
    tables: list[TableResult]
    items: list[ReviewItem] = field(default_factory=list)


def _unit_reasons(u: Unit, vlm_by_page: dict[int, VlmPageRecord]) -> list[str]:
    """The validator table, as code. Returns every reason that fires."""
    reasons: list[str] = []
    if u.source == Source.PYMUPDF and u.type in (UnitType.TEXT, UnitType.TITLE):
        if JUNK_CHARS_RE.search(u.content):
            reasons.append("junk chars in native text (broken font encoding)")
        elif orphan_combining_marks(u.content) > 0:
            reasons.append("orphan combining marks in native text (broken font encoding)")
    elif u.source == Source.GEMINI:
        rec = vlm_by_page.get(u.page)
        if rec and rec.review_reasons:
            reasons.extend(rec.review_reasons)
        if ILLEGIBLE_TOKEN in u.content:
            reasons.append("region transcribed as [ILLEGIBLE]")
    return reasons


def _table_reason(t: TableResult) -> str | None:
    reason = validate_cells(t.cells)
    if reason is None and t.source == "find_tables":
        # Native cells can carry mojibake on pages whose text layer is
        # only mildly broken — below the routing threshold (#29).
        n_junk = junk_cell_count(t.cells)
        if n_junk:
            reason = f"text layer junk in {n_junk} cell(s) — broken font encoding"
    return reason


def apply_quality_gate(
    units: list[Unit],
    tables: list[TableResult],
    vlm_pages: list[VlmPageRecord],
    pdf_path: Path,
    doc_out: Path,
    ext_cfg: ExtractionConfig,
) -> QualityResult:
    """Derive every flag; store crops for rejected tables; keep the
    audit trail. Units/tables are mutated in place and returned."""
    vlm_by_page = {r.page: r for r in vlm_pages}
    items: list[ReviewItem] = []

    for u in units:
        reasons = _unit_reasons(u, vlm_by_page)
        if reasons:
            u.needs_review = True
            items.append(
                ReviewItem(
                    page=u.page,
                    kind=u.type.value,
                    reason="; ".join(reasons),
                    excerpt=u.content[:120],
                )
            )

    flagged_tables = [(t, r) for t in tables if (r := _table_reason(t)) is not None]
    if flagged_tables:
        doc = pymupdf.open(pdf_path)
        try:
            for t, reason in flagged_tables:
                t.needs_review = True
                t.review_reason = reason
                # Rejected structure must not ship as confident renderings.
                t.markdown = ""
                t.grid = ""
                # The fallback tier is a human: store the exact region.
                key = f"figures/{t.table_id}_review.png"
                first_page, first_bbox = t.page_spans[0]
                pix = doc.load_page(first_page).get_pixmap(
                    clip=pymupdf.Rect(first_bbox), dpi=ext_cfg.figure_dpi
                )
                (doc_out / key).parent.mkdir(parents=True, exist_ok=True)
                (doc_out / key).write_bytes(pix.tobytes("png"))
                t.crop_key = key
                items.append(
                    ReviewItem(
                        page=first_page,
                        kind="table",
                        reason=reason,
                        excerpt=f"table {t.table_id}",
                        crop_key=key,
                    )
                )
        finally:
            doc.close()

    log.info(
        "quality gate: %d flagged item(s) (%d unit(s), %d table(s))",
        len(items),
        sum(1 for u in units if u.needs_review),
        len(flagged_tables),
    )
    return QualityResult(units=units, tables=tables, items=items)


def render_review_report(doc_id: str, result: QualityResult) -> str:
    """review_report.md: everything a reviewer needs, ordered by page,
    with reasons and crop links — never a bare boolean."""
    lines = [f"# Review report — {doc_id}", ""]
    if not result.items:
        lines.append("Nothing flagged. Every extracted item passed its validator.")
        return "\n".join(lines) + "\n"
    lines.append(f"{len(result.items)} item(s) need human eyes, grouped by page.")
    by_page: dict[int, list[ReviewItem]] = {}
    for item in result.items:
        by_page.setdefault(item.page, []).append(item)
    for page in sorted(by_page):
        lines.append(f"\n## Page {page + 1}")  # 1-based for humans, like citations
        for item in by_page[page]:
            lines.append(f"\n- **{item.kind}** — {item.reason}")
            if item.excerpt:
                lines.append(f"  - excerpt: `{item.excerpt}`")
            if item.crop_key:
                lines.append(f"  - crop: ![{item.excerpt}]({item.crop_key})")
    return "\n".join(lines) + "\n"
