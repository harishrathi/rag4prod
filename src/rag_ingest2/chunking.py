"""LAYER 7 — Chunking: the normalized pool -> retrieval-ready chunks +
merged.md. Unchanged from v1 behind the ``split_text()`` seam (#24); the
walk is just a (page, y0) sort now, because furniture, dedup, and
heading levels were all resolved in Layer 5 — this layer only consumes.

The two-text contract, row-group tables with repeated headers, and the
single 0->1-based page conversion are all v1 behavior verbatim (see the
v1 module docstring for the full rationale)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chonkie import SentenceChunker

from rag_ingest.models import Chunk, Source, Unit, UnitType

from .config import ChunkConfig
from .tables import TableResult
from .tables.validate import cells_to_markdown

log = logging.getLogger(__name__)

_chunker: SentenceChunker | None = None
_chunker_key: tuple | None = None


def _get_chunker(cfg: ChunkConfig) -> SentenceChunker:
    """Singleton per config: the tokenizer load is the expensive part.
    The constructor kwarg is `tokenizer` in chonkie 1.7 — renamed across
    releases, which is why the version is pinned."""
    global _chunker, _chunker_key
    key = (cfg.tokenizer, cfg.size_tokens, cfg.overlap_tokens)
    if _chunker is None or _chunker_key != key:
        _chunker = SentenceChunker(
            tokenizer=cfg.tokenizer,
            chunk_size=cfg.size_tokens,
            chunk_overlap=cfg.overlap_tokens,
        )
        _chunker_key = key
    return _chunker


def split_text(text: str, cfg: ChunkConfig) -> list[tuple[str, int]]:
    """Section prose -> (piece, token_count) pairs — the chunking-library
    seam; everything else is agnostic to what implements it."""
    if not text.strip():
        return []
    return [(c.text.strip(), c.token_count) for c in _get_chunker(cfg).chunk(text)]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class WalkItem:
    """One element of the document stream, in reading order."""

    kind: str  # "title" | "text" | "figure" | "table"
    page: int
    y0: float
    unit: Unit | None = None
    table: TableResult | None = None


def build_walk(units: list[Unit], tables: list[TableResult]) -> list[WalkItem]:
    """Normalized pool -> one (page, y0)-ordered stream. Sort is stable,
    so paid-lane units (which share a full-page y0) keep their emission
    order — which IS their reading order."""
    items: list[WalkItem] = []
    for u in units:
        kind = {UnitType.TITLE: "title", UnitType.TEXT: "text", UnitType.FIGURE: "figure"}.get(
            u.type
        )
        if kind is None:
            continue  # TABLE units became fragments in Layer 4
        items.append(WalkItem(kind=kind, page=u.page, y0=u.bbox[1], unit=u))
    for t in tables:
        first_page, first_bbox = t.page_spans[0]
        items.append(WalkItem(kind="table", page=first_page, y0=first_bbox[1], table=t))
    items.sort(key=lambda w: (w.page, w.y0))
    return items


class _ChunkBuilder:
    """Walk consumer: heading stack, section text accumulation, chunks +
    merged.md in one pass (v1 logic, config threaded through)."""

    def __init__(self, doc_id: str, cfg: ChunkConfig) -> None:
        self.doc_id = doc_id
        self.cfg = cfg
        self.chunks: list[Chunk] = []
        self.md_lines: list[str] = []
        self.stack: list[tuple[int, str]] = []
        self._texts: list[str] = []
        self._pages: set[int] = set()
        self._source: Source = Source.PYMUPDF
        self._review = False

    def _breadcrumb(self) -> str:
        return "[" + " > ".join(h for _, h in self.stack) + "]" if self.stack else ""

    def _next_id(self) -> str:
        return f"{self.doc_id}_c{len(self.chunks):04d}"

    def _flush_section_text(self) -> None:
        if not self._texts:
            return
        text = "\n\n".join(self._texts)
        pages = sorted(p + 1 for p in self._pages)  # THE 0->1-based conversion
        crumb = self._breadcrumb()
        for piece, piece_tokens in split_text(text, self.cfg):
            embedding = f"{crumb}\n\n{piece}" if crumb else piece
            self.chunks.append(
                Chunk(
                    chunk_id=self._next_id(),
                    type=UnitType.TEXT,
                    content=piece,
                    embedding_text=embedding,
                    headings=[h for _, h in self.stack],
                    pages=pages,
                    source=self._source,
                    # One flagged unit taints the section's chunks —
                    # better a spurious review than confident garbage.
                    needs_review=self._review,
                    token_count=piece_tokens + (_estimate_tokens(crumb) if crumb else 0),
                )
            )
        self._texts, self._pages, self._review = [], set(), False

    def on_title(self, item: WalkItem) -> None:
        self._flush_section_text()
        if item.unit is None:
            raise ValueError("title walk item without a unit")
        level = item.unit.level or 1
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, item.unit.content))
        self.md_lines.append(f"\n{'#' * level} {item.unit.content}\n")

    def on_text(self, item: WalkItem) -> None:
        if item.unit is None:
            raise ValueError("text walk item without a unit")
        self._texts.append(item.unit.content)
        self._pages.add(item.unit.page)
        self._source = item.unit.source
        self._review = self._review or item.unit.needs_review
        self.md_lines.append(item.unit.content + "\n")

    def on_figure(self, item: WalkItem) -> None:
        unit = item.unit
        if unit is None:
            raise ValueError("figure walk item without a unit")
        crumb = self._breadcrumb()
        page_1b = unit.page + 1
        embedding = (
            f"{crumb}\n\n[figure on page {page_1b}]" if crumb else f"[figure on page {page_1b}]"
        )
        self.chunks.append(
            Chunk(
                chunk_id=self._next_id(),
                type=UnitType.FIGURE,
                content="",
                embedding_text=embedding,
                headings=[h for _, h in self.stack],
                pages=[page_1b],
                bbox=unit.bbox,
                storage_key=unit.storage_key,
                source=unit.source,
                needs_review=unit.needs_review,
                token_count=_estimate_tokens(embedding),
            )
        )
        self.md_lines.append(f"\n![figure]({unit.storage_key})\n")

    def on_table(self, item: WalkItem) -> None:
        self._flush_section_text()
        t = item.table
        if t is None:
            raise ValueError("table walk item without a table")
        crumb = self._breadcrumb()
        pages = [p + 1 for p in t.pages]

        if t.needs_review or not t.cells:
            embedding = f"{crumb}\n\n[table needing review on page {pages[0]}]"
            self.chunks.append(
                Chunk(
                    chunk_id=self._next_id(),
                    type=UnitType.TABLE,
                    content="",
                    embedding_text=embedding,
                    headings=[h for _, h in self.stack],
                    pages=pages,
                    bbox=t.page_spans[0][1],
                    storage_key=t.crop_key,
                    needs_review=True,
                    table_id=t.table_id,
                    token_count=_estimate_tokens(embedding),
                )
            )
            self.md_lines.append(
                f"\n<!-- table {t.table_id}: NEEDS REVIEW ({t.review_reason}) -->\n"
            )
            return

        # ALL header rows repeat in every row group (#27).
        n_head = max(1, min(t.header_rows, len(t.cells) - 1))
        header, data = t.cells[:n_head], t.cells[n_head:]
        rows_per = self.cfg.table_rows_per_chunk
        for i in range(0, len(data), rows_per):
            md = cells_to_markdown([*header, *data[i : i + rows_per]])
            embedding = f"{crumb}\n\n{md}" if crumb else md
            self.chunks.append(
                Chunk(
                    chunk_id=self._next_id(),
                    type=UnitType.TABLE,
                    content=md,
                    embedding_text=embedding,
                    headings=[h for _, h in self.stack],
                    pages=pages,
                    bbox=t.page_spans[0][1],
                    table_id=t.table_id,
                    token_count=_estimate_tokens(embedding),
                )
            )
        if t.merges:
            self.md_lines.append("\n```text\n" + t.grid + "\n```\n")
        else:
            self.md_lines.append("\n" + t.markdown + "\n")


def chunk_document(
    doc_id: str, units: list[Unit], tables: list[TableResult], cfg: ChunkConfig
) -> tuple[list[Chunk], str]:
    """Consume the normalized pool; return (chunks, merged_markdown)."""
    builder = _ChunkBuilder(doc_id, cfg)
    handlers = {
        "title": builder.on_title,
        "text": builder.on_text,
        "figure": builder.on_figure,
        "table": builder.on_table,
    }
    for item in build_walk(units, tables):
        handlers[item.kind](item)
    builder._flush_section_text()

    counts: dict[str, int] = {}
    for c in builder.chunks:
        counts[c.type.value] = counts.get(c.type.value, 0) + 1
    log.info("chunking: %d chunk(s) -> %s", len(builder.chunks), counts)
    return builder.chunks, "".join(builder.md_lines)
