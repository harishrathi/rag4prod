"""STAGE 7b — Chunking: the walk stream -> retrieval-ready chunks + merged.md.

Chunking is PER HEADING SECTION (design spec §8.4): text accumulates
under the current heading stack and is chunked when the section closes.
Every chunk therefore inherits its section's breadcrumb and page range
directly — no offset arithmetic mapping chunker output back to sources,
which is the classic way citations drift.

The two-text contract (models.Chunk): ``content`` is what a human sees
when the chunk is cited; ``embedding_text`` is what the vector index
sees — the breadcrumb prefix ("[7. Payment Terms > 7.3 Liquidated
Damages]") is prepended there only, and is what makes a bare value like
"0.5% per week" retrievable for "what is the LD rate?".

Tables stay logically atomic but big ones are emitted as ROW GROUPS —
each group repeats the header row and shares the table's table_id, so a
retrieval hit lands on rows with their column meanings intact and a
consumer can reassemble the whole table.

The splitter is deliberately hand-rolled (~20 lines: paragraphs first,
sentences second, never mid-sentence): this repo exists to show the
internals. Production alternative: a chunking library (e.g. Chonkie's
recursive markdown recipe) — swap it inside split_text() and nothing
else changes. Token counts are estimated at ~4 chars/token; production
would use the embedding model's real tokenizer.

Page numbers: chunks carry 1-BASED pages — this module is the single
place the internal 0-based convention converts (models.py docstring).
"""

from __future__ import annotations

import logging
import re

from .assemble import WalkItem
from .config import CHUNK_MAX_CHARS, TABLE_ROWS_PER_CHUNK
from .models import Chunk, Source, UnitType
from .tables import cells_to_markdown

log = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Greedy split on paragraph boundaries, falling back to sentences.
    Never splits mid-sentence; a single sentence longer than max_chars is
    emitted whole (truncating it would corrupt meaning)."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    pieces: list[str] = []
    buf = ""
    for para in text.split("\n\n"):
        parts = [para] if len(para) <= max_chars else _SENTENCE_END.split(para)
        for part in parts:
            if buf and len(buf) + len(part) + 1 > max_chars:
                pieces.append(buf.strip())
                buf = ""
            buf = f"{buf} {part}".strip() if buf else part
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class _ChunkBuilder:
    """Walk consumer: maintains the heading stack, accumulates section
    text, and emits chunks + merged.md lines in one pass."""

    def __init__(self, doc_id: str, rows_per_chunk: int = TABLE_ROWS_PER_CHUNK) -> None:
        self.doc_id = doc_id
        self.rows_per_chunk = rows_per_chunk
        self.chunks: list[Chunk] = []
        self.md_lines: list[str] = []
        self.stack: list[tuple[int, str]] = []  # (level, heading text)
        self._texts: list[str] = []
        self._pages: set[int] = set()
        self._source: Source = Source.PYMUPDF

    # -- helpers ------------------------------------------------------------

    def _breadcrumb(self) -> str:
        return "[" + " > ".join(h for _, h in self.stack) + "]" if self.stack else ""

    def _next_id(self) -> str:
        return f"{self.doc_id}_c{len(self.chunks):04d}"

    def _emit(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)

    def _flush_section_text(self) -> None:
        if not self._texts:
            return
        text = "\n\n".join(self._texts)
        pages = sorted(p + 1 for p in self._pages)  # THE 0->1-based conversion
        crumb = self._breadcrumb()
        for piece in split_text(text):
            embedding = f"{crumb}\n\n{piece}" if crumb else piece
            self._emit(
                Chunk(
                    chunk_id=self._next_id(),
                    type=UnitType.TEXT,
                    content=piece,
                    embedding_text=embedding,
                    headings=[h for _, h in self.stack],
                    pages=pages,
                    source=self._source,
                    token_count=_estimate_tokens(embedding),
                )
            )
        self._texts, self._pages = [], set()

    # -- walk items ---------------------------------------------------------

    def on_title(self, item: WalkItem) -> None:
        self._flush_section_text()
        level = item.level or 1
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        assert item.unit is not None
        self.stack.append((level, item.unit.content))
        self.md_lines.append(f"\n{'#' * level} {item.unit.content}\n")

    def on_text(self, item: WalkItem) -> None:
        assert item.unit is not None
        self._texts.append(item.unit.content)
        self._pages.add(item.unit.page)
        self._source = item.unit.source
        self.md_lines.append(item.unit.content + "\n")

    def on_figure(self, item: WalkItem) -> None:
        unit = item.unit
        assert unit is not None
        crumb = self._breadcrumb()
        page_1b = unit.page + 1
        # No caption available (the vision tier was dropped): breadcrumb +
        # location is all the embedding has — an accepted retrievability
        # gap, ledger #18.
        embedding = (
            f"{crumb}\n\n[figure on page {page_1b}]" if crumb else f"[figure on page {page_1b}]"
        )
        self._emit(
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
                token_count=_estimate_tokens(embedding),
            )
        )
        self.md_lines.append(f"\n![figure]({unit.storage_key})\n")

    def on_table(self, item: WalkItem) -> None:
        # Section text seen so far belongs BEFORE the table in reading
        # order; flush it so ordering survives into the chunk list.
        self._flush_section_text()
        t = item.table
        assert t is not None
        crumb = self._breadcrumb()
        pages = [p + 1 for p in t.pages]

        if t.needs_review or not t.cells:
            # Unextracted table: emit a review chunk pointing at the crop.
            embedding = f"{crumb}\n\n[table needing review on page {pages[0]}]"
            self._emit(
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

        header, data = t.cells[0], t.cells[1:]
        groups = [
            data[i : i + self.rows_per_chunk] for i in range(0, len(data), self.rows_per_chunk)
        ]
        for group in groups:
            md = cells_to_markdown([header, *group])
            embedding = f"{crumb}\n\n{md}" if crumb else md
            self._emit(
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
        self.md_lines.append("\n" + t.markdown + "\n")


def chunk_document(doc_id: str, walk: list[WalkItem], rows_per_chunk: int = TABLE_ROWS_PER_CHUNK):
    """Consume the ordered walk; return (chunks, merged_markdown)."""
    builder = _ChunkBuilder(doc_id, rows_per_chunk)
    handlers = {
        "title": builder.on_title,
        "text": builder.on_text,
        "figure": builder.on_figure,
        "table": builder.on_table,
    }
    for item in walk:
        handlers[item.kind](item)
    builder._flush_section_text()

    counts: dict[str, int] = {}
    for c in builder.chunks:
        counts[c.type.value] = counts.get(c.type.value, 0) + 1
    log.info("chunking: %d chunk(s) -> %s", len(builder.chunks), counts)
    return builder.chunks, "".join(builder.md_lines)
