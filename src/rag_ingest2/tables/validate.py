"""Structural validation + the renderings of a cell matrix.

``validate_cells``/``junk_cell_count`` produce REASONS; turning a reason
into ``needs_review`` + a stored crop happens only in the quality gate
(theme B: trust is derived in one place). The renderings are shared by
the quality gate (markdown/grid onto TableResult) and chunking (row-group
markdown)."""

from __future__ import annotations

from rag_ingest.text_quality import JUNK_CHARS_RE, orphan_combining_marks


def validate_cells(cells: list[list[str]]) -> str | None:
    """Reason the table is NOT trustworthy, or None if it passes. Cheap
    structural checks — the gate between 'extracted' and 'reviewed'."""
    if len(cells) < 2:
        return "fewer than 2 rows (no grid found, or header-only)"
    widths = {len(r) for r in cells}
    if len(widths) != 1:
        return f"ragged rows: column counts {sorted(widths)}"
    if not any(c for c in cells[0]):
        return "empty header row"
    filled = sum(1 for r in cells for c in r if c)
    if filled / (len(cells) * len(cells[0])) < 0.4:
        return "mostly empty cells (grid without content?)"
    return None


def junk_cell_count(cells: list[list[str]]) -> int:
    """Cells carrying broken-CMap symptoms (junk chars or orphan marks) —
    native pages whose text layer is only mildly broken stay below the
    routing threshold but can still poison individual cells (#29)."""
    return sum(
        1
        for row in cells
        for c in row
        if JUNK_CHARS_RE.search(c) or orphan_combining_marks(c) > 0
    )


def cells_to_markdown(cells: list[list[str]]) -> str:
    esc = [[c.replace("|", "\\|") for c in row] for row in cells]
    lines = ["| " + " | ".join(esc[0]) + " |", "|" + "---|" * len(esc[0])]
    lines += ["| " + " | ".join(row) + " |" for row in esc[1:]]
    return "\n".join(lines)


def cells_to_grid(cells: list[list[str]], merges: list[list[int]]) -> str:
    """Visually faithful rendering: an ASCII box grid reconstructed from
    the unmerged matrix + merge list. A merged cell draws as ONE box —
    merged.md embeds this (fenced, monospace) whenever a table has spans,
    because pipe markdown cannot express them."""
    rows, cols = len(cells), len(cells[0]) if cells else 0
    if not rows or not cols:
        return ""
    anchor = {(m[0], m[1]): (m[2], m[3]) for m in merges}
    owner: dict[tuple[int, int], tuple[int, int]] = {}
    for r0, c0, rs, cs in merges:
        for r in range(r0, r0 + rs):
            for c in range(c0, c0 + cs):
                owner[(r, c)] = (r0, c0)

    def own(r: int, c: int) -> tuple[int, int]:
        return owner.get((r, c), (r, c))

    width = [3] * cols
    for r in range(rows):
        for c in range(cols):
            if own(r, c) == (r, c) and anchor.get((r, c), (1, 1))[1] == 1:
                width[c] = max(width[c], len(cells[r][c]))
    for (r0, c0), (_rs, cs) in anchor.items():
        if cs > 1:
            span = sum(width[c0 : c0 + cs]) + 3 * (cs - 1)
            if len(cells[r0][c0]) > span:
                width[c0 + cs - 1] += len(cells[r0][c0]) - span

    def v_border(r: int, b: int) -> bool:
        return b in (0, cols) or own(r, b - 1) != own(r, b)

    def h_border(line: int, c: int) -> bool:
        return line in (0, rows) or own(line - 1, c) != own(line, c)

    out: list[str] = []
    for line in range(rows + 1):
        s = ""
        for b in range(cols + 1):
            left = b > 0 and h_border(line, b - 1)
            right = b < cols and h_border(line, b)
            up = line > 0 and v_border(line - 1, b)
            down = line < rows and v_border(line, b)
            s += "+" if (left or right) else ("|" if (up or down) else " ")
            if b < cols:
                s += ("-" if h_border(line, b) else " ") * (width[b] + 2)
        out.append(s.rstrip())
        if line == rows:
            break
        s, c = "", 0
        while c < cols:
            s += "|" if v_border(line, c) else " "
            r0, c0 = own(line, c)
            cs = anchor.get((r0, c0), (1, 1))[1]
            span = sum(width[c : c + cs]) + 3 * (cs - 1)
            text = cells[line][c] if (r0, c0) == (line, c) else ""
            s += " " + text.ljust(span) + " "
            c += cs
        out.append(s + "|")
    return "\n".join(out)
