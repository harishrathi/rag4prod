"""Generate the complex-tables sample: merged cells and other table shapes
the base sample never exercises.

``sample_pdf.py`` proves the pipeline's happy paths; this file is the
torture test. Every page is a table scenario found in real schedules of
rates, and the *known-correct* cell matrix for each is recorded here
(``GT_*`` constants) so extraction quality is an assertion, not an
eyeball. Current behavior against these pages is documented in
docs/edge_cases.md (complex-tables findings).

    python -m rag_ingest.complex_pdf      # writes sample_data/complex_doc.pdf

Page map (0-based), all tables fully ruled unless stated:

    0  native, two-tier header: "Rate Breakdown" col-spans 2 sub-columns,
       "Item"/"Description" row-span both header rows
    1  native, vertical merges: category column with row-spans (3 rows / 2)
    2  native, combined: row-span Remarks cell + col-span Subtotal row +
       a data row whose description col-spans 3 columns
    3  SCANNED version of page 0 (two-tier header as pixels -> tier 2)
    4  SCANNED version of page 1 (row-span category as pixels -> tier 2)
    5  native BORDERLESS table (no rules at all -> yolo_only review path)
    6  native, table into the bottom margin, category row-span running
    7  ... INTO the page break; continuation repeats the header and leaves
       the category cell blank, exactly as print does
    8  SCANNED table page with page.set_rotation(90) — a landscape scan;
       exercises OCR behavior on rotated input
    9  native, 31-row table with two 15-row category spans — exercises
       TABLE_ROWS_PER_CHUNK row-grouping against merged labels
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

PAGE_W, PAGE_H = 595, 842  # A4 in points


def draw_merged_table(
    page: pymupdf.Page,
    x0: float,
    y0: float,
    col_ws: list[float],
    row_hs: list[float],
    cells: list[tuple[int, int, int, int, str]],
    fontsize: float = 9,
) -> None:
    """cells: (row, col, rowspan, colspan, text). Borders are drawn per
    LOGICAL cell, so merged cells have no interior rules — like print.
    Text is vertically centered in its (possibly merged) cell, which is
    also how real documents typeset merged labels."""
    xs = [x0]
    for w in col_ws:
        xs.append(xs[-1] + w)
    ys = [y0]
    for h in row_hs:
        ys.append(ys[-1] + h)
    shape = page.new_shape()
    for r, c, rs, cs, _ in cells:
        shape.draw_rect(pymupdf.Rect(xs[c], ys[r], xs[c + cs], ys[r + rs]))
    shape.finish(color=(0, 0, 0), width=0.7)
    shape.commit()
    for r, c, rs, _cs, text in cells:
        cy = (ys[r] + ys[r + rs]) / 2 + fontsize / 2 - 1
        page.insert_text((xs[c] + 5, cy), text, fontsize=fontsize, fontname="helv")


# ---------------------------------------------------------------------------
# Table specs + ground truth
# ---------------------------------------------------------------------------
# GT matrices are the UNMERGED logical content: a merged value repeated
# across every cell it covers. That is the target a merge-aware extractor
# should produce; the print convention (value once, blanks after) is what
# the current extractor produces and is derivable from these.

TWO_TIER = dict(
    col_ws=[50, 170, 100, 100],
    row_hs=[26] * 7,
    cells=[
        (0, 0, 2, 1, "Item"),
        (0, 1, 2, 1, "Description"),
        (0, 2, 1, 2, "Rate Breakdown"),
        (1, 2, 1, 1, "Material"),
        (1, 3, 1, 1, "Labour"),
        (2, 0, 1, 1, "1"), (2, 1, 1, 1, "Excavation"), (2, 2, 1, 1, "80.00"), (2, 3, 1, 1, "40.00"),
        (3, 0, 1, 1, "2"), (3, 1, 1, 1, "Concrete M25"), (3, 2, 1, 1, "4100.00"), (3, 3, 1, 1, "1300.00"),
        (4, 0, 1, 1, "3"), (4, 1, 1, 1, "Steel reinforcement"), (4, 2, 1, 1, "48.00"), (4, 3, 1, 1, "14.50"),
        (5, 0, 1, 1, "4"), (5, 1, 1, 1, "Formwork"), (5, 2, 1, 1, "210.00"), (5, 3, 1, 1, "100.00"),
        (6, 0, 1, 1, "5"), (6, 1, 1, 1, "Brickwork"), (6, 2, 1, 1, "60.00"), (6, 3, 1, 1, "35.00"),
    ],
)

GT_TWO_TIER = [
    ["Item", "Description", "Rate Breakdown", "Rate Breakdown"],
    ["Item", "Description", "Material", "Labour"],
    ["1", "Excavation", "80.00", "40.00"],
    ["2", "Concrete M25", "4100.00", "1300.00"],
    ["3", "Steel reinforcement", "48.00", "14.50"],
    ["4", "Formwork", "210.00", "100.00"],
    ["5", "Brickwork", "60.00", "35.00"],
]

ROWSPAN = dict(
    col_ws=[110, 190, 90],
    row_hs=[26] * 6,
    cells=[
        (0, 0, 1, 1, "Category"), (0, 1, 1, 1, "Work Item"), (0, 2, 1, 1, "Rate"),
        (1, 0, 3, 1, "Civil Works"),
        (1, 1, 1, 1, "Site clearance"), (1, 2, 1, 1, "75.00"),
        (2, 1, 1, 1, "Dewatering"), (2, 2, 1, 1, "150.00"),
        (3, 1, 1, 1, "Backfilling"), (3, 2, 1, 1, "88.00"),
        (4, 0, 2, 1, "Electrical"),
        (4, 1, 1, 1, "Cable trenching"), (4, 2, 1, 1, "62.00"),
        (5, 1, 1, 1, "Earthing grid"), (5, 2, 1, 1, "134.00"),
    ],
)

GT_ROWSPAN = [
    ["Category", "Work Item", "Rate"],
    ["Civil Works", "Site clearance", "75.00"],
    ["Civil Works", "Dewatering", "150.00"],
    ["Civil Works", "Backfilling", "88.00"],
    ["Electrical", "Cable trenching", "62.00"],
    ["Electrical", "Earthing grid", "134.00"],
]

COMBINED = dict(
    col_ws=[45, 165, 85, 140],
    row_hs=[26] * 5,
    cells=[
        (0, 0, 1, 1, "Item"), (0, 1, 1, 1, "Description"), (0, 2, 1, 1, "Rate"), (0, 3, 1, 1, "Remarks"),
        (1, 0, 1, 1, "1"), (1, 1, 1, 1, "Excavation"), (1, 2, 1, 1, "120.00"),
        (1, 3, 2, 1, "Rates incl. haulage to 5 km"),
        (2, 0, 1, 1, "2"), (2, 1, 1, 1, "Concrete M25"), (2, 2, 1, 1, "5400.00"),
        (3, 0, 1, 2, "Subtotal - earthworks"), (3, 2, 1, 1, "5520.00"), (3, 3, 1, 1, ""),
        (4, 0, 1, 1, "3"), (4, 1, 1, 3, "Provisional sum - contingency (all trades) 2000.00"),
    ],
)

BORDERLESS_ROWS = [
    ("Item", "Description", "Amount"),
    ("1", "Mobilisation", "12,000.00"),
    ("2", "Insurance and bonds", "8,500.00"),
    ("3", "Site establishment", "22,300.00"),
]

# Pages 6-7 after stitching: repeated header dropped, the Finishes span
# that crosses the page break filled onto the continuation rows.
GT_CONTINUATION = [
    ["Category", "Work Item", "Rate"],
    ["Civil Works", "Formwork", "310.00"],
    ["Civil Works", "Brickwork", "95.00"],
    ["Civil Works", "Plastering", "48.00"],
    ["Civil Works", "Flooring", "260.00"],
    ["Finishes", "Painting", "35.00"],
    ["Finishes", "Waterproofing", "410.00"],
    ["Finishes", "Roofing", "520.00"],
    ["Finishes", "Glazing", "180.00"],
    ["Finishes", "Joinery", "225.00"],
]

# Page 9 (31-row schedule): every data row carries its category.
GT_LONG = (
    [["Category", "Work Item", "Rate"]]
    + [["Civil Works", f"Civil item {i + 1}", f"{(i + 1) * 10}.00"] for i in range(15)]
    + [["Electrical", f"Electrical item {i + 1}", f"{(i + 1) * 7}.00"] for i in range(15)]
)


def _prose(page: pymupdf.Page, heading: str, y: float = 72) -> None:
    page.insert_text((72, y), heading, fontsize=14, fontname="hebo")
    page.insert_textbox(
        pymupdf.Rect(72, y + 20, PAGE_W - 72, y + 90),
        "The rates below apply as set out in the conditions of contract. "
        "Merged cells follow standard schedule-of-rates print conventions.",
        fontsize=10, fontname="helv",
    )


def _native_page(doc: pymupdf.Document, heading: str, spec: dict) -> None:
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    _prose(page, heading)
    draw_merged_table(page, 72, 200, **spec)


def _scanned_page(doc: pymupdf.Document, heading: str, spec: dict, rotate: int = 0) -> None:
    """Render the same content to pixels (200 DPI, per ledger #16's floor)
    and place it as a full-page image: no text layer, no vector lines."""
    tmp = pymupdf.open()
    p = tmp.new_page(width=PAGE_W, height=PAGE_H)
    _prose(p, heading)
    draw_merged_table(p, 72, 200, **{**spec, "fontsize": 10})
    png = p.get_pixmap(dpi=200).tobytes("png")
    tmp.close()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(pymupdf.Rect(0, 0, PAGE_W, PAGE_H), stream=png)
    if rotate:
        page.set_rotation(rotate)


def _borderless_page(doc: pymupdf.Document) -> None:
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    _prose(page, "8. Borderless Summary")
    for i, (a, b, c) in enumerate(BORDERLESS_ROWS):
        y = 220 + i * 24
        font = "hebo" if i == 0 else "helv"
        page.insert_text((72, y), a, fontsize=10, fontname=font)
        page.insert_text((140, y), b, fontsize=10, fontname=font)
        page.insert_text((400, y), c, fontsize=10, fontname=font)


def _continuation_pages(doc: pymupdf.Document) -> None:
    p = doc.new_page(width=PAGE_W, height=PAGE_H)
    _prose(p, "9. Rate Schedule by Category (full)")
    p.insert_textbox(
        pymupdf.Rect(72, 170, PAGE_W - 72, 560),
        "This schedule groups items by trade. " * 12,
        fontsize=10, fontname="helv",
    )
    # bottom edge 600 + 7*28 = 796 > 842*0.90 — continuation signal #1
    draw_merged_table(
        p, 72, 600,
        col_ws=[110, 190, 90], row_hs=[28] * 7,
        cells=[
            (0, 0, 1, 1, "Category"), (0, 1, 1, 1, "Work Item"), (0, 2, 1, 1, "Rate"),
            (1, 0, 4, 1, "Civil Works"),
            (1, 1, 1, 1, "Formwork"), (1, 2, 1, 1, "310.00"),
            (2, 1, 1, 1, "Brickwork"), (2, 2, 1, 1, "95.00"),
            (3, 1, 1, 1, "Plastering"), (3, 2, 1, 1, "48.00"),
            (4, 1, 1, 1, "Flooring"), (4, 2, 1, 1, "260.00"),
            (5, 0, 2, 1, "Finishes"),
            (5, 1, 1, 1, "Painting"), (5, 2, 1, 1, "35.00"),
            (6, 1, 1, 1, "Waterproofing"), (6, 2, 1, 1, "410.00"),
        ],
    )
    # continuation starts at y=72 < 842*0.12 — signal #2; header repeats;
    # the category cell is BLANK, as print renders a span crossing pages
    p = doc.new_page(width=PAGE_W, height=PAGE_H)
    draw_merged_table(
        p, 72, 72,
        col_ws=[110, 190, 90], row_hs=[28] * 4,
        cells=[
            (0, 0, 1, 1, "Category"), (0, 1, 1, 1, "Work Item"), (0, 2, 1, 1, "Rate"),
            (1, 0, 3, 1, ""),
            (1, 1, 1, 1, "Roofing"), (1, 2, 1, 1, "520.00"),
            (2, 1, 1, 1, "Glazing"), (2, 2, 1, 1, "180.00"),
            (3, 1, 1, 1, "Joinery"), (3, 2, 1, 1, "225.00"),
        ],
    )
    p.insert_textbox(
        pymupdf.Rect(72, 220, PAGE_W - 72, 340),
        "End of schedule. All rates are exclusive of applicable taxes.",
        fontsize=10, fontname="helv",
    )


def _long_rowspan_page(doc: pymupdf.Document) -> None:
    """31 rows, two 15-row category spans: rows_per_chunk grouping must
    not strand rows from their merged category label."""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text((72, 50), "11. Long Category Schedule", fontsize=14, fontname="hebo")
    cells = [(0, 0, 1, 1, "Category"), (0, 1, 1, 1, "Work Item"), (0, 2, 1, 1, "Rate")]
    cells.append((1, 0, 15, 1, "Civil Works"))
    for i in range(15):
        cells += [(1 + i, 1, 1, 1, f"Civil item {i + 1}"), (1 + i, 2, 1, 1, f"{(i + 1) * 10}.00")]
    cells.append((16, 0, 15, 1, "Electrical"))
    for i in range(15):
        cells += [(16 + i, 1, 1, 1, f"Electrical item {i + 1}"), (16 + i, 2, 1, 1, f"{(i + 1) * 7}.00")]
    draw_merged_table(page, 72, 70, col_ws=[110, 190, 90], row_hs=[24] * 31, cells=cells, fontsize=8)


def build_complex(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    _native_page(doc, "1. Two-Tier Header Schedule", TWO_TIER)              # p0
    _native_page(doc, "2. Category Schedule (row spans)", ROWSPAN)          # p1
    _native_page(doc, "3. Combined Merges", COMBINED)                       # p2
    _scanned_page(doc, "4. Two-Tier Header (scanned annex)", TWO_TIER)      # p3
    _scanned_page(doc, "5. Category Schedule (scanned annex)", ROWSPAN)     # p4
    _borderless_page(doc)                                                   # p5
    _continuation_pages(doc)                                                # p6, p7
    _scanned_page(doc, "10. Rotated Scanned Schedule", ROWSPAN, rotate=90)  # p8
    _long_rowspan_page(doc)                                                 # p9
    doc.save(path)
    doc.close()
    return path


if __name__ == "__main__":
    out = build_complex(Path("sample_data") / "complex_doc.pdf")
    print(f"wrote {out}")
