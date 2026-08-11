"""Generate a synthetic multi-layout PDF that exercises every triage branch.

Real large documents can't be committed to a public repo, so this builds a
small stand-in where the *correct* classification of every page is known in
advance (``EXPECTED_KINDS``) — which turns triage testing from eyeballing
into an assertion (see tests/test_triage.py).

    python -m rag_ingest.sample_pdf       # writes sample_data/sample_doc.pdf

Page map (0-based) and the triage branch each page exists to exercise:

    0  prose + headings + figure      -> TEXT_NATIVE  (easy case; embedded-figure
                                          extraction, Phase 2)
    1  prose + numbered headings       -> TEXT_NATIVE  (heading detection incl. a
                                          body-size bold numbered heading, Phase 2)
    2  raster image of printed text    -> SCANNED      (classic scan; OCR reads
                                          its body in Phase 4)
    3  same raster + header text layer -> SCANNED      (the header-over-scan TRAP:
                                          text length alone would say TEXT_NATIVE)
    4  dense vector line-work, no text -> DRAWING      (CAD-plan stand-in)
    5  title page, ~15 chars of text   -> SCANNED      (accepted misroute: costs
                                          one vision-API call, output still fine)
    6  ruled table + prose             -> TEXT_NATIVE  (table tier 1, find_tables)
    7  ruled table as pixels, no text  -> SCANNED      (table tier 2: image grid
                                          detection + per-cell OCR)
    8  table into the bottom margin    -> TEXT_NATIVE  (multi-page stitching:
    9  continuation + repeated header  -> TEXT_NATIVE   detect, merge, drop header)
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from .models import PageKind

PAGE_W, PAGE_H = 595, 842  # A4 in points

# Ground truth for tests: what a correct triage must say about each page.
EXPECTED_KINDS: dict[int, PageKind] = {
    0: PageKind.TEXT_NATIVE,
    1: PageKind.TEXT_NATIVE,
    2: PageKind.SCANNED,
    3: PageKind.SCANNED,
    4: PageKind.DRAWING,
    5: PageKind.SCANNED,
    6: PageKind.TEXT_NATIVE,
    7: PageKind.SCANNED,  # ruled table as pixels only (tier-2 case)
    8: PageKind.TEXT_NATIVE,  # table running into the bottom margin ...
    9: PageKind.TEXT_NATIVE,  # ... continuing at the top, repeated header
}

# Contract-clause-flavoured filler: gives later phases realistic prose with
# numbered clauses and cross-references to test heading detection and
# retrieval against, without being tied to any specific document domain.
LOREM = (
    "The supplier shall complete all services described in this section in "
    "accordance with the specifications and within the time stated in the "
    "agreement. Any delay attributable to the supplier shall attract "
    "liquidated damages as set out in clause 7.3 of these conditions. "
)


def _prose_page(doc: pymupdf.Document, heading: str, numbered: bool) -> None:
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = 72.0
    page.insert_text((72, y), heading, fontsize=18, fontname="hebo")
    y += 40
    sub = "7.3 Liquidated Damages" if numbered else "Background"
    page.insert_text((72, y), sub, fontsize=13, fontname="hebo")
    y += 24
    page.insert_textbox(
        pymupdf.Rect(72, y, PAGE_W - 72, y + 320), LOREM * 6, fontsize=10, fontname="helv"
    )
    if numbered:
        # Heading at BODY size: only detectable via bold + clause numbering,
        # exercising the second branch of the stage-2 heading rule.
        page.insert_text((72, y + 340), "7.3.1 Delay Notices", fontsize=10, fontname="hebo")
        page.insert_textbox(
            pymupdf.Rect(72, y + 350, PAGE_W - 72, y + 470), LOREM * 2, fontsize=10, fontname="helv"
        )
    else:
        # Embedded raster figure on a text-native page (~6% of page area:
        # well above FIGURE_MIN_AREA_FRAC, far below SCAN_IMAGE_COVERAGE).
        page.insert_image(pymupdf.Rect(72, y + 340, 292, y + 480), stream=_gray_png(220, 140, 180))


def _gray_png(w: int, h: int, value: int) -> bytes:
    """A flat gray PNG made with PyMuPDF alone (no Pillow dependency yet)."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h))
    pix.clear_with(value)
    return pix.tobytes("png")


def _printed_body_png(include_heading: bool = True) -> bytes:
    """A raster image OF printed text — what a real scanner produces.
    Rendered from a throwaway text page so OCR has genuine glyphs to
    read; the resulting page has an image, but NO text layer.

    Rendered at 200 DPI / 12pt on purpose: a first attempt at 150 DPI and
    11pt made Tesseract glue words together — inter-word gaps fell below
    the space-synthesis threshold. Same rule as real scanners: OCR quality
    is capped by the scan's native resolution, and OCR_DPI upsampling
    cannot restore detail the scan never captured (ledger #16)."""
    tmp = pymupdf.open()
    p = tmp.new_page(width=PAGE_W, height=PAGE_H)
    if include_heading:
        p.insert_text((72, 90), "4. Delivery Conditions", fontsize=16, fontname="hebo")
    p.insert_textbox(
        pymupdf.Rect(72, 120, PAGE_W - 72, 500), LOREM * 3, fontsize=12, fontname="helv"
    )
    png = p.get_pixmap(dpi=200).tobytes("png")
    tmp.close()
    return png


def _scan_page(doc: pymupdf.Document, header: str | None, include_heading: bool = True) -> None:
    """Full-page raster image of printed text, optionally with a
    text-layer header on top — the header variant is the trap
    SCAN_IMAGE_COVERAGE exists to catch. include_heading=False makes the
    page a CONTINUATION of the previous scan (body only), so consecutive
    scanned pages don't read as accidental duplicates in merged.md."""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(pymupdf.Rect(0, 0, PAGE_W, PAGE_H), stream=_printed_body_png(include_heading))
    if header:
        page.insert_text((72, 40), header, fontsize=9, fontname="helv")


def _drawing_page(doc: pymupdf.Document) -> None:
    """Vector line-work standing in for a CAD plan: no text layer, no
    raster image, 120 drawing segments."""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    shape = page.new_shape()
    for i in range(60):
        x = 72 + i * 7.5
        shape.draw_line(pymupdf.Point(x, 100), pymupdf.Point(x, 700))
        y = 100 + i * 10
        shape.draw_line(pymupdf.Point(72, y), pymupdf.Point(520, y))
    shape.finish(color=(0.2, 0.2, 0.2), width=0.5)
    shape.commit()


def _title_page(doc: pymupdf.Document) -> None:
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text((200, 400), "DOCUMENT No. 42", fontsize=24, fontname="hebo")


HEADER_ROW = ["Item", "Description", "Rate"]
PRICE_ROWS = [
    ["1", "Excavation", "120.00"],
    ["2", "Concrete M25", "5400.00"],
    ["3", "Steel reinforcement", "62.50"],
]
# Distinct rows for the scanned annex table (page 7) so merged.md doesn't
# read as an accidental duplicate of the page-6 table. Ground truth for
# the tier-2 (grid + OCR) tests.
ANNEX_ROWS = [
    ["A1", "Site clearance", "75.00"],
    ["A2", "Dewatering", "150.00"],
    ["A3", "Backfilling", "88.00"],
]
# Rows for the multi-page table (pages 8-9): header + 6 rows on page 8,
# repeated header + 4 rows on page 9. Ground truth for stitching tests.
CONT_ROWS_P8 = [
    ["4", "Formwork", "310.00"],
    ["5", "Brickwork", "95.00"],
    ["6", "Plastering", "48.00"],
    ["7", "Flooring", "260.00"],
    ["8", "Painting", "35.00"],
    ["9", "Waterproofing", "410.00"],
]
CONT_ROWS_P9 = [
    ["10", "Roofing", "520.00"],
    ["11", "Glazing", "180.00"],
    ["12", "Joinery", "225.00"],
    ["13", "Drainage", "140.00"],
]


def _draw_ruled_table(
    page: pymupdf.Page,
    cells: list[list[str]],
    y0: float,
    x0: float = 72.0,
    col_w: float = 150.0,
    row_h: float = 28.0,
    fontsize: float = 9,
) -> None:
    """Draw a fully ruled table with text in each cell."""
    rows, cols = len(cells), len(cells[0])
    shape = page.new_shape()
    for r in range(rows + 1):
        shape.draw_line(
            pymupdf.Point(x0, y0 + r * row_h), pymupdf.Point(x0 + cols * col_w, y0 + r * row_h)
        )
    for c in range(cols + 1):
        shape.draw_line(
            pymupdf.Point(x0 + c * col_w, y0), pymupdf.Point(x0 + c * col_w, y0 + rows * row_h)
        )
    shape.finish(color=(0, 0, 0), width=0.7)
    shape.commit()
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            page.insert_text(
                (x0 + c * col_w + 6, y0 + r * row_h + 18), cell, fontsize=fontsize, fontname="helv"
            )


def _table_page(doc: pymupdf.Document) -> None:
    """Prose above a ruled 4x3 table — text-native, with real grid lines."""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text((72, 72), "5. Price Schedule (extract)", fontsize=14, fontname="hebo")
    # NB: insert_textbox renders NOTHING if the text overflows the rect —
    # it refuses rather than truncates. One LOREM fits; two silently
    # vanished and took a test premise with them.
    page.insert_textbox(
        pymupdf.Rect(72, 100, PAGE_W - 72, 180), LOREM, fontsize=10, fontname="helv"
    )
    _draw_ruled_table(page, [HEADER_ROW, *PRICE_ROWS], y0=220.0)


def _scanned_table_page(doc: pymupdf.Document) -> None:
    """A ruled table that exists only as pixels — the tier-2 case: no
    vector lines for find_tables, no text layer; the grid must be found
    in the image and the cells read by OCR."""
    tmp = pymupdf.open()
    p = tmp.new_page(width=PAGE_W, height=PAGE_H)
    p.insert_text((72, 90), "5.2 Price Schedule (scanned annex)", fontsize=15, fontname="hebo")
    _draw_ruled_table(p, [HEADER_ROW, *ANNEX_ROWS], y0=160.0, fontsize=11)
    png = p.get_pixmap(dpi=200).tobytes("png")
    tmp.close()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(pymupdf.Rect(0, 0, PAGE_W, PAGE_H), stream=png)


def _continuation_pages(doc: pymupdf.Document) -> None:
    """A table split across two pages: runs into the bottom margin of the
    first page and resumes at the top of the next WITH a repeated header —
    the exact shape multi-page stitching must detect and merge."""
    p8 = doc.new_page(width=PAGE_W, height=PAGE_H)
    p8.insert_text((72, 72), "6. Rate Schedule (full)", fontsize=14, fontname="hebo")
    p8.insert_textbox(
        pymupdf.Rect(72, 100, PAGE_W - 72, 560), LOREM * 4, fontsize=10, fontname="helv"
    )
    # 7 rows x 28pt from y=600 -> bottom edge 796 > 842*0.90=758: lands in
    # the bottom margin zone, which is continuation signal #1.
    _draw_ruled_table(p8, [HEADER_ROW, *CONT_ROWS_P8], y0=600.0)

    p9 = doc.new_page(width=PAGE_W, height=PAGE_H)
    # Continuation starts at y=72 < 842*0.12=101: signal #2. First row
    # repeats the header — stitching must drop it, not duplicate it.
    _draw_ruled_table(p9, [HEADER_ROW, *CONT_ROWS_P9], y0=72.0)
    p9.insert_textbox(
        pymupdf.Rect(72, 260, PAGE_W - 72, 420), LOREM * 2, fontsize=10, fontname="helv"
    )


def build_sample(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    _prose_page(doc, "7. Payment Terms", numbered=False)  # p0
    _prose_page(doc, "7. Payment Terms (contd.)", numbered=True)  # p1
    _scan_page(doc, header=None)  # p2
    # Header must exceed MIN_TEXT_CHARS (50) or this page stops being a trap:
    # the point is that text length alone would say TEXT_NATIVE. Body-only
    # (continuation) so pages 2-3 don't read as duplicated content.
    _scan_page(
        doc,
        header="DOCUMENT NO 42/2026 - SECTION 4 - CONTINUED - PAGE 17 OF 300",
        include_heading=False,
    )  # p3
    _drawing_page(doc)  # p4
    _title_page(doc)  # p5
    _table_page(doc)  # p6
    _scanned_table_page(doc)  # p7
    _continuation_pages(doc)  # p8 + p9
    assert doc.page_count == len(EXPECTED_KINDS)
    doc.save(path)
    doc.close()
    return path


if __name__ == "__main__":
    out = build_sample(Path("sample_data") / "sample_doc.pdf")
    print(f"wrote {out} ({len(EXPECTED_KINDS)} pages)")
