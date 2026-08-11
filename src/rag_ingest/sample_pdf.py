"""Generate a synthetic multi-layout PDF that exercises every triage branch.

Real large documents can't be committed to a public repo, so this builds a
small stand-in where the *correct* classification of every page is known in
advance (``EXPECTED_KINDS``) — which turns triage testing from eyeballing
into an assertion (see tests/test_triage.py).

    python -m rag_ingest.sample_pdf       # writes sample_data/sample_doc.pdf

Page map (0-based) and the triage branch each page exists to exercise:

    0  prose + headings                -> TEXT_NATIVE  (the easy case)
    1  prose + numbered headings       -> TEXT_NATIVE  (heading detection, Phase 2)
    2  full-page raster, no text       -> SCANNED      (classic scan)
    3  full-page raster + header text  -> SCANNED      (the header-over-scan TRAP:
                                          text length alone would say TEXT_NATIVE)
    4  dense vector line-work, no text -> DRAWING      (CAD-plan stand-in)
    5  title page, ~15 chars of text   -> SCANNED      (accepted misroute: costs
                                          one vision-API call, output still fine)
    6  ruled table + prose             -> TEXT_NATIVE  (table extraction, Phase 2/3)
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


def _gray_png(w: int, h: int, value: int) -> bytes:
    """A flat gray PNG made with PyMuPDF alone (no Pillow dependency yet)."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h))
    pix.clear_with(value)
    return pix.tobytes("png")


def _scan_page(doc: pymupdf.Document, header: str | None) -> None:
    """Full-page raster image, optionally with a text-layer header on top —
    the header variant is the trap SCAN_IMAGE_COVERAGE exists to catch."""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(pymupdf.Rect(0, 0, PAGE_W, PAGE_H), stream=_gray_png(300, 424, 215))
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


def _table_page(doc: pymupdf.Document) -> None:
    """Prose above a ruled 4x3 table — text-native, with real grid lines."""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text((72, 72), "5. Price Schedule (extract)", fontsize=14, fontname="hebo")
    page.insert_textbox(
        pymupdf.Rect(72, 100, PAGE_W - 72, 180), LOREM * 2, fontsize=10, fontname="helv"
    )

    x0, y0, col_w, row_h, cols, rows = 72.0, 220.0, 150.0, 28.0, 3, 4
    cells = [
        ["Item", "Description", "Rate"],
        ["1", "Excavation", "120.00"],
        ["2", "Concrete M25", "5400.00"],
        ["3", "Steel reinforcement", "62.50"],
    ]
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
                (x0 + c * col_w + 6, y0 + r * row_h + 18), cell, fontsize=9, fontname="helv"
            )


def build_sample(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    _prose_page(doc, "7. Payment Terms", numbered=False)  # p0
    _prose_page(doc, "7. Payment Terms (contd.)", numbered=True)  # p1
    _scan_page(doc, header=None)  # p2
    # Header must exceed MIN_TEXT_CHARS (50) or this page stops being a trap:
    # the point is that text length alone would say TEXT_NATIVE.
    _scan_page(doc, header="DOCUMENT NO 42/2026 - SECTION 4 - CONTINUED - PAGE 17 OF 300")  # p3
    _drawing_page(doc)  # p4
    _title_page(doc)  # p5
    _table_page(doc)  # p6
    assert doc.page_count == len(EXPECTED_KINDS)
    doc.save(path)
    doc.close()
    return path


if __name__ == "__main__":
    out = build_sample(Path("sample_data") / "sample_doc.pdf")
    print(f"wrote {out} ({len(EXPECTED_KINDS)} pages)")
