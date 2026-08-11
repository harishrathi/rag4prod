"""v2 pure layers: boxes, routing, table ladder, normalization, quality.

Ported from v1's content-asserting suite (rewrite §5.2): every test
asserts EXPECTED values known by construction, not observed output."""

import pymupdf
import pytest

from rag_ingest.models import Source, Unit, UnitType
from rag_ingest.vlm_extract import VlmPageRecord
from rag_ingest2.boxes import PdfBox, PixelBox, to_pdf
from rag_ingest2.config import ExtractionConfig, NormalizeConfig, RoutingRules, TableConfig
from rag_ingest2.normalize import (
    build_tables,
    dedup_units_in_tables,
    resolve_heading_levels,
    strip_repeated_furniture,
    suppress_repeated_suspects,
)
from rag_ingest2.profiles import PageProfile
from rag_ingest2.quality import apply_quality_gate, render_review_report
from rag_ingest2.routing import Extractor, route
from rag_ingest2.tables import RawTable
from rag_ingest2.tables.stitch import merge_chain, stitch
from rag_ingest2.tables.validate import junk_cell_count, validate_cells

RULES = RoutingRules()
TCFG = TableConfig()
NCFG = NormalizeConfig()


# --- boxes -----------------------------------------------------------------


def test_pixel_to_pdf_uses_measured_scale():
    # 1000x2000 raster over a 500x1000pt page: scale is exactly 0.5,
    # derived from the dimensions carried BY the box — never from DPI.
    px = PixelBox(x0=100, y0=200, x1=300, y1=400, raster_w=1000, raster_h=2000)
    page = PdfBox(0, 0, 500, 1000)
    assert to_pdf(px, page) == PdfBox(50.0, 100.0, 150.0, 200.0)


def test_pixel_to_pdf_clamps_into_page():
    px = PixelBox(x0=-20, y0=0, x1=1100, y1=500, raster_w=1000, raster_h=1000)
    page = PdfBox(0, 0, 500, 500)
    out = to_pdf(px, page)
    assert (out.x0, out.x1) == (0.0, 500.0)


def test_degenerate_conversion_raises():
    # A box mapping entirely past the page edge clamps to x0 > x1 — must
    # raise, not return a silent empty crop.
    px = PixelBox(x0=2000, y0=2000, x1=2100, y1=2100, raster_w=1000, raster_h=1000)
    with pytest.raises(ValueError):
        to_pdf(px, PdfBox(0, 0, 100, 100))


# --- routing ---------------------------------------------------------------


def _profile(**kw) -> PageProfile:
    base = dict(
        page=0,
        text_chars=2000,
        text_compact_chars=1800,
        junk_chars=0,
        mojibake_chars=0,
        max_image_coverage=0.05,
        text_bbox_area_frac=0.5,
        vector_segments=None,
    )
    base.update(kw)
    return PageProfile(**base)


def test_clean_text_page_routes_native():
    r = route(_profile(), RULES)
    assert r.extractor == Extractor.NATIVE


def test_blanketing_raster_outranks_text_layer():
    # The header-over-scan trap: plenty of text, but one raster covers
    # 90% of the page.
    r = route(_profile(max_image_coverage=0.9), RULES)
    assert r.extractor == Extractor.VLM
    assert "raster" in r.reasons[0]


def test_junk_and_mojibake_both_recorded_when_both_fire():
    r = route(_profile(junk_chars=25, mojibake_chars=12), RULES)
    assert r.extractor == Extractor.VLM
    assert len(r.reasons) == 2
    assert "corrupt" in r.reasons[0] and "mojibake" in r.reasons[1]


def test_mojibake_alone_reroutes():
    r = route(_profile(mojibake_chars=10), RULES)
    assert r.extractor == Extractor.VLM


def test_drawing_page_routes_drawing():
    r = route(_profile(text_chars=10, text_compact_chars=9, vector_segments=150), RULES)
    assert r.extractor == Extractor.DRAWING


def test_near_blank_biases_to_vlm():
    r = route(_profile(text_chars=10, text_compact_chars=9, vector_segments=3), RULES)
    assert r.extractor == Extractor.VLM


# --- table ladder ----------------------------------------------------------

HEADER = ["Item", "Description", "Rate"]


def _frag(page, y0, y1, rows, source="find_tables", header_rows=1):
    return RawTable(
        page=page,
        bbox=(72.0, y0, 522.0, y1),
        cells=[HEADER, *rows],
        source=source,
        header_rows=header_rows,
    )


def test_continuation_stitched_and_header_dropped():
    a = _frag(0, 600.0, 800.0, [["1", "Excavation", "120.00"], ["2", "Concrete", "5400.00"]])
    b = _frag(1, 72.0, 200.0, [HEADER, ["3", "Steel", "62.50"]][1:] or [])
    # page-1 fragment repeats the header as its first row
    b.cells = [HEADER, ["3", "Steel", "62.50"]]
    chains = stitch([a, b], {0: 842.0, 1: 842.0}, TCFG)
    assert len(chains) == 1
    cells, _merges = merge_chain(chains[0], TCFG)
    assert cells == [
        HEADER,
        ["1", "Excavation", "120.00"],
        ["2", "Concrete", "5400.00"],
        ["3", "Steel", "62.50"],
    ]


def test_column_mismatch_refuses_to_stitch():
    a = _frag(0, 600.0, 800.0, [["1", "Excavation", "120.00"]])
    b = RawTable(page=1, bbox=(72.0, 72.0, 522.0, 200.0), cells=[["x", "y"]], source="find_tables")
    assert len(stitch([a, b], {0: 842.0, 1: 842.0}, TCFG)) == 2


def test_full_page_bbox_refuses_to_stitch():
    # Paid-lane fragments without YOLO boxes carry the full-page rect —
    # no geometry to test, so stitching must refuse rather than guess.
    a = RawTable(page=0, bbox=(0.0, 0.0, 595.0, 842.0), cells=[HEADER, ["1", "a", "2"]],
                 source="gemini")
    b = RawTable(page=1, bbox=(0.0, 0.0, 595.0, 842.0), cells=[HEADER, ["2", "b", "3"]],
                 source="gemini")
    assert len(stitch([a, b], {0: 842.0, 1: 842.0}, TCFG)) == 2


def test_validate_cells_expected_reasons():
    assert validate_cells([]) is not None
    assert validate_cells([HEADER]) == "fewer than 2 rows (no grid found, or header-only)"
    assert "ragged" in validate_cells([HEADER, ["a", "b"]])
    assert validate_cells([HEADER, ["1", "Excavation", "120.00"]]) is None


def test_junk_cell_count_sees_orphan_marks():
    cells = [HEADER, ["\x01बड", "िवभाग", "120.00"]]  # junk char + orphan matra
    assert junk_cell_count(cells) == 2


# --- normalization ---------------------------------------------------------


def _unit(page, y0, text, kind=UnitType.TEXT, source=Source.PYMUPDF, **kw):
    return Unit(
        page=page, bbox=(72.0, y0, 520.0, y0 + 12), type=kind, content=text, source=source, **kw
    )


def test_furniture_stripped_and_body_survives():
    heights = {p: 842.0 for p in range(5)}
    units = []
    for p in range(5):
        units.append(_unit(p, 30.0, "TENDER NOTICE NO. HY/M&S/10-RS/RC/22-23"))
        units.append(_unit(p, 400.0, f"Body paragraph unique to page {p}."))
        units.append(_unit(p, 810.0, f"Page {p + 1} of 5"))
    kept = strip_repeated_furniture(units, heights, NCFG)
    assert [u.content for u in kept] == [f"Body paragraph unique to page {p}." for p in range(5)]


def test_repeated_yolo_suspects_suppressed():
    box = (55.0, 26.0, 591.0, 79.0)
    raw = [RawTable(page=p, bbox=box, cells=[], source="yolo_only") for p in range(4)]
    unique = RawTable(page=9, bbox=(72.0, 300.0, 500.0, 480.0), cells=[], source="yolo_only")
    assert suppress_repeated_suspects([*raw, unique], NCFG) == [unique]


def test_heading_levels_explicit_beats_numbered_beats_size():
    units = [
        _unit(0, 100, "## style from VLM", kind=UnitType.TITLE, source=Source.GEMINI, level=2),
        _unit(0, 200, "7.3.1 Delay Notices", kind=UnitType.TITLE, font_size=10.0),
        _unit(0, 300, "Big Heading", kind=UnitType.TITLE, font_size=18.0),
        _unit(0, 400, "Small Heading", kind=UnitType.TITLE, font_size=13.0),
    ]
    resolve_heading_levels(units, NCFG)
    assert [u.level for u in units] == [2, 3, 1, 2]


def test_gemini_units_exempt_from_table_dedup():
    tables = build_tables(
        [_frag(0, 100.0, 400.0, [["1", "Excavation", "120.00"]])], {0: 842.0}, NCFG, TCFG
    )
    native_inside = _unit(0, 200.0, "native prose inside the table span")
    gemini_full_page = Unit(
        page=0,
        bbox=(0.0, 0.0, 595.0, 842.0),
        type=UnitType.TEXT,
        content="paid-lane prose, page-level bbox",
        source=Source.GEMINI,
    )
    kept = dedup_units_in_tables([native_inside, gemini_full_page], tables)
    assert kept == [gemini_full_page]


# --- quality gate ----------------------------------------------------------


def test_gate_flags_native_junk_and_gemini_page_reasons(tmp_path):
    units = [
        _unit(0, 100, "clean native prose"),
        _unit(0, 120, "broken \x01 native"),
        _unit(1, 0, "paid-lane text", source=Source.GEMINI),
        _unit(2, 0, "fine paid-lane page", source=Source.GEMINI),
        _unit(2, 10, "damaged [ILLEGIBLE] region", source=Source.GEMINI),
    ]
    vlm_pages = [
        VlmPageRecord(page=1, cached=False, input_tokens=1, output_tokens=1,
                      review_reasons=["repetition loop: a 20-char sequence recurs 40x"]),
        VlmPageRecord(page=2, cached=True, input_tokens=0, output_tokens=0),
    ]
    pdf = tmp_path / "t.pdf"
    d = pymupdf.open()
    d.new_page()
    d.save(pdf)
    d.close()
    result = apply_quality_gate(units, [], vlm_pages, pdf, tmp_path, ExtractionConfig())
    assert [u.needs_review for u in result.units] == [False, True, True, False, True]
    report = render_review_report("t", result)
    assert "repetition loop" in report and "[ILLEGIBLE]" in report


def test_gate_rejects_junk_table_with_crop(tmp_path):
    pdf = tmp_path / "t.pdf"
    d = pymupdf.open()
    d.new_page()
    d.save(pdf)
    d.close()
    tables = build_tables(
        [
            RawTable(
                page=0,
                bbox=(72.0, 100.0, 500.0, 200.0),
                cells=[HEADER, ["\x01बड", "िवभाग/Bid", "120.00"]],
                source="find_tables",
            )
        ],
        {0: 842.0},
        NCFG,
        TCFG,
    )
    result = apply_quality_gate([], tables, [], pdf, tmp_path, ExtractionConfig())
    t = result.tables[0]
    assert t.needs_review and "junk" in t.review_reason
    assert t.markdown == "" and t.grid == ""
    assert t.crop_key and (tmp_path / t.crop_key).exists()
