"""Real-corpus hardening tests (ledgers #29-30): broken text layers
and repeating page furniture.

Fixtures here are synthetic but the expected behavior is declared from
the failure shapes observed on real documents: bilingual pages whose
Devanagari font has a broken ToUnicode CMap (text layer emits C0
control characters), and bordered page-title boxes that YOLO flags as a
table on every page.
"""

from rag_ingest.assemble import strip_repeated_furniture
from rag_ingest.models import Source, Unit, UnitType
from rag_ingest.tables import RawTable, _suppress_repeated_suspects
from rag_ingest.triage import text_layer_junk

# Real mojibake from a broken-CMap page: control chars interleaved with
# misassigned Devanagari codepoints.
MOJIBAKE = "\x01बड \x01ववरण/Bid Details \x01बड बंद होने क\x10 तार\x13ख"
CLEAN_HINDI = "बिड विवरण बंद होने की तारीख और समय"
CLEAN_ENGLISH = "The bid closes at 12:00 on 15-06-2026 with EMD amount 28224."


# --- Broken text layer detection (triage reroute, ledger #29) ----------------


def test_junk_detector_fires_on_mojibake_only():
    # MOJIBAKE carries exactly 5 control chars: \x01 x3, \x10, \x13.
    n, ratio = text_layer_junk(MOJIBAKE)
    assert n == 5 and ratio > 0.005  # over threshold -> reroutes to the VLM lane
    assert text_layer_junk(CLEAN_HINDI) == (0, 0.0)
    assert text_layer_junk(CLEAN_ENGLISH) == (0, 0.0)


# --- Repeating page furniture (ledger #26/#30) -------------------------------


def _unit(page: int, y0: float, text: str, kind=UnitType.TEXT) -> Unit:
    return Unit(page=page, bbox=(72.0, y0, 520.0, y0 + 12), type=kind, content=text)


def test_repeating_headers_and_numbered_footers_stripped():
    heights = {p: 842.0 for p in range(5)}
    units = []
    for p in range(5):
        units.append(_unit(p, 30.0, "TENDER NOTICE NO. HY/M&S/10-RS/RC/22-23"))  # header
        units.append(_unit(p, 400.0, f"Body paragraph unique to page {p}."))
        units.append(_unit(p, 810.0, f"Page {p + 1} of 5"))  # footer, digits vary
    kept = strip_repeated_furniture(units, heights)
    assert [u.content for u in kept] == [f"Body paragraph unique to page {p}." for p in range(5)]


def test_body_text_and_rare_repeats_survive():
    heights = {p: 842.0 for p in range(5)}
    units = [_unit(p, 400.0, "The supplier shall complete all services.") for p in range(5)]
    units += [_unit(p, 30.0, "Annex heading") for p in range(2)]  # only 2 repeats
    kept = strip_repeated_furniture(units, heights)
    assert len(kept) == 7  # mid-page repetition and sub-threshold repeats survive


def test_repeated_yolo_suspects_suppressed_but_unique_kept():
    header_box = (55.0, 26.0, 591.0, 79.0)
    raw = [
        RawTable(page=p, bbox=header_box, cells=[], source="yolo_only") for p in range(4)
    ]
    unique = RawTable(page=9, bbox=(72.0, 300.0, 500.0, 480.0), cells=[], source="yolo_only")
    real = RawTable(
        page=2,
        bbox=(80.0, 100.0, 500.0, 200.0),
        cells=[["a", "b"], ["1", "2"]],
        source="find_tables",
    )
    kept = _suppress_repeated_suspects([*raw, unique, real])
    assert kept == [unique, real]  # 4 repeating suspects gone, genuine suspect stays


def test_tier1_table_with_junk_cells_flagged(tmp_path):
    import pymupdf

    from rag_ingest.tables import finalize

    doc = pymupdf.open()
    doc.new_page()
    raw = RawTable(
        page=0,
        bbox=(72.0, 100.0, 500.0, 200.0),
        cells=[["Item", "Rate"], ["\x01बड \x01ववरण/Bid", "120.00"]],
        source="find_tables",
    )
    results = finalize([raw], {0: 842.0}, doc, tmp_path)
    doc.close()
    assert results[0].needs_review
    assert "junk" in (results[0].review_reason or "")


# --- Residual mojibake units are flagged, never silent -----------------------


def test_mojibake_unit_flag_reaches_chunks():
    from rag_ingest.assemble import build_walk
    from rag_ingest.chunking import chunk_document

    units = [
        Unit(
            page=0,
            bbox=(72.0, 100.0, 500.0, 120.0),
            type=UnitType.TEXT,
            content=MOJIBAKE,
            source=Source.PYMUPDF,
            needs_review=True,  # as local_extract now sets for junk content
        )
    ]
    chunks, _ = chunk_document("t", build_walk(units, []))
    assert chunks and all(c.needs_review for c in chunks)
