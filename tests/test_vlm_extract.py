"""The paid lane, tested entirely offline: a fake VlmClient stands in for
Gemini, so parser, verification, caching, and the page seam are all
exercised with expected values and zero network."""

import pymupdf
import pytest

from rag_ingest.models import Source, UnitType
from rag_ingest.vlm_extract import (
    PROMPT,
    CachedVlmClient,
    VlmError,
    VlmPageRecord,
    VlmResponse,
    ink_fraction,
    markdown_table_cells,
    parse_page_markdown,
    strip_outer_fence,
    verify_page_markdown,
    vlm_page_units,
)

PAGE_RECT = (0.0, 0.0, 595.0, 842.0)

SAMPLE_MD = """# बोली दस्तावेज़ / Bid Document

## 1. General Terms

पहला पैराग्राफ की पंक्ति एक
और पंक्ति दो

| विवरण/Item | मात्रा/Qty |
| --- | --- |
| पेन | 5 |

Second paragraph."""


class FakeClient:
    model_id = "fake-model"

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def generate(self, png: bytes, prompt: str) -> VlmResponse:
        self.calls += 1
        return VlmResponse(text=self.text, input_tokens=100, output_tokens=50)


class FailingClient:
    model_id = "fake-model"

    def generate(self, png: bytes, prompt: str) -> VlmResponse:
        raise VlmError("engine down")


@pytest.fixture(scope="module")
def blank_png() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    png = page.get_pixmap(dpi=72, alpha=False).tobytes("png")
    doc.close()
    return png


# --- parser ----------------------------------------------------------------


def test_parse_units_types_and_order():
    units = parse_page_markdown(SAMPLE_MD, 3, PAGE_RECT, [])
    assert [u.type for u in units] == [
        UnitType.TITLE,
        UnitType.TITLE,
        UnitType.TEXT,
        UnitType.TABLE,
        UnitType.TEXT,
    ]
    assert all(u.page == 3 and u.source == Source.GEMINI for u in units)


def test_parse_heading_levels_are_explicit_with_no_font_size():
    units = parse_page_markdown(SAMPLE_MD, 0, PAGE_RECT, [])
    h1, h2 = units[0], units[1]
    assert (h1.level, h1.font_size, h1.content) == (1, None, "बोली दस्तावेज़ / Bid Document")
    assert (h2.level, h2.content) == (2, "1. General Terms")


def test_parse_paragraph_lines_kept_together():
    units = parse_page_markdown(SAMPLE_MD, 0, PAGE_RECT, [])
    assert units[2].content == "पहला पैराग्राफ की पंक्ति एक\nऔर पंक्ति दो"
    assert units[4].content == "Second paragraph."


def test_prose_and_titles_get_full_page_bbox():
    units = parse_page_markdown(SAMPLE_MD, 0, PAGE_RECT, [])
    assert units[0].bbox == PAGE_RECT
    assert units[2].bbox == PAGE_RECT


def test_table_gets_yolo_bbox_when_counts_match():
    yolo_box = (50.0, 300.0, 500.0, 400.0)
    units = parse_page_markdown(SAMPLE_MD, 0, PAGE_RECT, [yolo_box])
    table = next(u for u in units if u.type == UnitType.TABLE)
    assert table.bbox == yolo_box


def test_table_falls_back_to_page_bbox_on_count_mismatch():
    boxes = [(50.0, 300.0, 500.0, 400.0), (50.0, 500.0, 500.0, 600.0)]
    units = parse_page_markdown(SAMPLE_MD, 0, PAGE_RECT, boxes)  # 1 table, 2 boxes
    table = next(u for u in units if u.type == UnitType.TABLE)
    assert table.bbox == PAGE_RECT


def test_pipe_line_without_delimiter_row_is_prose():
    md = "| this is just | prose with pipes |\n\nreal text"
    units = parse_page_markdown(md, 0, PAGE_RECT, [])
    assert [u.type for u in units] == [UnitType.TEXT, UnitType.TEXT]


def test_strip_outer_fence():
    fenced = "```markdown\n# Heading\n\nText.\n```"
    assert strip_outer_fence(fenced) == "# Heading\n\nText."
    # Inner fences are content, not wrapping — untouched.
    inner = "para\n\n```text\ncode\n```\n\nmore"
    assert strip_outer_fence(inner) == inner


def test_markdown_table_cells_expected_matrix():
    table = next(
        u for u in parse_page_markdown(SAMPLE_MD, 0, PAGE_RECT, []) if u.type == UnitType.TABLE
    )
    assert markdown_table_cells(table.content) == [["विवरण/Item", "मात्रा/Qty"], ["पेन", "5"]]


def test_markdown_table_cells_unescapes_pipes():
    md = "| a \\| b | c |\n| --- | --- |\n| 1 | 2 |"
    assert markdown_table_cells(md) == [["a | b", "c"], ["1", "2"]]


# --- verification ----------------------------------------------------------


def test_clean_page_produces_no_reasons():
    assert verify_page_markdown(SAMPLE_MD, text_layer_chars=len(SAMPLE_MD)) == []


def test_repetition_loop_flagged():
    md = "the same runaway phrase again " * 30
    reasons = verify_page_markdown(md)
    assert len(reasons) == 1 and "repetition" in reasons[0]


def test_length_bounds_against_text_layer():
    assert any(
        "length" in r for r in verify_page_markdown("tiny", text_layer_chars=1000)
    )  # omission
    varied = " ".join(f"word{i}" for i in range(700))  # ~6000 varied chars
    assert any(
        "length" in r for r in verify_page_markdown(varied, text_layer_chars=1000)
    )  # runaway
    in_bounds = " ".join(f"word{i}" for i in range(150))  # ~1000 varied chars
    assert verify_page_markdown(in_bounds, text_layer_chars=1000) == []


def test_dense_ink_with_empty_output_flagged():
    assert any("ink" in r for r in verify_page_markdown("almost nothing", ink_frac=0.05))
    # Near-blank page with near-empty output is fine — blank pages exist.
    assert verify_page_markdown("almost nothing", ink_frac=0.001) == []


def test_mojibake_echo_flagged():
    assert any("mojibake" in r for r in verify_page_markdown("स]म िवभाग 3ितशत"))
    assert any("mojibake" in r for r in verify_page_markdown("bad \x01 control"))


def test_yolo_cross_check_one_directional():
    # YOLO saw a table, markdown has none -> flag.
    assert any("YOLO" in r for r in verify_page_markdown("prose only", yolo_table_count=1))
    # Markdown has a table YOLO missed -> NOT flagged.
    assert verify_page_markdown(SAMPLE_MD, text_layer_chars=len(SAMPLE_MD)) == []


# --- caching ---------------------------------------------------------------


def test_cache_hit_skips_engine(tmp_path):
    inner = FakeClient("# Heading\n\nBody.")
    client = CachedVlmClient(inner, tmp_path / "vlm")
    first = client.generate(b"png-bytes", PROMPT)
    second = client.generate(b"png-bytes", PROMPT)
    assert inner.calls == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.text == "# Heading\n\nBody."
    assert second.input_tokens == 100 and second.output_tokens == 50


def test_cache_keys_on_png_bytes(tmp_path):
    inner = FakeClient("text")
    client = CachedVlmClient(inner, tmp_path / "vlm")
    client.generate(b"page-one", PROMPT)
    client.generate(b"page-two", PROMPT)
    assert inner.calls == 2


# --- page seam -------------------------------------------------------------


def test_vlm_page_units_happy_path(blank_png):
    client = FakeClient(SAMPLE_MD)
    units, record = vlm_page_units(
        blank_png, 7, PAGE_RECT, [], client, text_layer_chars=len(SAMPLE_MD)
    )
    assert len(units) == 5
    assert all(not u.needs_review for u in units)
    assert record == VlmPageRecord(
        page=7, cached=False, input_tokens=100, output_tokens=50, review_reasons=[]
    )


def test_failed_page_becomes_one_flagged_empty_unit(blank_png):
    units, record = vlm_page_units(blank_png, 2, PAGE_RECT, [], FailingClient())
    assert len(units) == 1
    u = units[0]
    assert (u.type, u.content, u.needs_review, u.source) == (
        UnitType.TEXT,
        "",
        True,
        Source.GEMINI,
    )
    assert any("vlm call failed" in r for r in record.review_reasons)


def test_illegible_flags_only_its_unit(blank_png):
    md = "Clean paragraph.\n\nDamaged [ILLEGIBLE] region."
    client = FakeClient(md)
    units, record = vlm_page_units(
        blank_png, 0, PAGE_RECT, [], client, text_layer_chars=len(md)
    )
    assert record.review_reasons == []
    assert [u.needs_review for u in units] == [False, True]


def test_page_level_failure_flags_every_unit(blank_png):
    client = FakeClient(SAMPLE_MD)
    # Length check fails (text layer said 10x more chars) -> whole page flagged.
    units, record = vlm_page_units(
        blank_png, 0, PAGE_RECT, [], client, text_layer_chars=len(SAMPLE_MD) * 10
    )
    assert all(u.needs_review for u in units)
    assert any("length" in r for r in record.review_reasons)


def test_ink_fraction_blank_page_is_zero(blank_png):
    assert ink_fraction(blank_png) == 0.0
