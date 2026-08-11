"""Triage against the synthetic sample PDF, where every page's correct
classification is known ground truth (sample_pdf.EXPECTED_KINDS).

The interesting assertions are the trap pages:
  * page 3 has >50 chars of real text but must still be SCANNED
    (header-over-scan guard);
  * page 4 has no text and no raster image but must be DRAWING, not SCANNED
    (vector segment count).
"""

import pymupdf
import pytest

from rag_ingest.models import PageKind
from rag_ingest.sample_pdf import EXPECTED_KINDS, build_sample
from rag_ingest.triage import (
    interleaved_ascii_symbols,
    mojibake_score,
    orphan_combining_marks,
    triage,
    triage_page,
)


@pytest.fixture(scope="module")
def sample_doc(tmp_path_factory):
    path = build_sample(tmp_path_factory.mktemp("pdf") / "sample_doc.pdf")
    doc = pymupdf.open(path)
    yield doc
    doc.close()


def test_every_page_classified_correctly(sample_doc):
    records = triage(sample_doc)
    got = {r.page: r.kind for r in records}
    assert got == EXPECTED_KINDS


def test_header_over_scan_trap_caught_by_coverage_not_text(sample_doc):
    record = triage(sample_doc)[3]
    # The trap is only meaningful if the text layer alone WOULD have passed:
    assert record.text_chars >= 50
    assert record.max_image_coverage > 0.7


def test_drawing_page_detected_by_segments(sample_doc):
    record = triage(sample_doc)[4]
    assert record.drawing_segments is not None
    assert record.drawing_segments >= 100


# --- Script-agnostic mojibake scoring (gemini_extractor_spec.md §3) --------
# Expected counts, not observed ones: each string is built so the correct
# answer is known by construction.


def test_healthy_devanagari_scores_zero():
    # Logical-order Hindi: matras follow their consonants, anusvara stacks
    # on a matra (में = म + े + ं) — none of it is an orphan.
    assert orphan_combining_marks("राशि विभाग निविदा में") == 0


def test_nfd_latin_diacritics_are_not_orphans():
    # Decomposed é (e + U+0301): the mark's script is Inherited, so the
    # cross-script test must not fire on ordinary accented Latin.
    assert orphan_combining_marks("café résumé") == 0


def test_word_initial_matra_is_orphan():
    # Visual-order encoding puts the i-matra BEFORE its consonant, so it
    # lands after a space at word start: िवभाग (for विभाग).
    assert orphan_combining_marks("िवभाग") == 1


def test_matra_after_digit_and_latin_base_are_orphans():
    # Broken CMaps map Devanagari glyphs to digits/Latin, leaving the
    # surviving matras attached to the wrong script: 3ितशत (for प्रतिशत).
    assert orphan_combining_marks("3ितशत") == 1
    assert orphan_combining_marks("aि") == 1


def test_interleaved_ascii_symbols_counts_only_non_latin_tokens():
    # स]म, म=, चा<हए carry one mojibake symbol each; the pure-ASCII
    # tokens x=y and [a] must not count.
    assert interleaved_ascii_symbols("स]म म= चा<हए x=y [a]") == 3


def test_legitimate_punctuation_in_indic_words_not_counted():
    # Slashes, hyphens, percent, parens occur in real bilingual text.
    assert interleaved_ascii_symbols("विवरण/Bid निविदा-प्रपत्र 5% (राशि)") == 0


def test_mojibake_score_combines_both_signals():
    # One word-initial matra + one interleaved symbol over 9 compact chars.
    text = "िवभाग म= ab"
    count, ratio = mojibake_score(text)
    assert count == 2
    assert ratio == pytest.approx(2 / 9)


class _FakeTextPage:
    """Just enough of pymupdf.Page for triage_page's native branch."""

    rect = pymupdf.Rect(0, 0, 595, 842)

    def __init__(self, text: str):
        self._text = text

    def get_text(self, mode: str) -> str:
        return self._text

    def get_image_info(self) -> list:
        return []


def test_printable_mojibake_page_reroutes_to_vlm_lane():
    # A page long enough to pass MIN_TEXT_CHARS, zero junk chars, but
    # full of printable CMap garbage — MUST reroute (the exact failure
    # that shipped unflagged before the VLM lane existed).
    broken = "लाभाथ\\ के प] म= होनी चा<हए। स]म 3ािधकार िवभाग िनिवदा " * 4
    record = triage_page(_FakeTextPage(broken), 0)
    assert record.kind == PageKind.SCANNED
    assert "mojibake" in record.reason


def test_healthy_bilingual_page_stays_native():
    healthy = "विवरण/Bid Number: GEM/2025/B/123 राशि की गणना निविदा में दी गई है। " * 4
    record = triage_page(_FakeTextPage(healthy), 0)
    assert record.kind == PageKind.TEXT_NATIVE
