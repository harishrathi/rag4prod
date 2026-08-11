"""Text-layer quality signals, shared by v1 triage, v2 profiling, and the
VLM lane's verification.

These are pure functions over extracted text — evidence collectors, not
decision makers. They live outside triage.py because three consumers need
them: v1's triage (reroute decisions), the VLM verification gate (mojibake
echo check), and the v2 rewrite's page profiling (rag_ingest2). All are
Unicode-general on purpose: per-language tables defeat the goal of one
code path for every script (gemini_extractor_spec.md §3).

Two distinct broken-CMap symptoms, two detectors:

* ``text_layer_junk`` — glyphs mapping to NON-printable garbage: C0
  controls, U+FFFD, Private Use Area (ledger #29).
* ``mojibake_score`` — glyphs mapping to PRINTABLE garbage: orphan
  combining marks and ASCII symbols interleaved inside non-Latin words
  (`लाभाथ\\ के प] म= होनी चा<हए`), which the junk test cannot see.
"""

from __future__ import annotations

import re
import unicodedata

# Junk characters that healthy text layers NEVER contain: C0 controls
# (except tab/newline/CR, which the whitespace strip removes anyway),
# U+FFFD replacement chars, and Private Use Area codepoints. Their
# presence means the font's ToUnicode CMap is broken — the page renders
# fine but its text layer is lying (ledger #29).
JUNK_CHARS_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f�-]")


def text_layer_junk(text: str) -> tuple[int, float]:
    """(junk char count, junk ratio) over non-whitespace text."""
    compact = "".join(text.split())
    if not compact:
        return 0, 0.0
    n = len(JUNK_CHARS_RE.findall(compact))
    return n, n / len(compact)


# Marks whose Unicode script is Inherited attach to letters of ANY script
# (NFD `é` carries U+0301) — exempt from the cross-script test.
_INHERITED_MARK_RANGES = (
    (0x0300, 0x036F),  # Combining Diacritical Marks
    (0x1AB0, 0x1AFF),  # Combining Diacritical Marks Extended
    (0x1DC0, 0x1DFF),  # Combining Diacritical Marks Supplement
    (0x20D0, 0x20FF),  # Combining Diacritical Marks for Symbols
    (0xFE20, 0xFE2F),  # Combining Half Marks
)

_MOJIBAKE_SYMBOLS = frozenset("\\][=<>^_@#|~")


def orphan_combining_marks(text: str) -> int:
    """Count combining marks (Mn/Mc) not attached to a same-script letter.

    The base of a mark is the most recent non-mark, non-format char: marks
    stack (र + ि + ं) and ZWJ/ZWNJ are transparent, so simply looking at
    the preceding char would miscount healthy Indic text. Script identity
    is approximated by the 128-codepoint block (exact for the Indic blocks
    this was built against; coarse elsewhere, which only softens the
    signal, never inflates it for healthy text)."""
    orphans = 0
    base: str | None = None
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Mn", "Mc"):
            cp = ord(ch)
            if base is None or not base.isalpha():
                orphans += 1
            elif not any(lo <= cp <= hi for lo, hi in _INHERITED_MARK_RANGES) and (
                cp // 0x80 != ord(base) // 0x80
            ):
                orphans += 1
        elif cat != "Cf":  # format chars (ZWJ/ZWNJ) are transparent
            base = ch
    return orphans


def interleaved_ascii_symbols(text: str) -> int:
    """Count mojibake-symbol chars inside tokens that contain non-Latin
    letters — `प]` and `म=` are CMap shrapnel; a bare `x=y` is not.
    Only symbols that never legitimately appear mid-word count;
    ./,-%() occur in real text (dates, abbreviations) and are excluded."""
    count = 0
    for token in text.split():
        if any(ch.isalpha() and not ch.isascii() for ch in token):
            count += sum(1 for ch in token if ch in _MOJIBAKE_SYMBOLS)
    return count


def mojibake_score(text: str) -> tuple[int, float]:
    """(mojibake char count, ratio) over non-whitespace text — the
    printable-garbage counterpart of text_layer_junk().

    Measured on the real corpus (2026-08-11): healthy pages score
    EXACTLY 0; broken-CMap pages score 10-20 (see config.MOJIBAKE_MIN)."""
    compact = "".join(text.split())
    if not compact:
        return 0, 0.0
    n = orphan_combining_marks(text) + interleaved_ascii_symbols(text)
    return n, n / len(compact)
