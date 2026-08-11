"""Central configuration for the PDF ingestion pipeline.

Every threshold in this pipeline is a *tuning knob*, not a law of nature.
They are collected here — named, typed, and documented — so that:

  1. An interviewer asking "why 50 characters?" gets a real answer
     (see each constant's comment), and
  2. Tuning against a real corpus means editing one file, not hunting
     magic numbers across five modules.

Constants are added phase by phase; each block is labelled with the
pipeline stage that consumes it.
"""

# ---------------------------------------------------------------------------
# STAGE 1 — Triage (rag_ingest/triage.py)
# ---------------------------------------------------------------------------

# A page whose extractable text is shorter than this is presumed to have no
# usable text layer (i.e. it is a scanned image, a drawing, or blank).
#
# Why 50: a genuine prose page yields hundreds-to-thousands of characters;
# a scanned page with no OCR layer yields 0-ish. The awkward middle ground
# (10..50 chars) is almost always a page number + running header sitting on
# top of a scanned body — which we WANT classified as scanned. A title page
# with only "DOCUMENT No. 42" also lands here and gets sent to Gemini
# unnecessarily; that costs one cheap API call and is an accepted trade.
MIN_TEXT_CHARS = 50

# Trap this guards against: a scanned page where only the header/footer has
# a real text layer (common when a scanner stamps page numbers). Such a page
# passes the MIN_TEXT_CHARS test and would be "extracted" as garbage.
# If any single raster image covers more than this fraction of the page,
# the page is treated as SCANNED regardless of its text length.
SCAN_IMAGE_COVERAGE = 0.70

# Pages that are almost pure vector graphics (CAD plans, engineering
# drawings). They have no text layer and no raster image, so the
# two rules above would misroute them to Gemini, which would return junk
# at real cost. If a near-textless page contains at least this many vector
# line/curve segments, classify it as DRAWING: render to PNG, store as a
# figure, never send to Gemini for text extraction.
# (Counted as individual segments across all paths, not path objects —
# PyMuPDF may report one committed shape as a single path with N items.)
DRAWING_MIN_SEGMENTS = 100

# ---------------------------------------------------------------------------
# Debug artifacts (all stages)
# ---------------------------------------------------------------------------
# Every stage writes its output to output/<doc_id>/stages/NN_*.json[l] so a
# document can be traced through the pipeline file by file, and any stage
# can be re-run from its predecessor's artifact without recomputing the
# world (--from-stage). Images saved under debug/ are *copies for humans*:
# they are downscaled JPEGs to keep a 100-page run in tens of MB, while the
# pipeline itself always works on full-quality PNGs in memory.
DEBUG_IMAGE_MAX_DIM = 1200  # px, longest side of a debug image copy
DEBUG_JPEG_QUALITY = 70  # good enough to eyeball, ~10x smaller than PNG
