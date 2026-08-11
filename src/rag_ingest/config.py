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

# ---------------------------------------------------------------------------
# STAGE 2 — Local extraction (rag_ingest/local_extract.py)
# ---------------------------------------------------------------------------

# Body font size is estimated from a sample of text-native pages spread
# across the WHOLE document — front matter (covers, tables of contents,
# forms) is typographically unrepresentative, so "first N pages" sampling
# would skew the estimate on real documents.
BODY_FONT_SAMPLE_PAGES = 60

# A line is a heading when its font size is >= body_size * this ratio.
# 1.15 catches 12pt headings over 10pt body. Headings set at exactly body
# size are missed unless they are bold AND numbered (see regex below) —
# an accepted gap, documented in docs/edge_cases.md.
HEADING_SIZE_RATIO = 1.15

# Bold lines that look like numbered clauses ("7.3 Liquidated Damages")
# are headings even at body size. Numbering is also the strongest signal
# stage 6 has for heading depth, so this regex is shared with assembly.
HEADING_NUMBERED_RE = r"^\d+(\.\d+)*\.?\s+\S"

# Embedded raster images smaller than this fraction of the page area are
# skipped: they are almost always logos, watermarks, or bullet glyphs.
# Storing them as figures adds noise chunks that retrieval can hit
# instead of real content.
FIGURE_MIN_AREA_FRAC = 0.005

# DPI used when cropping embedded figures to PNG. Figures are stored, not
# OCR'd, so this only affects visual quality of the stored artifact.
FIGURE_DPI = 200

# A page is marked as having a ruled grid when its vector drawings contain
# at least this many axis-aligned horizontal AND vertical segments. Grids
# are a cross-check for stage 4's table boxes, NOT a router — page borders
# and letterhead rules false-positive too easily to gate YOLO on this
# (see docs/design_spec.md §5).
RULED_MIN_H_SEGMENTS = 4
RULED_MIN_V_SEGMENTS = 4

# ---------------------------------------------------------------------------
# STAGE 3 — Rendering (rag_ingest/render.py)
# ---------------------------------------------------------------------------

# Render DPI for the vision paths. 200 is the accuracy-leaning default;
# 150 nearly halves render time and memory. This is a REAL tuning knob:
# the design spec (§11.1) says to benchmark table extraction at both on
# real documents before committing a production value.
RENDER_DPI = 200

# ---------------------------------------------------------------------------
# STAGE 4 — Layout detection (rag_ingest/layout.py)
# ---------------------------------------------------------------------------

# DocLayout-YOLO checkpoint, fetched once from Hugging Face and cached in
# the standard HF cache (~/.cache/huggingface). ~40 MB.
YOLO_HF_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
YOLO_HF_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"

# The model was fine-tuned at this input size; feeding other sizes
# degrades accuracy (the wrapper letterboxes internally).
YOLO_IMG_SIZE = 1024

# Detections below this confidence are dropped. Low on purpose: a missed
# table is silent data loss, a false positive costs one wasted Gemini
# crop that comes back as prose. Errors should fall toward "wasted call".
YOLO_CONF_THRESHOLD = 0.2

# Only these DocLayNet-style labels become regions; prose/captions/etc.
# are already covered by local extraction or the full-page Gemini pass.
YOLO_KEEP_LABELS = ("table", "figure")

# Padding added to each detection (in rendered-image pixels) before the
# coordinate conversion, so tight boxes don't clip caption lines or the
# last table row. Tune against real documents (design spec §11.5).
YOLO_BOX_PAD_PX = 10

# "cpu" is fine at our page counts; set "cuda" / "mps" when available.
YOLO_DEVICE = "cpu"
