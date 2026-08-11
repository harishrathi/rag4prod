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

# Broken text layers (ledger #29): some PDF toolchains embed fonts whose
# glyphs RENDER correctly but map to garbage codepoints in the text
# layer (broken ToUnicode CMap) — endemic in bilingual documents with
# custom-encoded Indic fonts. Healthy text layers contain ZERO C0
# control characters, so any occurrence is diagnostic. A page whose
# extracted text crosses either bound below has a lying text layer and
# is rerouted to the VLM lane (the rendered glyphs are fine — the page
# is re-read from pixels). Below both bounds the page stays native and
# the few mojibake units are flagged needs_review.
# Measured on real documents: healthy pages 0 junk chars, broken pages
# 20-451 per page.
TEXT_LAYER_JUNK_MIN = 20  # absolute junk chars per page
TEXT_LAYER_JUNK_RATIO = 0.005  # junk chars / non-whitespace chars

# Printable mojibake (gemini_extractor_spec.md §3): the OTHER broken-CMap
# symptom — glyphs mapping to printable garbage (orphan combining marks,
# ASCII symbols inside non-Latin words), which the junk-char test cannot
# see. Pages crossing either bound reroute to the VLM lane.
# Measured on the real corpus (640 native-eligible pages, 2026-08-11):
# healthy pages score EXACTLY 0 (629/640); broken-CMap pages score 10-20
# (GeM bilingual pages: orphan matras + `म=`/`स]म` interleave; IREPS
# booklet cover: visual-order matras — a break the junk test missed).
# The 1-7 band is empty, so 8 sits below every observed broken page and
# above every healthy one. The ratio bound is for short broken pages a
# small absolute count would miss; healthy pages are all 0, so any
# positive ratio is safe there.
MOJIBAKE_MIN = 8  # absolute mojibake chars per page
MOJIBAKE_RATIO = 0.01  # mojibake chars / non-whitespace chars

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

# ---------------------------------------------------------------------------
# STAGE 5 — VLM extraction (rag_ingest/vlm_extract.py)
# ---------------------------------------------------------------------------

# Engine model id. Verified against ai.google.dev on 2026-08-11:
# gemini-3.6-flash is the current recommended stable Flash model (the
# spec's "gemini-3-flash" exists only as a preview id). Flash tier on
# purpose — benchmarks show no OCR gain from Pro at ~10x the cost.
# Engines are perishable; when this id retires, swap it here and the
# response cache re-keys automatically.
VLM_MODEL = "gemini-3.6-flash"

# Retries on 429/5xx/network errors: exponential backoff with jitter.
# A page exhausting retries becomes one empty needs_review unit — pages
# never silently vanish.
VLM_MAX_RETRIES = 3

# Verification (vlm_extract.verify_page_markdown). VLMs fail as fluent
# lies, not symbol soup — so the checks look for repetition loops,
# implausible lengths, and silent omission rather than junk tokens.
#
# Repetition: a normalized 20-char sequence recurring more than this many
# times is a runaway loop (the documented failure mode emits up to 71x
# the reference length), not natural prose. 10 tolerates boilerplate-
# heavy legal pages while catching real loops, which recur hundreds of
# times.
VLM_MAX_REPEATS = 10

# Length sanity for rerouted lying-CMap pages: the garbled text layer
# still COUNTS characters correctly, so output far shorter (omission) or
# far longer (loop/hallucination) than the layer is suspect. Wide bounds
# on purpose: markdown adds table/heading syntax, and Devanagari-vs-
# mojibake char counts differ legitimately.
VLM_LEN_LO = 0.3
VLM_LEN_HI = 3.0

# Length sanity for true scans (no text layer to count): a render whose
# ink coverage says "dense page" but whose response is near-empty is a
# silent-omission suspect. Ink coverage reuses GRID_DARK_THRESHOLD.
# Measured on the real corpus at RENDER_DPI (2026-08-11): a truly blank
# scanned page = 0.0000; the LIGHTEST content-bearing page = 0.0196
# (645 chars); typical text scans 0.02-0.07. 0.015 sits between blank
# and lightest-content. One outlier: a dark-background cover page hit
# 0.58 with ~214 chars — such pages may flag spuriously, which errs the
# right way (a review flag, not silent garbage).
VLM_DENSE_INK_FRAC = 0.015
VLM_MIN_CHARS_DENSE = 200


# ---------------------------------------------------------------------------
# STAGE 6 — Tables (rag_ingest/tables.py)
# ---------------------------------------------------------------------------

# "Ink" on a 0-255 gray scale: pixels darker than this. Originally the
# tier-2 grid detector's knob; survives as the darkness bound behind
# vlm_extract.ink_fraction() — the length-sanity proxy for true scans
# (VLM spec §5.2).
GRID_DARK_THRESHOLD = 128

# Multi-page continuation: table on page N is a continuation CANDIDATE
# into page N+1 when its bbox bottom reaches below BOTTOM_FRAC of the
# page height AND page N+1 has a table starting above TOP_FRAC. Both
# checks are cheap geometry; the column-count match does the real work.
TABLE_CONT_BOTTOM_FRAC = 0.90
TABLE_CONT_TOP_FRAC = 0.12

# A continuation page often repeats the header row. Rows are compared
# after whitespace normalization with a similarity ratio (not equality:
# paid-lane cells can carry transcription noise). Above this ratio -> treated as a
# repeated header and dropped from the continuation fragment.
HEADER_MATCH_RATIO = 0.8

# Page furniture (ledger #26/#30): headers/footers repeat at the same
# position across pages. Two consumers: (a) text units in the top/bottom
# band whose normalized text repeats on enough pages are stripped before
# chunking; (b) YOLO table *suspects* at the same position on enough
# pages are page decoration (bordered title boxes), not tables — they
# would otherwise each open a needs_review item. Digits are normalized
# out before matching ("Page 5 of 33" == "Page 6 of 33").
FURNITURE_MIN_REPEATS = 3  # distinct pages before something is "repeating"
FURNITURE_BAND_FRAC = 0.2  # top/bottom fraction of page height searched

# ---------------------------------------------------------------------------
# STAGE 7 — Assembly + chunking (rag_ingest/assemble.py, chunking.py)
# ---------------------------------------------------------------------------

# Text chunk sizing, enforced by Chonkie's SentenceChunker with a REAL
# tokenizer — "target 512 tokens" becomes "guaranteed under 512 tokens".
# gpt2 is a proxy tokenizer until the retrieval side picks an embedding
# model; swap CHUNK_TOKENIZER for that model's tokenizer then (different
# tokenizers disagree by ~10-30% on number-dense text).
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 0
CHUNK_TOKENIZER = "gpt2"

# A merged multi-page table can exceed any chunk size. It stays
# LOGICALLY atomic but is emitted as row groups of this many data rows,
# each group repeating the header row, all sharing a table_id — so a
# retrieval hit lands on rows WITH their column meanings intact, and a
# consumer can reassemble the full table by table_id.
TABLE_ROWS_PER_CHUNK = 20

# Heading depth is capped: anything deeper flattens to this level.
# Markdown only renders 6; retrieval gains nothing below that.
MAX_HEADING_LEVEL = 6
