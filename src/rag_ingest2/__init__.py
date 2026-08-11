"""rag_ingest2 — the v2 pipeline (docs/rewrite_design.md).

Coexists with v1 (``rag_ingest``) during the migration overlap; shares
only ``models.py``, ``vlm_extract.py``, and ``text_quality.py`` with it
(rewrite_design.md §4). Everything else is rebuilt along the eight-layer
architecture:

    0  ingest gate      reject encrypted / corrupt / zero-page
    1  page profiling   evidence per page, no decisions      (profiles.py)
    2  routing          PageProfile -> PageRoute, pure       (routing.py)
    3  extraction       per (page, route) workers            (workers.py)
    4  table ladder     extractors behind validation gates   (tables/)
    5  doc normalization  whole-document passes              (normalize.py)
    6  quality gate     one pass derives needs_review        (quality.py)
    7  chunking         unchanged behind split_text()        (chunking.py)

The orchestrator (pipeline.py + stages.py) owns run-vs-load, timing,
artifact I/O, and config snapshotting uniformly.
"""
