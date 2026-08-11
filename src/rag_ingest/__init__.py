"""PDF ingestion pipeline for RAG — PDF -> retrieval-ready chunks.

Stages (one module each, orchestrated by pipeline.py):

    1. triage.py         which pages have a usable text layer?      [Phase 1]
    2. local_extract.py  free, exact extraction on text pages       [Phase 2]
    3. render.py         page images for the vision paths           [Phase 3]
    4. layout.py         YOLO region detection + coord conversion   [Phase 3]
    5. gemini_client.py  vision extraction of tables/scanned pages  [Phase 4]
    6. stitch.py         multi-page table merging                   [Phase 5]
       assemble.py       merge walk + heading normalization         [Phase 5]
       chunking.py       per-section chunking, row-group tables     [Phase 5]
"""
