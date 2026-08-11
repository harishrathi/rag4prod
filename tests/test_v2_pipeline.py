"""v2 integration: profiling + routing against the synthetic sample's
known ground truth, the Layer-0 gate, and an offline end-to-end run.

The e2e fixture DISABLES the VLM engine explicitly (it must not depend
on whether a real GEMINI_API_KEY happens to be configured): VLM-routed
pages must degrade to flagged empty units (never crash, never vanish)
while the free lane produces the same content v1's tests assert on."""

import json

import pymupdf
import pytest

from rag_ingest.models import PageKind
from rag_ingest.sample_pdf import EXPECTED_KINDS, build_sample
from rag_ingest2.config import IngestConfig, RoutingRules
from rag_ingest2.pipeline import run
from rag_ingest2.profiles import profile_document
from rag_ingest2.routing import Extractor, route_document

# v1 PageKind ground truth -> v2 Extractor ground truth.
KIND_TO_EXTRACTOR = {
    PageKind.TEXT_NATIVE: Extractor.NATIVE,
    PageKind.SCANNED: Extractor.VLM,
    PageKind.DRAWING: Extractor.DRAWING,
}


@pytest.fixture(scope="module")
def sample_path(tmp_path_factory):
    return build_sample(tmp_path_factory.mktemp("pdf") / "sample_doc.pdf")


def test_every_page_routed_correctly(sample_path):
    doc = pymupdf.open(sample_path)
    routes = route_document(profile_document(doc, RoutingRules()), RoutingRules())
    doc.close()
    got = {r.page: r.extractor for r in routes}
    assert got == {p: KIND_TO_EXTRACTOR[k] for p, k in EXPECTED_KINDS.items()}


def test_header_over_scan_trap_caught_by_coverage(sample_path):
    doc = pymupdf.open(sample_path)
    profiles = profile_document(doc, RoutingRules())
    doc.close()
    trap = profiles[3]
    # The trap is only meaningful if the text layer alone WOULD have passed.
    assert trap.text_chars >= 50
    assert trap.max_image_coverage > 0.7
    r = route_document(profiles, RoutingRules())[3]
    assert r.extractor == Extractor.VLM and "raster" in r.reasons[0]


def test_corrupt_pdf_rejected_with_manifest(tmp_path):
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    manifest = run(bad, tmp_path / "out")
    assert manifest["status"] == "rejected"
    assert "unreadable" in manifest["reason"]


def test_encrypted_pdf_rejected_with_manifest(tmp_path):
    locked = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret content")
    doc.save(locked, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    doc.close()
    manifest = run(locked, tmp_path / "out")
    assert manifest["status"] == "rejected"
    assert "encrypted" in manifest["reason"]


@pytest.fixture(scope="module")
def e2e(sample_path, tmp_path_factory):
    from rag_ingest.vlm_extract import GeminiClient, VlmError

    def _engine_disabled(self, png, prompt):
        raise VlmError("offline test: engine disabled")

    mp = pytest.MonkeyPatch()
    mp.setattr(GeminiClient, "generate", _engine_disabled)
    try:
        out = tmp_path_factory.mktemp("out2")
        manifest = run(sample_path, out, cfg=IngestConfig(debug_images=False))
    finally:
        mp.undo()
    return manifest, out / "sample_doc"


def test_e2e_route_counts_match_ground_truth(e2e):
    manifest, _ = e2e
    assert manifest["routes"] == {"native": 5, "vlm": 4, "drawing": 1}
    assert manifest["status"] == "ok"


def test_e2e_native_content_reaches_chunks(e2e):
    _, doc_out = e2e
    merged = (doc_out / "merged.md").read_text(encoding="utf-8")
    assert "supplier shall complete" in merged  # native prose (p0)
    assert "# 7. Payment Terms" in merged  # heading with resolved level
    assert "Excavation" in merged  # native table content (p6)


def test_e2e_multipage_table_stitched(e2e):
    _, doc_out = e2e
    normalized = json.loads((doc_out / "stages" / "05_normalized.json").read_text("utf-8"))
    multi = [t for t in normalized["tables"] if len(t["pages"]) > 1]
    assert len(multi) == 1
    t = multi[0]
    assert t["pages"] == [8, 9]
    # header + 6 rows (p8) + 4 rows (p9), repeated header dropped
    assert t["row_count"] == 11
    assert any("Roofing" in " ".join(r) for r in t["cells"])  # p9 row survived the merge


def test_e2e_offline_vlm_pages_flagged_never_lost(e2e):
    manifest, doc_out = e2e
    # No API key: all 4 VLM pages fail permanently -> one empty flagged
    # unit each, visible in the review report and chunk flags.
    reviewed = json.loads((doc_out / "stages" / "06_reviewed.json").read_text("utf-8"))
    vlm_units = [u for u in reviewed["units"] if u["source"] == "gemini"]
    assert len(vlm_units) == 4
    assert all(u["needs_review"] for u in vlm_units)
    assert {u["page"] for u in vlm_units} == {2, 3, 5, 7}
    assert (doc_out / "review_report.md").exists()
    assert manifest["review_items"] >= 4


def test_e2e_config_snapshot_in_manifest(e2e):
    manifest, _ = e2e
    assert manifest["config"]["routing"]["mojibake_min"] == 8
    assert manifest["config"]["chunking"]["size_tokens"] == 512
