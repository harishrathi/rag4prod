"""Pipeline-level guards: clean rejection of undigestible PDFs (ledger #5).

A rejected document must produce a manifest recording status="rejected"
and a human-readable reason — never a raw library traceback, and never a
partial artifact tree that a later --from-stage run would trust.
"""

import json

import pymupdf

from rag_ingest.pipeline import run


def test_corrupt_pdf_rejected_with_manifest(tmp_path):
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    manifest = run(bad, tmp_path / "out")
    assert manifest["status"] == "rejected"
    assert "unreadable" in manifest["reason"]
    on_disk = json.loads((tmp_path / "out" / "corrupt" / "manifest.json").read_text())
    assert on_disk["status"] == "rejected"


def test_encrypted_pdf_rejected_with_manifest(tmp_path):
    locked = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "secret content")
    doc.save(locked, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    doc.close()
    manifest = run(locked, tmp_path / "out")
    assert manifest["status"] == "rejected"
    assert "encrypted" in manifest["reason"]
