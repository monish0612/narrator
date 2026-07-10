from __future__ import annotations

import pymupdf
import pytest

from narrator.core.errors import IngestError
from narrator.pipeline.ingest import ingest_pdf_bytes, ingest_text, normalize


def _pdf_with_text(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _blank_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def _encrypted_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "confidential contents here")
    data = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="secret",
        owner_pw="secret",
    )
    doc.close()
    return data


def test_normalize_dehyphenates_and_reflows():
    raw = "This sen-\ntence was split.\nSame paragraph.\n\nNext paragraph."
    out = normalize(raw)
    assert "sentence" in out
    assert out.count("\n\n") == 1  # two paragraphs
    assert "\n" not in out.split("\n\n")[0]  # intra-paragraph newlines removed


def test_ingest_text_empty_raises():
    with pytest.raises(IngestError):
        ingest_text("   \n\n   ")


def test_pdf_with_text_ok():
    doc = ingest_pdf_bytes(_pdf_with_text("Hello world. This is a real text layer."))
    assert doc.source_type == "pdf"
    assert "Hello world" in doc.text
    assert doc.char_count > 0


def test_scanned_pdf_needs_ocr_422():
    with pytest.raises(IngestError) as ei:
        ingest_pdf_bytes(_blank_pdf())
    assert ei.value.status_code == 422
    assert "OCR" in str(ei.value)


def test_encrypted_pdf_422():
    with pytest.raises(IngestError) as ei:
        ingest_pdf_bytes(_encrypted_pdf())
    assert ei.value.status_code == 422
    assert "encrypted" in str(ei.value).lower()
