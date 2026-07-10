"""Document ingestion (ARCHITECTURE.md sections 1 / 15-I).

Streams txt/md/pdf into a normalized :class:`Document`. Normalization: NFC,
de-hyphenation across line breaks, header/footer/page-number stripping,
paragraph reflow. Typed 422 errors for the input pathologies we do not handle
on 2 vCPUs (scanned/zero-text PDF -> "needs OCR", encrypted PDF).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

from narrator.core.errors import IngestError
from narrator.core.models import Document

_MIN_PDF_CHARS = 8  # below this we assume no real text layer (scanned)
_PAGE_NUM_RE = re.compile(r"^(?:page\s+)?\d+$", re.IGNORECASE)
_PAGE_DASH_RE = re.compile(r"^[-\u2013\u2014]\s*\d+\s*[-\u2013\u2014]$")
_DEHYPHEN_RE = re.compile(r"(\w)-\n\s*(\w)")


def _is_page_number(line: str) -> bool:
    return bool(_PAGE_NUM_RE.match(line) or _PAGE_DASH_RE.match(line))


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _DEHYPHEN_RE.sub(r"\1\2", text)
    paragraphs = re.split(r"\n\s*\n+", text)
    out: list[str] = []
    for para in paragraphs:
        collapsed = re.sub(r"\s*\n\s*", " ", para)
        collapsed = re.sub(r"[ \t]+", " ", collapsed).strip()
        if collapsed:
            out.append(collapsed)
    return "\n\n".join(out)


def _decode(data: bytes) -> tuple[str, list[str]]:
    warnings: list[str] = []
    try:
        return data.decode("utf-8"), warnings
    except UnicodeDecodeError:
        pass
    replaced = data.decode("utf-8", errors="replace")
    ratio = replaced.count("\ufffd") / max(1, len(replaced))
    try:
        text = data.decode("cp1252")
    except UnicodeDecodeError:
        text = replaced
    if ratio > 0.05:
        warnings.append(f"encoding: {ratio:.0%} of characters could not be decoded as UTF-8")
    return text, warnings


def _strip_repeated_lines(pages: list[str]) -> str:
    n = len(pages)
    firsts: Counter[str] = Counter()
    lasts: Counter[str] = Counter()
    per_page: list[list[str]] = []
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        per_page.append(lines)
        if lines:
            firsts[lines[0]] += 1
            lasts[lines[-1]] += 1
    repeated: set[str] = set()
    if n >= 3:
        threshold = n * 0.5
        repeated.update(line for line, c in firsts.items() if c > threshold)
        repeated.update(line for line, c in lasts.items() if c > threshold)
    rebuilt: list[str] = []
    for lines in per_page:
        kept = [ln for ln in lines if ln not in repeated and not _is_page_number(ln)]
        rebuilt.append("\n".join(kept))
    return "\n\n".join(rebuilt)


def ingest_text(raw: str, *, title: str | None = None, source_type: str = "text") -> Document:
    text = normalize(raw)
    if len(text.strip()) == 0:
        raise IngestError("input contains no readable text")
    return Document(text=text, char_count=len(text), source_type=source_type, title=title)


def ingest_pdf_bytes(data: bytes, *, title: str | None = None) -> Document:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # malformed PDF
        raise IngestError(f"could not open PDF: {exc}") from exc
    try:
        if getattr(doc, "needs_pass", False) or getattr(doc, "is_encrypted", False):
            raise IngestError("encrypted PDF: remove the password and retry")
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    text = normalize(_strip_repeated_lines(pages))
    if len(text.strip()) < _MIN_PDF_CHARS:
        raise IngestError("PDF has no extractable text layer (needs OCR)")
    return Document(text=text, char_count=len(text), source_type="pdf", title=title)


def ingest_file(path: str | Path, *, title: str | None = None) -> Document:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return ingest_pdf_bytes(p.read_bytes(), title=title or p.stem)
    if ext in (".txt", ".md", ""):
        text, warnings = _decode(p.read_bytes())
        doc = ingest_text(text, title=title or p.stem, source_type="txt" if ext != ".md" else "md")
        doc.warnings.extend(warnings)
        return doc
    raise IngestError(f"unsupported file type: {ext}", status_code=415)
