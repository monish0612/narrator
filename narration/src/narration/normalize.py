"""Cache-key hashing and article-text normalization."""

from __future__ import annotations

import hashlib
import re
import unicodedata


_WS = re.compile(r"\s+")


def normalize_article_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return _WS.sub(" ", text).strip()


def build_cache_key(
    *,
    article_text: str,
    voice: str,
    speed: float,
    model_version: str,
    audio_format: str,
    bitrate: int,
) -> str:
    blob = "|".join(
        [
            normalize_article_text(article_text),
            voice.strip(),
            f"{float(speed):.2f}",
            model_version.strip(),
            audio_format.strip().lower(),
            str(int(bitrate)),
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
