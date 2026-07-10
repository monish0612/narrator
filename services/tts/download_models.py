"""Download + verify Kokoro model files (ARCHITECTURE.md sections 12 / 15-C).

Resumable download (HTTP Range) into ``MODELS_DIR`` with optional sha256
verification. Defaults point at the official kokoro-onnx v1.0 release assets
(int8 model + voices bin). Runs on container start; the ``tts-models`` volume
makes it a once-ever cost.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.int8.onnx"
)
DEFAULT_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)

_BLOCK = 1024 * 1024  # 1 MB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def _remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download(url: str, dest: Path, *, sha256: str | None = None, max_retries: int = 5) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and sha256 and _sha256(dest) == sha256:
        print(f"[download] {dest.name} present + verified, skipping")
        return dest

    total = _remote_size(url)
    attempt = 0
    while True:
        attempt += 1
        have = dest.stat().st_size if dest.exists() else 0
        if total is not None and have == total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
                mode = "ab" if have and resp.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(dest, mode) as fh:
                    while True:
                        block = resp.read(_BLOCK)
                        if not block:
                            break
                        fh.write(block)
                        have += len(block)
            if total is None or dest.stat().st_size >= total:
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= max_retries:
                raise
            print(f"[download] {dest.name} attempt {attempt} failed ({exc}); resuming")

    if sha256:
        actual = _sha256(dest)
        if actual != sha256:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"sha256 mismatch for {dest.name}: {actual} != {sha256}")
        print(f"[download] {dest.name} verified sha256")
    print(f"[download] {dest.name} ready ({dest.stat().st_size} bytes)")
    return dest


def main() -> int:
    models_dir = Path(os.getenv("MODELS_DIR", "/models"))
    model_url = os.getenv("MODEL_URL") or DEFAULT_MODEL_URL
    voices_url = os.getenv("VOICES_URL") or DEFAULT_VOICES_URL
    model_sha = os.getenv("MODEL_SHA256") or None
    voices_sha = os.getenv("VOICES_SHA256") or None

    model_path = Path(os.getenv("MODEL_PATH", str(models_dir / "kokoro-v1.0.int8.onnx")))
    voices_path = Path(os.getenv("VOICES_PATH", str(models_dir / "voices-v1.0.bin")))

    download(model_url, model_path, sha256=model_sha)
    download(voices_url, voices_path, sha256=voices_sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
