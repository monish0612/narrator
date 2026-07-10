"""Test bootstrap: set the minimal required env before narrator is imported.

``narrator.core.config`` validates fail-fast at import, so required env vars
must exist before the first import in the test process.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("API_KEYS", "test-key-1,test-key-2")
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("PROCESSOR_DEFAULT", "verbatim")
os.environ.setdefault("DATA_DIR", str(Path(tempfile.gettempdir()) / "narrator-test-data"))
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TTS_BASE_URL", "http://tts:8880/v1")
