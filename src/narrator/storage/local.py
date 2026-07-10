"""Local storage backend (ARCHITECTURE.md section 11).

Copies the finished artifact into ``/data/outputs/{YYYY}/{MM}/`` and returns a
:class:`StoredFile` whose ``local_path`` the API streams for download. This is
the active backend for the initial deploy (``STORAGE_BACKEND=local``).
"""

from __future__ import annotations

import asyncio
import shutil
from datetime import UTC, datetime
from pathlib import Path

from narrator.core.logging import get_logger
from narrator.core.models import StoredFile

log = get_logger(__name__)


def _copy(src: Path, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return dest.stat().st_size


class LocalStorage:
    """Implements the ``StorageBackend`` protocol against the local filesystem."""

    def __init__(self, outputs_dir: str | Path) -> None:
        self._outputs = Path(outputs_dir)

    async def upload(
        self,
        local_path: str,
        *,
        filename: str,
        mime_type: str,
        meta: dict,
    ) -> StoredFile:
        now = datetime.now(UTC)
        dest = self._outputs / f"{now:%Y}" / f"{now:%m}" / filename
        size = await asyncio.to_thread(_copy, Path(local_path), dest)
        log.info("storage.local.stored", filename=filename, size_bytes=size)
        return StoredFile(
            backend="local",
            filename=filename,
            size_bytes=size,
            local_path=str(dest),
        )
