"""Google Drive storage backend (ARCHITECTURE.md sections 11 / 14).

Resumable upload into a cached monthly folder. Every Drive call goes through the
drive retry policy + breaker. ``invalid_grant`` / ``storageQuotaExceeded`` map to
``DeliveryTerminal`` so the job goes UPLOAD_PENDING (downloadable local artifact)
instead of FAILED. Built now but dormant until STORAGE_BACKEND=gdrive and the
one-time OAuth bootstrap (scripts/gdrive_auth.py) provides a refresh token.

The Drive service is injectable so the folder-cache / error-mapping logic is
unit-tested with a fake (no network); the google client libs import lazily.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from narrator.core.breaker import CircuitBreaker
from narrator.core.errors import DeliveryRetryable, DeliveryTerminal
from narrator.core.logging import get_logger
from narrator.core.models import StoredFile
from narrator.core.retry import classify_drive_error, drive_retrying
from narrator.core.state import StateStore

log = get_logger(__name__)

_FOLDER_MIME = "application/vnd.google-apps.folder"
_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


class GoogleDriveStorage:
    def __init__(
        self,
        *,
        breaker: CircuitBreaker,
        state: StateStore,
        root_folder_name: str = "Narrator",
        service: Any = None,
        credentials: dict[str, str] | None = None,
    ) -> None:
        self._breaker = breaker
        self._state = state
        self._root = root_folder_name
        self._service = service
        self._creds = credentials or {}

    # --- service -------------------------------------------------------------
    def _svc(self) -> Any:
        if self._service is None:
            self._service = self._build_service()
        return self._service

    def _build_service(self) -> Any:  # pragma: no cover - needs network/libs
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials(
            None,
            refresh_token=self._creds.get("refresh_token"),
            client_id=self._creds.get("client_id"),
            client_secret=self._creds.get("client_secret"),
            token_uri="https://oauth2.googleapis.com/token",
            scopes=_SCOPES,
        )
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # --- error mapping -------------------------------------------------------
    @staticmethod
    def _classify(exc: Exception) -> None:
        reason = str(exc)
        if "invalid_grant" in reason.lower():
            raise DeliveryTerminal("invalid_grant") from exc
        status = None
        resp = getattr(exc, "resp", None)
        if resp is not None:
            status = getattr(resp, "status", None)
        status = status or getattr(exc, "status_code", None)
        if status is not None:
            try:
                classify_drive_error(int(status), reason)
            except (DeliveryTerminal, DeliveryRetryable):
                raise
        # Unknown Drive failure: treat as transient so it retries then trips.
        raise DeliveryRetryable(reason) from exc

    async def _execute(self, request: Any) -> dict:
        async def _call() -> dict:
            try:
                return await asyncio.to_thread(request.execute)
            except (DeliveryTerminal, DeliveryRetryable):
                raise
            except Exception as exc:
                self._classify(exc)
                raise  # unreachable; _classify always raises

        async with self._breaker.guard():
            return await drive_retrying()(_call)

    async def _execute_resumable(self, request: Any) -> dict:
        response = None

        async def _step() -> tuple[Any, Any]:
            try:
                return await asyncio.to_thread(request.next_chunk)
            except (DeliveryTerminal, DeliveryRetryable):
                raise
            except Exception as exc:
                self._classify(exc)
                raise

        while response is None:
            async with self._breaker.guard():
                _status, response = await drive_retrying()(_step)
        return response

    # --- folders -------------------------------------------------------------
    async def _ensure_named_folder(self, name: str, parent_id: str | None) -> str:
        q = f"name = '{name}' and mimeType = '{_FOLDER_MIME}' and trashed = false"
        if parent_id:
            q += f" and '{parent_id}' in parents"
        res = await self._execute(
            self._svc().files().list(q=q, fields="files(id,name)", spaces="drive")
        )
        files = res.get("files", [])
        if files:
            return files[0]["id"]
        body: dict[str, Any] = {"name": name, "mimeType": _FOLDER_MIME}
        if parent_id:
            body["parents"] = [parent_id]
        created = await self._execute(self._svc().files().create(body=body, fields="id"))
        return created["id"]

    async def resolve_month_folder(self, when: datetime | None = None) -> str:
        now = when or datetime.now(UTC)
        key = f"gdrive:folder:{now:%Y-%m}"
        cached = await self._state.kv_get(key)
        if cached:
            return cached
        root_id = await self._ensure_named_folder(self._root, None)
        month_id = await self._ensure_named_folder(f"{now:%Y-%m}", root_id)
        await self._state.kv_set(key, month_id)
        log.info("storage.gdrive.folder_resolved", folder=f"{now:%Y-%m}", id=month_id)
        return month_id

    # --- upload --------------------------------------------------------------
    def _media(self, local_path: str, mime_type: str) -> Any:  # pragma: no cover
        from googleapiclient.http import MediaFileUpload

        return MediaFileUpload(local_path, mimetype=mime_type, resumable=True, chunksize=8 * 1024 * 1024)

    async def upload(
        self,
        local_path: str,
        *,
        filename: str,
        mime_type: str,
        meta: dict,
    ) -> StoredFile:
        folder_id = meta.get("drive_folder_id") or await self.resolve_month_folder()
        media = self._media(local_path, mime_type)
        request = self._svc().files().create(
            body={"name": filename, "parents": [folder_id]},
            media_body=media,
            fields="id,webViewLink,size",
        )
        result = await self._execute_resumable(request)
        log.info("storage.gdrive.uploaded", filename=filename, file_id=result.get("id"))
        return StoredFile(
            backend="gdrive",
            filename=filename,
            drive_file_id=result.get("id"),
            web_view_link=result.get("webViewLink"),
            size_bytes=int(result.get("size", 0) or 0),
        )
