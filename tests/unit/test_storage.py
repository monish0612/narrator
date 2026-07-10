from __future__ import annotations

from pathlib import Path

import fakeredis
import pytest
import pytest_asyncio

from narrator.core.breaker import CircuitBreaker
from narrator.core.errors import DeliveryTerminal
from narrator.core.state import StateStore
from narrator.storage.gdrive import GoogleDriveStorage
from narrator.storage.local import LocalStorage


@pytest_asyncio.fixture
async def state(tmp_path: Path):
    r = fakeredis.FakeAsyncRedis()
    st = StateStore(tmp_path / "ledger.db", r)
    await st.connect()
    try:
        yield st
    finally:
        await st.close()
        await r.aclose()


# --- local -------------------------------------------------------------------
async def test_local_delivers_and_returns_path(tmp_path: Path):
    src = tmp_path / "final.mp3"
    src.write_bytes(b"ID3" + b"\x00" * 2048)
    storage = LocalStorage(tmp_path / "outputs")
    stored = await storage.upload(str(src), filename="my_narration.mp3", mime_type="audio/mpeg", meta={})
    assert stored.backend == "local"
    assert stored.local_path is not None
    assert Path(stored.local_path).exists()
    assert stored.size_bytes == src.stat().st_size


# --- drive (fake service) ----------------------------------------------------
class _Req:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    def execute(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class _Files:
    def __init__(self, list_exc=None):
        self.list_calls = 0
        self.create_calls = 0
        self._list_exc = list_exc

    def list(self, q=None, fields=None, spaces=None):
        self.list_calls += 1
        if self._list_exc is not None:
            return _Req(exc=self._list_exc)
        return _Req(result={"files": []})  # nothing exists yet

    def create(self, body=None, fields=None, media_body=None):
        self.create_calls += 1
        return _Req(result={"id": f"folder-{self.create_calls}", "webViewLink": "http://d/x", "size": "10"})


class _Service:
    def __init__(self, files: _Files):
        self._files = files

    def files(self):
        return self._files


def _breaker() -> CircuitBreaker:
    return CircuitBreaker(fakeredis.FakeAsyncRedis(), "drive")


async def test_drive_folder_cache(state: StateStore):
    files = _Files()
    storage = GoogleDriveStorage(breaker=_breaker(), state=state, service=_Service(files))

    first = await storage.resolve_month_folder()
    assert files.create_calls == 2  # root + month
    assert files.list_calls == 2

    second = await storage.resolve_month_folder()
    assert second == first
    # Second call is served from the kv cache: no extra Drive traffic.
    assert files.create_calls == 2
    assert files.list_calls == 2


async def test_drive_invalid_grant_is_terminal(state: StateStore):
    files = _Files(list_exc=Exception("invalid_grant: token revoked"))
    storage = GoogleDriveStorage(breaker=_breaker(), state=state, service=_Service(files))
    with pytest.raises(DeliveryTerminal) as ei:
        await storage.resolve_month_folder()
    assert "invalid_grant" in str(ei.value)
