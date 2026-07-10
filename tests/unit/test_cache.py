from __future__ import annotations

import asyncio
from pathlib import Path

import fakeredis
import pytest_asyncio

from narrator.core.cache import TTSCache, cache_key
from narrator.core.state import StateStore


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    r = fakeredis.FakeAsyncRedis()
    st = StateStore(tmp_path / "ledger.db", r)
    await st.connect()
    try:
        yield st
    finally:
        await st.close()
        await r.aclose()


def test_key_is_deterministic():
    k1 = cache_key(model="kokoro-int8", voice="af_heart", speed=1.0, sanitized_text="hello")
    k2 = cache_key(model="kokoro-int8", voice="af_heart", speed=1.0, sanitized_text="hello")
    k3 = cache_key(model="kokoro-int8", voice="af_bella", speed=1.0, sanitized_text="hello")
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 64


async def test_hit_avoids_synth_callable(store: StateStore, tmp_path: Path):
    cache = TTSCache(store, tmp_path / "cache", max_gb=5)
    key = cache_key(model="m", voice="v", speed=1.0, sanitized_text="the quick brown fox")
    calls = {"n": 0}

    async def get_or_synth(dest: Path) -> None:
        if await cache.get(key, dest):
            return
        calls["n"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 1200)
        await cache.put(key, dest)

    d1 = tmp_path / "job1" / "000001.wav"
    await get_or_synth(d1)
    assert calls["n"] == 1

    d2 = tmp_path / "job2" / "000001.wav"
    hit = await cache.get(key, d2)
    assert hit is True
    assert d2.read_bytes() == d1.read_bytes()

    d3 = tmp_path / "job3" / "000001.wav"
    await get_or_synth(d3)
    assert calls["n"] == 1  # still cached, no new synthesis


async def test_eviction_respects_lru_order(store: StateStore, tmp_path: Path):
    # ~2 KB budget so only two 1 KB entries survive.
    cache = TTSCache(store, tmp_path / "cache", max_gb=2048 / (1024**3))
    keys = []
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    for i in range(3):
        key = cache_key(model="m", voice="v", speed=1.0, sanitized_text=f"chunk-{i}")
        keys.append(key)
        src = src_dir / f"{i}.wav"
        src.write_bytes(b"\x00" * 1024)
        await cache.put(key, src)
        await asyncio.sleep(0.01)  # ensure distinct last_used ordering

    evicted = await cache.evict_to_limit()
    assert evicted == 1
    # Oldest (keys[0]) evicted; newest two remain.
    assert not cache.path_for(keys[0]).exists()
    assert await store.cache_get(keys[0]) is None
    assert cache.path_for(keys[1]).exists()
    assert cache.path_for(keys[2]).exists()
