from __future__ import annotations

import fakeredis
import pytest
import pytest_asyncio

from narrator.core.breaker import CLOSED, HALF_OPEN, OPEN, CircuitBreaker
from narrator.core.errors import BreakerOpen, TTSError, UnspeakableError


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest_asyncio.fixture
async def redis():
    r = fakeredis.FakeAsyncRedis()
    yield r
    await r.aclose()


async def _fail(br: CircuitBreaker) -> None:
    async with br.guard():
        raise TTSError("boom")


async def test_success_keeps_closed(redis):
    br = CircuitBreaker(redis, "tts", threshold=3)
    async with br.guard():
        pass
    assert await br.get_state() == CLOSED


async def test_opens_after_threshold(redis):
    clock = FakeClock()
    br = CircuitBreaker(redis, "tts", threshold=3, clock=clock)
    for _ in range(3):
        with pytest.raises(TTSError):
            await _fail(br)
    assert await br.get_state() == OPEN
    # Now blocked without executing the body.
    with pytest.raises(BreakerOpen):
        async with br.guard():
            raise AssertionError("body must not run while open")


async def test_half_open_probe_success_closes(redis):
    clock = FakeClock()
    br = CircuitBreaker(redis, "tts", threshold=2, base_cooldown=60, clock=clock)
    for _ in range(2):
        with pytest.raises(TTSError):
            await _fail(br)
    assert await br.get_state() == OPEN
    clock.advance(61)  # cooldown elapsed
    async with br.guard():
        # This is the single half-open probe; it succeeds.
        assert await br.get_state() == HALF_OPEN
    assert await br.get_state() == CLOSED


async def test_probe_failure_reopens_with_doubled_cooldown(redis):
    clock = FakeClock()
    br = CircuitBreaker(redis, "tts", threshold=2, base_cooldown=60, max_cooldown=600, clock=clock)
    for _ in range(2):
        with pytest.raises(TTSError):
            await _fail(br)
    clock.advance(61)
    with pytest.raises(TTSError):
        await _fail(br)  # probe fails
    assert await br.get_state() == OPEN
    cooldown = await br._get("cooldown")
    assert float(cooldown) == 120.0


async def test_terminal_error_does_not_trip(redis):
    br = CircuitBreaker(redis, "tts", threshold=2)
    for _ in range(5):
        with pytest.raises(UnspeakableError):
            async with br.guard():
                raise UnspeakableError("symbols")
    assert await br.get_state() == CLOSED
