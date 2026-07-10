from __future__ import annotations

import asyncio

import fakeredis
import pytest_asyncio

from narrator.core.gate import SynthGate


@pytest_asyncio.fixture
async def gate():
    r = fakeredis.FakeAsyncRedis()
    yield SynthGate(r, poll_interval=0.01)
    await r.aclose()


async def test_no_fast_pending_returns_immediately(gate: SynthGate):
    stalled = await asyncio.wait_for(gate.bulk_wait_turn(), timeout=1)
    assert stalled is False


async def test_bulk_parks_until_fast_exits(gate: SynthGate):
    await gate.fast_enter()
    stalls: list[int] = []

    async def on_stall() -> None:
        stalls.append(1)

    task = asyncio.create_task(gate.bulk_wait_turn(on_stall=on_stall))
    await asyncio.sleep(0.05)
    assert not task.done()  # parked while fast pending
    await gate.fast_exit()
    stalled = await asyncio.wait_for(task, timeout=1)
    assert stalled is True
    assert stalls == [1]  # emitted exactly once


async def test_fast_lane_context(gate: SynthGate):
    assert await gate.fast_pending() == 0
    async with gate.fast_lane():
        assert await gate.fast_pending() == 1
    assert await gate.fast_pending() == 0


async def test_fast_preempts_bulk_loop_within_one_iteration(gate: SynthGate):
    processed: list[int] = []

    async def bulk_loop() -> None:
        for i in range(3):
            await gate.bulk_wait_turn()
            processed.append(i)
            await asyncio.sleep(0.01)

    # Fast job arrives before the loop starts its second chunk.
    task = asyncio.create_task(bulk_loop())
    await asyncio.sleep(0.005)
    await gate.fast_enter()
    await asyncio.sleep(0.05)
    # Bulk must have parked - not all chunks processed yet.
    assert len(processed) < 3
    await gate.fast_exit()
    await asyncio.wait_for(task, timeout=1)
    assert processed == [0, 1, 2]
