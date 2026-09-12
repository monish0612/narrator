"""Host RAM gate. Reads /proc/meminfo (host view on Linux Docker)."""

from __future__ import annotations

from pathlib import Path

MEMINFO = Path("/proc/meminfo")


def mem_available_bytes(source: str | None = None) -> int | None:
    raw = source
    if raw is None:
        try:
            raw = MEMINFO.read_text(encoding="utf-8")
        except OSError:
            return None
    for line in raw.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                unit = parts[2].lower() if len(parts) > 2 else "kb"
                kb = int(parts[1])
                if unit == "kb":
                    return kb * 1024
                if unit == "mb":
                    return kb * 1024 * 1024
                return kb
    return None


def should_defer(floor_bytes: int, *, available: int | None = None) -> bool:
    avail = mem_available_bytes() if available is None else available
    if avail is None:
        return False
    return avail < floor_bytes


def next_backoff_s(defer_count: int) -> int:
    steps = (30, 60, 120, 240, 480)
    if defer_count <= 0:
        return steps[0]
    idx = min(defer_count, len(steps) - 1)
    return steps[idx]
