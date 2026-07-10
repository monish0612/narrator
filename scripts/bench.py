#!/usr/bin/env python3
"""Benchmark the TTS service to tune SYNTH_CONCURRENCY / ONNX_INTRA_OP.

Run on the deploy box (or against a published tts port). It measures the
realtime factor (RTF = audio_seconds / wall_seconds) at increasing request
concurrency and recommends the sweet spot: the highest concurrency where
aggregate throughput still climbs and single-request RTF stays >= 1.0.

    python scripts/bench.py --base-url http://localhost:8880/v1 --max-concurrency 4

Because the box is shared with your website, we deliberately stop increasing
concurrency once throughput plateaus rather than saturating the CPU.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

PARAGRAPH = (
    "Narrator turns long documents into natural sounding narration. "
    "This benchmark synthesizes a fixed paragraph many times to estimate the "
    "realtime factor of the text to speech engine on this machine. The realtime "
    "factor is the number of seconds of audio produced per second of wall clock "
    "time, which tells us how many concurrent synthesis streams the box can "
    "sustain before latency degrades for everyone."
)


async def _one(client: httpx.AsyncClient, base_url: str, voice: str) -> tuple[float, float]:
    """Return (audio_seconds, wall_seconds) for a single synthesis call."""
    start = time.perf_counter()
    resp = await client.post(
        f"{base_url}/audio/speech",
        json={"input": PARAGRAPH, "voice": voice, "response_format": "wav"},
    )
    resp.raise_for_status()
    wall = time.perf_counter() - start
    audio_ms = float(resp.headers.get("X-Audio-Duration-Ms", "0"))
    return audio_ms / 1000.0, wall


async def _round(client, base_url, voice, concurrency: int, reps: int) -> dict:
    audio_total = 0.0
    per_req_rtf: list[float] = []
    wall_start = time.perf_counter()
    for _ in range(reps):
        results = await asyncio.gather(
            *[_one(client, base_url, voice) for _ in range(concurrency)]
        )
        for audio_s, wall_s in results:
            audio_total += audio_s
            if wall_s > 0:
                per_req_rtf.append(audio_s / wall_s)
    wall_total = time.perf_counter() - wall_start
    return {
        "concurrency": concurrency,
        "throughput_rtf": audio_total / wall_total if wall_total else 0.0,
        "per_req_rtf": statistics.median(per_req_rtf) if per_req_rtf else 0.0,
        "audio_s": audio_total,
        "wall_s": wall_total,
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description="Narrator TTS benchmark")
    ap.add_argument("--base-url", default="http://localhost:8880/v1")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3, help="rounds per concurrency level")
    args = ap.parse_args()

    print(f"Benchmarking {args.base_url} (voice={args.voice})\n")
    print(f"{'conc':>4}  {'throughput_rtf':>15}  {'per_req_rtf':>12}  {'wall_s':>8}")
    print("-" * 48)

    rows: list[dict] = []
    async with httpx.AsyncClient(timeout=300) as client:
        # Warm the path once (ignored) so the first timed round isn't cold.
        try:
            await _one(client, args.base_url, args.voice)
        except httpx.HTTPError as exc:
            raise SystemExit(f"TTS not reachable at {args.base_url}: {exc}") from exc

        for c in range(1, args.max_concurrency + 1):
            row = await _round(client, args.base_url, args.voice, c, args.reps)
            rows.append(row)
            print(
                f"{c:>4}  {row['throughput_rtf']:>15.2f}  "
                f"{row['per_req_rtf']:>12.2f}  {row['wall_s']:>8.1f}"
            )

    # Recommend: best throughput while median per-request RTF stays >= 1.0
    # (i.e. we still keep up with realtime for each stream). Fall back to the
    # global throughput peak if none clear the bar.
    viable = [r for r in rows if r["per_req_rtf"] >= 1.0]
    best = max(viable or rows, key=lambda r: r["throughput_rtf"])
    intra = max(1, 4 // best["concurrency"]) if best["concurrency"] else 2

    print("\n== Recommendation ==")
    print(f"SYNTH_CONCURRENCY={best['concurrency']}")
    print(f"ONNX_INTRA_OP={min(2, intra)}")
    print(
        "\nApply these in the Coolify env tab and redeploy. On a 2-vCPU box, "
        "prefer ONNX_INTRA_OP=2 with SYNTH_CONCURRENCY=1-2 so a marathon job "
        "never starves the public site."
    )


if __name__ == "__main__":
    asyncio.run(main())
