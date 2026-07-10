from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from narrator.core.errors import AssemblyError
from narrator.core.models import Chunk, Manifest, ManifestChunk, ManifestEngine
from narrator.pipeline.assemble import Assembler, ffmpeg_available
from narrator.pipeline.wavtools import read_wav_duration_ms


def _wav_bytes(ms: int, rate: int = 24000) -> bytes:
    n = int(rate * ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


def _chunks() -> list[Chunk]:
    return [
        Chunk(idx=1, text="one", sha256="a", pause_ms_after=600),
        Chunk(idx=2, text="two", sha256="b", pause_ms_after=0),
    ]


def _manifest() -> Manifest:
    return Manifest(
        job_id="jb_1",
        engine=ManifestEngine(base_url="", model="m", voice="af_heart", speed=1.0),
        chunks=[
            ManifestChunk(idx=1, sha256="a", chars=3, status="done", file="chunks/000001.wav"),
            ManifestChunk(idx=2, sha256="b", chars=3, status="done", file="chunks/000002.wav"),
        ],
        created_at="t",
        updated_at="t",
    )


def _write_chunks(job_dir: Path, *, valid=(1, 2)) -> None:
    cdir = job_dir / "chunks"
    cdir.mkdir(parents=True, exist_ok=True)
    for idx in (1, 2):
        data = _wav_bytes(500) if idx in valid else b"RIFF"
        (cdir / f"{idx:06d}.wav").write_bytes(data)


def test_ensure_silence_generates_correct_duration(tmp_path: Path):
    asm = Assembler(tmp_path / "silence")
    p = asm.ensure_silence(600)
    assert p.exists()
    assert abs(read_wav_duration_ms(p) - 600) <= 5
    # idempotent
    assert asm.ensure_silence(600) == p


def test_build_concat_interleaves_silence(tmp_path: Path):
    asm = Assembler(tmp_path / "silence")
    _write_chunks(tmp_path)
    concat = asm.build_concat_file(_chunks(), tmp_path)
    lines = concat.read_text().strip().splitlines()
    # chunk1, silence(600), chunk2  => 3 lines (chunk2 has no pause)
    assert len(lines) == 3
    assert "000001.wav" in lines[0]
    assert "p600.wav" in lines[1]
    assert "000002.wav" in lines[2]


async def test_validate_and_repair_invokes_repair(tmp_path: Path):
    asm = Assembler(tmp_path / "silence")
    _write_chunks(tmp_path, valid=(1,))  # chunk 2 corrupt

    repaired: list[int] = []

    async def repair(idx: int) -> None:
        repaired.append(idx)
        (tmp_path / "chunks" / f"{idx:06d}.wav").write_bytes(_wav_bytes(500))

    await asm.validate_and_repair(_manifest(), tmp_path, repair=repair)
    assert repaired == [2]


async def test_validate_and_repair_raises_when_unrepairable(tmp_path: Path):
    asm = Assembler(tmp_path / "silence")
    _write_chunks(tmp_path, valid=(1,))

    async def repair(idx: int) -> None:
        pass  # fails to fix

    with pytest.raises(AssemblyError):
        await asm.validate_and_repair(_manifest(), tmp_path, repair=repair)


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not installed")
async def test_ffmpeg_end_to_end(tmp_path: Path):
    asm = Assembler(tmp_path / "silence")
    _write_chunks(tmp_path)
    result = await asm.assemble(
        _chunks(), _manifest(), tmp_path, title="Test", output_format="mp3"
    )
    assert result.path.exists()
    assert result.path.suffix == ".mp3"
    # two 500 ms chunks + one 600 ms pause ~= 1.6 s
    assert result.duration_seconds > 1.0
    assert result.size_bytes > 0
