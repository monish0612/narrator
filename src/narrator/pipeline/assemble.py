"""Assembly stage (ARCHITECTURE.md section 10).

Concatenates chunk wavs + global silence files into a single loudness-normalized
output via one ffmpeg pass. ffmpeg runs in its own process group so cancel /
shutdown can kill the whole group (no zombies). No audio ever enters RAM - the
concat demuxer reads files from disk. Corrupt chunk => one re-synthesis, then a
typed ``AssemblyError``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

from narrator.core.errors import AssemblyError, JobCancelled
from narrator.core.logging import get_logger
from narrator.core.models import Chunk, Manifest
from narrator.pipeline.wavtools import is_valid_wav, write_silence_wav

log = get_logger(__name__)

_FORMAT_ARGS = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "64k", "-id3v2_version", "3"],
    "opus": ["-c:a", "libopus", "-b:a", "32k"],
}
_EXT = {"mp3": "mp3", "opus": "opus"}


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _ffmpeg_bin() -> str:
    return os.getenv("FFMPEG_BIN", "ffmpeg")


def _ffprobe_bin() -> str:
    return os.getenv("FFPROBE_BIN", "ffprobe")


def ffmpeg_available() -> bool:
    return shutil.which(_ffmpeg_bin()) is not None and shutil.which(_ffprobe_bin()) is not None


class AssemblyResult:
    def __init__(self, path: Path, duration_seconds: float, size_bytes: int) -> None:
        self.path = path
        self.duration_seconds = duration_seconds
        self.size_bytes = size_bytes


class Assembler:
    def __init__(self, silence_dir: str | Path) -> None:
        self._silence_dir = Path(silence_dir)

    # --- silence -------------------------------------------------------------
    def ensure_silence(self, ms: int, *, sample_rate: int = 24000) -> Path:
        path = self._silence_dir / f"p{ms}.wav"
        if not path.exists():
            write_silence_wav(path, ms, sample_rate=sample_rate)
        return path

    # --- concat --------------------------------------------------------------
    @staticmethod
    def _concat_quote(path: Path) -> str:
        # ffmpeg concat demuxer: single-quote and escape embedded quotes.
        p = str(path.resolve()).replace("'", "'\\''")
        return f"file '{p}'"

    def build_concat_file(self, chunks: list[Chunk], job_dir: Path) -> Path:
        chunks_dir = Path(job_dir) / "chunks"
        lines: list[str] = []
        for chunk in chunks:
            lines.append(self._concat_quote(chunks_dir / f"{chunk.idx:06d}.wav"))
            if chunk.pause_ms_after > 0:
                lines.append(self._concat_quote(self.ensure_silence(chunk.pause_ms_after)))
        concat_path = Path(job_dir) / "concat.txt"
        concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return concat_path

    # --- validation ----------------------------------------------------------
    async def validate_and_repair(
        self,
        manifest: Manifest,
        job_dir: Path,
        *,
        repair: Callable[[int], Awaitable[None]] | None = None,
    ) -> None:
        chunks_dir = Path(job_dir) / "chunks"
        for entry in manifest.chunks:
            path = chunks_dir / f"{entry.idx:06d}.wav"
            if is_valid_wav(path):
                continue
            if repair is not None:
                log.warning("assemble.repair_chunk", chunk_idx=entry.idx)
                await repair(entry.idx)
            if not is_valid_wav(path):
                raise AssemblyError(f"chunk {entry.idx} invalid after repair")

    # --- ffmpeg --------------------------------------------------------------
    async def _run(
        self,
        args: list[str],
        *,
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> tuple[int, bytes]:
        popen_kwargs: dict = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = 0x00000200  # CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True  # own process group

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **popen_kwargs,
        )
        comm = asyncio.ensure_future(proc.communicate())
        while True:
            done, _ = await asyncio.wait({comm}, timeout=0.5)
            if comm in done:
                break
            if cancel_check is not None and await cancel_check():
                self._kill(proc)
                await comm
                raise JobCancelled("assembly cancelled")
        stdout, stderr = await comm
        return proc.returncode or 0, (stderr or b"")

    @staticmethod
    def _kill(proc: asyncio.subprocess.Process) -> None:
        try:
            if sys.platform == "win32":
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    async def assemble(
        self,
        chunks: list[Chunk],
        manifest: Manifest,
        job_dir: str | Path,
        *,
        title: str,
        output_format: str = "mp3",
        repair: Callable[[int], Awaitable[None]] | None = None,
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> AssemblyResult:
        if not ffmpeg_available():
            raise AssemblyError("ffmpeg/ffprobe not found on PATH")
        if output_format not in _FORMAT_ARGS:
            raise AssemblyError(f"unsupported output format: {output_format}")

        job_dir = Path(job_dir)
        await self.validate_and_repair(manifest, job_dir, repair=repair)
        concat_path = self.build_concat_file(chunks, job_dir)
        out_path = job_dir / f"final.{_EXT[output_format]}"

        args = [
            _ffmpeg_bin(), "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "24000", "-ac", "1",
            *_FORMAT_ARGS[output_format],
            "-metadata", f"title={title}", str(out_path),
        ]
        code, stderr = await self._run(args, cancel_check=cancel_check)
        if code != 0 or not out_path.exists():
            tail = stderr.decode("utf-8", "replace")[-800:]
            raise AssemblyError(f"ffmpeg exit {code}: {tail}")

        duration, size = await self._ffprobe(out_path)
        log.info("assemble.done", duration_seconds=duration, size_bytes=size)
        return AssemblyResult(out_path, duration, size)

    async def _ffprobe(self, path: Path) -> tuple[float, int]:
        args = [
            _ffprobe_bin(), "-v", "quiet", "-print_format", "json",
            "-show_entries", "format=duration,size", str(path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, _ = await proc.communicate()
        fallback_size = await asyncio.to_thread(_file_size, path)
        try:
            fmt = json.loads(out.decode("utf-8")).get("format", {})
            duration = float(fmt.get("duration", 0.0))
            size = int(fmt.get("size", fallback_size))
        except (json.JSONDecodeError, ValueError, KeyError):
            duration, size = 0.0, fallback_size
        return duration, size
