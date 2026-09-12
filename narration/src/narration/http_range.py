"""HTTP Range / 206 for Opus files."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import Request
from fastapi.responses import Response, StreamingResponse

_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


def range_file_response(path: Path, request: Request, *, media_type: str = "audio/opus") -> Response:
    size = path.stat().st_size
    header = request.headers.get("range") or request.headers.get("Range")
    if not header:
        return StreamingResponse(
            path.open("rb"),
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(size),
                "Cache-Control": "private, max-age=3600",
            },
        )
    m = _RANGE.match(header.strip())
    if not m:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    start_s, end_s = m.group(1), m.group(2)
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else size - 1
    if start >= size or end < start:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    end = min(end, size - 1)
    length = end - start + 1

    def chunks():
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                block = fh.read(min(64 * 1024, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block

    return StreamingResponse(
        chunks(),
        status_code=206,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Cache-Control": "private, max-age=3600",
        },
    )
