from __future__ import annotations

from narration.http_range import range_file_response


class _Req:
    def __init__(self, range_header: str | None = None) -> None:
        self.headers = {"range": range_header} if range_header else {}


def test_full_body_when_no_range(tmp_path):
    p = tmp_path / "a.opus"
    p.write_bytes(b"0123456789")
    resp = range_file_response(p, _Req())
    assert resp.status_code == 200
    assert resp.headers["Accept-Ranges"] == "bytes"
    assert resp.headers["Content-Length"] == "10"


def test_bytes_zero_open_end(tmp_path):
    p = tmp_path / "a.opus"
    p.write_bytes(b"0123456789")
    resp = range_file_response(p, _Req("bytes=0-"))
    assert resp.status_code == 206
    assert resp.headers["Content-Range"] == "bytes 0-9/10"
    assert resp.headers["Content-Length"] == "10"


def test_mid_seek(tmp_path):
    p = tmp_path / "a.opus"
    p.write_bytes(b"0123456789")
    resp = range_file_response(p, _Req("bytes=4-7"))
    assert resp.status_code == 206
    assert resp.headers["Content-Range"] == "bytes 4-7/10"
    assert resp.headers["Content-Length"] == "4"


def test_suffix_is_last_n_bytes(tmp_path):
    p = tmp_path / "a.opus"
    p.write_bytes(b"0123456789")
    resp = range_file_response(p, _Req("bytes=-4"))
    assert resp.status_code == 206
    assert resp.headers["Content-Range"] == "bytes 6-9/10"
    assert resp.headers["Content-Length"] == "4"


def test_suffix_longer_than_file(tmp_path):
    p = tmp_path / "a.opus"
    p.write_bytes(b"0123456789")
    resp = range_file_response(p, _Req("bytes=-99"))
    assert resp.status_code == 206
    assert resp.headers["Content-Range"] == "bytes 0-9/10"


def test_unsatisfiable_and_malformed(tmp_path):
    p = tmp_path / "a.opus"
    p.write_bytes(b"0123456789")
    assert range_file_response(p, _Req("bytes=99-")).status_code == 416
    assert range_file_response(p, _Req("bytes=8-2")).status_code == 416
    assert range_file_response(p, _Req("bytes=abc")).status_code == 416
    assert range_file_response(p, _Req("bytes=-0")).status_code == 416
