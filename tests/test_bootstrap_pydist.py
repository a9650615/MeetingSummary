import io
import json
import os
import tarfile
import urllib.request

import bootstrap


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_tarball():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("python/bin/python3")
        data = b"#!/bin/sh\necho fake python\n"
        info.size = len(data)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_fetch_pydist_skips_on_non_apple_silicon(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "_is_apple_silicon_hw", lambda: False)

    def boom(*a, **k):
        raise AssertionError("must not hit network on non-arm64")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert bootstrap._fetch_pydist() is False


def test_fetch_pydist_false_when_asset_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "_is_apple_silicon_hw", lambda: True)
    release = {"tag_name": "v0.2.0", "assets": [{"name": "other.tar.gz", "browser_download_url": "https://x/o"}]}
    monkeypatch.setattr(urllib.request, "urlopen",
                         lambda req, timeout=10: _Resp(json.dumps(release).encode()))
    assert bootstrap._fetch_pydist() is False


def test_fetch_pydist_extracts_as_venv_and_writes_reqhash(monkeypatch, tmp_path):
    (tmp_path / "requirements-app.txt").write_text("fastapi\nuvicorn\n")
    monkeypatch.setattr(bootstrap, "HERE", str(tmp_path))
    monkeypatch.setattr(bootstrap, "REQ", "requirements-app.txt")
    monkeypatch.setattr(bootstrap, "_is_apple_silicon_hw", lambda: True)

    release = {
        "tag_name": "v0.2.0",
        "assets": [{"name": "pydist-arm64.tar.gz", "browser_download_url": "https://x/pydist-arm64.tar.gz"}],
    }
    tar_bytes = _fake_tarball()
    calls = {"n": 0}

    def fake_urlopen(req, timeout=10):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(json.dumps(release).encode())
        return _Resp(tar_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert bootstrap._fetch_pydist() is True

    venv = tmp_path / ".venv"
    assert (venv / "bin" / "python3").is_file()
    assert (venv / "bin" / "python").is_symlink()
    assert os.readlink(venv / "bin" / "python") == "python3"
    assert (venv / ".reqhash").exists()
    assert not (tmp_path / ".venv.tmp").exists()
