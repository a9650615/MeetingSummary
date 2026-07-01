"""Tests for /native/capability endpoint."""
import subprocess

import pytest
from fastapi.testclient import TestClient

import backends
from app import create_app
from store import Store


def make_client(tmp_path):
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "SUMMARY", asr_backend=None)
    return TestClient(app)


def test_capability_no_helper(tmp_path, monkeypatch):
    # When audiocap binary is absent, both flags are False.
    monkeypatch.setattr(backends, "audiocap_bin", lambda: None)
    client = make_client(tmp_path)
    r = client.get("/native/capability")
    assert r.status_code == 200
    body = r.json()
    assert body == {"audiocap": False, "granted": False}


def test_capability_helper_granted(tmp_path, monkeypatch, tmp_path_factory):
    # audiocap present, --check exits 0 and prints GRANTED.
    fake_bin = tmp_path_factory.mktemp("bin") / "audiocap"
    fake_bin.write_text("#!/bin/sh\necho GRANTED\nexit 0\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(backends, "audiocap_bin", lambda: str(fake_bin))
    client = make_client(tmp_path)
    r = client.get("/native/capability")
    assert r.status_code == 200
    body = r.json()
    assert body == {"audiocap": True, "granted": True}


def test_capability_helper_denied(tmp_path, monkeypatch, tmp_path_factory):
    # audiocap present, --check exits 1 and prints DENIED.
    fake_bin = tmp_path_factory.mktemp("bin") / "audiocap"
    fake_bin.write_text("#!/bin/sh\necho DENIED\nexit 1\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(backends, "audiocap_bin", lambda: str(fake_bin))
    client = make_client(tmp_path)
    r = client.get("/native/capability")
    assert r.status_code == 200
    body = r.json()
    assert body == {"audiocap": True, "granted": False}


def test_capability_helper_timeout(tmp_path, monkeypatch, tmp_path_factory):
    # Old helper without --check would hang (sleep); must return granted=False within timeout.
    fake_bin = tmp_path_factory.mktemp("bin") / "audiocap"
    fake_bin.write_text("#!/bin/sh\nsleep 60\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(backends, "audiocap_bin", lambda: str(fake_bin))
    # Patch timeout to 0.1s so the test doesn't block.
    import app as _app
    orig_popen = subprocess.Popen

    class FastTimeoutPopen(orig_popen):
        def communicate(self, input=None, timeout=None):  # noqa: A002
            return super().communicate(input=input, timeout=0.1)

    monkeypatch.setattr(subprocess, "Popen", FastTimeoutPopen)
    client = make_client(tmp_path)
    r = client.get("/native/capability")
    assert r.status_code == 200
    body = r.json()
    assert body["audiocap"] is True
    assert body["granted"] is False
