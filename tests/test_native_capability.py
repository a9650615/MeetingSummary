"""Tests for /native/capability endpoint. Simplified since audiocap was
removed: floatpanel is now the only native-capture front-end (in-process mic
+ system-audio, relayed to /ws/native-capture), so the endpoint just reports
whether floatpanel is installed — no more subprocess-based TCC pre-flighting."""
import backends
from app import create_app
from store import Store


def make_client(tmp_path):
    from fastapi.testclient import TestClient
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "SUMMARY", asr_backend=None)
    return TestClient(app)


def test_capability_no_floatpanel(tmp_path, monkeypatch):
    monkeypatch.setattr(backends, "floatpanel_bin", lambda: None)
    client = make_client(tmp_path)
    r = client.get("/native/capability")
    assert r.status_code == 200
    assert r.json() == {"floatpanel": False}


def test_capability_floatpanel_installed(tmp_path, monkeypatch, tmp_path_factory):
    fake_bin = tmp_path_factory.mktemp("bin") / "floatpanel"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(backends, "floatpanel_bin", lambda: str(fake_bin))
    client = make_client(tmp_path)
    r = client.get("/native/capability")
    assert r.status_code == 200
    assert r.json() == {"floatpanel": True}
