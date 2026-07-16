from fastapi import FastAPI
from fastapi.testclient import TestClient
from store import Store
from viewer.routes import mount_viewer


def _app(tmp_path):
    store = Store(tmp_path / "s.db")
    mid = store.create_meeting("週會", 1721111111.0, "zh-TW")
    store.add_segment(mid, 0, "", 1721111111.0, 5.0, "live")
    store.add_transcript(mid, "accurate", "mixed", 0, 1000, "我", "hello")
    store.finalize_meeting(mid)
    data = tmp_path / "data"; (data / str(mid)).mkdir(parents=True)
    (data / str(mid) / "mixed.m4a").write_bytes(b"AUDIOBYTES")
    app = FastAPI(); mount_viewer(app, store, str(data))
    return TestClient(app), mid


def test_index_and_detail(tmp_path):
    c, mid = _app(tmp_path)
    assert "週會" in c.get("/").text
    assert "hello" in c.get(f"/m/{mid}").text
    assert c.get("/m/9999").status_code == 404


def test_audio_served_with_range(tmp_path):
    c, mid = _app(tmp_path)
    r = c.get(f"/meetings/{mid}/audio/mixed.m4a")
    assert r.status_code == 200 and r.content == b"AUDIOBYTES"
    rng = c.get(f"/meetings/{mid}/audio/mixed.m4a", headers={"Range": "bytes=0-3"})
    assert rng.status_code == 206 and rng.content == b"AUDI"
    assert c.get(f"/meetings/{mid}/audio/nope.m4a").status_code == 404


def test_export_download(tmp_path):
    c, mid = _app(tmp_path)
    r = c.get(f"/meetings/{mid}/export")
    assert r.status_code == 200 and "我：hello" in r.text
