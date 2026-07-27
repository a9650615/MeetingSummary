"""Session modes and pause, end to end over /ws/native-capture.

both      = record + live ASR (the default)
record    = 純錄音, PCM saved, no inference
transcribe = 純字幕, captions only, NOTHING written to disk
pause     = keep the session/meeting open but stop consuming audio
"""
import struct
import time

from fastapi.testclient import TestClient

import recorder
from app import create_app
from store import Store
from tests.test_live_start import _FakeLiveManager, _wait_until


def _frames(n=5):
    return (struct.pack("<BI", recorder.TRACK_SYSTEM, 2) + b"\x00\x00") * n


def test_transcribe_mode_keeps_no_audio_and_no_segment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "x", asr_backend=None,
                     live_manager=_FakeLiveManager())
    with TestClient(app) as c:
        with c.websocket_connect(
                "/ws/native-capture?source=system&mode=transcribe") as ws:
            mid = ws.receive_json()["id"]
            ws.send_bytes(_frames())
            assert _wait_until(lambda: c.get("/live/state").json()["recording"] is True)
        assert _wait_until(lambda: c.get("/live/state").json()["recording"] is False)

    assert list((tmp_path / "data").glob("*/*.pcm")) == []  # nothing recorded
    assert store.list_segments(mid) == []                   # and no phantom segment


def test_default_mode_still_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "x", asr_backend=None,
                     live_manager=_FakeLiveManager())
    with TestClient(app) as c:
        with c.websocket_connect("/ws/native-capture?source=system") as ws:
            mid = ws.receive_json()["id"]
            ws.send_bytes(_frames())
            assert _wait_until(lambda: c.get("/live/state").json()["recording"] is True)
        assert _wait_until(lambda: c.get("/live/state").json()["recording"] is False)

    assert list((tmp_path / "data").glob("*/system.pcm"))
    assert len(store.list_segments(mid)) == 1


def test_live_mode_setting_is_validated(tmp_path):
    store = Store(tmp_path / "m.db")  # not ":memory:" — Store is per-thread, so an
    # in-memory DB gives the threadpool a fresh empty one (no settings table)
    app = create_app(store, summary_backend=lambda p: "x", asr_backend=None)
    with TestClient(app) as c:
        assert c.get("/settings/live_mode").json()["value"] == "both"
        assert c.post("/settings/live_mode",
                      json={"value": "transcribe"}).json()["value"] == "transcribe"
        # anything unknown falls back to the safe default (still records)
        assert c.post("/settings/live_mode",
                      json={"value": "nonsense"}).json()["value"] == "both"


def test_pause_endpoint_marks_the_session_and_survives_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "x", asr_backend=None,
                     live_manager=_FakeLiveManager())
    with TestClient(app) as c:
        assert c.post("/live/pause").json() == {"paused": True, "sessions": 0}
        with c.websocket_connect("/ws/native-capture?source=system") as ws:
            ws.receive_json()
            ws.send_bytes(_frames())
            assert _wait_until(lambda: c.get("/live/state").json()["recording"] is True)
            # a pause set before this session began must NOT leak into it
            assert c.get("/live/state").json()["paused"] is False

            assert c.post("/live/pause").json()["sessions"] == 1
            assert _wait_until(lambda: c.get("/live/state").json()["paused"] is True)
            assert c.get("/live/state").json()["recording"] is True  # still live

            c.post("/live/pause?on=false")
            assert _wait_until(lambda: c.get("/live/state").json()["paused"] is False)
        assert _wait_until(lambda: c.get("/live/state").json()["recording"] is False)


def test_paused_session_records_no_audio_while_paused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "x", asr_backend=None,
                     live_manager=_FakeLiveManager())
    with TestClient(app) as c:
        with c.websocket_connect("/ws/native-capture?source=system") as ws:
            ws.receive_json()
            assert _wait_until(lambda: c.get("/live/state").json()["recording"] is True)
            c.post("/live/pause")
            assert _wait_until(lambda: c.get("/live/state").json()["paused"] is True)
            ws.send_bytes(_frames(200))       # all of this must be dropped
            time.sleep(0.3)
            pcm = list((tmp_path / "data").glob("*/system.pcm"))
            assert pcm and pcm[0].stat().st_size == 0
        assert _wait_until(lambda: c.get("/live/state").json()["recording"] is False)
