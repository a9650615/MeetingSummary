"""Tests for POST /ws/native-capture — the floatpanel's in-process capture
relay (mic via AVCaptureSession + system audio via a Core Audio process tap,
both captured natively INSIDE floatpanel and relayed here framed, per
recorder.py's <track><len><payload> protocol). Server-side this demuxes into
the SAME live_session.py pipeline /ws/live uses.

Uses `with TestClient(app) as c:` (not a bare TestClient()) everywhere here:
the relay's consume/finalize work happens in a detached asyncio task that
outlives the request/socket, and only the `with` form keeps one event loop
alive across calls so that task actually gets to run between our polls -- a
bare TestClient() tears its loop down after every single call, silently
discarding such tasks."""
import struct
import time

from fastapi.testclient import TestClient

import recorder
from app import create_app
from store import Store


class _FakeBackend:
    current_model = "fake"

    def __call__(self, pcm_bytes):
        return []  # no speech recognized -- fine, we're testing the plumbing, not ASR


class _FakeLiveManager:
    """Just enough of backends.LiveModelManager's surface for TwoPassSession
    and live_session.build_track_sessions, without loading a real model."""

    def __init__(self):
        self.backend = _FakeBackend()
        self.requested = "fake"
        self.language = None

    @property
    def current(self):
        return "fake"

    def set_language(self, lang):
        self.language = lang

    def __call__(self, pcm_bytes):
        return self.backend(pcm_bytes)


def _wait_until(fn, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(0.05)
    return False


def test_meeting_transcripts_after_returns_only_newer_rows(tmp_path):
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "x", asr_backend=None)
    mid = store.create_meeting("t", time.time(), "zh-TW")
    id1 = store.add_transcript(mid, "live", "mic", 0, 100, "我", "hello")
    store.add_transcript(mid, "live", "mic", 100, 200, "我", "world")

    with TestClient(app) as c:
        rows = c.get(f"/meetings/{mid}/transcripts").json()["rows"]
        assert [r["text"] for r in rows] == ["hello", "world"]

        rows_after = c.get(f"/meetings/{mid}/transcripts?after={id1}").json()["rows"]
        assert [r["text"] for r in rows_after] == ["world"]

        assert c.get("/meetings/999999/transcripts").status_code == 404


def test_ws_native_capture_ingests_relayed_frames(tmp_path, monkeypatch):
    # The floatpanel captures system audio in-process and relays it framed to
    # /ws/native-capture. Server ingests it through the SAME pipeline as
    # /ws/live (no subprocess spawned server-side) and persists the PCM.
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "x", asr_backend=None,
                     live_manager=_FakeLiveManager())
    frame = struct.pack("<BI", recorder.TRACK_SYSTEM, 2) + b"\x00\x00"
    with TestClient(app) as c:
        with c.websocket_connect("/ws/native-capture?source=system") as ws:
            ws.send_bytes(frame * 5)
            assert _wait_until(lambda: c.get("/live/state").json()["recording"] is True)
        # closing the socket ends the session like an explicit /live/stop
        assert _wait_until(lambda: c.get("/live/state").json()["recording"] is False)

    seg_dirs = list((tmp_path / "data").glob("*-*"))
    assert seg_dirs, "no segment dir created for the relayed native session"
    assert (seg_dirs[0] / "system.pcm").read_bytes() != b""


def test_ws_native_capture_needs_a_live_backend(tmp_path):
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "x", asr_backend=None)  # live_manager=None
    with TestClient(app) as c:
        with c.websocket_connect("/ws/native-capture?source=mic") as ws:
            # server closes immediately (1011) — receiving raises on the closed socket
            import pytest as _pytest
            from starlette.websockets import WebSocketDisconnect
            with _pytest.raises(WebSocketDisconnect):
                ws.receive_bytes()


def test_ws_native_capture_both_tracks_creates_two_pcm_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "x", asr_backend=None,
                     live_manager=_FakeLiveManager())
    frame = (struct.pack("<BI", recorder.TRACK_MIC, 2) + b"\x00\x00" +
             struct.pack("<BI", recorder.TRACK_SYSTEM, 2) + b"\x00\x00")
    with TestClient(app) as c:
        with c.websocket_connect("/ws/native-capture?source=both") as ws:
            ws.send_bytes(frame * 3)
            assert _wait_until(lambda: c.get("/live/state").json()["recording"] is True)

    seg_dirs = list((tmp_path / "data").glob("*-*"))
    assert seg_dirs
    assert (seg_dirs[0] / "mic.pcm").exists()
    assert (seg_dirs[0] / "system.pcm").exists()
