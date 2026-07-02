"""Tests for POST /live/start — browserless (native) recording: the server
spawns a fake "audiocap" helper itself (a tiny script standing in for the
real Swift binary) and the /live/start pipeline demuxes its framed stdout,
runs it through live_session, and persists like any other live session.

Uses `with TestClient(app) as c:` (not a bare TestClient()) everywhere here:
/live/start's work happens in a detached asyncio task that outlives the
request, and only the `with` form keeps one event loop alive across calls so
that task actually gets to run between our polls -- a bare TestClient() tears
its loop down after every single call, silently discarding such tasks."""
import struct
import time

from fastapi.testclient import TestClient

import backends
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


def _fake_audiocap(path, body):
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(0o755)


def _make_app(tmp_path, monkeypatch, audiocap_body):
    fake_bin = tmp_path / "audiocap"
    _fake_audiocap(fake_bin, audiocap_body)
    monkeypatch.setattr(backends, "audiocap_bin", lambda: str(fake_bin))
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "SUMMARY", asr_backend=None,
                     live_manager=_FakeLiveManager())
    return app, store


def _wait_until(fn, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(0.05)
    return False


def test_live_start_requires_audiocap(tmp_path, monkeypatch):
    monkeypatch.setattr(backends, "audiocap_bin", lambda: None)
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "x", asr_backend=None,
                     live_manager=_FakeLiveManager())
    with TestClient(app) as c:
        r = c.post("/live/start", json={"source": "mic"})
        assert r.status_code == 400


def test_live_start_requires_a_live_backend(tmp_path):
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "x", asr_backend=None)  # live_manager=None
    with TestClient(app) as c:
        r = c.post("/live/start", json={"source": "mic"})
        assert r.status_code == 400


def test_live_start_runs_end_to_end_and_self_stops(tmp_path, monkeypatch):
    # Helper emits one framed mic frame then exits (EOF) -- the session must
    # end itself (like a crash/permission-loss would), close its PCM file, and
    # clear live_active, with no explicit /live/stop needed.
    frame = struct.pack("<BI", recorder.TRACK_MIC, 2) + b"\x00\x00"
    script = "import sys\n" f"sys.stdout.buffer.write({frame!r})\n"
    app, store = _make_app(tmp_path, monkeypatch, script)

    with TestClient(app) as c:
        r = c.post("/live/start", json={"source": "mic"})
        assert r.status_code == 200
        body = r.json()
        mid, source = body["id"], body["source"]
        assert source == "mic"
        assert c.get("/live/state").json()["recording"] is True

        assert _wait_until(lambda: c.get("/live/state").json()["recording"] is False)

    assert store.get_meeting(mid) is not None
    seg_dirs = list((tmp_path / "data").glob(f"{mid}-*"))
    assert seg_dirs, "no segment dir created for the native session"
    assert (seg_dirs[0] / "mic.pcm").read_bytes() != b""


def test_live_stop_terminates_a_running_native_session(tmp_path, monkeypatch):
    # Helper never exits on its own -- like real system audio, it streams
    # continuously (here: silence frames every 50ms) until told to stop, which
    # is also what lets the consumer loop wake up often enough to notice a
    # pending /live/stop (it only rechecks between arrivals of new audio).
    script = (
        "import struct, sys, time\n"
        f"frame = struct.pack('<BI', {recorder.TRACK_SYSTEM}, 2) + b'\\x00\\x00'\n"
        "while True:\n"
        "    sys.stdout.buffer.write(frame)\n"
        "    sys.stdout.buffer.flush()\n"
        "    time.sleep(0.05)\n"
    )
    app, store = _make_app(tmp_path, monkeypatch, script)

    with TestClient(app) as c:
        r = c.post("/live/start", json={"source": "system"})
        assert r.status_code == 200
        mid = r.json()["id"]
        assert c.get("/live/state").json()["recording"] is True

        assert c.post("/live/stop").json()["stopping"] == 1
        assert _wait_until(lambda: c.get("/live/state").json()["recording"] is False)

    assert store.get_meeting(mid) is not None


def test_live_state_includes_started_at_for_attach_on_load(tmp_path, monkeypatch):
    # /live page attach-on-load needs the session's real start time (not
    # page-load time) to show a correct elapsed timer after a refresh.
    script = (
        "import struct, sys, time\n"
        f"frame = struct.pack('<BI', {recorder.TRACK_MIC}, 2) + b'\\x00\\x00'\n"
        "while True:\n"
        "    sys.stdout.buffer.write(frame)\n"
        "    sys.stdout.buffer.flush()\n"
        "    time.sleep(0.05)\n"
    )
    app, store = _make_app(tmp_path, monkeypatch, script)

    with TestClient(app) as c:
        t0 = time.time()
        r = c.post("/live/start", json={"source": "mic"})
        mid = r.json()["id"]
        st = c.get("/live/state").json()
        assert st["mid"] == mid
        assert st["started_at"] == store.get_meeting(mid)["created_at"]
        assert abs(st["started_at"] - t0) < 5  # sanity: not None, roughly "now"

        c.post("/live/stop")
        assert _wait_until(lambda: c.get("/live/state").json()["recording"] is False)

    # idle: no active session -> no started_at
    assert c.get("/live/state").json()["started_at"] is None


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


def test_live_start_both_tracks_creates_two_pcm_files(tmp_path, monkeypatch):
    frame = (struct.pack("<BI", recorder.TRACK_MIC, 2) + b"\x00\x00" +
             struct.pack("<BI", recorder.TRACK_SYSTEM, 2) + b"\x00\x00")
    script = "import sys\n" f"sys.stdout.buffer.write({frame!r})\n"
    app, store = _make_app(tmp_path, monkeypatch, script)

    with TestClient(app) as c:
        r = c.post("/live/start", json={"source": "both"})
        assert r.status_code == 200
        mid = r.json()["id"]
        assert _wait_until(lambda: c.get("/live/state").json()["recording"] is False)

    seg_dirs = list((tmp_path / "data").glob(f"{mid}-*"))
    assert (seg_dirs[0] / "mic.pcm").exists()
    assert (seg_dirs[0] / "system.pcm").exists()
