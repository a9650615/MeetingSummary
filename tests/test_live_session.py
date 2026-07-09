"""Tests for live_session.py — the pipeline shared between /ws/live (browser)
and /live/start (native, browserless). Focus: the frame-fed reader (framed
audiocap stdout -> WallClockPump) and the consumer/flush loops, all driven
with fakes so no real ASR backend or subprocess is needed."""
import asyncio
import io
import struct
import time

import live_session
import recorder
from store import Store


def frame(track, payload):
    return struct.pack("<BI", track, len(payload)) + payload


class StubSession:
    """Fake TwoPassSession: records feed() calls, returns canned events."""

    def __init__(self, events=None, flush_events=None):
        self.feed_calls = []
        self._events = events if events is not None else []
        self._flush_events = flush_events if flush_events is not None else []

    def feed(self, chunk, want_interim):
        self.feed_calls.append((chunk, want_interim))
        return self._events

    def flush(self):
        return self._flush_events


# --- WallClockPump -----------------------------------------------------

def test_pump_feed_appends_payload_after_padding():
    f = io.BytesIO()
    pump = live_session.WallClockPump({"t": ("mic", "我")}, {"t": f}, t0=time.time())
    pump.feed("t", b"\xab\xcd")
    assert pump.buffers["t"].endswith(b"\xab\xcd")
    assert f.getvalue().endswith(b"\xab\xcd")
    assert pump.got.is_set()


def test_pump_feed_ignores_unknown_track():
    f = io.BytesIO()
    pump = live_session.WallClockPump({"t": ("mic", "我")}, {"t": f}, t0=time.time())
    pump.feed("other", b"\x01")
    assert pump.buffers == {"t": bytearray()}
    assert not pump.got.is_set()


def test_pump_pad_to_equalizes_untouched_tracks():
    # "b" never gets fed real data, but every feed() pads ALL tracks to the
    # same wall-clock target -- so a track that's fallen behind (or a track
    # whose source hasn't started yet) still stays the same length as one
    # that's actively receiving audio.
    fa, fb = io.BytesIO(), io.BytesIO()
    t0 = time.time() - 1.0  # pretend the session started 1s ago
    pump = live_session.WallClockPump({"a": ("mic", "我"), "b": ("system", "對方")},
                                      {"a": fa, "b": fb}, t0)
    pump.feed("a", b"\x00\x00")
    assert pump.written["b"] > 0
    assert pump.written["a"] == pump.written["b"] + 2


# --- frame-fed reader (audiocap stdout -> pump) -------------------------

def test_pump_framed_stdout_demuxes_into_pump():
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data(frame(recorder.TRACK_MIC, b"\x01\x02") +
                         frame(recorder.TRACK_SYSTEM, b"\x03"))
        reader.feed_eof()
        tracks = {recorder.TRACK_MIC: ("mic", "我"), recorder.TRACK_SYSTEM: ("system", "對方")}
        audio_files = {recorder.TRACK_MIC: io.BytesIO(), recorder.TRACK_SYSTEM: io.BytesIO()}
        pump = live_session.WallClockPump(tracks, audio_files, time.time())
        await live_session.pump_framed_stdout(reader, pump)
        return pump

    pump = asyncio.run(run())
    assert pump.buffers[recorder.TRACK_MIC].endswith(b"\x01\x02")
    assert pump.buffers[recorder.TRACK_SYSTEM].endswith(b"\x03")


def test_pump_framed_stdout_ignores_unknown_track():
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack("<BI", 99, 2) + b"\xff\xff")  # not a track in `tracks`
        reader.feed_eof()
        tracks = {recorder.TRACK_MIC: ("mic", "我")}
        pump = live_session.WallClockPump(tracks, {recorder.TRACK_MIC: io.BytesIO()}, time.time())
        await live_session.pump_framed_stdout(reader, pump)
        return pump

    pump = asyncio.run(run())
    assert pump.buffers[recorder.TRACK_MIC] == bytearray()


def test_pump_raw_stdout_feeds_the_given_tag():
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data(b"\x01\x02\x03")
        reader.feed_eof()
        pump = live_session.WallClockPump({"t": ("system", "對方")}, {"t": io.BytesIO()}, time.time())
        await live_session.pump_raw_stdout(reader, pump, "t")
        return pump

    pump = asyncio.run(run())
    assert bytes(pump.buffers["t"]) == b"\x01\x02\x03"


# --- NativeCaptureSupervisor ----------------------------------------------

class FakeHelperProc:
    """Stand-in for asyncio.subprocess.Process: stdout/stderr are real
    StreamReaders (so reader_fn/the supervisor's stderr drain work unchanged),
    terminate() just marks it dead."""

    def __init__(self, out=b"", err=()):
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(out)
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        for line in err:
            self.stderr.feed_data(line.encode() + b"\n")
        self.stderr.feed_eof()
        self.returncode = None

    def terminate(self):
        self.returncode = -15


async def _drain_to_pump(stdout, pump, tag="t"):
    while True:
        data = await stdout.read(8192)
        if not data:
            break
        pump.feed(tag, data)


def test_supervisor_respawns_on_death_and_keeps_feeding(monkeypatch):
    # Freeze the wall clock pump.feed() reads: real time elapses between the
    # two fake helpers' writes (event-loop scheduling, the backoff sleep),
    # and pad_to() would otherwise insert a few bytes of real silence for
    # that gap -- frozen time keeps the byte comparison below exact.
    monkeypatch.setattr(live_session.time, "time", lambda: 1000.0)

    async def run():
        pump = live_session.WallClockPump({"t": ("mic", "我")}, {"t": io.BytesIO()}, live_session.time.time())
        procs = [FakeHelperProc(out=b"\x00\x00"), FakeHelperProc(out=b"\x11\x11")]
        spawned = []

        async def spawn():
            p = procs.pop(0)
            spawned.append(p)
            return p

        notices = []

        async def on_notice(msg):
            notices.append(msg)

        sup = live_session.NativeCaptureSupervisor(
            spawn=spawn, reader_fn=_drain_to_pump, pump=pump, on_notice=on_notice,
            backoff=(0,), max_fast_fails=5, fast_fail_window=5.0)
        result = await asyncio.wait_for(
            sup.run(should_stop=lambda: len(spawned) >= 2), timeout=2)
        return result, pump.buffers["t"], notices

    result, buf, notices = asyncio.run(run())
    assert result == "stopped"
    assert bytes(buf) == b"\x00\x00\x11\x11"          # frames from BOTH the dead and the respawned proc
    assert any("重啟" in n for n in notices)


def test_supervisor_gives_up_after_max_fast_fails():
    async def run():
        pump = live_session.WallClockPump({"t": ("mic", "我")}, {"t": io.BytesIO()}, time.time())

        async def spawn():
            return FakeHelperProc(out=b"")  # dies immediately, every time

        gave_up = []

        async def on_give_up():
            gave_up.append(True)

        sup = live_session.NativeCaptureSupervisor(
            spawn=spawn, reader_fn=_drain_to_pump, pump=pump, on_give_up=on_give_up,
            backoff=(0,), max_fast_fails=3, fast_fail_window=5.0)
        result = await asyncio.wait_for(sup.run(should_stop=lambda: False), timeout=2)
        return result, gave_up, sup.gave_up

    result, gave_up, flag = asyncio.run(run())
    assert result == "gave_up"
    assert gave_up == [True]
    assert flag is True


def test_supervisor_does_not_respawn_once_should_stop():
    async def run():
        pump = live_session.WallClockPump({"t": ("mic", "我")}, {"t": io.BytesIO()}, time.time())
        spawn_count = []

        async def spawn():
            spawn_count.append(1)
            return FakeHelperProc(out=b"")

        sup = live_session.NativeCaptureSupervisor(spawn=spawn, reader_fn=_drain_to_pump, pump=pump)
        result = await sup.run(should_stop=lambda: True)
        return result, spawn_count

    result, spawn_count = asyncio.run(run())
    assert result == "stopped"
    assert spawn_count == [1]  # the initial spawn ran once -- no respawn attempted


def test_supervisor_terminate_before_any_spawn_is_a_noop():
    sup = live_session.NativeCaptureSupervisor(spawn=None, reader_fn=None, pump=None)
    sup.terminate()  # must not raise -- nothing to terminate yet


def test_supervisor_forwards_stderr_lines():
    async def run():
        pump = live_session.WallClockPump({"t": ("mic", "我")}, {"t": io.BytesIO()}, time.time())
        proc = FakeHelperProc(out=b"", err=["READY", "ERR NOPERM 需要螢幕錄製權限"])
        lines = []

        async def spawn():
            return proc

        async def on_stderr_line(s):
            lines.append(s)

        sup = live_session.NativeCaptureSupervisor(
            spawn=spawn, reader_fn=_drain_to_pump, pump=pump, on_stderr_line=on_stderr_line)
        await sup.run(should_stop=lambda: True)
        return lines

    lines = asyncio.run(run())
    assert lines == ["ERR NOPERM 需要螢幕錄製權限"]  # READY is filtered out


# --- consume() -----------------------------------------------------------

def test_consume_feeds_session_and_emits_finals():
    async def run():
        tracks = {"t": ("mic", "我")}
        pump = live_session.WallClockPump(tracks, {"t": io.BytesIO()}, t0=time.time())
        pump.feed("t", b"\x00" * 10)
        session = StubSession(events=[{"kind": "final", "start_ms": 0, "end_ms": 100, "text": "hi"}])
        emitted = []

        async def emit(ev, label):
            emitted.append((ev, label))

        await live_session.consume(pump, {"t": session}, tracks, rec_on=lambda: False,
                                   emit=emit, should_stop=lambda: True,
                                   interim_lag_bytes=10**9)
        return emitted, session.feed_calls

    emitted, feed_calls = asyncio.run(run())
    assert emitted == [({"kind": "final", "start_ms": 0, "end_ms": 100, "text": "hi"}, ("mic", "我"))]
    assert len(feed_calls) == 1


def test_consume_skips_asr_when_record_only():
    async def run():
        tracks = {"t": ("mic", "我")}
        pump = live_session.WallClockPump(tracks, {"t": io.BytesIO()}, t0=time.time())
        pump.feed("t", b"\x00" * 10)
        session = StubSession(events=[{"kind": "final", "start_ms": 0, "end_ms": 1, "text": "x"}])

        async def emit(ev, label):
            raise AssertionError("should not emit in record-only mode")

        await live_session.consume(pump, {"t": session}, tracks, rec_on=lambda: True,
                                   emit=emit, should_stop=lambda: True,
                                   interim_lag_bytes=10**9)
        return session.feed_calls

    assert asyncio.run(run()) == []  # PCM saved via pump.feed, but ASR never called


def test_consume_trims_backlog_to_max_bytes():
    async def run():
        tracks = {"t": ("mic", "我")}
        pump = live_session.WallClockPump(tracks, {"t": io.BytesIO()}, t0=time.time())
        big = b"x" * (live_session.TRACK_BACKLOG_MAXB + 100)
        pump.buffers["t"].extend(big)
        pump.got.set()
        session = StubSession(events=[])

        async def emit(ev, label):
            pass

        await live_session.consume(pump, {"t": session}, tracks, rec_on=lambda: False,
                                   emit=emit, should_stop=lambda: True, interim_lag_bytes=0)
        return session.feed_calls, big

    feed_calls, big = asyncio.run(run())
    chunk, _want_interim = feed_calls[0]
    assert len(chunk) == live_session.TRACK_BACKLOG_MAXB
    assert chunk == big[-live_session.TRACK_BACKLOG_MAXB:]


def test_consume_should_abort_drops_buffered_audio_immediately():
    async def run():
        tracks = {"t": ("mic", "我")}
        pump = live_session.WallClockPump(tracks, {"t": io.BytesIO()}, t0=time.time())
        pump.buffers["t"].extend(b"\x00" * 10)
        pump.got.set()
        session = StubSession(events=[{"kind": "final", "start_ms": 0, "end_ms": 1, "text": "x"}])
        emitted = []

        async def emit(ev, label):
            emitted.append(ev)

        await live_session.consume(pump, {"t": session}, tracks, rec_on=lambda: False,
                                   emit=emit, should_stop=lambda: True,
                                   interim_lag_bytes=10**9, should_abort=lambda: True)
        return emitted, session.feed_calls

    emitted, feed_calls = asyncio.run(run())
    assert emitted == [] and feed_calls == []


def test_consume_calls_on_stop_hook_when_stopping():
    async def run():
        tracks = {"t": ("mic", "我")}
        pump = live_session.WallClockPump(tracks, {"t": io.BytesIO()}, t0=time.time())
        pump.got.set()
        stopped = []

        async def on_stop():
            stopped.append(True)

        async def emit(ev, label):
            pass

        await live_session.consume(pump, {"t": StubSession()}, tracks, rec_on=lambda: False,
                                   emit=emit, should_stop=lambda: True,
                                   interim_lag_bytes=0, on_stop=on_stop)
        return stopped

    assert asyncio.run(run()) == [True]


def test_consume_pops_and_forwards_notices():
    async def run():
        tracks = {"t": ("mic", "我")}
        pump = live_session.WallClockPump(tracks, {"t": io.BytesIO()}, t0=time.time())
        pump.got.set()
        notices = []

        async def on_notice(msg):
            notices.append(msg)

        async def emit(ev, label):
            pass

        await live_session.consume(pump, {"t": StubSession()}, tracks, rec_on=lambda: False,
                                   emit=emit, should_stop=lambda: True, interim_lag_bytes=0,
                                   on_notice=on_notice, pop_notice=lambda: "downgraded model")
        return notices

    assert asyncio.run(run()) == ["downgraded model"]


# --- flush + persistence --------------------------------------------------

def test_flush_sessions_persists_finals(tmp_path):
    store = Store(tmp_path / "m.db")
    mid = store.create_meeting("t", 0.0, "zh-TW")
    tracks = {"t": ("mic", "我")}
    session = StubSession(flush_events=[{"kind": "final", "start_ms": 0, "end_ms": 500, "text": "hello"}])
    asyncio.run(live_session.flush_sessions({"t": session}, tracks, mid, conn_offset_ms=0, store=store))
    assert store.latest_transcript(mid) == "我：hello"


def test_flush_sessions_swallows_a_wedged_backend(tmp_path):
    store = Store(tmp_path / "m.db")
    mid = store.create_meeting("t", 0.0, "zh-TW")
    tracks = {"t": ("mic", "我")}

    class Wedged:
        def flush(self):
            raise RuntimeError("backend stuck")

    # Must not raise -- a wedged backend can't hang stop forever.
    asyncio.run(live_session.flush_sessions({"t": Wedged()}, tracks, mid, conn_offset_ms=0, store=store))
    assert store.latest_transcript(mid) is None


def test_make_store_emit_persists_only_final(tmp_path):
    store = Store(tmp_path / "m.db")
    mid = store.create_meeting("t", 0.0, "zh-TW")
    emit = live_session.make_store_emit(mid, conn_offset_ms=0, store=store)
    asyncio.run(emit({"kind": "interim", "start_ms": 0, "end_ms": 100, "text": "partial"}, ("mic", "我")))
    assert store.latest_transcript(mid) is None
    asyncio.run(emit({"kind": "final", "start_ms": 0, "end_ms": 100, "text": "done"}, ("mic", "我")))
    assert store.latest_transcript(mid) == "我：done"


def test_resolve_speaker_collapses_placeholder_to_side_keeps_name():
    r = live_session.resolve_speaker
    assert r("說話者1", "對方") == "對方"      # auto cluster -> side label
    assert r("對方2", "對方") == "對方"         # numbered side placeholder -> bare side
    assert r("我3", "我") == "我"
    assert r(None, "對方") == "對方"            # diarize off -> side label
    assert r("", "我") == "我"
    assert r("Scott", "對方") == "Scott"        # recognized name wins
    assert r("對方", "對方") == "對方"           # bare side label is not a placeholder


# --- enable_diarization wiring (省效能: one embed per utterance by default) ---

def _diar_sessions_tracks():
    sessions = {"sys": StubSession(), "mic": StubSession()}
    tracks = {"sys": (0, "對方"), "mic": (1, "我")}
    return sessions, tracks


def test_enable_diarization_default_wires_speaker_fn_not_splitter(tmp_path, monkeypatch):
    # Default: label each utterance ONCE (speaker_fn), no per-window split.
    import diarize
    monkeypatch.setattr(diarize, "embedding_extractor", lambda *a, **k: (lambda b, sr=16000: b))
    monkeypatch.delenv("LIVE_DIAR_SPLIT", raising=False)
    store = Store(tmp_path / "m.db")
    sessions, tracks = _diar_sessions_tracks()
    live_session.enable_diarization(sessions, tracks, store)
    assert callable(getattr(sessions["sys"], "speaker_fn", None))
    assert getattr(sessions["sys"], "splitter", None) is None
    # The mic ("我") track is never diarized.
    assert getattr(sessions["mic"], "speaker_fn", None) is None


def test_enable_diarization_split_opt_in_wires_splitter(tmp_path, monkeypatch):
    import diarize
    monkeypatch.setattr(diarize, "embedding_extractor", lambda *a, **k: (lambda b, sr=16000: b))
    monkeypatch.setenv("LIVE_DIAR_SPLIT", "1")
    store = Store(tmp_path / "m.db")
    sessions, tracks = _diar_sessions_tracks()
    live_session.enable_diarization(sessions, tracks, store)
    assert callable(getattr(sessions["sys"], "splitter", None))
    assert getattr(sessions["sys"], "speaker_fn", None) is None
