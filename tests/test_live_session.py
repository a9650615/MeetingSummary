"""Tests for live_session.py — the pipeline shared between /ws/live (browser)
and /ws/native-capture (floatpanel, in-process capture relayed framed).
Focus: the frame-fed reader (framed relay -> WallClockPump) and the
consumer/flush loops, all driven with fakes so no real ASR backend or
subprocess is needed."""
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


# --- frame-fed reader (floatpanel relay -> pump) -------------------------

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
