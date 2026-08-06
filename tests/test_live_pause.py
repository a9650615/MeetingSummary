"""Pause/resume and transcribe-only for a live session (WallClockPump).

The pump's invariant is that bytes fed to the ASR session == bytes on disk ==
wall clock since t0 — TwoPassSession derives every timestamp from it. Both
features have to hold that invariant: pause freezes the clock (and elides the
gap on resume, instead of dumping the whole pause into the buffer as silence),
transcribe-only drops only the disk write.
"""
import io

import live_session


def _pump(paused=None, files=None, t0=None):
    # t0 must be a real wall-clock stamp like the handlers pass: pad_to fills
    # silence from t0 to now, so t0=0 would try to allocate decades of it.
    tracks = {0: ("mic", "我")}
    return live_session.WallClockPump(tracks, files or {0: io.BytesIO()},
                                      live_session.time.time() if t0 is None else t0,
                                      paused=paused)


def test_paused_pump_saves_nothing_and_freezes_the_clock():
    flag = {"on": True}
    f = io.BytesIO()
    p = _pump(paused=lambda: flag["on"], files={0: f})
    p.feed(0, b"\x01\x02" * 100)
    assert f.getvalue() == b""      # nothing recorded
    assert p.buffers[0] == b""      # and nothing sent to ASR
    assert p.written[0] == 0        # clock did not advance


def test_resume_elides_the_paused_span_instead_of_padding_it(monkeypatch):
    # A long pause must NOT come back as an equally long block of silence: that
    # would be written to disk AND pushed through the ASR buffer on resume.
    # Drive a fake clock so 30s "passes" while paused without sleeping.
    now = {"t": 1000.0}
    monkeypatch.setattr(live_session.time, "time", lambda: now["t"])
    flag = {"on": False}
    f = io.BytesIO()
    p = _pump(paused=lambda: flag["on"], files={0: f}, t0=now["t"])
    p.feed(0, b"\x01\x02" * 100)
    before = p.written[0]

    flag["on"] = True
    p.feed(0, b"\x01\x02" * 100)     # marks paused_at
    now["t"] += 30                    # 30s elapse while paused
    p.feed(0, b"\x01\x02" * 100)
    assert p.written[0] == before, "clock advanced while paused"

    flag["on"] = False
    p.feed(0, b"\x01\x02" * 100)
    grew = p.written[0] - before
    # only the 200 bytes actually fed — not 30s (960000 bytes) of silence
    assert grew == 200, f"resume back-filled the pause as silence ({grew} bytes)"


def test_unpaused_pump_is_unchanged():
    f = io.BytesIO()
    p = _pump(files={0: f})          # paused=None -> the old behaviour exactly
    p.feed(0, b"\x01\x02" * 100)
    assert f.getvalue().endswith(b"\x01\x02")
    assert p.written[0] == 200


def test_null_sink_keeps_the_clock_but_writes_no_audio():
    p = _pump(files={0: live_session.NullSink()})
    p.feed(0, b"\x01\x02" * 100)
    assert p.written[0] == 200        # timestamps unaffected...
    assert bytes(p.buffers[0]) == b"\x01\x02" * 100   # ...ASR still fed
