"""Live-first scheduling: batch recognition yields the Neural Engine to a live
recording and resumes when it ends (app.wait_while_live + iter_transcribe's
per-window gate). Batch and live ASR are separate processes with nothing else
arbitrating between them, so without this they fight over the ANE."""
import threading
import time

import app


def test_wait_while_live_returns_immediately_when_idle():
    assert app.wait_while_live(lambda: False) is False
    assert app.wait_while_live(None) is False  # no predicate wired -> never waits


def test_wait_while_live_blocks_until_recording_ends():
    recording = {"on": True}
    started, finished = threading.Event(), threading.Event()

    def _worker():
        started.set()
        app.wait_while_live(lambda: recording["on"], poll_s=0.01)
        finished.set()

    threading.Thread(target=_worker, daemon=True).start()
    assert started.wait(2)
    assert not finished.wait(0.2), "did not wait while live was recording"
    recording["on"] = False
    assert finished.wait(2), "did not resume once the recording ended"


def test_wait_while_live_marks_the_job_paused():
    recording = {"on": True}
    jobs = {7: {"state": "running", "text": "辨識中…"}}
    done = threading.Event()

    def _worker():
        app.wait_while_live(lambda: recording["on"], jobs, 7, poll_s=0.01)
        done.set()

    threading.Thread(target=_worker, daemon=True).start()
    time.sleep(0.1)
    assert jobs[7]["text"] == app._PAUSED_TEXT  # /jobs shows why it stalled
    recording["on"] = False
    assert done.wait(2)


def test_iter_transcribe_pauses_between_windows(tmp_path, monkeypatch):
    # Two 30s windows of a fake mic track; live is "recording" for the first gate
    # check only, so the run emits paused/resumed and still finishes both windows.
    from store import Store
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "m.db")
    mid = store.create_meeting("t", time.time(), "zh-TW")
    seg = tmp_path / "seg"
    seg.mkdir()
    # exactly 2 windows of real (non-silent) audio — iter_transcribe windows at 29s
    # and skips windows that are pure silence, neither of which this test is about
    import numpy as np
    n = 2 * 29 * 16000
    t = np.arange(n) / 16000
    (seg / "mic.pcm").write_bytes((6000 * np.sin(2 * np.pi * 220 * t)).astype("<i2").tobytes())
    store.add_segment(mid, idx=0, dir_path=str(seg), started_at=time.time(),
                      duration_s=58, origin="recorded")
    monkeypatch.setattr(app.asr, "transcribe",
                        lambda *a, **k: [{"start_ms": 0, "end_ms": 1, "text": "x",
                                          "profile": "accurate", "track": "mic"}])

    calls = {"n": 0}

    def _live_busy():           # busy on the first gate check only
        calls["n"] += 1
        return calls["n"] == 1

    kinds = [ev["type"] for ev in
             app.iter_transcribe(store, mid, backend=None, live_busy=_live_busy)]
    assert "paused" in kinds and "resumed" in kinds
    assert kinds[-1] == "done"                    # still completed after resuming
    assert kinds.count("progress") == 2           # both windows transcribed
