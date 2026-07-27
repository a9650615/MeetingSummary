"""Live-freeze guards on the ANE helper (backends.ane_live_backend).

The live loop abandons a threadpool feed after FEED_TIMEOUT_S but the thread
keeps running — and used to keep holding the ANE lock, so every later window
queued behind it forever ("辨識偶爾突然卡住"). Likewise a helper that never
printed READY blocked the lock for good. Both waits are now bounded.
"""
import threading

import pytest

import backends


@pytest.fixture(autouse=True)
def _fresh_helper_state(monkeypatch):
    monkeypatch.setattr(backends, "_ANE_HELP", {"proc": None, "lock": None})


def test_skips_window_instead_of_blocking_on_a_wedged_predecessor(monkeypatch):
    monkeypatch.setattr(backends, "ANE_LOCK_WAIT_S", 0.1)
    lock = threading.Lock()
    backends._ANE_HELP["lock"] = lock
    lock.acquire()  # stand in for the abandoned thread still holding it

    run = backends.ane_live_backend()
    done = threading.Event()
    out = []

    def _call():
        out.append(run(b"\x00" * 4000))
        done.set()

    threading.Thread(target=_call, daemon=True).start()
    assert done.wait(5), "run() blocked on the held lock instead of giving up"
    assert out == [[]]  # window skipped (audio is on disk), no exception


def test_helper_that_never_signals_ready_times_out(monkeypatch):
    # /bin/cat reads stdin forever and never prints READY on stderr — the old
    # `for line in p.stderr` waited on that with no bound, holding the lock.
    monkeypatch.setattr(backends, "ane_helper_bin", lambda: "/bin/cat")
    monkeypatch.setattr(backends, "ANE_READY_WAIT_S", 0.3)
    backends._ANE_HELP["lock"] = threading.Lock()

    run = backends.ane_live_backend()
    done = threading.Event()
    err = []

    def _call():
        try:
            run(b"\x00" * 4000)
        except Exception as e:  # noqa: BLE001
            err.append(e)
        done.set()

    threading.Thread(target=_call, daemon=True).start()
    assert done.wait(5), "_ensure() hung waiting for READY"
    assert err and "READY" in str(err[0])
    # and the lock is free again, so the next call isn't wedged behind it
    assert backends._ANE_HELP["lock"].acquire(timeout=0.1)
