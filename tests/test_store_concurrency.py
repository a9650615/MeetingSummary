"""Regression test for the SQLite shared-connection crash/spin under FastAPI's
threadpool: many threads calling Store.get_meeting concurrently while a writer
thread updates notes, on the SAME db_path/Store instance. Before the fix
(single shared sqlite3 connection stepped from multiple threads) this reliably
segfaults or hangs the process within a few seconds. After the fix (one
connection per thread) it must complete cleanly and fast."""
import threading
import time

import pytest

from store import Store

N_READERS = 8
DURATION_S = 3
TIMEOUT_S = 10


def test_concurrent_reads_and_writes_dont_hang_or_crash(tmp_path):
    s = Store(tmp_path / "meetings.db")
    mids = [s.create_meeting(title=f"m{i}", created_at=float(i), lang="zh-TW")
            for i in range(N_READERS)]

    iters = [0] * N_READERS
    errors = []
    stop = threading.Event()

    def reader(idx):
        mid = mids[idx]
        try:
            while not stop.is_set():
                s.get_meeting(mid)
                iters[idx] += 1
        except Exception as e:  # noqa: BLE001 - want to see ANY failure mode
            errors.append((idx, e))

    def writer():
        n = 0
        try:
            while not stop.is_set():
                s.set_notes(mids[n % N_READERS], f"note{n}")
                n += 1
                time.sleep(0.001)
        except Exception as e:  # noqa: BLE001
            errors.append(("writer", e))

    threads = [threading.Thread(target=reader, args=(i,), daemon=True)
               for i in range(N_READERS)]
    threads.append(threading.Thread(target=writer, daemon=True))
    for t in threads:
        t.start()

    time.sleep(DURATION_S)
    stop.set()
    for t in threads:
        t.join(timeout=TIMEOUT_S)
        # A thread stuck in the corrupted-cursor spin never exits -> join times
        # out -> this is the "hang" failure mode on the old shared-connection code.
        assert not t.is_alive(), "worker thread failed to stop (hung/spinning)"

    assert not errors, f"worker thread(s) raised: {errors}"
    # Every reader must have kept making forward progress the whole time —
    # the corrupted-cursor failure mode is a thread that silently stops
    # completing iterations while still burning CPU.
    for idx, count in enumerate(iters):
        assert count > 0, f"reader {idx} made no progress at all"
