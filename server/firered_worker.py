"""Background FireRed re-correction on the VM. Long-running, so it is resumable:
corrected lines land under profile="firered_staging" one at a time, and only when
every source sentence is staged does the worker promote them to profile="firered"
in one step (so the viewer never shows a half-corrected transcript). Progress and
a stop request live in the store's key/value settings — durable across restarts.
CPU-only (sherpa-onnx). Speaker labels + timing are inherited from each source
row; the VM never diarizes."""
import glob
import json
import os
import queue
import subprocess
import threading
import traceback


def decode_span_pcm(m4a_path, start_ms, end_ms, ffmpeg="ffmpeg"):
    dur = max(0.0, (end_ms - start_ms) / 1000.0)
    if dur <= 0:
        return b""
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error",
           "-ss", f"{start_ms/1000.0:.3f}", "-t", f"{dur:.3f}",
           "-i", m4a_path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return b""


def chunk_ranges(start_ms, end_ms, max_ms=30000):
    out, s = [], start_ms
    while s < end_ms:
        e = min(s + max_ms, end_ms)
        out.append((s, e))
        s = e
    return out or [(start_ms, end_ms)]


def _pkey(mid):
    return f"firered:{mid}"


def _skey(mid):
    return f"firered_stop:{mid}"


def get_progress(store, mid):
    raw = store.get_setting(_pkey(mid))
    if not raw:
        return {"state": "idle", "done": 0, "total": 0}
    return json.loads(raw)


def set_progress(store, mid, state=None, done=None, total=None):
    p = get_progress(store, mid)
    if state is not None:
        p["state"] = state
    if done is not None:
        p["done"] = done
    if total is not None:
        p["total"] = total
    store.set_setting(_pkey(mid), json.dumps(p))
    return p


def request_stop(store, mid):
    store.set_setting(_skey(mid), "1")


def clear_stop(store, mid):
    store.set_setting(_skey(mid), "0")


def stop_requested(store, mid):
    return store.get_setting(_skey(mid)) == "1"


def _track_path(data_dir, mid, track):
    direct = os.path.join(data_dir, str(mid), f"{track}.m4a")
    if os.path.exists(direct):
        return direct
    others = sorted(glob.glob(os.path.join(data_dir, str(mid), "*.m4a")))
    return others[0] if others else None


def _source_rows(store, mid):
    return [r for r in store.list_transcripts(mid)
            if r["profile"] not in ("firered", "firered_staging")]


def _staged_keys(store, mid):
    return {(r["track"], r["start_ms"], r["end_ms"])
            for r in store.list_transcripts(mid) if r["profile"] == "firered_staging"}


def _promote(store, mid):
    """Atomically swap the staged set in as the firered profile."""
    store.clear_transcripts(mid, profile="firered")
    store.db.execute(
        "UPDATE transcripts SET profile='firered' "
        "WHERE meeting_id=? AND profile='firered_staging'", (mid,))
    store.db.commit()


def correct_meeting(store, data_dir, mid, recognize, *, decode=None,
                    should_stop=None, restart=False):
    """Re-transcribe each source sentence with recognize(pcm)->str, staging as we
    go, and promote when complete. Resumable: rows already staged are skipped.
    restart=True wipes prior staging + firered and starts over."""
    decode = decode or decode_span_pcm
    if restart:
        store.clear_transcripts(mid, profile="firered_staging")
        store.clear_transcripts(mid, profile="firered")
    source = _source_rows(store, mid)
    total = len(source)
    staged = _staged_keys(store, mid)
    set_progress(store, mid, state="running", done=len(staged), total=total)
    done = len(staged)
    for r in source:
        key = (r["track"], r["start_ms"], r["end_ms"])
        if key in staged:
            continue
        if should_stop and should_stop():
            return set_progress(store, mid, state="paused")
        path = _track_path(data_dir, mid, r["track"])
        text = ""
        if path is not None:
            parts = []
            for cs, ce in chunk_ranges(r["start_ms"], r["end_ms"]):
                pcm = decode(path, cs, ce)
                if pcm:
                    t = (recognize(pcm) or "").strip()
                    if t:
                        parts.append(t)
            text = "".join(parts)
        # stage even an empty result so the row counts as done and won't be retried
        store.add_transcript(mid, "firered_staging", r["track"], r["start_ms"],
                             r["end_ms"], r["speaker"], text)
        done += 1
        set_progress(store, mid, done=done)
    _promote(store, mid)
    return set_progress(store, mid, state="done", done=total, total=total)


class FireRedWorker:
    """One-at-a-time background corrector. enqueue(mid), stop(mid), start()."""
    def __init__(self, store, data_dir):
        self.store = store
        self.data_dir = data_dir
        self.q = queue.Queue()
        self._backend = None

    def _recognize(self, pcm_bytes):
        if self._backend is None:
            import backends  # sherpa-onnx CPU; lazy
            self._backend = backends.firered_batch_backend("firered")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f:
            f.write(pcm_bytes)
            tmp = f.name
        try:
            res = self._backend(tmp)
            return res[0]["text"] if res else ""
        finally:
            os.remove(tmp)

    def enqueue(self, mid, restart=False):
        self.q.put((mid, restart))

    def resume_incomplete(self):
        """Re-enqueue meetings whose correction was running/paused when the
        process died — the in-memory queue does not survive a restart/redeploy,
        so without this every redeploy orphans an in-flight FireRed pass. Progress
        is persisted, so a re-enqueued run resumes from the staged rows."""
        for m in self.store.list_meetings():
            if get_progress(self.store, m["id"])["state"] in ("running", "paused"):
                self.enqueue(m["id"])

    def stop(self, mid):
        request_stop(self.store, mid)

    def _loop(self):
        while True:
            mid, restart = self.q.get()
            try:
                clear_stop(self.store, mid)
                correct_meeting(self.store, self.data_dir, mid, self._recognize,
                                should_stop=lambda: stop_requested(self.store, mid),
                                restart=restart)
            except Exception:
                traceback.print_exc()  # a bad meeting must not kill the worker
                # don't leave progress stuck at "running" after a crash; "paused"
                # is resumable and honest about the interruption.
                try:
                    set_progress(self.store, mid, state="paused")
                except Exception:
                    pass
            finally:
                self.q.task_done()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
