"""Build a meeting bundle from the local store and POST it to the VM.
Heavy Mac-only deps (app helpers, recorder) are imported lazily so this module
imports cleanly in tests and on non-Mac machines."""
import os
import tempfile

from viewer import bundle


def _default_assemble(store, mid, track):
    from app import _assemble_track
    return _assemble_track(store, mid, track)


def _default_tracks(store, mid):
    from app import _meeting_tracks
    return _meeting_tracks(store, mid)


def _default_to_m4a(pcm_path, m4a_path):
    import recorder
    recorder.pcm_to_m4a(pcm_path, m4a_path)


def _default_post(url, files=None, timeout=None):
    import requests
    return requests.post(url, files=files, timeout=timeout)


def build_and_push(store, mid, vm_url, *, assemble=None, to_m4a=None,
                   http_post=None, tracks=None):
    assemble = assemble or _default_assemble
    to_m4a = to_m4a or _default_to_m4a
    http_post = http_post or _default_post
    track_names = tracks if tracks is not None else _default_tracks(store, mid)

    with tempfile.TemporaryDirectory() as td:
        track_files = {}
        present = []
        for t in track_names:
            pcm = assemble(store, mid, t)
            if not pcm:
                continue
            pcm_path = os.path.join(td, f"{t}.pcm")
            with open(pcm_path, "wb") as f:
                f.write(pcm)
            m4a_path = os.path.join(td, f"{t}.m4a")
            to_m4a(pcm_path, m4a_path)
            if os.path.exists(m4a_path) and os.path.getsize(m4a_path) > 0:
                track_files[t] = m4a_path
                present.append(t)

        # No audio assembled — the meeting has no usable track (never recorded,
        # or its pcm/m4a is missing on disk). Pushing a track-less bundle would
        # look "successful" but upload no audio, so refuse with a clear reason
        # instead of a silent no-op that reads as an upload failure.
        if not track_files:
            return {"ok": False, "status": 0, "mid": None,
                    "reason": "此會議沒有可上傳的音檔"}

        b = bundle.meeting_to_bundle(store, mid, present)
        zip_path = os.path.join(td, "bundle.zip")
        bundle.write_bundle_zip(zip_path, b, track_files)

        with open(zip_path, "rb") as zf:
            resp = http_post(f"{vm_url}/ingest-bundle",
                             files={"bundle": ("bundle.zip", zf, "application/zip")},
                             timeout=300)
        ok = 200 <= resp.status_code < 300
        new_mid = None
        if ok:
            try:
                new_mid = resp.json().get("mid")
            except Exception:
                new_mid = None
        return {"ok": ok, "status": resp.status_code, "mid": new_mid}
