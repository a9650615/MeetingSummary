"""Meeting <-> portable bundle (zip). The single source of the bundle schema,
imported by both the Mac push plugin and the VM server. No Apple/ASR deps."""
import json
import os
import shutil
import zipfile


def _rows(rows):
    return [dict(r) for r in rows]


def meeting_to_bundle(store, mid, track_names):
    """The meeting.json dict for a meeting. Does not touch audio files."""
    m = dict(store.get_meeting(mid))
    return {
        "meeting": {"title": m.get("title"), "created_at": m.get("created_at"),
                    "lang": m.get("lang"), "status": m.get("status", "finalized"),
                    "notes": m.get("notes") or ""},
        "segments": [{"idx": s["idx"], "started_at": s["started_at"],
                      "duration_s": s["duration_s"], "origin": s["origin"]}
                     for s in store.list_segments(mid)],
        "transcripts": [{"profile": t["profile"], "track": t["track"],
                         "start_ms": t["start_ms"], "end_ms": t["end_ms"],
                         "speaker": t["speaker"], "text": t["text"]}
                        for t in store.list_transcripts(mid)],
        "summaries": [{"kind": s["kind"], "lang": s["lang"], "text": s["text"],
                       "model": s["model"], "created_at": s["created_at"]}
                      for s in store.list_summaries(mid)],
        "tracks": list(track_names),
    }


def write_bundle_zip(zip_path, bundle_dict, track_files):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("meeting.json", json.dumps(bundle_dict, ensure_ascii=False))
        for track, path in (track_files or {}).items():
            z.write(path, f"tracks/{track}.m4a")


def read_bundle_zip(zip_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        # reject path traversal in member names before extracting
        for name in z.namelist():
            if name.startswith("/") or ".." in name.split("/"):
                raise ValueError(f"unsafe zip member: {name}")
        z.extractall(dest_dir)
        bundle_dict = json.loads(z.read("meeting.json").decode("utf-8"))
    tracks = {}
    for t in bundle_dict.get("tracks", []):
        p = os.path.join(dest_dir, "tracks", f"{t}.m4a")
        if os.path.exists(p):
            tracks[t] = p
    return bundle_dict, tracks


def ingest_bundle(store, data_dir, bundle_dict, track_files):
    """Insert a bundle into store; copy track m4a to data_dir/<mid>/<t>.m4a.
    Idempotent by created_at: an existing meeting with the same start is deleted
    first (re-push replaces)."""
    m = bundle_dict["meeting"]
    created_at = m["created_at"]
    to_replace = [e for e in store.list_meetings() if e["created_at"] == created_at]
    # ponytail: meetings.id has no AUTOINCREMENT, so SQLite reuses a freed rowid.
    # Create the replacement first (while the old row still holds the max id),
    # then delete the old one, so the new mid is guaranteed to differ.
    mid = store.create_meeting(m["title"], created_at, m["lang"])
    for existing in to_replace:
        for d in store.delete_meeting(existing["id"]):
            if d and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(os.path.join(data_dir, str(existing["id"])),
                      ignore_errors=True)
    if m.get("notes"):
        store.set_notes(mid, m["notes"])
    if m.get("status") == "finalized":
        store.finalize_meeting(mid)
    for s in bundle_dict.get("segments", []):
        store.add_segment(mid, s["idx"], "", s["started_at"],
                          s["duration_s"], s["origin"])
    for t in bundle_dict.get("transcripts", []):
        store.add_transcript(mid, t["profile"], t["track"], t["start_ms"],
                             t["end_ms"], t["speaker"], t["text"])
    for s in bundle_dict.get("summaries", []):
        store.add_summary(mid, s["kind"], s["lang"], s["text"],
                          s["model"], s["created_at"])
    dst_dir = os.path.join(data_dir, str(mid))
    os.makedirs(dst_dir, exist_ok=True)
    for track, path in (track_files or {}).items():
        shutil.copyfile(path, os.path.join(dst_dir, f"{track}.m4a"))
    return mid
