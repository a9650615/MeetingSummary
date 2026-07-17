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


def _copy_tracks(data_dir, mid, track_files):
    dst_dir = os.path.join(data_dir, str(mid))
    os.makedirs(dst_dir, exist_ok=True)
    for track, path in (track_files or {}).items():
        shutil.copyfile(path, os.path.join(dst_dir, f"{track}.m4a"))


def ingest_bundle(store, data_dir, bundle_dict, track_files):
    """Insert or top-up a bundle, keyed by created_at. Returns (mid, is_new).

    NEW meeting (unseen created_at) -> full insert (segments + transcripts +
    summaries + tracks).

    EXISTING meeting -> top-up: refresh the non-transcript info the user may add
    after the first push (title, notes, status, summaries, tracks) but PRESERVE
    the transcripts, including any FireRed correction rows / progress. Summaries
    are replaced only when the bundle actually carries some (an early push before
    the summary exists must not wipe a later one). The transcript is owned by the
    first push + the VM's FireRed pass; a top-up never touches it."""
    m = bundle_dict["meeting"]
    created_at = m["created_at"]
    existing = next((e for e in store.list_meetings()
                     if e["created_at"] == created_at), None)

    if existing is not None:  # top-up: metadata/summary/tracks only
        mid = existing["id"]
        store.update_title(mid, m.get("title") or existing["title"])
        if m.get("notes"):
            store.set_notes(mid, m["notes"])
        if m.get("status") == "finalized":
            store.finalize_meeting(mid)
        if bundle_dict.get("summaries"):  # replace only when bundle brings some
            store.db.execute("DELETE FROM summaries WHERE meeting_id=?", (mid,))
            store.db.commit()
            for s in bundle_dict["summaries"]:
                store.add_summary(mid, s["kind"], s["lang"], s["text"],
                                  s["model"], s["created_at"])
        # sync speaker labels (人員命名 the user did on the Mac after the first
        # push). Speakers live on transcript rows, but a top-up must NOT rewrite
        # transcript text (preserving FireRed). So match rows by (track, start_ms,
        # end_ms) and update ONLY the speaker — this propagates a rename to both
        # the local rows AND the FireRed rows (which inherited the same span).
        spk = {(t["track"], t["start_ms"], t["end_ms"]): t["speaker"]
               for t in bundle_dict.get("transcripts", [])}
        if spk:
            changed = False
            for r in store.list_transcripts(mid):
                key = (r["track"], r["start_ms"], r["end_ms"])
                if key in spk and spk[key] != r["speaker"]:
                    store.db.execute("UPDATE transcripts SET speaker=? WHERE id=?",
                                     (spk[key], r["id"]))
                    changed = True
            if changed:
                store.db.commit()
        _copy_tracks(data_dir, mid, track_files)
        return mid, False

    mid = store.create_meeting(m["title"], created_at, m["lang"])
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
    _copy_tracks(data_dir, mid, track_files)
    return mid, True
