"""Meeting <-> portable bundle (zip). The single source of the bundle schema,
imported by both the Mac push plugin and the VM server. No Apple/ASR deps."""
import base64
import json
import os
import re
import shutil
import zipfile

# an unnamed session cluster label (說話者1, 說話者2…) — NOT a real person, so its
# voiceprint must never be synced to the central 聲紋庫 nor treated as a name.
_CLUSTER_LABEL = re.compile(r"^說話者\d+$")


def _rows(rows):
    return [dict(r) for r in rows]


def _bundle_speakers(store):
    """The global voiceprint library (聲紋庫) as portable JSON — name + base64
    centroid (raw np.float32 bytes, 192-d) + count. Pure numeric data, so it
    ports to the VM's store.db verbatim; matching there is plain numpy cosine.
    Only 有標記 (真名) voiceprints are synced — a session cluster placeholder
    (說話者N) or an empty name is not a person, so it never goes to the VM."""
    out = []
    for s in store.list_speakers():
        c, name = s["centroid"], (s["name"] or "").strip()
        if not c or not name or _CLUSTER_LABEL.match(name):
            continue
        out.append({"name": name, "count": s["count"],
                    "centroid_b64": base64.b64encode(bytes(c)).decode("ascii")})
    return out


def _sync_speakers(store, bundle_dict):
    """Merge the bundle's voiceprint library into the store, NON-DESTRUCTIVELY:
    add only names the store doesn't already have. Never overwrites a voiceprint
    the VM already holds (a name a human curated locally on the VM stays put)."""
    have = {s["name"] for s in store.list_speakers()}
    for sp in bundle_dict.get("speakers", []):
        name, b64 = sp.get("name"), sp.get("centroid_b64")
        if name and b64 and name not in have:
            store.add_speaker(name, base64.b64decode(b64))
            have.add(name)


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
        "tags": store.tags_for(mid),   # meeting 標籤 (#客戶 #1on1…), synced on ingest
        # the whole voiceprint library rides along so the VM stays in sync (central
        # 聲紋庫). Small (192 floats/speaker); merged non-destructively on ingest.
        "speakers": _bundle_speakers(store),
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


def _sync_tags(store, mid, bundle_dict):
    """Make the meeting's tags on the VM match the bundle's (add missing, drop
    removed). Tags are the meeting's own labels, so they mirror the Mac exactly."""
    want = set(bundle_dict.get("tags") or [])
    have = set(store.tags_for(mid))
    for t in have - want:
        store.remove_tag(mid, t)
    for t in want - have:
        store.add_tag(mid, t)


def _copy_tracks(data_dir, mid, track_files):
    dst_dir = os.path.join(data_dir, str(mid))
    os.makedirs(dst_dir, exist_ok=True)
    for track, path in (track_files or {}).items():
        shutil.copyfile(path, os.path.join(dst_dir, f"{track}.m4a"))


def _local_sig(transcripts):
    """Identity of the local (non-FireRed) transcript for change detection:
    (track, start_ms, end_ms, text) per row, sorted. Speaker is excluded — a
    pure rename is handled in place, not treated as a transcript replacement."""
    return sorted((t["track"], t["start_ms"], t["end_ms"], t["text"])
                  for t in transcripts
                  if t["profile"] not in ("firered", "firered_staging"))


def _replace_transcript(store, mid, bundle_dict):
    """The Mac transcript text changed (re-transcribe / edits) — swap it in and
    drop the now-stale FireRed correction so it re-runs against the new text."""
    store.db.execute(
        "DELETE FROM transcripts WHERE meeting_id=? "
        "AND profile NOT IN ('firered','firered_staging')", (mid,))
    store.db.commit()
    store.clear_transcripts(mid, profile="firered")
    store.clear_transcripts(mid, profile="firered_staging")
    for t in bundle_dict.get("transcripts", []):
        store.add_transcript(mid, t["profile"], t["track"], t["start_ms"],
                             t["end_ms"], t["speaker"], t["text"])


def ingest_bundle(store, data_dir, bundle_dict, track_files):
    """Insert or top-up a bundle, keyed by created_at. Returns
    (mid, is_new, retranscribe) — retranscribe True means the FireRed pass must
    (re-)run (a new meeting, or a top-up whose transcript text changed).

    NEW meeting -> full insert.

    EXISTING meeting -> top-up: refresh title/notes/status/summaries/tracks +
    voiceprint library. Transcript handling:
      - text UNCHANGED -> preserve it (and any FireRed rows/progress); a pure
        speaker rename is applied in place.
      - text CHANGED (re-transcribe/edit on the Mac) -> replace the local
        transcript and drop the stale FireRed so it re-runs on the new text."""
    m = bundle_dict["meeting"]
    created_at = m["created_at"]
    existing = next((e for e in store.list_meetings()
                     if e["created_at"] == created_at), None)

    if existing is not None:  # top-up
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

        retranscribe = bool(bundle_dict.get("transcripts")) and \
            _local_sig(bundle_dict["transcripts"]) != \
            _local_sig(_rows(store.list_transcripts(mid)))
        if retranscribe:
            _replace_transcript(store, mid, bundle_dict)
        else:
            # text same -> only sync speaker renames in place (never touch text),
            # keyed by (track, start_ms, end_ms); propagates to local + FireRed rows.
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
        _sync_speakers(store, bundle_dict)
        _sync_tags(store, mid, bundle_dict)
        _copy_tracks(data_dir, mid, track_files)
        return mid, False, retranscribe

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
    _sync_speakers(store, bundle_dict)
    _sync_tags(store, mid, bundle_dict)
    _copy_tracks(data_dir, mid, track_files)
    return mid, True, True
