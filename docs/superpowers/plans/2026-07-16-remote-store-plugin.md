# Remote-Store Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in capability to push finished meetings from the Mac to a shared Azure VM that hosts a lightweight web viewer (list/search/playback/transcript/summary/export) and slowly re-corrects transcripts with FireRedASR — non-destructively, no ASR/voiceprint on the critical path.

**Architecture:** A shared `viewer/` module (Linux-safe, zero Apple deps) provides bundle (de)serialization, transcript-profile selection, HTML render, and read-only routes over `store.py`. A standalone `server/` folder mounts `viewer/` + an `/ingest-bundle` write endpoint and runs a CPU-pinned FireRed worker, deployed to VM `10.102.0.7:5556` in its own Docker container. A Mac-side `plugins/remote_store/` package (absent in the base build) adds a "上傳到 server" button via a minimal try-import seam in `app.py`.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, SQLite (`store.py`), sherpa-onnx + onnxruntime (FireRed, CPU), ffmpeg (M4A decode on Linux), Docker Compose, `az ssh` + rsync deploy.

## Global Constraints

- The Mac app (`app.py`, ASR/live/recording pipeline) is **unchanged** except ONE try-import seam + ONE guarded button in the detail page.
- `viewer/` and `server/` MUST NOT import `app`, `asr`, `backends`, `live_session`, `live`, `diarize`, or `recorder`'s decode path — they must import cleanly on x86 Linux with no MLX/Qwen/afconvert.
- The VM runs **FireRed batch correction only**. No live ASR, no voiceprint, no diarization on the VM.
- FireRed is **non-destructive**: results stored under `profile="firered"`; Mac-pushed rows are never cleared. Speaker labels + start/end are inherited from the pushed row.
- No auth (vnet boundary). VM server on port **5556**, isolated from acp's container on :5555.
- Audio on the VM is served as **M4A directly** to the browser; the VM never assembles WAV.
- Storage is M4A-only. Raw PCM/WAV is never uploaded.
- Follow repo test convention: `pytest` + `tmp_path`, assert-based, no new frameworks.
- Reuse `store.py` verbatim on both sides (it is already pure SQLite with a `profile` column).

---

## File Structure

- `viewer/__init__.py` — package marker.
- `viewer/bundle.py` — `meeting_to_bundle`, `write_bundle_zip` (Mac), `read_bundle_zip`, `ingest_bundle` (server). The one place that knows the bundle schema.
- `viewer/render.py` — `pick_transcripts`, `group_lines`, `render_index`, `render_detail`, `export_md`. Pure functions, no I/O.
- `viewer/routes.py` — `mount_viewer(app, store, data_dir)`; read-only GET routes + M4A serving.
- `plugins/remote_store/__init__.py` — `register(app, store)`, `enabled()`.
- `plugins/remote_store/push.py` — `build_and_push(store, mid, vm_url)`.
- `server/main.py` — `build_server()` app factory + `POST /ingest-bundle`.
- `server/firered_worker.py` — `FireRedWorker` (queue, per-row re-transcribe, ffmpeg decode, fixed-window fallback, non-destructive write).
- `server/requirements.txt`, `server/Dockerfile`, `server/docker/docker-compose.yml`, `server/deploy-vm.sh` — deploy.
- Tests under `tests/`: `test_viewer_bundle.py`, `test_viewer_render.py`, `test_viewer_routes.py`, `test_remote_push.py`, `test_server_ingest.py`, `test_firered_worker.py`.

---

## Bundle format (reference for all tasks)

A `.zip` containing:

- `meeting.json`:
  ```json
  {
    "meeting": {"title": "...", "created_at": 1721111111.0, "lang": "zh-TW",
                "status": "finalized", "notes": ""},
    "segments": [{"idx": 0, "started_at": 1721111111.0, "duration_s": 61.0, "origin": "live"}],
    "transcripts": [{"profile": "accurate", "track": "mixed", "start_ms": 0,
                     "end_ms": 3200, "speaker": "我", "text": "開始開會"}],
    "summaries": [{"kind": "minutes", "lang": "zh-TW", "text": "...",
                   "model": "...", "created_at": 1721111500.0}],
    "tracks": ["mixed"]
  }
  ```
- `tracks/<track>.m4a` for each name in `meeting["tracks"]` (pre-assembled/pre-mixed on the Mac).

---

## Phase A — `viewer/` shared module

### Task 1: `viewer/bundle.py` — bundle (de)serialization

**Files:**
- Create: `viewer/__init__.py` (empty), `viewer/bundle.py`
- Test: `tests/test_viewer_bundle.py`

**Interfaces:**
- Produces:
  - `meeting_to_bundle(store, mid, track_names) -> dict` — the `meeting.json` dict (does NOT read audio).
  - `write_bundle_zip(zip_path, bundle_dict, track_files) -> None` where `track_files: dict[str, str]` maps track name → local `.m4a` path.
  - `read_bundle_zip(zip_path, dest_dir) -> tuple[dict, dict]` — returns `(bundle_dict, {track: extracted_m4a_path})`.
  - `ingest_bundle(store, data_dir, bundle_dict, track_files) -> int` — inserts into `store`, copies each `track_files[t]` to `data_dir/<mid>/<t>.m4a`, returns the new `mid`. Idempotent by (title, created_at): if a meeting with the same `created_at` exists, replace it.
- Consumes: `store.py` API (`create_meeting`, `add_segment`, `add_transcript`, `add_summary`, `set_notes`, `finalize_meeting`, `list_meetings`, `delete_meeting`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_viewer_bundle.py
import json, zipfile
from pathlib import Path
from store import Store
from viewer import bundle


def _seed(store):
    mid = store.create_meeting("測試會議", 1721111111.0, "zh-TW")
    store.add_segment(mid, 0, "", 1721111111.0, 61.0, "live")
    store.add_transcript(mid, "accurate", "mixed", 0, 3200, "我", "開始開會")
    store.add_summary(mid, "minutes", "zh-TW", "重點", "m", 1721111500.0)
    store.finalize_meeting(mid)
    return mid


def test_meeting_to_bundle_shape(tmp_path):
    store = Store(tmp_path / "a.db")
    mid = _seed(store)
    b = bundle.meeting_to_bundle(store, mid, ["mixed"])
    assert b["meeting"]["title"] == "測試會議"
    assert b["tracks"] == ["mixed"]
    assert b["transcripts"][0]["text"] == "開始開會"
    assert b["summaries"][0]["kind"] == "minutes"


def test_roundtrip_zip_and_ingest(tmp_path):
    src = Store(tmp_path / "src.db")
    mid = _seed(src)
    b = bundle.meeting_to_bundle(src, mid, ["mixed"])
    (tmp_path / "mixed.m4a").write_bytes(b"FAKE_M4A")
    zp = tmp_path / "bundle.zip"
    bundle.write_bundle_zip(zp, b, {"mixed": str(tmp_path / "mixed.m4a")})

    dst = Store(tmp_path / "dst.db")
    data_dir = tmp_path / "data"
    b2, tracks = bundle.read_bundle_zip(zp, tmp_path / "extract")
    new_mid = bundle.ingest_bundle(dst, str(data_dir), b2, tracks)

    rows = dst.list_transcripts(new_mid)
    assert rows[0]["text"] == "開始開會"
    assert (data_dir / str(new_mid) / "mixed.m4a").read_bytes() == b"FAKE_M4A"


def test_ingest_is_idempotent_by_created_at(tmp_path):
    src = Store(tmp_path / "src.db"); mid = _seed(src)
    b = bundle.meeting_to_bundle(src, mid, [])
    dst = Store(tmp_path / "dst.db")
    m1 = bundle.ingest_bundle(dst, str(tmp_path / "d"), b, {})
    m2 = bundle.ingest_bundle(dst, str(tmp_path / "d"), b, {})
    assert len([m for m in dst.list_meetings() if m["created_at"] == 1721111111.0]) == 1
    assert m2 != m1  # replaced, new row id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_viewer_bundle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'viewer'`

- [ ] **Step 3: Write minimal implementation**

```python
# viewer/__init__.py
# (empty package marker)
```

```python
# viewer/bundle.py
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
    for existing in store.list_meetings():
        if existing["created_at"] == created_at:
            for d in store.delete_meeting(existing["id"]):
                if d and os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
            shutil.rmtree(os.path.join(data_dir, str(existing["id"])),
                          ignore_errors=True)
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
    dst_dir = os.path.join(data_dir, str(mid))
    os.makedirs(dst_dir, exist_ok=True)
    for track, path in (track_files or {}).items():
        shutil.copyfile(path, os.path.join(dst_dir, f"{track}.m4a"))
    return mid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_viewer_bundle.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add viewer/__init__.py viewer/bundle.py tests/test_viewer_bundle.py
git commit -m "feat(viewer): bundle serialize/ingest for remote store"
```

---

### Task 2: `viewer/render.py` — profile selection + HTML + export

**Files:**
- Create: `viewer/render.py`
- Test: `tests/test_viewer_render.py`

**Interfaces:**
- Produces:
  - `pick_transcripts(transcripts) -> list[dict]` — if any row has `profile == "firered"`, return only those; else return all non-firered rows. Sorted by `start_ms`.
  - `active_profile(transcripts) -> str` — `"firered"` or `"local"` (for the badge).
  - `group_lines(transcripts) -> list[dict]` — `pick_transcripts` mapped to `{"speaker","text","start_ms","track"}`.
  - `export_md(meeting, transcripts, summaries) -> str`.
  - `render_index(meetings) -> str`, `render_detail(meeting, transcripts, summaries, tracks, tags) -> str` — full HTML strings.
- Consumes: plain row dicts (from `_rows`) / sqlite Rows (index via `["key"]`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_viewer_render.py
from viewer import render


def test_pick_prefers_firered():
    ts = [{"profile": "accurate", "track": "mixed", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "local text"},
          {"profile": "firered", "track": "mixed", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "corrected"}]
    picked = render.pick_transcripts(ts)
    assert [p["text"] for p in picked] == ["corrected"]
    assert render.active_profile(ts) == "firered"


def test_pick_falls_back_to_local():
    ts = [{"profile": "accurate", "track": "mixed", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "local"}]
    assert [p["text"] for p in render.pick_transcripts(ts)] == ["local"]
    assert render.active_profile(ts) == "local"


def test_export_md_has_title_and_lines():
    md = render.export_md({"title": "週會"},
                          [{"profile": "accurate", "track": "mixed", "start_ms": 0,
                            "end_ms": 1, "speaker": "我", "text": "hi"}],
                          [{"kind": "minutes", "text": "重點"}])
    assert "# 週會" in md and "我：hi" in md and "重點" in md


def test_render_detail_contains_audio_and_text():
    html = render.render_detail(
        {"id": 7, "title": "週會", "created_at": 1721111111.0},
        [{"profile": "firered", "track": "mixed", "start_ms": 0, "end_ms": 1,
          "speaker": "我", "text": "corrected"}],
        [], ["mixed"], [])
    assert "/meetings/7/audio/mixed.m4a" in html
    assert "corrected" in html and "FireRed" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_viewer_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'viewer.render'`

- [ ] **Step 3: Write minimal implementation**

```python
# viewer/render.py
"""Pure render helpers for the remote viewer. No I/O, no Apple/ASR deps."""
import html
import time


def pick_transcripts(transcripts):
    rows = [dict(t) for t in transcripts]
    fr = [t for t in rows if t.get("profile") == "firered"]
    chosen = fr if fr else [t for t in rows if t.get("profile") != "firered"]
    return sorted(chosen, key=lambda t: t.get("start_ms") or 0)


def active_profile(transcripts):
    return "firered" if any(dict(t).get("profile") == "firered"
                            for t in transcripts) else "local"


def group_lines(transcripts):
    return [{"speaker": t.get("speaker") or "", "text": t.get("text") or "",
             "start_ms": t.get("start_ms") or 0, "track": t.get("track")}
            for t in pick_transcripts(transcripts)]


def export_md(meeting, transcripts, summaries):
    out = [f"# {meeting['title']}", ""]
    for s in summaries:
        out += [f"## 摘要（{s.get('kind','')}）", s.get("text") or "", ""]
    out += ["## 逐字稿", ""]
    for line in group_lines(transcripts):
        out.append(f"{line['speaker']}：{line['text']}")
    return "\n".join(out) + "\n"


def _e(s):
    return html.escape(str(s) if s is not None else "")


def _page(title, body):
    return (f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title><style>body{{font-family:system-ui,"
            f"'PingFang TC',sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;"
            f"line-height:1.6}}a{{color:#06c;text-decoration:none}}.badge{{font-size:.75rem;"
            f"background:#eee;border-radius:6px;padding:.1rem .4rem;margin-left:.5rem}}"
            f".line{{margin:.3rem 0}}.spk{{color:#888;margin-right:.4rem}}"
            f"audio{{width:100%;margin:.5rem 0}}</style></head><body>{body}</body></html>")


def render_index(meetings):
    items = []
    for m in meetings:
        d = dict(m)
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(d.get("created_at") or 0))
        items.append(f"<li><a href='/m/{d['id']}'>{_e(d.get('title') or '未命名')}</a>"
                     f" <span class='badge'>{ts}</span></li>")
    body = ("<h1>會議</h1>"
            "<form action='/search'><input name='q' placeholder='搜尋…'>"
            "<button>搜尋</button></form>"
            f"<ul>{''.join(items) or '<li>（無會議）</li>'}</ul>")
    return _page("會議", body)


def render_search(q, results):
    items = [f"<li><a href='/m/{r['id']}'>{_e(r['title'])}</a> "
             f"<small>{_e(r.get('snippet') or '')}</small></li>" for r in results]
    return _page(f"搜尋 {q}",
                 f"<h1>搜尋：{_e(q)}</h1><p><a href='/'>← 全部</a></p>"
                 f"<ul>{''.join(items) or '<li>無結果</li>'}</ul>")


def render_detail(meeting, transcripts, summaries, tracks, tags):
    m = dict(meeting)
    prof = active_profile(transcripts)
    badge = ("<span class='badge'>FireRed 校正版</span>" if prof == "firered"
             else "<span class='badge'>本地版（校正中…）</span>")
    audio = "".join(
        f"<div><b>{_e(t)}</b><audio controls preload='none' "
        f"src='/meetings/{m['id']}/audio/{_e(t)}.m4a'></audio></div>" for t in tracks)
    sums = "".join(f"<h2>摘要（{_e(s.get('kind',''))}）</h2><p>{_e(s.get('text'))}</p>"
                   for s in summaries)
    lines = "".join(f"<div class='line'><span class='spk'>{_e(l['speaker'])}</span>"
                    f"{_e(l['text'])}</div>" for l in group_lines(transcripts))
    body = (f"<p><a href='/'>← 全部</a></p><h1>{_e(m.get('title') or '未命名')}{badge}</h1>"
            f"<p><a href='/meetings/{m['id']}/export'>下載逐字稿 (.md)</a></p>"
            f"{audio}{sums}<h2>逐字稿</h2>{lines}")
    return _page(m.get("title") or "會議", body)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_viewer_render.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add viewer/render.py tests/test_viewer_render.py
git commit -m "feat(viewer): profile-aware render + md export"
```

---

### Task 3: `viewer/routes.py` — read-only routes + M4A serving

**Files:**
- Create: `viewer/routes.py`
- Test: `tests/test_viewer_routes.py`

**Interfaces:**
- Produces: `mount_viewer(app, store, data_dir)` — registers on a FastAPI `app`:
  - `GET /` → index HTML
  - `GET /m/{mid}` → detail HTML (404 if missing)
  - `GET /search?q=` → search HTML
  - `GET /meetings` → JSON list (parity with base app's shape, minus audio helpers)
  - `GET /meetings/{mid}/audio/{track}.m4a` → `FileResponse(data_dir/<mid>/<track>.m4a)` (native HTTP Range) or 404
  - `GET /meetings/{mid}/export` → markdown download
  - `meeting_tracks(data_dir, mid) -> list[str]` helper (glob `data_dir/<mid>/*.m4a`).
- Consumes: `viewer.render`, `store.py`, `viewer.bundle._rows` pattern.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_viewer_routes.py
from fastapi import FastAPI
from fastapi.testclient import TestClient
from store import Store
from viewer.routes import mount_viewer


def _app(tmp_path):
    store = Store(tmp_path / "s.db")
    mid = store.create_meeting("週會", 1721111111.0, "zh-TW")
    store.add_segment(mid, 0, "", 1721111111.0, 5.0, "live")
    store.add_transcript(mid, "accurate", "mixed", 0, 1000, "我", "hello")
    store.finalize_meeting(mid)
    data = tmp_path / "data"; (data / str(mid)).mkdir(parents=True)
    (data / str(mid) / "mixed.m4a").write_bytes(b"AUDIOBYTES")
    app = FastAPI(); mount_viewer(app, store, str(data))
    return TestClient(app), mid


def test_index_and_detail(tmp_path):
    c, mid = _app(tmp_path)
    assert "週會" in c.get("/").text
    assert "hello" in c.get(f"/m/{mid}").text
    assert c.get("/m/9999").status_code == 404


def test_audio_served_with_range(tmp_path):
    c, mid = _app(tmp_path)
    r = c.get(f"/meetings/{mid}/audio/mixed.m4a")
    assert r.status_code == 200 and r.content == b"AUDIOBYTES"
    rng = c.get(f"/meetings/{mid}/audio/mixed.m4a", headers={"Range": "bytes=0-3"})
    assert rng.status_code == 206 and rng.content == b"AUDI"
    assert c.get(f"/meetings/{mid}/audio/nope.m4a").status_code == 404


def test_export_download(tmp_path):
    c, mid = _app(tmp_path)
    r = c.get(f"/meetings/{mid}/export")
    assert r.status_code == 200 and "我：hello" in r.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_viewer_routes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'viewer.routes'`

- [ ] **Step 3: Write minimal implementation**

```python
# viewer/routes.py
"""Read-only viewer routes over a Store, serving M4A directly. Linux-safe."""
import glob
import os
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

from viewer import render


def meeting_tracks(data_dir, mid):
    paths = sorted(glob.glob(os.path.join(data_dir, str(mid), "*.m4a")))
    order = {"mixed": 0, "system": 1, "mic": 2}
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return sorted(names, key=lambda n: order.get(n, 9))


def mount_viewer(app, store, data_dir):
    @app.get("/", response_class=HTMLResponse)
    def index():
        return render.render_index(store.list_meetings())

    @app.get("/search", response_class=HTMLResponse)
    def search(q: str = ""):
        results = [{"id": r["id"], "title": r["title"],
                    "snippet": (r["t_snip"] or r["s_snip"] or "")}
                   for r in store.search(q)]
        return render.render_search(q, results)

    @app.get("/meetings")
    def list_meetings():
        out = []
        for m in store.list_meetings():
            d = dict(m)
            segs = store.list_segments(m["id"])
            d["tracks"] = meeting_tracks(data_dir, m["id"])
            d["duration_s"] = round(sum((s["duration_s"] or 0) for s in segs))
            d["n_segments"] = len(segs)
            out.append(d)
        return out

    @app.get("/m/{mid}", response_class=HTMLResponse)
    def detail(mid: int):
        m = store.get_meeting(mid)
        if m is None:
            raise HTTPException(404, "meeting not found")
        return render.render_detail(dict(m), store.list_transcripts(mid),
                                    store.list_summaries(mid),
                                    meeting_tracks(data_dir, mid),
                                    store.tags_for(mid))

    @app.get("/meetings/{mid}/audio/{track}.m4a")
    def audio(mid: int, track: str):
        path = os.path.join(data_dir, str(mid), f"{os.path.basename(track)}.m4a")
        if not os.path.exists(path):
            raise HTTPException(404, "track not found")
        return FileResponse(path, media_type="audio/mp4")  # FileResponse => Range

    @app.get("/meetings/{mid}/export")
    def export(mid: int):
        m = store.get_meeting(mid)
        if m is None:
            raise HTTPException(404, "meeting not found")
        md = render.export_md(dict(m), store.list_transcripts(mid),
                              store.list_summaries(mid))
        safe = "".join(c for c in m["title"] if c.isalnum() or c in " -_")[:40].strip()
        fname = (safe or f"meeting-{mid}") + ".md"
        return Response(md, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition":
                                 f"attachment; filename*=UTF-8''{quote(fname)}"})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_viewer_routes.py -v`
Expected: PASS (3 tests). Note: `FileResponse` returns 206 for a `Range` request automatically.

- [ ] **Step 5: Commit**

```bash
git add viewer/routes.py tests/test_viewer_routes.py
git commit -m "feat(viewer): read-only routes + direct M4A range serving"
```

---

## Phase B — Mac push plugin

### Task 4: `plugins/remote_store/push.py` — build + upload bundle

**Files:**
- Create: `plugins/__init__.py` (empty), `plugins/remote_store/__init__.py`, `plugins/remote_store/push.py`
- Test: `tests/test_remote_push.py`

**Interfaces:**
- Produces:
  - `build_and_push(store, mid, vm_url, *, assemble=None, to_m4a=None, http_post=None) -> dict` — assembles each playable track to a temp `.m4a`, builds the bundle, zips it, POSTs to `f"{vm_url}/ingest-bundle"` as multipart file field `bundle`. Returns `{"ok": bool, "status": int, "mid": int|None}`. The `assemble`/`to_m4a`/`http_post` params are injection seams for tests; defaults wire to the real Mac helpers lazily.
- Consumes: `viewer.bundle`, and (lazily, at call time) `app._meeting_tracks`, `app._assemble_track`, `recorder.pcm_to_m4a`. Never import these at module top (keeps the plugin importable in tests without the Apple stack).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_remote_push.py
import zipfile
from store import Store
from plugins.remote_store import push


def test_build_and_push_posts_zip(tmp_path):
    store = Store(tmp_path / "s.db")
    mid = store.create_meeting("週會", 1721111111.0, "zh-TW")
    store.add_segment(mid, 0, "", 1721111111.0, 5.0, "live")
    store.add_transcript(mid, "accurate", "mixed", 0, 1000, "我", "hi")
    store.finalize_meeting(mid)

    def fake_assemble(_store, _mid, track):   # pretend one mixed track exists
        return b"\x00\x01" * 100 if track == "mixed" else None

    def fake_to_m4a(pcm_path, m4a_path):
        open(m4a_path, "wb").write(b"M4A:" + open(pcm_path, "rb").read()[:4])

    captured = {}

    class Resp:  # minimal requests.Response stand-in
        status_code = 200
        def json(self):
            return {"mid": 42}

    def fake_post(url, files=None, timeout=None):
        captured["url"] = url
        captured["zip"] = files["bundle"][1].read()
        return Resp()

    res = push.build_and_push(store, mid, "http://vm:5556",
                              assemble=fake_assemble, to_m4a=fake_to_m4a,
                              http_post=fake_post,
                              tracks=["mixed"])
    assert res["ok"] and res["status"] == 200 and res["mid"] == 42
    assert captured["url"] == "http://vm:5556/ingest-bundle"
    # the posted bytes are a valid zip carrying meeting.json + the track
    import io
    with zipfile.ZipFile(io.BytesIO(captured["zip"])) as z:
        names = z.namelist()
    assert "meeting.json" in names and "tracks/mixed.m4a" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_remote_push.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'plugins.remote_store'`

- [ ] **Step 3: Write minimal implementation**

```python
# plugins/__init__.py
# (empty)
```

```python
# plugins/remote_store/__init__.py
"""Opt-in remote-store plugin. Present == feature on. A minimal try-import seam
in app.py calls register(); the base build ships without this folder."""
import os

VM_URL = os.environ.get("REMOTE_STORE_URL", "http://10.102.0.7:5556")


def enabled():
    return True  # presence of this package is the switch


def register(app, store):
    """Add the push route. The detail-page button is injected by the app.py seam
    guarded on this module importing."""
    from fastapi import HTTPException
    from plugins.remote_store import push as _push

    @app.post("/remote/push/{mid}")
    def remote_push(mid: int):
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        res = _push.build_and_push(store, mid, VM_URL)
        if not res["ok"]:
            raise HTTPException(502, f"push failed ({res['status']})")
        return res
```

```python
# plugins/remote_store/push.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_remote_push.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add plugins/__init__.py plugins/remote_store/ tests/test_remote_push.py
git commit -m "feat(plugin): remote-store Mac push (build+upload bundle)"
```

---

### Task 5: `app.py` seam — register plugin + detail button

**Files:**
- Modify: `app.py` (inside `create_app`, near the end where routes are registered ~app.py:3120; and the detail page builder that returns at ~app.py:2249)
- Test: `tests/test_plugin_seam.py`

**Interfaces:**
- Consumes: `plugins.remote_store.register` (optional).
- Produces: a module-level `REMOTE_PLUGIN` flag on the app module set True when the plugin imported, so the detail page can render the button.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plugin_seam.py
import importlib


def test_seam_present_and_guarded():
    src = open("app.py", encoding="utf-8").read()
    # the seam must be a try-import that never hard-fails the base app
    assert "plugins.remote_store" in src
    assert "REMOTE_PLUGIN" in src
    # button must be guarded on the flag, not unconditional
    assert "/remote/push/" in src


def test_plugin_imports_standalone():
    # base-build safety: importing the plugin must not require the Apple stack
    mod = importlib.import_module("plugins.remote_store")
    assert mod.enabled() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_plugin_seam.py -v`
Expected: FAIL — `assert "plugins.remote_store" in src` fails (seam not added yet)

- [ ] **Step 3: Add the seam and the guarded button**

In `create_app`, immediately before `return app` (find the final `return app` inside `create_app`), add:

```python
    # --- opt-in remote-store plugin (absent in the base build) ---
    try:
        import plugins.remote_store as _remote_plugin
        _remote_plugin.register(app, store)
        globals()["REMOTE_PLUGIN"] = True
    except Exception:
        globals()["REMOTE_PLUGIN"] = False
```

Add a module-level default near the other module globals (top of `app.py`, after imports):

```python
REMOTE_PLUGIN = False
```

In the detail page HTML builder (the function returning the `/m/{mid}` page, around app.py:2249), add the button where the other per-meeting action links are rendered. Insert into the actions area:

```python
    remote_btn = ""
    if globals().get("REMOTE_PLUGIN"):
        remote_btn = (f"<button onclick=\"fetch('/remote/push/{mid}',{{method:'POST'}})"
                      f".then(r=>r.json()).then(j=>alert(j.ok?'已上傳到 server':'上傳失敗'))"
                      f".catch(()=>alert('上傳失敗'))\">上傳到 server</button>")
```

Then include `remote_btn` in the actions HTML string for that page (concatenate it alongside the existing export/rename buttons).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_plugin_seam.py -v`
Expected: PASS

Then verify the base suite still green (seam must not break anything):

Run: `.venv/bin/pytest -q`
Expected: PASS (all existing tests + the new ones)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_plugin_seam.py
git commit -m "feat(app): opt-in remote-store plugin seam + guarded push button"
```

---

## Phase C — `server/` standalone VM host

### Task 6: `server/main.py` — app factory + `/ingest-bundle`

**Files:**
- Create: `server/__init__.py` (empty), `server/main.py`
- Test: `tests/test_server_ingest.py`

**Interfaces:**
- Produces:
  - `build_server(db_path=None, data_dir=None) -> FastAPI` — creates a `Store`, mounts viewer routes, adds `POST /ingest-bundle` (accepts multipart file field `bundle`, a zip; runs `viewer.bundle.read_bundle_zip` + `ingest_bundle`; returns `{"mid": int}`). On enqueue it calls `app.state.on_ingest(mid)` if set (Task 9 wires the worker).
  - `GET /health` → `{"ok": true}`.
- Consumes: `store.Store`, `viewer.routes.mount_viewer`, `viewer.bundle`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server_ingest.py
import io, json, zipfile
from fastapi.testclient import TestClient
from server.main import build_server


def _bundle_zip():
    meta = {"meeting": {"title": "遠端會議", "created_at": 1721111111.0,
                        "lang": "zh-TW", "status": "finalized", "notes": ""},
            "segments": [{"idx": 0, "started_at": 1721111111.0,
                          "duration_s": 5.0, "origin": "live"}],
            "transcripts": [{"profile": "accurate", "track": "mixed", "start_ms": 0,
                             "end_ms": 1000, "speaker": "我", "text": "hi"}],
            "summaries": [], "tracks": ["mixed"]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("meeting.json", json.dumps(meta, ensure_ascii=False))
        z.writestr("tracks/mixed.m4a", b"AUDIO")
    return buf.getvalue()


def test_ingest_then_view(tmp_path):
    app = build_server(db_path=str(tmp_path / "s.db"),
                       data_dir=str(tmp_path / "data"))
    c = TestClient(app)
    r = c.post("/ingest-bundle",
               files={"bundle": ("b.zip", _bundle_zip(), "application/zip")})
    assert r.status_code == 200
    mid = r.json()["mid"]
    assert "遠端會議" in c.get("/").text
    assert "hi" in c.get(f"/m/{mid}").text
    assert c.get(f"/meetings/{mid}/audio/mixed.m4a").content == b"AUDIO"


def test_ingest_rejects_bad_zip(tmp_path):
    app = build_server(db_path=str(tmp_path / "s.db"),
                       data_dir=str(tmp_path / "data"))
    c = TestClient(app)
    r = c.post("/ingest-bundle",
               files={"bundle": ("b.zip", b"not a zip", "application/zip")})
    assert r.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_server_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.main'`

- [ ] **Step 3: Write minimal implementation**

```python
# server/__init__.py
# (empty)
```

```python
# server/main.py
"""Standalone remote-store server: read-only viewer + bundle ingest. Runs on the
VM, x86 Linux, no Apple/ASR deps. FireRed worker wired in a later task."""
import os
import tempfile
import zipfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from store import Store
from viewer import bundle
from viewer.routes import mount_viewer

DB_PATH = os.environ.get("STORE_DB", "data/store.db")
DATA_DIR = os.environ.get("STORE_DATA", "data")


def build_server(db_path=None, data_dir=None):
    store = Store(db_path or DB_PATH)
    data = data_dir or DATA_DIR
    os.makedirs(data, exist_ok=True)
    app = FastAPI()
    app.state.store = store
    app.state.data_dir = data
    app.state.on_ingest = None  # Task 9 sets this to the FireRed worker's enqueue
    mount_viewer(app, store, data)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/ingest-bundle")
    async def ingest_bundle(bundle_file: UploadFile = File(..., alias="bundle")):
        raw = await bundle_file.read()
        with tempfile.TemporaryDirectory() as td:
            zp = os.path.join(td, "in.zip")
            with open(zp, "wb") as f:
                f.write(raw)
            try:
                bd, tracks = bundle.read_bundle_zip(zp, os.path.join(td, "x"))
            except (zipfile.BadZipFile, KeyError, ValueError) as e:
                raise HTTPException(400, f"bad bundle: {e}")
            mid = bundle.ingest_bundle(store, data, bd, tracks)
        if app.state.on_ingest:
            app.state.on_ingest(mid)
        return {"mid": mid}

    return app


app = build_server()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_server_ingest.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add server/__init__.py server/main.py tests/test_server_ingest.py
git commit -m "feat(server): standalone remote-store host + ingest-bundle"
```

---

### Task 7: `server/` Docker + deploy

**Files:**
- Create: `server/requirements.txt`, `server/Dockerfile`, `server/docker/docker-compose.yml`, `server/deploy-vm.sh`
- No unit test (config/deploy) — a smoke check step instead.

- [ ] **Step 1: Write `server/requirements.txt`**

```
fastapi
uvicorn[standard]
python-multipart
sherpa-onnx
onnxruntime
numpy
```

- [ ] **Step 2: Write `server/Dockerfile`**

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# shared modules copied from the repo root at build context
COPY store.py ./store.py
COPY viewer ./viewer
COPY server ./server
RUN useradd -u 1001 -m appuser && mkdir -p /app/data /app/models \
    && chown -R 1001:1001 /app
USER 1001
ENV STORE_DB=/app/data/store.db STORE_DATA=/app/data \
    OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
EXPOSE 3000
# taskset pins FireRed (and everything) to core 1, leaving core 0 for acp/viewer.
CMD ["taskset", "-c", "1", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "3000"]
```

- [ ] **Step 3: Write `server/docker/docker-compose.yml`**

```yaml
services:
  app:
    build:
      context: ../..           # repo root, so store.py + viewer/ are in context
      dockerfile: server/Dockerfile
    ports:
      - "${HOST_PORT:-5556}:3000"
    volumes:
      - ./data:/app/data       # meetings DB + M4A live only on the VM
      - ./models:/app/models   # FireRed model cache, persisted across redeploys
    restart: unless-stopped
```

- [ ] **Step 4: Write `server/deploy-vm.sh` (clone of acp's, port 5556)**

```bash
#!/usr/bin/env bash
# Deploy the remote-store server to the Azure VM over `az ssh`. rsync the repo
# subset it needs (store.py + viewer/ + server/), build + run docker compose.
# Own container + port, isolated from acp's on :5555. Data/model volumes live
# only on the VM and survive redeploys.
set -euo pipefail
VM_IP="${VM_IP:-10.102.0.7}"
HOST_PORT="${HOST_PORT:-5556}"
REMOTE_DIR="${REMOTE_DIR:-meeting_store}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_CFG="/tmp/az_ssh_cfg_${VM_IP}"
COMPOSE="docker compose -f server/docker/docker-compose.yml"

echo "==> az ssh config ($VM_IP)"
az ssh config --ip "$VM_IP" --file "$SSH_CFG" --overwrite >/dev/null
run() { az ssh vm --ip "$VM_IP" -- "$@"; }

echo "==> rsync (store.py + viewer/ + server/) -> $VM_IP:$REMOTE_DIR/"
rsync -az --delete -e "ssh -F $SSH_CFG" \
  --exclude '__pycache__' --exclude '*.pyc' \
  --exclude data --exclude models \
  "$REPO_ROOT/store.py" "$REPO_ROOT/viewer" "$REPO_ROOT/server" \
  "$VM_IP:$REMOTE_DIR/"

echo "==> build + recreate (HOST_PORT=$HOST_PORT)"
run 'cd '"$REMOTE_DIR"' && HOST_PORT='"$HOST_PORT"' '"$COMPOSE"' up -d --build'

echo "==> health"
run 'cd '"$REMOTE_DIR"' && '"$COMPOSE"' ps | tail -2
    for i in $(seq 1 15); do
      code=$(curl -sSL -o /dev/null -w "%{http_code}" http://localhost:'"$HOST_PORT"'/health 2>/dev/null || echo 000)
      [ "$code" = "200" ] && { echo "HTTP 200"; break; }
      sleep 2
    done
    [ "$code" = "200" ] || echo "HTTP $code (not healthy after 30s)"'
echo "==> done: http://$VM_IP:$HOST_PORT"
```

- [ ] **Step 5: Make executable + local smoke check (no VM needed)**

```bash
chmod +x server/deploy-vm.sh
# smoke: the server image boots and /health responds, purely locally
docker build -f server/Dockerfile -t meeting-store-test .
docker run --rm -d -p 5599:3000 --name mstest meeting-store-test
sleep 3 && curl -sf http://localhost:5599/health && echo OK
docker stop mstest
```
Expected: `{"ok":true}OK`

- [ ] **Step 6: Commit**

```bash
git add server/requirements.txt server/Dockerfile server/docker/docker-compose.yml server/deploy-vm.sh
git commit -m "feat(server): dockerize + az-ssh deploy (:5556, isolated from acp)"
```

---

## Phase D — FireRed correction worker (VM, background, resumable)

FireRed on the VM is long-running (~20–30 min per hour of audio on 1 core), so
the worker must (a) report fine-grained progress, and (b) support **stop** and
**resume/continue** without losing completed work.

**Design — staging profile + durable progress (no store.py changes):**
- Corrected lines are written incrementally under `profile="firered_staging"`
  (one row per source sentence). This is durable: a stop/restart keeps what's
  done, and resume skips rows already staged.
- When ALL source rows are staged, the worker **promotes** in one step:
  `clear_transcripts(mid, "firered")` then a single `UPDATE transcripts SET
  profile='firered' WHERE meeting_id=? AND profile='firered_staging'`. Only then
  does the viewer flip to the corrected version — no half-corrected transcript
  is ever shown.
- Progress + stop are persisted via the existing `store` key/value settings
  (`get_setting`/`set_setting`) — no schema change:
  - `firered:{mid}` → JSON `{state, done, total}`; `state ∈
    idle|running|paused|done`.
  - `firered_stop:{mid}` → `"1"` requests a stop; the loop checks it between rows.
- Because staging rows carry `profile="firered_staging"` (not `"firered"`), the
  viewer's `pick_transcripts` must exclude them from BOTH the firered set and the
  local fallback — a one-line fix in `viewer/render.py` (Task 9).

### Task 8: `server/firered_worker.py` — resumable per-row re-transcribe

**Files:**
- Create: `server/firered_worker.py`
- Test: `tests/test_firered_worker.py`

**Interfaces:**
- Produces:
  - `decode_span_pcm(m4a_path, start_ms, end_ms, ffmpeg="ffmpeg") -> bytes` — ffmpeg slice+decode to raw s16le/16k/mono; `b""` on failure.
  - `chunk_ranges(start_ms, end_ms, max_ms=30000) -> list[tuple[int,int]]`.
  - `get_progress(store, mid) -> dict` — `{state, done, total}` (defaults `{"state":"idle","done":0,"total":0}`).
  - `set_progress(store, mid, state=None, done=None, total=None)` — merge+persist.
  - `request_stop(store, mid)` / `clear_stop(store, mid)` / `stop_requested(store, mid) -> bool`.
  - `correct_meeting(store, data_dir, mid, recognize, *, decode=None, should_stop=None, restart=False) -> dict` — returns `{state, done, total}`. Resumable + non-destructive (see steps).
  - `class FireRedWorker` — `enqueue(mid, restart=False)`, `stop(mid)`, `start()`; lazy sherpa backend.
- Consumes: `store.py` (`list_transcripts`, `add_transcript`, `clear_transcripts`, `get_setting`, `set_setting`, `.db`), `backends.firered_batch_backend` (worker only, lazy).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_firered_worker.py
import json
from store import Store
from server import firered_worker as fw


def _seed(tmp_path):
    store = Store(tmp_path / "s.db")
    mid = store.create_meeting("m", 1721111111.0, "zh-TW")
    store.add_transcript(mid, "accurate", "mixed", 0, 1000, "我", "wrong a")
    store.add_transcript(mid, "accurate", "mixed", 1000, 2000, "對方", "wrong b")
    d = tmp_path / "data" / str(mid); d.mkdir(parents=True)
    (d / "mixed.m4a").write_bytes(b"AUDIO")
    return store, mid


def test_chunk_ranges_splits_long_spans():
    assert fw.chunk_ranges(0, 5000) == [(0, 5000)]
    assert fw.chunk_ranges(0, 70000, max_ms=30000) == [(0, 30000), (30000, 60000), (60000, 70000)]


def test_full_run_promotes_and_inherits_labels(tmp_path):
    store, mid = _seed(tmp_path)
    res = fw.correct_meeting(store, str(tmp_path / "data"), mid,
                             lambda pcm: "corrected", decode=lambda *a, **k: b"P")
    rows = store.list_transcripts(mid)
    local = [r for r in rows if r["profile"] == "accurate"]
    fr = [r for r in rows if r["profile"] == "firered"]
    staging = [r for r in rows if r["profile"] == "firered_staging"]
    assert res == {"state": "done", "done": 2, "total": 2}
    assert len(local) == 2 and [r["text"] for r in local] == ["wrong a", "wrong b"]
    assert not staging                                   # promoted, staging cleared
    assert [(r["speaker"], r["start_ms"], r["end_ms"]) for r in fr] == \
           [("我", 0, 1000), ("對方", 1000, 2000)]        # labels + timing inherited
    assert all(r["text"] == "corrected" for r in fr)
    assert fw.get_progress(store, mid)["state"] == "done"


def test_stop_pauses_and_keeps_partial(tmp_path):
    store, mid = _seed(tmp_path)
    # stop after the first row is staged
    calls = {"n": 0}
    def should_stop():
        calls["n"] += 1
        return calls["n"] > 1          # allow row 0, stop before row 1
    res = fw.correct_meeting(store, str(tmp_path / "data"), mid,
                             lambda pcm: "c", decode=lambda *a, **k: b"P",
                             should_stop=should_stop)
    rows = store.list_transcripts(mid)
    assert res["state"] == "paused" and res["done"] == 1
    assert len([r for r in rows if r["profile"] == "firered_staging"]) == 1
    assert not [r for r in rows if r["profile"] == "firered"]   # not promoted
    assert fw.get_progress(store, mid) == {"state": "paused", "done": 1, "total": 2}


def test_resume_continues_from_staging(tmp_path):
    store, mid = _seed(tmp_path)
    # first pass: stop after row 0
    calls = {"n": 0}
    fw.correct_meeting(store, str(tmp_path / "data"), mid, lambda pcm: "c",
                       decode=lambda *a, **k: b"P",
                       should_stop=lambda: (calls.__setitem__("n", calls["n"] + 1) or calls["n"] > 1))
    seen = []
    def recognize(pcm):
        seen.append(pcm); return "c2"
    # resume: must only re-transcribe the ONE remaining row, then promote
    res = fw.correct_meeting(store, str(tmp_path / "data"), mid, recognize,
                             decode=lambda *a, **k: b"P")
    assert res == {"state": "done", "done": 2, "total": 2}
    assert len(seen) == 1                                  # only the un-staged row ran
    fr = [r for r in store.list_transcripts(mid) if r["profile"] == "firered"]
    assert len(fr) == 2


def test_restart_rebuilds_from_scratch(tmp_path):
    store, mid = _seed(tmp_path)
    fw.correct_meeting(store, str(tmp_path / "data"), mid, lambda pcm: "old",
                       decode=lambda *a, **k: b"P")
    fw.correct_meeting(store, str(tmp_path / "data"), mid, lambda pcm: "new",
                       decode=lambda *a, **k: b"P", restart=True)
    fr = [r for r in store.list_transcripts(mid) if r["profile"] == "firered"]
    assert len(fr) == 2 and all(r["text"] == "new" for r in fr)   # replaced, not doubled
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_firered_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.firered_worker'`

- [ ] **Step 3: Write minimal implementation**

```python
# server/firered_worker.py
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
                pass  # a bad meeting must not kill the worker
            finally:
                self.q.task_done()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_firered_worker.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add server/firered_worker.py tests/test_firered_worker.py
git commit -m "feat(server): resumable FireRed worker (staging + progress + stop/resume)"
```

---

### Task 9: Wire worker + progress/stop/resume routes + viewer progress UI

**Files:**
- Modify: `server/main.py`, `viewer/render.py`
- Test: `tests/test_server_worker_wire.py`

**Interfaces:**
- Consumes: `server.firered_worker` (`FireRedWorker`, `get_progress`).
- Produces (on `server/main.py`'s app):
  - startup: create+start a `FireRedWorker`, set `app.state.on_ingest = worker.enqueue` (unless `FIRERED_DISABLED=1`), store it on `app.state.firered`.
  - `GET /meetings/{mid}/firered/progress` → `{state, done, total}`.
  - `POST /meetings/{mid}/firered/stop` → `{"stopped": True}`.
  - `POST /meetings/{mid}/firered/resume` → `{"resumed": True}` (re-enqueues; `?restart=1` for a full redo).
- `viewer/render.py`: `pick_transcripts` excludes `firered_staging`; `render_detail` shows a progress line + a tiny poller that hits the progress endpoint.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server_worker_wire.py
import io, json, zipfile
from fastapi.testclient import TestClient
import server.main as sm


def _zip(created_at=1721111111.0):
    meta = {"meeting": {"title": "m", "created_at": created_at, "lang": "zh-TW",
                        "status": "finalized", "notes": ""},
            "segments": [], "transcripts": [], "summaries": [], "tracks": []}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("meeting.json", json.dumps(meta))
    return buf.getvalue()


def test_ingest_enqueues(tmp_path):
    app = sm.build_server(db_path=str(tmp_path / "s.db"), data_dir=str(tmp_path / "d"))
    seen = []
    app.state.on_ingest = lambda mid, restart=False: seen.append(mid)
    c = TestClient(app)
    c.post("/ingest-bundle", files={"bundle": ("b.zip", _zip(), "application/zip")})
    assert len(seen) == 1


def test_worker_started_by_default(tmp_path, monkeypatch):
    started = {}
    class FakeWorker:
        def __init__(self, *a): pass
        def start(self): started["yes"] = True
        def enqueue(self, mid, restart=False): pass
        def stop(self, mid): pass
    monkeypatch.setattr(sm.firered_worker, "FireRedWorker", FakeWorker)
    monkeypatch.delenv("FIRERED_DISABLED", raising=False)
    app = sm.build_server(db_path=str(tmp_path / "s.db"), data_dir=str(tmp_path / "d"))
    assert started.get("yes") and app.state.on_ingest is not None


def test_progress_and_stop_resume_routes(tmp_path, monkeypatch):
    stops, resumes = [], []
    class FakeWorker:
        def __init__(self, *a): pass
        def start(self): pass
        def enqueue(self, mid, restart=False): resumes.append((mid, restart))
        def stop(self, mid): stops.append(mid)
    monkeypatch.setattr(sm.firered_worker, "FireRedWorker", FakeWorker)
    app = sm.build_server(db_path=str(tmp_path / "s.db"), data_dir=str(tmp_path / "d"))
    c = TestClient(app)
    assert c.get("/meetings/1/firered/progress").json() == {"state": "idle", "done": 0, "total": 0}
    assert c.post("/meetings/1/firered/stop").json() == {"stopped": True}
    assert stops == [1]
    assert c.post("/meetings/1/firered/resume").json() == {"resumed": True}
    assert resumes[-1] == (1, False)
    c.post("/meetings/1/firered/resume?restart=1")
    assert resumes[-1] == (1, True)
```

Also add to `tests/test_viewer_render.py`:

```python
def test_pick_excludes_firered_staging():
    ts = [{"profile": "accurate", "track": "m", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "local"},
          {"profile": "firered_staging", "track": "m", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "half"}]
    picked = render.pick_transcripts(ts)
    assert [p["text"] for p in picked] == ["local"]   # staging never shown
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_server_worker_wire.py tests/test_viewer_render.py::test_pick_excludes_firered_staging -v`
Expected: FAIL — worker attr / staging rows leak into picked set.

- [ ] **Step 3: Implement**

In `viewer/render.py`, change `pick_transcripts` to exclude staging from both sets:

```python
def pick_transcripts(transcripts):
    rows = [dict(t) for t in transcripts if dict(t).get("profile") != "firered_staging"]
    fr = [t for t in rows if t.get("profile") == "firered"]
    chosen = fr if fr else [t for t in rows if t.get("profile") != "firered"]
    return sorted(chosen, key=lambda t: t.get("start_ms") or 0)
```

In `viewer/render.py`, `render_detail`: add a progress line + poller. Insert into `body` (after the `badge`, before the audio block):

```python
    progress = (f"<div id='fr-prog' class='badge' style='display:none'></div>"
                f"<script>(function(){{var el=document.getElementById('fr-prog');"
                f"function tick(){{fetch('/meetings/{m['id']}/firered/progress')"
                f".then(r=>r.json()).then(p=>{{"
                f"if(p.state==='running'||p.state==='paused'){{el.style.display='';"
                f"el.textContent='FireRed 校正 '+(p.done||0)+'/'+(p.total||0)+"
                f"(p.state==='paused'?'（已暫停）':'…');setTimeout(tick,3000);}}"
                f"else if(p.state==='done'){{el.style.display='';el.textContent='FireRed 校正完成';}}"
                f"}}).catch(()=>{{}});}}tick();}})();</script>")
```

Then include `{progress}` in the `body` f-string next to `{badge}`.

In `server/main.py`, add the import and wiring. Replace the `app.state.on_ingest = None` block with:

```python
    from server import firered_worker  # (add at top of file with other imports)
    ...
    if os.environ.get("FIRERED_DISABLED") == "1":
        app.state.on_ingest = None
        app.state.firered = None
    else:
        worker = firered_worker.FireRedWorker(store, data)
        worker.start()
        app.state.firered = worker
        app.state.on_ingest = worker.enqueue
```

And add the routes (after the ingest route):

```python
    @app.get("/meetings/{mid}/firered/progress")
    def firered_progress(mid: int):
        return firered_worker.get_progress(store, mid)

    @app.post("/meetings/{mid}/firered/stop")
    def firered_stop(mid: int):
        if app.state.firered:
            app.state.firered.stop(mid)
        return {"stopped": True}

    @app.post("/meetings/{mid}/firered/resume")
    def firered_resume(mid: int, restart: int = 0):
        if app.state.firered:
            app.state.firered.enqueue(mid, restart=bool(restart))
        return {"resumed": True}
```

Note: the ingest route calls `app.state.on_ingest(mid)`. Since `on_ingest` is now `worker.enqueue(mid, restart=False)`, that call stays valid. Confirm the ingest handler passes only `mid`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_server_worker_wire.py tests/test_viewer_render.py -v`
Expected: PASS

Then the full server-side suite:

Run: `.venv/bin/pytest tests/test_viewer_bundle.py tests/test_viewer_render.py tests/test_viewer_routes.py tests/test_server_ingest.py tests/test_firered_worker.py tests/test_server_worker_wire.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/main.py viewer/render.py tests/test_server_worker_wire.py tests/test_viewer_render.py
git commit -m "feat(server): FireRed progress + stop/resume routes + viewer progress poller"
```

## Post-implementation (manual, after all tasks)

1. **Deploy**: `cd <repo> && ./server/deploy-vm.sh` → expect `HTTP 200` + `done: http://10.102.0.7:5556`.
2. **First push**: on the Mac, ensure `plugins/remote_store/` is present, open a finalized meeting, click "上傳到 server" → expect `已上傳到 server`.
3. **Verify viewer**: browse `http://10.102.0.7:5556/`, open the meeting, play audio, read transcript (badge shows 本地版 first).
4. **Verify FireRed upgrade**: wait for the worker (~20–30 min/hr of audio on 1 core); reload → badge flips to `FireRed 校正版`, local rows untouched.
5. **CHANGELOG**: add a 未發佈 line via the `release` skill (plugin is opt-in; note it ships separately from the base build).

## Self-Review notes (addressed)

- **Spec coverage**: viewer module (T1–3), Mac plugin + seam (T4–5), server + deploy (T6–7), FireRed worker + wiring (T8–9). All spec sections mapped.
- **No-voiceprint / inherit-labels**: enforced in `correct_meeting` (writes row's own speaker/start/end). VM never diarizes.
- **Non-destructive**: `clear_transcripts(mid, profile="firered")` only; verified by `test_correct_meeting_reruns_clean`.
- **Linux-safe**: `viewer/` + `server/` import only fastapi/store/stdlib; Mac-only deps are lazy inside `plugins/remote_store/push.py` and the worker's `_recognize`.
- **Deviation from spec**: Silero VAD replaced by "re-transcribe per existing sentence row + fixed-window fallback for >30s rows" — simpler, since the Mac already sentence-split. Noted here rather than adding a VAD dependency.
