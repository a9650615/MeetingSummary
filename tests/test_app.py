from fastapi.testclient import TestClient

from app import create_app
from store import Store


def make_client(tmp_path, summary_backend=None, asr_backend=None):
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=summary_backend or (lambda p: "SUMMARY"),
                     asr_backend=asr_backend)
    return TestClient(app), store


def test_run_upload_job_async_pipeline(tmp_path, monkeypatch):
    # Upload is now async: the bg job transcribes -> summarizes -> finalizes and
    # records progress in jobs[mid] (the dict the meeting page polls).
    import app
    import asr
    store = Store(tmp_path / "m.db")
    monkeypatch.setattr(asr, "transcribe", lambda *a, **k: [
        {"profile": "accurate", "track": "mic", "start_ms": 0,
         "end_ms": 1000, "text": "討論預算"}])
    monkeypatch.setattr(app, "_save_upload_pcm", lambda *a, **k: None)  # skip ffmpeg
    mid = store.create_meeting("實測", 1.0, "zh-TW")
    jobs = {}
    app._run_upload_job(store, mid, "x.m4a", asr_backend=None,
                        summary_backend=lambda p: "會議摘要：談預算",
                        summary_model="mlx-lm", kind="minutes", title="實測",
                        jobs=jobs)
    assert jobs[mid]["state"] == "done"
    assert store.list_transcripts(mid)[0]["text"] == "討論預算"
    assert store.list_summaries(mid)                      # a summary landed
    assert store.get_meeting(mid)["status"] == "finalized"


def test_run_upload_job_records_error(tmp_path, monkeypatch):
    import app
    import asr
    store = Store(tmp_path / "m.db")
    monkeypatch.setattr(asr, "transcribe",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    mid = store.create_meeting("實測", 1.0, "zh-TW")
    jobs = {}
    app._run_upload_job(store, mid, "x.m4a", asr_backend=None,
                        summary_backend=lambda p: "x", summary_model="mlx-lm",
                        kind="minutes", title="實測", jobs=jobs)
    assert jobs[mid]["state"] == "error" and "boom" in jobs[mid]["msg"]


def test_upload_pcm_anchored_at_created_at_not_now(tmp_path, monkeypatch):
    # The decode runs minutes after create (transcribe first). The segment must
    # anchor at the meeting's created_at, NOT time.time() — else _assemble_track
    # prepends those minutes as silence and the audio won't line up / "won't play".
    import shutil
    import subprocess

    import app
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "m.db")
    created = 1000.0
    mid = store.create_meeting("m", created, "zh-TW")
    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/ffmpeg")

    def fake_run(cmd, **k):
        open(cmd[-1], "wb").write(b"\x00\x00")  # cmd[-1] = pcm out path
        return type("R", (), {"returncode": 0})()
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(app.time, "time", lambda: created + 9999)  # decode finishes late
    app._save_upload_pcm("x.m4a", mid, store)
    seg = store.list_segments(mid)[0]
    assert seg["started_at"] == created   # offset 0 -> no silence prefix


def test_run_diarize_job_reports_progress_then_done(tmp_path, monkeypatch):
    # Diarization runs as a bg job; sherpa's chunk callback -> jobs[mid] progress.
    import app
    import diarize as diar
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "m.db")
    mid = store.create_meeting("m", 1.0, "zh-TW")
    store.add_transcript(mid, "accurate", "mixed", 0, 1000, "對方", "你好")
    monkeypatch.setattr(app, "_meeting_tracks", lambda s, m: ["mixed"])
    monkeypatch.setattr(app, "_assemble_track", lambda s, m, t: b"\x00\x00")
    def fake_diar(tmp, *, num_speakers=-1, seg_model=None, emb_model=None,
                  enroll=False, on_progress=None, on_phase=None, provider="cpu"):
        on_progress(1, 2)
        on_progress(2, 2)  # callback drives the progress dict
        return [{"start": 0.0, "end": 1.0, "speaker": 0}], None  # (segments, embeddings)
    monkeypatch.setattr(diar, "diarize_with_progress", fake_diar)
    body = app.DiarizeIn(track="all", enroll=False)  # skip cross-meeting voiceprints
    jobs = {}
    app._run_diarize_job(store, mid, body, jobs)
    assert jobs[mid]["state"] == "done" and jobs[mid]["speakers"] == 1
    assert store.list_transcripts(mid)[0]["speaker"] != "對方"   # got a diarized label


def test_run_diarize_job_errors_without_audio(tmp_path, monkeypatch):
    import app
    store = Store(tmp_path / "m.db")
    mid = store.create_meeting("m", 1.0, "zh-TW")
    monkeypatch.setattr(app, "_meeting_tracks", lambda s, m: [])  # no playable audio
    jobs = {}
    app._run_diarize_job(store, mid, app.DiarizeIn(), jobs)
    assert jobs[mid]["state"] == "error"


def test_detect_ignores_background_app_without_mic(tmp_path, monkeypatch):
    import meeting_watch as mw
    c, _ = make_client(tmp_path)
    micbusy = tmp_path / "micbusy"
    micbusy.write_text("x")  # helper present -> mic is the real signal
    monkeypatch.setattr(mw, "_MICBUSY", str(micbusy))
    monkeypatch.setattr(mw, "meeting_app_running", lambda: "MSTeams")  # runs all day
    monkeypatch.setattr(mw, "mic_in_use", lambda: False)
    d = c.get("/detect").json()
    assert d["meeting"] is False and d["app"] == "MSTeams"  # bg app alone != meeting
    monkeypatch.setattr(mw, "mic_in_use", lambda: True)
    assert c.get("/detect").json()["meeting"] is True       # real mic use -> meeting


def test_speakers_routes_list_rename_merge_delete(tmp_path):
    import struct
    c, store = make_client(tmp_path)
    store.add_speaker("對方1", struct.pack("2f", 1.0, 0.0))
    store.add_speaker("對方2", struct.pack("2f", 0.0, 1.0))
    mid = store.create_meeting("M", 1.0, "zh-TW")
    store.add_transcript(mid, "accurate", "system", 0, 1, "對方1", "hi")
    store.add_transcript(mid, "accurate", "system", 1, 2, "對方2", "yo")

    body = c.get("/speakers").json()
    assert body["persist"] is True and len(body["speakers"]) == 2

    c.post("/speakers/rename", json={"old": "對方1", "new": "Scott"})  # name-based rename
    assert store.list_transcripts(mid)[0]["speaker"] == "Scott"        # transcript moved too

    assert c.post("/speakers/merge", json={"keep": "Scott", "drop": "對方2"}).json()["moved"] == 1
    assert {s["name"] for s in store.list_speakers()} == {"Scott"}
    assert len(c.get("/speakers/Scott/utterances").json()["utterances"]) == 2

    c.post("/speakers/delete", json={"name": "Scott"})
    assert store.list_speakers() == []


def test_speakers_reconcile_route_merges_close_and_purges_noise(tmp_path):
    import struct
    c, store = make_client(tmp_path)
    store.add_speaker("Jimmy", struct.pack("2f", 1.0, 0.0))     # fragmented same-name row
    store.add_speaker("Jimmy", struct.pack("2f", 0.9, 0.3))     # cohesive with the row above
    store.add_speaker("對方5", struct.pack("2f", 1.0, 0.0))      # placeholder, very close to Jimmy
    store.add_speaker("Ann", struct.pack("2f", 0.0, 1.0))       # different real name -> untouched
    store.add_speaker("對方77", struct.pack("2f", 0.0, -1.0))    # noise: unmatched, never reinforced

    r = c.post("/speakers/reconcile").json()
    assert r["merged"] == 2   # Jimmy's own duplicate + the close placeholder
    assert r["purged"] == 1   # the orthogonal placeholder

    names = {s["name"] for s in store.list_speakers()}
    assert names == {"Jimmy", "Ann"}

    # a full-DB snapshot was taken first, so the whole op is reversible.
    import os
    assert r["backup"] and os.path.exists(r["backup"])
    import sqlite3
    bak = sqlite3.connect(r["backup"])
    assert bak.execute("SELECT COUNT(*) FROM speakers").fetchone()[0] == 5   # pre-reconcile state preserved
    bak.close()


def test_persistent_names_matches_named_person_via_consolidated_centroid(tmp_path):
    # A named person accrues several fragmented voiceprints. A new same-voice sample
    # can sit just below threshold vs any SINGLE fragment yet clearly match the
    # per-person CONSOLIDATED (averaged) centroid. _persistent_names must reuse the
    # name (recall) instead of spawning yet another placeholder ("同一人被拆開").
    import math
    import numpy as np
    import app
    c, store = make_client(tmp_path)
    store.set_setting("speaker_threshold", "0.97")

    def unit(v):
        v = np.asarray(v, dtype=np.float32)
        return (v / (np.linalg.norm(v) + 1e-9))

    g1 = unit([1.0, 0.0])
    g2 = unit([math.cos(math.radians(40)), math.sin(math.radians(40))])  # 40° apart, cohesive
    store.add_speaker("Alice", g1.tobytes())        # id 1
    store.set_speaker_name(1, "Alice")
    store.update_speaker_centroid(1, g1.tobytes(), 3)
    store.add_speaker("Alice", g2.tobytes())        # id 2
    store.set_speaker_name(2, "Alice")
    store.update_speaker_centroid(2, g2.tobytes(), 3)

    sample = unit([math.cos(math.radians(20)), math.sin(math.radians(20))])  # bisector ≈ consolidated
    # sanity: sample vs each raw fragment is ~cos20≈0.94 < 0.97, so raw match alone misses
    assert float(sample @ g1) < 0.97 and float(sample @ g2) < 0.97

    names = app._persistent_names(store, {0: sample}, prefix="對方")
    assert names[0] == "Alice"                       # matched via consolidated centroid, not a new 對方N
    assert not any(s["name"].startswith("對方") for s in store.list_speakers())


def test_speakers_grouped_by_name(tmp_path):
    import struct
    c, store = make_client(tmp_path)
    store.add_speaker("Alice", struct.pack("2f", 1.0, 0.0))
    store.add_speaker("Alice", struct.pack("2f", 0.9, 0.1))  # 2nd voiceprint, same name
    mid = store.create_meeting("A", 1.0, "zh-TW")
    store.add_transcript(mid, "accurate", "mic", 0, 1, "Alice", "hi")  # 1ms: shows, but no sample
    rows = c.get("/speakers").json()["speakers"]
    jasons = [r for r in rows if r["name"] == "Alice"]
    assert len(jasons) == 1 and jasons[0]["voiceprints"] == 2  # one row, not two
    assert jasons[0]["has_sample"] is False  # span <800ms -> no 試聽 button


def test_detail_shows_recognized_cross_meeting_speaker(tmp_path):
    import struct
    c, store = make_client(tmp_path)
    store.add_speaker("張三", struct.pack("2f", 1.0, 0.0))  # known voiceprint
    m1 = store.create_meeting("A", 1.0, "zh-TW")
    m2 = store.create_meeting("B", 2.0, "zh-TW")
    store.add_transcript(m1, "accurate", "system", 0, 1, "張三", "hi")
    store.add_transcript(m2, "accurate", "system", 0, 1, "張三", "yo")  # 2nd meeting
    html = c.get(f"/m/{m2}").text
    assert "本場認得" in html and "張三" in html  # cross-meeting recognition surfaced


def test_speaker_sample_wav(tmp_path):
    c, store = make_client(tmp_path)
    mid = store.create_meeting("M", 1.0, "zh-TW")
    seg = tmp_path / "seg"
    seg.mkdir()
    (seg / "mic.pcm").write_bytes(b"\x11\x11" * 16000 * 5)  # 5s nonzero
    store.add_segment(mid, idx=0, dir_path=str(seg),
                      started_at=store.get_meeting(mid)["created_at"],
                      duration_s=5, origin="recorded")
    store.add_transcript(mid, "accurate", "mic", 1000, 4000, "張三", "hi")  # 3s span
    r = c.get("/speakers/張三/sample.wav")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content
    assert c.get("/speakers/nobody/sample.wav").status_code == 404  # no utterance span


def test_speaker_suggestions_route(tmp_path):
    import struct
    c, store = make_client(tmp_path)
    mid = store.create_meeting(title="m", created_at=0.0, lang="zh-TW")
    store.add_speaker("A", struct.pack("2f", 1.0, 0.0))
    store.add_speaker("對方5", struct.pack("2f", 0.95, 0.31))  # unnamed cluster ~ A
    store.add_transcript(mid, "accurate", "mic", 0, 2000, "A", "hi")  # playable samples
    store.add_transcript(mid, "accurate", "mic", 0, 2000, "對方5", "yo")
    pairs = c.get("/speakers/suggestions").json()["pairs"]
    names = {pairs[0]["a"], pairs[0]["b"]}
    assert pairs and names == {"A", "對方5"} and pairs[0]["sim"] > 0.8


def test_persist_speakers_toggle_route(tmp_path):
    c, store = make_client(tmp_path)
    assert c.get("/settings/persist_speakers").json()["value"] == "1"
    assert c.post("/settings/persist_speakers", json={"value": "0"}).json()["value"] == "0"
    assert store.get_setting("persist_speakers") == "0"


def test_live_language_setting_persists_and_validates(tmp_path):
    c, store = make_client(tmp_path)
    assert c.get("/settings/live_language").json()["value"] == ""   # default 自動偵測
    assert c.post("/settings/live_language", json={"value": "ja"}).json()["value"] == "ja"
    assert store.get_setting("live_language") == "ja"
    # unknown code is rejected -> falls back to "" (auto), never stored raw
    assert c.post("/settings/live_language", json={"value": "klingon"}).json()["value"] == ""
    assert store.get_setting("live_language") == ""


def test_auto_pipeline_uses_ane_when_toggled(tmp_path, monkeypatch):
    # ANE toggle on -> the default/auto ASR (no explicit model) routes to the ANE
    # backend, not just manual dropdown picks.
    import app
    import backends
    calls = []
    monkeypatch.setattr(app, "_ane_available", lambda: True)
    monkeypatch.setattr(backends, "make_batch_backend",
                        lambda model, language=None: calls.append(model) or (lambda p: []))
    c, store = make_client(tmp_path, asr_backend=lambda p: [])
    mid = store.create_meeting("m", 1.0, "zh-TW")
    store.set_setting("ane", "1")
    c.post(f"/meetings/{mid}/transcribe", json={})         # no model -> _default_asr
    assert "ane-qwen3-0.6b" in calls
    calls.clear()
    store.set_setting("ane", "0")
    c.post(f"/meetings/{mid}/transcribe", json={})         # toggle off -> configured default
    assert "ane-qwen3-0.6b" not in calls


def test_merge_nearby_route(tmp_path):
    c, store = make_client(tmp_path)
    store.create_meeting("A", 1000.0, "zh-TW")
    store.create_meeting("B", 1060.0, "zh-TW")   # 1 min later -> within 10 min
    store.create_meeting("C", 9000.0, "zh-TW")   # far -> untouched
    r = c.post("/meetings/merge-nearby?gap_min=10")
    assert r.status_code == 200 and r.json()["merged_groups"] == 1
    assert len(store.list_meetings()) == 2       # A+B merged, C kept


def test_create_and_list_meeting(tmp_path):
    c, _ = make_client(tmp_path)
    r = c.post("/meetings", json={"title": "standup", "lang": "zh-TW"})
    assert r.status_code == 200
    mid = r.json()["id"]
    assert any(m["id"] == mid for m in c.get("/meetings").json())


def test_get_meeting_detail(tmp_path):
    c, store = make_client(tmp_path)
    mid = store.create_meeting("m", 1.0, "zh-TW")
    store.add_transcript(mid, "accurate", "mic", 0, 1000, "我", "你好")
    body = c.get(f"/meetings/{mid}").json()
    assert body["meeting"]["id"] == mid
    assert body["transcripts"][0]["text"] == "你好"


def test_meeting_page_rename_uses_raw_stored_speaker_not_display_label(tmp_path):
    # Live diarization stores the raw cluster label (說話者N); the review page
    # (/m/{mid}) must display it collapsed (對方) but click-to-rename has to key
    # off the RAW stored value — else it silently updates 0 rows because no row
    # literally has speaker="對方" in the database.
    c, store = make_client(tmp_path)
    mid = store.create_meeting("m", 1.0, "zh-TW")
    store.add_transcript(mid, "accurate", "system", 0, 1000, "說話者1", "你好")

    html = c.get(f"/m/{mid}").text
    assert "data-spk='說話者1'" in html
    assert ">對方<" in html

    r = c.post(f"/meetings/{mid}/speaker", json={"old": "說話者1", "new": "Scott", "track": "system"})
    assert r.json()["renamed"] == 1
    assert store.list_transcripts(mid)[0]["speaker"] == "Scott"


def test_get_missing_meeting_404(tmp_path):
    c, _ = make_client(tmp_path)
    assert c.get("/meetings/999").status_code == 404


def test_summary_job_runs_backend_and_stores(tmp_path):
    import app
    captured = {}

    def backend(p):
        captured["p"] = p
        return "會議記錄"

    store = Store(tmp_path / "m.db")
    store.set_setting("summary_correct", "0")  # isolate summary flow from the correction pass
    mid = store.create_meeting("m", 1.0, "zh-TW")
    store.add_transcript(mid, "accurate", "mic", 0, 1000, "我", "討論預算")
    jobs = {}
    app._run_summary_job(store, mid, "minutes", backend, "mlx-lm", jobs)
    assert jobs[mid]["state"] == "done" and jobs[mid]["text"] == "會議記錄"
    assert "討論預算" in captured["p"]  # transcript fed into the prompt
    assert store.list_summaries(mid)[0]["text"] == "會議記錄"


def test_summary_route_is_async_started(tmp_path):
    # The route now spawns a bg job and returns at once (no blocking on the LLM).
    c, store = make_client(tmp_path, summary_backend=lambda p: "S")
    mid = store.create_meeting("m", 1.0, "zh-TW")
    store.add_transcript(mid, "accurate", "mic", 0, 1000, "我", "x")
    r = c.post(f"/meetings/{mid}/summary", json={"kind": "minutes"})
    assert r.status_code == 200 and r.json().get("started") is True
    assert c.get(f"/meetings/{mid}/summary/progress").json()["state"] in (
        "running", "done")


def test_transcribe_route_runs_asr_and_stores(tmp_path):
    seg_dir = tmp_path / "seg0"
    seg_dir.mkdir()
    (seg_dir / "system.pcm").write_bytes(b"\x00" * 32000)
    (seg_dir / "mic.pcm").write_bytes(b"\x00" * 32000)

    def asr_backend(path):
        text = "你好" if "system" in str(path) else "hello"
        return [{"start": 0.0, "end": 1.0, "text": text}]

    c, store = make_client(tmp_path, asr_backend=asr_backend)
    mid = store.create_meeting("m", 1.0, "zh-TW")
    store.add_segment(mid, idx=0, dir_path=str(seg_dir), started_at=1.0,
                      duration_s=1.0, origin="recorded")
    r = c.post(f"/meetings/{mid}/transcribe")
    assert r.status_code == 200
    assert r.json()["transcripts"] == 2
    rows = {row["track"]: row["text"] for row in store.list_transcripts(mid)}
    assert rows == {"system": "你好", "mic": "hello"}


def test_transcribe_without_backend_returns_503(tmp_path):
    c, store = make_client(tmp_path)  # no asr_backend
    mid = store.create_meeting("m", 1.0, "zh-TW")
    assert c.post(f"/meetings/{mid}/transcribe").status_code == 503


def test_delete_meeting_route(tmp_path):
    c, store = make_client(tmp_path)
    mid = store.create_meeting("m", 1.0, "zh-TW")
    assert c.delete(f"/meetings/{mid}").status_code == 200
    assert store.get_meeting(mid) is None
    assert c.delete(f"/meetings/{mid}").status_code == 404


def test_delete_preserves_dir_shared_with_another_meeting(tmp_path):
    import os
    c, store = make_client(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "mic.pcm").write_bytes(b"\x00" * 100)
    a = store.create_meeting("A", 1.0, "zh")
    b = store.create_meeting("B", 2.0, "zh")
    store.add_segment(a, 0, str(shared), 1.0, 0, "recorded")
    store.add_segment(b, 0, str(shared), 2.0, 0, "recorded")
    c.delete(f"/meetings/{a}")
    assert (shared / "mic.pcm").exists()       # still referenced by B -> kept
    c.delete(f"/meetings/{b}")
    assert not (shared / "mic.pcm").exists()    # last ref gone -> removed


def test_list_meetings_has_audio_flag(tmp_path):
    import os
    c, store = make_client(tmp_path)
    mid = store.create_meeting("m", 1.0, "zh-TW")
    seg_dir = tmp_path / "seg"
    seg_dir.mkdir()
    (seg_dir / "mic.pcm").write_bytes(b"\x00" * 100)
    store.add_segment(mid, 0, str(seg_dir), 1.0, 0, "recorded")
    row = next(m for m in c.get("/meetings").json() if m["id"] == mid)
    assert row["has_audio"] is True


def test_meeting_not_finalized_until_explicit(tmp_path):
    c, store = make_client(tmp_path)
    mid = store.create_meeting("m", 1.0, "zh-TW")
    assert store.get_meeting(mid)["status"] == "recording"  # not finalized on create
    r = c.post(f"/meetings/{mid}/finalize")
    assert r.status_code == 200
    assert store.get_meeting(mid)["status"] == "finalized"  # only after the call


def test_health_endpoint(tmp_path):
    c, _ = make_client(tmp_path)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_index_page_served(tmp_path):
    c, _ = make_client(tmp_path)
    r = c.get("/")
    assert r.status_code == 200
    assert "MeetingSummary" in r.text


def test_src_labels_maps_source_to_track_and_speaker():
    from app import _src_labels
    assert _src_labels("mic") == ("mic", "我")
    assert _src_labels("system") == ("system", "對方")
    assert _src_labels("both") == ("mixed", "混合")
    assert _src_labels("garbage") == ("mic", "我")  # default


def test_meeting_page_renders_without_auto_summary(tmp_path):
    c, store = make_client(tmp_path)
    mid = store.create_meeting("m", 1.0, "zh-TW")
    store.add_transcript(mid, "live", "mic", 0, 1000, "我", "測試內容")
    r = c.get(f"/m/{mid}")
    assert r.status_code == 200
    assert "測試內容" in r.text and "產生摘要" in r.text
    assert store.list_summaries(mid) == []  # viewing must NOT trigger summary


def test_suggestions_skip_speakers_without_audio_sample(tmp_path):
    # "沒有語音就不應該拿來比較" — a voiceprint with no playable sample (no >800ms
    # utterance) must not appear in merge suggestions, even if its centroid matches.
    import numpy as np
    client, store = make_client(tmp_path)
    mid = store.create_meeting(title="m", created_at=0.0, lang="zh-TW")
    v = np.array([1, 0, 0], dtype=np.float32).tobytes()  # identical -> cos sim 1.0
    # only 未命名群<->真名 is a valid suggestion now (two named people are never
    # compared), so pair a name with a placeholder cluster.
    for nm in ("Alice", "對方5", "對方9"):
        store.add_speaker(nm, v)
    store.add_transcript(mid, "accurate", "mic", 0, 2000, "Alice", "hi")
    store.add_transcript(mid, "accurate", "mic", 0, 2000, "對方5", "yo")
    store.add_transcript(mid, "accurate", "mic", 0, 300, "對方9", "x")  # <800ms = no sample
    pairs = client.get("/speakers/suggestions").json()["pairs"]
    names = {n for p in pairs for n in (p["a"], p["b"])}
    assert "對方9" not in names
    assert {"Alice", "對方5"} <= names


def test_nonmatch_dismisses_suggestion(tmp_path):
    # "不是同一人" — after dismissing, the pair stops being suggested.
    import struct
    c, store = make_client(tmp_path)
    mid = store.create_meeting("m", 0.0, "zh-TW")
    store.add_speaker("Alice", struct.pack("2f", 1.0, 0.0))
    store.add_speaker("對方5", struct.pack("2f", 0.95, 0.31))   # unnamed cluster ~ Alice
    store.add_transcript(mid, "accurate", "mic", 0, 2000, "Alice", "hi")
    store.add_transcript(mid, "accurate", "mic", 0, 2000, "對方5", "yo")
    assert c.get("/speakers/suggestions").json()["pairs"]  # suggested first
    c.post("/speakers/nonmatch", json={"keep": "對方5", "drop": "Alice"})  # any order
    assert c.get("/speakers/suggestions").json()["pairs"] == []  # gone


def test_storage_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c, store = make_client(tmp_path)
    mid = store.create_meeting("big", 1.0, "zh-TW")
    d = tmp_path / "data" / f"{mid}-1"
    d.mkdir(parents=True)
    (d / "mic.pcm").write_bytes(b"\x00" * 4096)
    store.add_segment(mid, idx=0, dir_path=str(d), started_at=1.0, duration_s=0, origin="recorded")
    j = c.get("/storage").json()
    assert j["total"] >= 4096
    assert any(m["id"] == mid and m["bytes"] >= 4096 for m in j["meetings"])
    assert len(j["categories"]) == 4


def test_screenshot_routes(tmp_path, monkeypatch):
    # Browser uploads the captured PNG (server doesn't run screencapture — TCC is
    # per-app, a headless server can't be granted Screen-Recording).
    monkeypatch.chdir(tmp_path)
    c, store = make_client(tmp_path)
    mid = store.create_meeting("m", 1.0, "zh-TW")
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    r = c.post(f"/meetings/{mid}/screenshot", files={"img": ("shot.png", png, "image/png")})
    name = r.json()["name"]
    assert name.endswith(".png")
    assert name in c.get(f"/meetings/{mid}/shots").json()["shots"]
    assert c.get(f"/meetings/{mid}/shots/{name}").status_code == 200
    c.post(f"/meetings/{mid}/shots/delete", json={"name": name})
    assert c.get(f"/meetings/{mid}/shots").json()["shots"] == []


def test_live_state_and_stop(tmp_path):
    # float control panel endpoints: nothing recording -> state false, stop 0
    c, _ = make_client(tmp_path)
    assert c.get("/live/state").json()["recording"] is False
    assert c.post("/live/stop").json()["stopping"] == 0


def test_floatpanel_open_requires_install(tmp_path, monkeypatch):
    import backends
    c, _ = make_client(tmp_path)
    monkeypatch.setattr(backends, "floatpanel_bin", lambda: None)
    assert c.post("/floatpanel/open").status_code == 400  # not installed -> guided error


def test_ingest_filename_sanitized(tmp_path, monkeypatch):
    # a path-traversal filename must not escape data/uploads
    import app
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "m.db")
    monkeypatch.setattr(app, "_run_upload_job", lambda *a, **k: None)  # skip bg pipeline
    c = TestClient(create_app(store, summary_backend=lambda p: "x", asr_backend=lambda *a, **k: []))
    r = c.post("/ingest", data={"title": "t", "kind": "minutes", "lang": "zh-TW"},
               files={"audio": ("../../escape.wav", b"RIFFxxxx", "audio/wav")},
               follow_redirects=False)
    assert r.status_code in (200, 303)
    assert (tmp_path / "data" / "uploads" / "escape.wav").exists()  # basename, inside uploads
    assert not (tmp_path / "escape.wav").exists()                   # did NOT escape


def test_live_prewarm_noop_without_ane(tmp_path):
    # no live_manager / non-ANE -> prewarm is a harmless no-op
    c, _ = make_client(tmp_path)
    assert c.post("/live/prewarm").json()["warming"] is False


def test_notes_append_route(tmp_path):
    c, store = make_client(tmp_path)
    mid = store.create_meeting("m", 1.0, "zh-TW")
    c.post(f"/meetings/{mid}/notes/append", json={"value": "a"})
    c.post(f"/meetings/{mid}/notes/append", json={"value": "b"})
    assert store.get_meeting(mid)["notes"] == "a\nb"
    assert c.post("/meetings/99999/notes/append", json={"value": "x"}).status_code == 404


def test_rename_speaker_into_existing_person_backs_up_first(tmp_path):
    # Renaming a speaker onto a name ALREADY used by another person merges the two
    # unsplittably. The endpoint must snapshot the DB first so it's recoverable.
    import os
    c, store = make_client(tmp_path)
    mid = store.create_meeting("m", 1.0, "zh-TW")
    store.add_transcript(mid, "accurate", "mic", 0, 1, "對方", "hi")
    store.add_transcript(mid, "accurate", "mic", 1, 2, "王先生", "yo")

    # merge: 對方 -> 王先生 (王先生 already present) -> backup taken
    r = c.post(f"/meetings/{mid}/speaker",
               json={"old": "對方", "new": "王先生", "track": "mic"}).json()
    assert r["merged"] is True and r["backup"] and os.path.exists(r["backup"])

    # non-merge: fresh name -> no backup, no wasted snapshot
    store.add_transcript(mid, "accurate", "mic", 2, 3, "李四", "sup")
    r2 = c.post(f"/meetings/{mid}/speaker",
                json={"old": "李四", "new": "陌生人", "track": "mic"}).json()
    assert r2["merged"] is False and r2["backup"] is None


def test_speaker_split_check_and_apply(tmp_path, monkeypatch):
    # Global voiceprint split: one name is actually two voices. Fake the audio
    # sampling so no sherpa/audio is needed — feed two clean voice groups.
    import numpy as np
    import app
    import diarize
    store = Store(tmp_path / "m.db")
    rng = np.random.default_rng(0)
    va = rng.normal(size=16); va /= np.linalg.norm(va)
    vb = rng.normal(size=16); vb /= np.linalg.norm(vb)
    store.add_speaker("Hank", va.astype(np.float32).tobytes())   # polluted single row
    monkeypatch.setattr(diarize, "embedding_extractor", lambda *a, **k: (lambda b: va))

    def fake_sample(st, name, ext, sample_rate=16000):
        sp = lambda: {"meeting_id": 1, "track": "mic", "start_ms": 0, "end_ms": 3000}
        return [(sp(), (va + 0.03 * rng.normal(size=16)).astype(np.float32)) for _ in range(4)] + \
               [(sp(), (vb + 0.03 * rng.normal(size=16)).astype(np.float32)) for _ in range(4)]
    monkeypatch.setattr(app, "_sample_speaker_embeddings", fake_sample)

    chk = app._speaker_split_check(store, "Hank")
    assert chk["enough"] and chk["split"] is True and chk["sep"] < 0.5
    assert len(chk["groups"]) == 2

    res = app._apply_speaker_split(store, "Hank", "Amy")
    assert res["ok"] and res["kept"] == "Hank" and res["new"] == "Amy"
    names = sorted(r["name"] for r in store.list_speakers())
    assert names == ["Amy", "Hank"]            # one clean centroid each


def test_speaker_split_refuses_single_voice(tmp_path, monkeypatch):
    import numpy as np
    import app
    import diarize
    store = Store(tmp_path / "m.db")
    rng = np.random.default_rng(1)
    v = rng.normal(size=16); v /= np.linalg.norm(v)
    store.add_speaker("Solo", v.astype(np.float32).tobytes())
    monkeypatch.setattr(diarize, "embedding_extractor", lambda *a, **k: (lambda b: v))
    monkeypatch.setattr(app, "_sample_speaker_embeddings", lambda st, n, e, sample_rate=16000:
                        [({"meeting_id": 1, "track": "mic", "start_ms": 0, "end_ms": 3000},
                          (v + 0.03 * rng.normal(size=16)).astype(np.float32)) for _ in range(8)])
    assert app._speaker_split_check(store, "Solo")["split"] is False
    assert app._apply_speaker_split(store, "Solo", "Ghost")["ok"] is False
    assert [r["name"] for r in store.list_speakers()] == ["Solo"]   # untouched


def test_persistent_names_does_not_enroll_unrecognized(tmp_path):
    # New model: an unrecognized cluster gets NO global voiceprint (stays 對方N via
    # assign_speakers). Only a human assignment enrolls.
    import numpy as np
    import app
    from store import Store
    s = Store(tmp_path / "m.db")
    names = app._persistent_names(s, {0: np.array([1, 0, 0, 0], dtype=np.float32)}, "對方")
    assert names == {}                                  # not labelled here -> local 對方N
    assert list(s.list_speakers()) == []                # nothing written to global 語者庫


def test_enroll_meeting_speaker_on_assignment(tmp_path, monkeypatch):
    import numpy as np
    import app
    import diarize
    from store import Store
    s = Store(tmp_path / "m.db")
    mid = s.create_meeting("m", 1.0, "zh-TW")
    s.add_transcript(mid, "accurate", "system", 0, 3000, "Alice", "hi there")
    v = np.array([0.1, 0.9, 0.2, 0.0], dtype=np.float32)
    monkeypatch.setattr(diarize, "embedding_extractor", lambda *a, **k: (lambda b: v))
    monkeypatch.setattr(app, "_assemble_track", lambda st, m, t, sr=16000: b"\x01\x02" * (16000 * 3))
    assert app._enroll_meeting_speaker(s, mid, "system", "Alice") is True
    assert any(r["name"] == "Alice" for r in s.list_speakers())
    # naming again REINFORCES (learns) the existing voiceprint — no dup row, no skip
    assert app._enroll_meeting_speaker(s, mid, "system", "Alice") is True
    assert sum(1 for r in s.list_speakers() if r["name"] == "Alice") == 1


def test_set_line_speaker_reassigns_one_line(tmp_path, monkeypatch):
    # 抽離: a single mislabeled line moves to the right person; siblings stay put.
    import app
    import diarize
    client, store = make_client(tmp_path)
    mid = store.create_meeting(title="m", created_at=0.0, lang="zh-TW")
    t1 = store.add_transcript(mid, "accurate", "system", 0, 3000, "Imp", "line one")
    t2 = store.add_transcript(mid, "accurate", "system", 3000, 6000, "Imp", "line two")
    monkeypatch.setattr(app, "_enroll_meeting_speaker", lambda *a, **k: False)  # skip audio
    r = client.post(f"/meetings/{mid}/transcript/{t2}/speaker",
                    json={"speaker": "Angle", "track": "system"})
    assert r.status_code == 200 and r.json()["changed"] == 1
    by_id = {row["id"]: row["speaker"] for row in store.list_transcripts(mid)}
    assert by_id[t1] == "Imp" and by_id[t2] == "Angle"      # only the one line moved


def test_upload_runs_diarize_before_summary(tmp_path, monkeypatch):
    # Upload pipeline must call the voiceprint step (so summary gets speakers),
    # and must transcribe with the backend it was GIVEN (accurate), not ANE.
    import app, asr
    store = Store(tmp_path / "m.db")
    seen = {}
    def fake_tx(path, **k):
        seen["backend"] = k.get("backend")
        return [{"profile": "accurate", "track": "mic", "start_ms": 0,
                 "end_ms": 1000, "text": "討論預算"}]
    monkeypatch.setattr(asr, "transcribe", fake_tx)
    monkeypatch.setattr(app, "_save_upload_pcm", lambda *a, **k: None)
    monkeypatch.setattr(app, "_diarize_meeting",
                        lambda *a, **k: seen.__setitem__("diarized", True))
    mid = store.create_meeting("實測", 1.0, "zh-TW")
    jobs = {}
    app._run_upload_job(store, mid, "x.m4a", asr_backend="ACCURATE_BE",
                        summary_backend=lambda p: "摘要", summary_model="m",
                        kind="minutes", title="實測", jobs=jobs)
    assert seen.get("diarized") is True            # voiceprint step ran
    assert seen.get("backend") == "ACCURATE_BE"    # used given (accurate) backend
    assert jobs[mid]["state"] == "done"


def test_retranscribe_runs_diarize_for_speaker_breaks(tmp_path, monkeypatch):
    # Re-transcribe must also diarize (speaker-aware line breaks), like upload —
    # otherwise batch lines only split on punctuation, not on speaker change.
    import app
    store = Store(tmp_path / "m.db")
    mid = store.create_meeting("實測", 1.0, "zh-TW")
    monkeypatch.setattr(app, "iter_transcribe",
                        lambda *a, **k: iter([{"type": "start", "total": 1},
                                              {"type": "done", "transcripts": 3}]))
    called = {}
    monkeypatch.setattr(app, "_diarize_meeting",
                        lambda s, m, j, **k: called.setdefault("mid", m))
    jobs = {}
    app._run_transcribe_job(store, mid, backend=None, jobs=jobs)
    assert called.get("mid") == mid            # diarize ran after transcription
    assert jobs[mid]["state"] == "done" and jobs[mid]["done"] == 3
