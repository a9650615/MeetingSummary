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
                  enroll=False, on_progress=None, on_phase=None):
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


def test_speakers_grouped_by_name(tmp_path):
    import struct
    c, store = make_client(tmp_path)
    store.add_speaker("Jason", struct.pack("2f", 1.0, 0.0))
    store.add_speaker("Jason", struct.pack("2f", 0.9, 0.1))  # 2nd voiceprint, same name
    rows = c.get("/speakers").json()["speakers"]
    jasons = [r for r in rows if r["name"] == "Jason"]
    assert len(jasons) == 1 and jasons[0]["voiceprints"] == 2  # one row, not two


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
    store.add_speaker("A", struct.pack("2f", 1.0, 0.0))
    store.add_speaker("B", struct.pack("2f", 0.95, 0.31))  # near-duplicate voice
    pairs = c.get("/speakers/suggestions").json()["pairs"]
    assert pairs and pairs[0]["a"] == "A" and pairs[0]["b"] == "B" and pairs[0]["sim"] > 0.8


def test_persist_speakers_toggle_route(tmp_path):
    c, store = make_client(tmp_path)
    assert c.get("/settings/persist_speakers").json()["value"] == "1"
    assert c.post("/settings/persist_speakers", json={"value": "0"}).json()["value"] == "0"
    assert store.get_setting("persist_speakers") == "0"


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
