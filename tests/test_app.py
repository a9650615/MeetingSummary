from fastapi.testclient import TestClient

from app import create_app
from store import Store


def make_client(tmp_path, summary_backend=None, asr_backend=None):
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=summary_backend or (lambda p: "SUMMARY"),
                     asr_backend=asr_backend)
    return TestClient(app), store


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


def test_summary_route_runs_backend_and_stores(tmp_path):
    captured = {}

    def backend(p):
        captured["p"] = p
        return "會議記錄"

    c, store = make_client(tmp_path, summary_backend=backend)
    mid = store.create_meeting("m", 1.0, "zh-TW")
    store.add_transcript(mid, "accurate", "mic", 0, 1000, "我", "討論預算")
    r = c.post(f"/meetings/{mid}/summary", json={"kind": "minutes"})
    assert r.status_code == 200
    assert r.json()["text"] == "會議記錄"
    assert "討論預算" in captured["p"]  # transcript fed into the prompt
    assert store.list_summaries(mid)[0]["text"] == "會議記錄"


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
