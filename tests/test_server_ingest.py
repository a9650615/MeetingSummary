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


def test_ingest_rejects_valid_zip_missing_meeting_key(tmp_path):
    app = build_server(db_path=str(tmp_path / "s.db"),
                       data_dir=str(tmp_path / "data"))
    c = TestClient(app)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("meeting.json", json.dumps({"tracks": []}))
    r = c.post("/ingest-bundle",
               files={"bundle": ("b.zip", buf.getvalue(), "application/zip")})
    assert r.status_code == 400
