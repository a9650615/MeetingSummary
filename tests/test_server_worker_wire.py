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


def test_resume_noop_when_already_done(tmp_path, monkeypatch):
    resumes = []
    class FakeWorker:
        def __init__(self, *a): pass
        def start(self): pass
        def enqueue(self, mid, restart=False): resumes.append((mid, restart))
        def stop(self, mid): pass
    monkeypatch.setattr(sm.firered_worker, "FireRedWorker", FakeWorker)
    app = sm.build_server(db_path=str(tmp_path / "s.db"), data_dir=str(tmp_path / "d"))
    store = app.state.store
    sm.firered_worker.set_progress(store, 1, state="done", done=5, total=5)
    c = TestClient(app)
    resp = c.post("/meetings/1/firered/resume").json()
    assert resp == {"resumed": False, "reason": "already done"}
    assert resumes == []
    resp2 = c.post("/meetings/1/firered/resume?restart=1").json()
    assert resp2 == {"resumed": True}
    assert resumes[-1] == (1, True)
