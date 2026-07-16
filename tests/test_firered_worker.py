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
