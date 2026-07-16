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
