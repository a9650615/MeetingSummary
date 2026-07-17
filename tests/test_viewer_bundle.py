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
    new_mid, is_new = bundle.ingest_bundle(dst, str(data_dir), b2, tracks)

    assert is_new is True
    rows = dst.list_transcripts(new_mid)
    assert rows[0]["text"] == "開始開會"
    assert (data_dir / str(new_mid) / "mixed.m4a").read_bytes() == b"FAKE_M4A"


def test_reingest_same_created_at_is_a_topup(tmp_path):
    # Same created_at -> same meeting id, is_new False (top-up, not a new row).
    src = Store(tmp_path / "src.db"); _seed(src)
    b = bundle.meeting_to_bundle(src, 1, [])
    dst = Store(tmp_path / "dst.db")
    m1, new1 = bundle.ingest_bundle(dst, str(tmp_path / "d"), b, {})
    m2, new2 = bundle.ingest_bundle(dst, str(tmp_path / "d"), b, {})
    assert new1 is True and new2 is False
    assert m2 == m1
    assert len([m for m in dst.list_meetings() if m["created_at"] == 1721111111.0]) == 1


def test_topup_updates_meta_and_summary_but_preserves_transcript(tmp_path):
    dst = Store(tmp_path / "dst.db")
    # first push: transcript present, no summary yet, title A
    first = {"meeting": {"title": "會議 A", "created_at": 99.0, "lang": "zh-TW",
                         "status": "recording", "notes": ""},
             "segments": [], "summaries": [],
             "transcripts": [{"profile": "accurate", "track": "mic", "start_ms": 0,
                              "end_ms": 1000, "speaker": "我", "text": "原始逐字"}],
             "tracks": []}
    mid, new1 = bundle.ingest_bundle(dst, str(tmp_path / "d"), first, {})
    # simulate FireRed having corrected a line on the VM
    dst.add_transcript(mid, "firered", "mic", 0, 1000, "我", "校正後")
    # top-up: renamed title, added notes + a summary, status finalized; no transcript
    topup = {"meeting": {"title": "會議 A（正式）", "created_at": 99.0, "lang": "zh-TW",
                         "status": "finalized", "notes": "補充筆記"},
             "segments": [], "transcripts": [],
             "summaries": [{"kind": "minutes", "lang": "zh-TW", "text": "會議摘要",
                            "model": "m", "created_at": 100.0}],
             "tracks": []}
    mid2, new2 = bundle.ingest_bundle(dst, str(tmp_path / "d"), topup, {})
    assert mid2 == mid and new2 is False
    meeting = dst.get_meeting(mid)
    assert meeting["title"] == "會議 A（正式）"
    assert meeting["notes"] == "補充筆記"
    assert meeting["status"] == "finalized"
    assert dst.list_summaries(mid)[0]["text"] == "會議摘要"
    # transcript (incl the FireRed row) untouched — top-up never wipes it
    texts = {r["text"] for r in dst.list_transcripts(mid)}
    assert texts == {"原始逐字", "校正後"}


def test_topup_syncs_speaker_names_without_touching_text(tmp_path):
    dst = Store(tmp_path / "s.db")
    first = {"meeting": {"title": "m", "created_at": 5.0, "lang": "zh-TW",
                         "status": "recording", "notes": ""},
             "segments": [], "summaries": [],
             "transcripts": [{"profile": "accurate", "track": "mic", "start_ms": 0,
                              "end_ms": 1000, "speaker": "說話者1", "text": "哈囉"}],
             "tracks": []}
    mid, _ = bundle.ingest_bundle(dst, str(tmp_path / "d"), first, {})
    # FireRed corrected the same span, inheriting the speaker label
    dst.add_transcript(mid, "firered", "mic", 0, 1000, "說話者1", "哈囉（校正）")
    # re-push after renaming 說話者1 -> Scott on the Mac (same span, same text)
    topup = {"meeting": {"title": "m", "created_at": 5.0, "lang": "zh-TW",
                         "status": "recording", "notes": ""},
             "segments": [], "summaries": [],
             "transcripts": [{"profile": "accurate", "track": "mic", "start_ms": 0,
                              "end_ms": 1000, "speaker": "Scott", "text": "哈囉"}],
             "tracks": []}
    bundle.ingest_bundle(dst, str(tmp_path / "d"), topup, {})
    rows = {r["profile"]: r for r in dst.list_transcripts(mid)}
    assert rows["accurate"]["speaker"] == "Scott"        # renamed
    assert rows["firered"]["speaker"] == "Scott"         # rename propagated to FireRed
    assert rows["accurate"]["text"] == "哈囉"             # text untouched
    assert rows["firered"]["text"] == "哈囉（校正）"       # FireRed text preserved


def test_bundle_carries_voiceprints_and_ingest_merges_nondestructively(tmp_path):
    import numpy as np
    src = Store(tmp_path / "src.db")
    mid = src.create_meeting("m", 7.0, "zh-TW")
    src.add_transcript(mid, "accurate", "mic", 0, 1, "Scott", "hi")
    cA = np.ones(192, dtype=np.float32).tobytes()
    cB = (np.ones(192, dtype=np.float32) * 2).tobytes()
    src.add_speaker("Scott", cA)
    src.add_speaker("Mia", cB)

    b = bundle.meeting_to_bundle(src, mid, [])
    names = {s["name"] for s in b["speakers"]}
    assert names == {"Scott", "Mia"}
    assert b["speakers"][0]["centroid_b64"]              # base64 present

    dst = Store(tmp_path / "dst.db")
    # VM already has a curated "Scott" voiceprint that must NOT be clobbered
    cOwn = (np.ones(192, dtype=np.float32) * 9).tobytes()
    dst.add_speaker("Scott", cOwn)
    bundle.ingest_bundle(dst, str(tmp_path / "d"), b, {})

    got = {s["name"]: bytes(s["centroid"]) for s in dst.list_speakers()}
    assert "Mia" in got and got["Mia"] == cB             # new name added
    assert got["Scott"] == cOwn                          # existing VM voiceprint untouched
