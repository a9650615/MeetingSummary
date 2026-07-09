from store import Store


def _store(tmp_path):
    return Store(tmp_path / "meetings.db")


def test_caption_hides_placeholder_but_keeps_named(tmp_path):
    s = _store(tmp_path)
    mid = s.create_meeting(title="m", created_at=1.0, lang="zh-TW")
    s.add_transcript(mid, "live", "system", 0, 500, "說話者1", "未辨識的一句話")
    assert s.latest_transcript(mid) == "未辨識的一句話"          # placeholder -> no name
    s.add_transcript(mid, "live", "mic", 500, 900, "我", "我講的話")
    assert s.latest_transcript(mid) == "我：我講的話"            # bare track label kept
    s.add_transcript(mid, "live", "system", 900, 1300, "Scott", "有命名的一句話")
    assert s.latest_transcript(mid) == "Scott：有命名的一句話"    # human name kept
    assert s.recent_transcripts(mid, limit=3) == [
        "未辨識的一句話", "我：我講的話", "Scott：有命名的一句話"]


def test_create_and_get_meeting(tmp_path):
    s = _store(tmp_path)
    mid = s.create_meeting(title="standup", created_at=1.0, lang="zh-TW")
    m = s.get_meeting(mid)
    assert m["title"] == "standup"
    assert m["lang"] == "zh-TW"
    assert m["status"] == "recording"  # default on create


def test_add_segment_and_list(tmp_path):
    s = _store(tmp_path)
    mid = s.create_meeting(title="m", created_at=1.0, lang="zh-TW")
    s.add_segment(mid, idx=0, dir_path="data/m/0", started_at=1.0,
                  duration_s=5.0, origin="recorded")
    segs = s.list_segments(mid)
    assert len(segs) == 1
    assert segs[0]["idx"] == 0
    assert segs[0]["origin"] == "recorded"


def test_update_title(tmp_path):
    s = _store(tmp_path)
    mid = s.create_meeting(title="舊", created_at=1.0, lang="zh-TW")
    s.update_title(mid, "週會 6/24")
    assert s.get_meeting(mid)["title"] == "週會 6/24"


def test_rename_speaker_across_meeting(tmp_path):
    s = _store(tmp_path)
    mid = s.create_meeting(title="m", created_at=1.0, lang="zh-TW")
    for i in range(3):
        s.add_transcript(mid, "live", "system", i * 1000, i * 1000 + 500,
                         "說話者1", f"t{i}")
    s.add_transcript(mid, "live", "system", 9000, 9500, "說話者2", "other")
    n = s.rename_speaker(mid, "說話者1", "Scott")
    assert n == 3
    spk = {r["speaker"] for r in s.list_transcripts(mid)}
    assert spk == {"Scott", "說話者2"}


def test_transcripts_roundtrip(tmp_path):
    s = _store(tmp_path)
    mid = s.create_meeting(title="m", created_at=1.0, lang="zh-TW")
    s.add_transcript(mid, profile="accurate", track="mic",
                     start_ms=0, end_ms=1000, speaker="我", text="你好")
    rows = s.list_transcripts(mid)
    assert rows[0]["text"] == "你好"
    assert rows[0]["track"] == "mic"


def test_summary_roundtrip(tmp_path):
    s = _store(tmp_path)
    mid = s.create_meeting(title="m", created_at=1.0, lang="zh-TW")
    s.add_summary(mid, kind="minutes", lang="zh-TW", text="...",
                  model="qwen2.5-14b", created_at=2.0)
    assert s.list_summaries(mid)[0]["kind"] == "minutes"


def test_list_unfinalized_for_recovery(tmp_path):
    s = _store(tmp_path)
    live = s.create_meeting(title="crashed", created_at=1.0, lang="zh-TW")
    done = s.create_meeting(title="ok", created_at=1.0, lang="zh-TW")
    s.finalize_meeting(done)
    ids = [m["id"] for m in s.list_unfinalized()]
    assert live in ids and done not in ids


def test_set_notes_roundtrip(tmp_path):
    s = _store(tmp_path)
    mid = s.create_meeting(title="m", created_at=1.0, lang="zh-TW")
    assert (s.get_meeting(mid)["notes"] or "") == ""  # default empty
    s.set_notes(mid, "負責人 Amy")
    assert s.get_meeting(mid)["notes"] == "負責人 Amy"


def test_append_note(tmp_path):
    s = _store(tmp_path)
    mid = s.create_meeting("m", 1.0, "zh-TW")
    s.append_note(mid, "第一行")
    s.append_note(mid, "第二行")
    assert s.get_meeting(mid)["notes"] == "第一行\n第二行"


def test_latest_transcript(tmp_path):
    s = _store(tmp_path)
    mid = s.create_meeting("m", 1.0, "zh-TW")
    assert s.latest_transcript(mid) is None
    s.add_transcript(mid, "live", "mic", 0, 1000, "我", "你好")
    s.add_transcript(mid, "live", "mic", 1000, 2000, "我", "最新一句")
    assert s.latest_transcript(mid) == "我：最新一句"
