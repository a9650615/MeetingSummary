from store import Store


def _store(tmp_path):
    return Store(tmp_path / "meetings.db")


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
