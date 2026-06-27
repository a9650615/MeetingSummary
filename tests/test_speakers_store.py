import struct

from store import Store


def _cen(*xs):
    return struct.pack(f"{len(xs)}f", *xs)


def test_global_speakers_roundtrip(tmp_path):
    s = Store(str(tmp_path / "m.db"))
    sid = s.add_speaker("說話者", _cen(1.0, 0.0))
    s.set_speaker_name(sid, "對方3")
    assert s.list_speakers()[0]["name"] == "對方3"
    s.update_speaker_centroid(sid, _cen(0.0, 1.0), 4)
    row = s.list_speakers()[0]
    assert row["count"] == 4 and struct.unpack("2f", row["centroid"]) == (0.0, 1.0)
    # rename propagates by name (unique placeholder) -> 1 row
    assert s.rename_global_speaker("對方3", "Scott") == 1
    assert s.list_speakers()[0]["name"] == "Scott"


def test_speakers_with_stats(tmp_path):
    s = Store(str(tmp_path / "m.db"))
    s.add_speaker("張三", _cen(1.0, 0.0))
    m1 = s.create_meeting("A", 1.0, "zh-TW")
    m2 = s.create_meeting("B", 2.0, "zh-TW")
    s.add_transcript(m1, "accurate", "mic", 0, 1, "張三", "你好")
    s.add_transcript(m1, "accurate", "mic", 1, 2, "張三", "再見")
    s.add_transcript(m2, "accurate", "mic", 0, 1, "張三", "早安")
    st = {r["name"]: r for r in s.speakers_with_stats()}["張三"]
    assert st["meetings"] == 2 and st["utterances"] == 3


def test_speakers_with_stats_hides_orphan_voiceprints(tmp_path):
    # A voiceprint no transcript references (0 utterances) must not show on the
    # speaker page — "根本沒有紀錄就不應該進語者頁面" (e.g. orphaned by re-transcribe).
    s = Store(str(tmp_path / "m.db"))
    s.add_speaker("我68", _cen(1.0, 0.0))  # orphan: no transcripts
    s.add_speaker("張三", _cen(0.0, 1.0))
    m = s.create_meeting("A", 1.0, "zh-TW")
    s.add_transcript(m, "accurate", "mic", 0, 1, "張三", "hi")
    names = [r["name"] for r in s.speakers_with_stats()]
    assert names == ["張三"]  # 我68 hidden


def test_merge_speakers_reassigns_and_drops(tmp_path):
    s = Store(str(tmp_path / "m.db"))
    s.add_speaker("張三", _cen(1.0, 0.0))
    s.add_speaker("對方2", _cen(0.9, 0.1))   # same person, different label
    m = s.create_meeting("A", 1.0, "zh-TW")
    s.add_transcript(m, "accurate", "mic", 0, 1, "張三", "a")
    s.add_transcript(m, "accurate", "mic", 1, 2, "對方2", "b")
    assert s.merge_speakers("張三", "對方2") == 1          # 1 transcript moved
    assert [r["name"] for r in s.list_speakers()] == ["張三"]  # 對方2 voiceprint gone
    assert s.speaker_utterances("張三")[0]["text"] in ("a", "b")
    assert len(s.speaker_utterances("張三")) == 2


def test_delete_speaker_keeps_transcript_label(tmp_path):
    s = Store(str(tmp_path / "m.db"))
    sid = s.add_speaker("張三", _cen(1.0, 0.0))
    m = s.create_meeting("A", 1.0, "zh-TW")
    s.add_transcript(m, "accurate", "mic", 0, 1, "張三", "a")
    s.delete_speaker(sid)
    assert s.list_speakers() == []                       # voiceprint forgotten
    assert s.list_transcripts(m)[0]["speaker"] == "張三"   # label untouched


def test_settings_kv(tmp_path):
    s = Store(str(tmp_path / "m.db"))
    assert s.get_setting("persist_speakers", "1") == "1"  # default
    s.set_setting("persist_speakers", "0")
    assert s.get_setting("persist_speakers") == "0"
    s.set_setting("persist_speakers", "1")                # upsert
    assert s.get_setting("persist_speakers") == "1"


def test_rename_updates_nonmatches(tmp_path):
    s = Store(str(tmp_path / "m.db"))
    s.add_speaker_nonmatch("Alice", "Bob")
    s.rename_speaker_global("Bob", "Bobby")
    assert s.list_speaker_nonmatches() == [("Alice", "Bobby")]  # not stale "Bob"


def test_merge_meetings_clears_tags(tmp_path):
    s = Store(str(tmp_path / "m.db"))
    t = s.create_meeting("T", 1.0, "zh-TW"); src = s.create_meeting("S", 2.0, "zh-TW")
    s.add_tag(src, "x")
    s.merge_meetings(t, [src])
    # the source's meeting_tags row is gone (no orphan)
    n = s.db.execute("SELECT COUNT(*) c FROM meeting_tags WHERE meeting_id=?", (src,)).fetchone()["c"]
    assert n == 0
