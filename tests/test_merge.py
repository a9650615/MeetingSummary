import os

from app import _assemble_track, _meeting_tracks
from store import Store, group_by_proximity


def test_group_by_proximity():
    # gap 300s: 100 & 200 close; 1000 close to 1100; 5000 alone.
    meetings = [{"id": 1, "created_at": 100}, {"id": 2, "created_at": 200},
                {"id": 3, "created_at": 1000}, {"id": 4, "created_at": 1100},
                {"id": 5, "created_at": 5000}]
    groups = group_by_proximity(meetings, gap_s=300)
    assert groups == [[1, 2], [3, 4]]   # only groups of >=2, singletons dropped


def test_group_by_proximity_unsorted_input():
    meetings = [{"id": 9, "created_at": 250}, {"id": 8, "created_at": 100}]
    assert group_by_proximity(meetings, gap_s=300) == [[8, 9]]


def _store(tmp_path):
    return Store(tmp_path / "m.db")


def test_merge_meetings_combines_and_rebases(tmp_path):
    s = _store(tmp_path)
    a = s.create_meeting("A", 1000.0, "zh-TW")
    b = s.create_meeting("B", 1005.0, "zh-TW")   # 5s after A
    s.add_transcript(a, "live", "mic", 0, 500, "我", "a0")
    s.add_transcript(b, "live", "mic", 2000, 2500, "我", "b2")  # 2s into B
    s.add_segment(b, 0, "data/2", 1005.0, 0, "recorded")

    s.merge_meetings(a, [b])

    rows = s.list_transcripts(a)
    assert [r["text"] for r in rows] == ["a0", "b2"]
    # b's transcript rebased: 5s gap *1000 + 2000 = 7000ms
    assert rows[1]["start_ms"] == 7000
    assert len(s.list_segments(a)) == 1          # b's segment moved to a
    assert s.get_meeting(b) is None              # source deleted


def test_merge_into_earliest(tmp_path):
    s = _store(tmp_path)
    a = s.create_meeting("A", 1000.0, "zh-TW")
    b = s.create_meeting("B", 1005.0, "zh-TW")
    target = s.merge_into_earliest([b, a])       # order-agnostic
    assert target == a and s.get_meeting(b) is None


def _pcm(d, track, val, n):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{track}.pcm"), "wb") as f:
        f.write(bytes([val, 0]) * n)  # n int16 samples


def test_assemble_track_places_segments_at_time_offsets(tmp_path):
    s = _store(tmp_path)
    mid = s.create_meeting("M", 1000.0, "zh-TW")
    d0, d1 = str(tmp_path / "s0"), str(tmp_path / "s1")
    _pcm(d0, "mic", 1, 16000)   # 1 s at t0
    _pcm(d1, "mic", 2, 16000)   # 1 s, source started 2 s after target
    s.add_segment(mid, 0, d0, started_at=1000.0, duration_s=1, origin="recorded")
    s.add_segment(mid, 1, d1, started_at=1002.0, duration_s=1, origin="recorded")
    pcm = _assemble_track(s, mid, "mic")
    # layout: [seg0 1s][1s silence gap][seg1 1s] = 3 s
    assert len(pcm) == 16000 * 2 * 3
    assert pcm[:2] == b"\x01\x00"                 # seg0 at 0
    assert pcm[16000 * 2:16000 * 2 + 2] == b"\x00\x00"          # 1-2s silence
    assert pcm[16000 * 2 * 2:16000 * 2 * 2 + 2] == b"\x02\x00"  # seg1 at 2 s


def test_assemble_track_dedupes_shared_dir(tmp_path):
    # Two segments from the SAME source dir (session resume) -> place its pcm once.
    s = _store(tmp_path)
    mid = s.create_meeting("M", 1000.0, "zh-TW")
    d = str(tmp_path / "shared")
    _pcm(d, "mic", 1, 16000)
    s.add_segment(mid, 0, d, started_at=1000.0, duration_s=1, origin="recorded")
    s.add_segment(mid, 1, d, started_at=1001.0, duration_s=1, origin="recorded")
    pcm = _assemble_track(s, mid, "mic")
    assert len(pcm) == 16000 * 2          # one copy, not two
    assert _meeting_tracks(s, mid) == ["mic"]


def test_assemble_track_none_when_absent(tmp_path):
    s = _store(tmp_path)
    mid = s.create_meeting("M", 1000.0, "zh-TW")
    s.add_segment(mid, 0, str(tmp_path / "empty"), started_at=1000.0,
                  duration_s=0, origin="recorded")
    assert _assemble_track(s, mid, "mic") is None
    assert _meeting_tracks(s, mid) == []
