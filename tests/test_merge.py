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
