from diarize import assign_speakers


def test_assign_speakers_by_time():
    transcripts = [
        {"start_ms": 500, "speaker": "對方", "text": "a"},
        {"start_ms": 3000, "speaker": "對方", "text": "b"},
        {"start_ms": 9000, "speaker": "對方", "text": "c"},
    ]
    segs = [{"start": 0.0, "end": 2.0, "speaker": 0},
            {"start": 2.0, "end": 5.0, "speaker": 1}]
    out = assign_speakers(transcripts, segs)
    assert out[0]["speaker"] == "說話者1"   # 0.5s -> seg 0
    assert out[1]["speaker"] == "說話者2"   # 3.0s -> seg 1
    assert out[2]["speaker"] == "對方"      # 9.0s -> no segment, unchanged
    assert out[0]["text"] == "a"           # other fields preserved


def test_assign_speakers_custom_prefix():
    out = assign_speakers([{"start_ms": 100, "speaker": "x"}],
                          [{"start": 0, "end": 1, "speaker": 2}], prefix="對方")
    assert out[0]["speaker"] == "對方3"


def test_assign_speakers_empty_segments_noop():
    t = [{"start_ms": 0, "speaker": "我", "text": "hi"}]
    assert assign_speakers(t, []) == t
