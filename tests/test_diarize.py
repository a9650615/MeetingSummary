import numpy as np

from diarize import SpeakerTracker, assign_speakers


def test_speaker_tracker_online_clustering():
    t = SpeakerTracker(threshold=0.5)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert t.assign(a) == 0                       # first speaker
    assert t.assign(a + np.array([0.1, 0, 0])) == 0  # similar -> same
    assert t.assign(b) == 1                       # different -> new speaker
    assert t.assign(b + np.array([0, 0.1, 0])) == 1  # similar to 2nd -> same
    assert t.assign(a) == 0                       # back to speaker 0


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


def test_assign_speakers_remaps_ids_to_1_based():
    # sherpa cluster ids need not start at 0; remap distinct ids -> 1..N.
    out = assign_speakers([{"start_ms": 100, "speaker": "x"}],
                          [{"start": 0, "end": 1, "speaker": 2}], prefix="對方")
    assert out[0]["speaker"] == "對方1"  # the only cluster -> 1, not 3


def test_assign_speakers_empty_segments_noop():
    t = [{"start_ms": 0, "speaker": "我", "text": "hi"}]
    assert assign_speakers(t, []) == t
