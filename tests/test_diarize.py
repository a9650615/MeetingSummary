import numpy as np

import diarize
from diarize import SpeakerTracker, assign_speakers, match_speaker


def test_match_speaker_cosine():
    a = np.array([1.0, 0, 0], dtype=np.float32)
    b = np.array([0, 1.0, 0], dtype=np.float32)
    known = [(1, a), (2, b)]
    assert match_speaker(a, known, 0.55)[0] == 1            # same vector -> match
    assert match_speaker(b + np.array([0, 0, .1]), known, 0.55)[0] == 2
    assert match_speaker(np.array([0, 0, 1.0], np.float32), known, 0.55)[0] is None  # orthogonal -> new


def test_assign_speakers_uses_names_map():
    line = [{"start_ms": 0, "end_ms": 2000, "speaker": "x"}]
    segs = [{"start": 0.0, "end": 2.0, "speaker": 7}]   # cluster id 7
    out = assign_speakers(line, segs, names={7: "Alice"})
    assert out[0]["speaker"] == "Alice"


def test_resolve_models_prefers_existing_paths(tmp_path, monkeypatch):
    # explicit args that exist must win and never trigger a download
    seg = tmp_path / "seg.onnx"; seg.write_bytes(b"x")
    emb = tmp_path / "emb.onnx"; emb.write_bytes(b"x")
    monkeypatch.setattr(diarize, "_ensure_seg", lambda *a, **k: (_ for _ in ()).throw(AssertionError("downloaded")))
    monkeypatch.setattr(diarize, "_ensure_emb", lambda *a, **k: (_ for _ in ()).throw(AssertionError("downloaded")))
    assert diarize._resolve_models(str(seg), str(emb)) == (str(seg), str(emb))
    # missing explicit path -> falls back to the (mocked) provisioner
    monkeypatch.setattr(diarize, "_ensure_seg", lambda *a, **k: "SEG")
    monkeypatch.setattr(diarize, "_ensure_emb", lambda *a, **k: "EMB")
    assert diarize._resolve_models("/nope", "/nope") == ("SEG", "EMB")


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


def test_assign_speakers_dominant_over_interjection():
    # a line spans [0,5]s; a 0.5s interjection by speaker B at the start, rest is
    # speaker A -> the line should be labeled A (max overlap), not B.
    line = [{"start_ms": 0, "end_ms": 5000, "speaker": "我", "text": "x"}]
    segs = [{"start": 0.0, "end": 0.5, "speaker": 1},   # B interjects briefly
            {"start": 0.5, "end": 5.0, "speaker": 0}]   # A dominates
    out = assign_speakers(line, segs)
    assert out[0]["speaker"] == "說話者1"  # speaker 0 -> id 1 (the dominant one)


def test_assign_speakers_remaps_ids_to_1_based():
    # sherpa cluster ids need not start at 0; remap distinct ids -> 1..N.
    out = assign_speakers([{"start_ms": 100, "speaker": "x"}],
                          [{"start": 0, "end": 1, "speaker": 2}], prefix="對方")
    assert out[0]["speaker"] == "對方1"  # the only cluster -> 1, not 3


def test_assign_speakers_empty_segments_noop():
    t = [{"start_ms": 0, "speaker": "我", "text": "hi"}]
    assert assign_speakers(t, []) == t


def test_assign_speakers_mark_overlap():
    # line [0,4]s split ~evenly between two speakers -> 2nd covers >=30% -> 🔀 tag
    line = [{"start_ms": 0, "end_ms": 4000, "speaker": "對方", "text": "x"}]
    segs = [{"start": 0.0, "end": 2.0, "speaker": 0},
            {"start": 2.0, "end": 4.0, "speaker": 1}]
    off = assign_speakers(line, segs)                       # default: dominant only
    assert off[0]["speaker"] == "說話者1" and "overlap" not in off[0]
    on = assign_speakers(line, segs, mark_overlap=True)
    assert on[0]["speaker"] == "🔀 說話者1+說話者2" and on[0]["overlap"] is True


def test_mark_overlap_ignores_brief_interjection():
    # 0.4s second speaker on a 5s line (<30%) -> still single dominant, no tag
    line = [{"start_ms": 0, "end_ms": 5000, "speaker": "我", "text": "x"}]
    segs = [{"start": 0.0, "end": 0.4, "speaker": 1},
            {"start": 0.4, "end": 5.0, "speaker": 0}]
    out = assign_speakers(line, segs, mark_overlap=True)
    assert out[0]["speaker"] == "說話者1" and "overlap" not in out[0]
