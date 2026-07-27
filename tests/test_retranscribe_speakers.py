"""Re-transcribe must not come back with worse speakers than live had.

Two defects this pins down:
  * _diarize_meeting derived its track list from _meeting_tracks, which collapses
    mic+system to a synthetic "mixed". No transcript row is ever tagged "mixed",
    so its row filter matched nothing and the pass relabelled zero rows.
  * clear_transcripts wipes the live rows, destroying names live had already
    recognized, with nothing putting them back.
"""
import app
from store import Store


def _meeting_with_live_rows(tmp_path):
    store = Store(tmp_path / "m.db")
    mid = store.create_meeting("M", 1000.0, "zh-TW")
    store.add_transcript(mid, "live", "mic", 0, 2000, "Alice", "早安")
    store.add_transcript(mid, "live", "system", 2000, 4000, "對方", "你好")
    store.add_transcript(mid, "live", "system", 4000, 6000, "Bob", "收到")
    store.add_transcript(mid, "live", "mic", 6000, 8000, "說話者2", "嗯嗯")
    return store, mid


def test_real_speaker_name_ignores_side_and_cluster_labels():
    assert app._real_speaker_name("Alice") == "Alice"
    for junk in ("我", "對方", "混合", "說話者2", "對方3", "我1", "", None):
        assert app._real_speaker_name(junk) is None


def test_live_name_spans_keeps_only_recognized_people(tmp_path):
    store, mid = _meeting_with_live_rows(tmp_path)
    spans = app.live_name_spans(store, mid)
    assert {(s[2], s[3]) for s in spans} == {("mic", "Alice"), ("system", "Bob")}


def test_names_are_reapplied_to_the_rebuilt_transcript(tmp_path):
    store, mid = _meeting_with_live_rows(tmp_path)
    spans = app.live_name_spans(store, mid)
    store.clear_transcripts(mid)          # what re-transcribe does
    # rebuilt rows carry only the side label, on the same timeline
    store.add_transcript(mid, "accurate", "mic", 0, 2000, "我", "早安")
    store.add_transcript(mid, "accurate", "system", 2000, 4000, "對方", "你好")
    store.add_transcript(mid, "accurate", "system", 4000, 6000, "對方", "收到")

    assert app.apply_name_spans(store, mid, spans) == 2
    got = {(r["start_ms"], r["speaker"]) for r in store.list_transcripts(mid)}
    assert got == {(0, "Alice"), (2000, "對方"), (4000, "Bob")}


def test_reapply_does_not_overwrite_a_name_the_new_pass_found(tmp_path):
    store, mid = _meeting_with_live_rows(tmp_path)
    spans = app.live_name_spans(store, mid)
    store.clear_transcripts(mid)
    store.add_transcript(mid, "accurate", "mic", 0, 2000, "Carol", "早安")
    assert app.apply_name_spans(store, mid, spans) == 0
    assert store.list_transcripts(mid)[0]["speaker"] == "Carol"


def test_reapply_never_crosses_tracks(tmp_path):
    store, mid = _meeting_with_live_rows(tmp_path)
    spans = app.live_name_spans(store, mid)
    store.clear_transcripts(mid)
    # same time span as live's Alice(mic), but on the system track
    store.add_transcript(mid, "accurate", "system", 0, 2000, "對方", "早安")
    assert app.apply_name_spans(store, mid, spans) == 0


def test_diarize_uses_the_tracks_the_transcript_actually_carries(tmp_path, monkeypatch):
    # mic+system present -> _meeting_tracks says ["mixed"], but every row is
    # mic/system. _diarize_meeting must follow the rows, or it relabels nothing.
    store = Store(tmp_path / "m.db")
    mid = store.create_meeting("M", 1000.0, "zh-TW")
    store.add_transcript(mid, "accurate", "mic", 0, 2000, "我", "a")
    store.add_transcript(mid, "accurate", "system", 0, 2000, "對方", "b")

    seen = []
    monkeypatch.setattr(app, "_assemble_track",
                        lambda s, m, track: seen.append(track) or None)
    app._diarize_meeting(store, mid, {mid: {}})
    assert seen == ["mic", "system"]      # not ["mixed"]
