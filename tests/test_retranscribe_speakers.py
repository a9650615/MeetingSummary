"""Re-transcribe must not come back with worse speakers than live had.

_diarize_meeting derived its track list from _meeting_tracks, which collapses
mic+system to a synthetic "mixed". No transcript row is ever tagged "mixed", so
its row filter matched nothing and the pass relabelled zero rows.
"""
import app
from store import Store


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
