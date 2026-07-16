# tests/test_viewer_render.py
from viewer import render


def test_pick_prefers_firered():
    ts = [{"profile": "accurate", "track": "mixed", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "local text"},
          {"profile": "firered", "track": "mixed", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "corrected"}]
    picked = render.pick_transcripts(ts)
    assert [p["text"] for p in picked] == ["corrected"]
    assert render.active_profile(ts) == "firered"


def test_pick_falls_back_to_local():
    ts = [{"profile": "accurate", "track": "mixed", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "local"}]
    assert [p["text"] for p in render.pick_transcripts(ts)] == ["local"]
    assert render.active_profile(ts) == "local"


def test_export_md_has_title_and_lines():
    md = render.export_md({"title": "週會"},
                          [{"profile": "accurate", "track": "mixed", "start_ms": 0,
                            "end_ms": 1, "speaker": "我", "text": "hi"}],
                          [{"kind": "minutes", "text": "重點"}])
    assert "# 週會" in md and "我：hi" in md and "重點" in md


def test_render_detail_contains_audio_and_text():
    html = render.render_detail(
        {"id": 7, "title": "週會", "created_at": 1721111111.0},
        [{"profile": "firered", "track": "mixed", "start_ms": 0, "end_ms": 1,
          "speaker": "我", "text": "corrected"}],
        [], ["mixed"], [])
    assert "/meetings/7/audio/mixed.m4a" in html
    assert "corrected" in html and "FireRed" in html
