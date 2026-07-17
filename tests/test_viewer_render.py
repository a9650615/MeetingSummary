# tests/test_viewer_render.py
from viewer import render


def test_pick_defaults_to_local_not_firered():
    # FireRed is a comparison, NOT an auto-upgrade: default view stays local.
    ts = [{"profile": "accurate", "track": "mixed", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "local text"},
          {"profile": "firered", "track": "mixed", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "corrected"}]
    assert [p["text"] for p in render.pick_transcripts(ts)] == ["local text"]
    assert render.has_firered(ts) is True


def test_pick_firered_only_when_asked():
    ts = [{"profile": "accurate", "track": "mixed", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "local text"},
          {"profile": "firered", "track": "mixed", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "corrected"}]
    assert [p["text"] for p in render.pick_transcripts(ts, "firered")] == ["corrected"]


def test_has_firered_false_without_firered():
    ts = [{"profile": "accurate", "track": "mixed", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "local"}]
    assert render.has_firered(ts) is False
    assert [p["text"] for p in render.pick_transcripts(ts)] == ["local"]


def test_export_md_has_title_and_lines():
    md = render.export_md({"title": "週會"},
                          [{"profile": "accurate", "track": "mixed", "start_ms": 0,
                            "end_ms": 1, "speaker": "我", "text": "hi"}],
                          [{"kind": "minutes", "text": "重點"}])
    assert "# 週會" in md and "我：hi" in md and "重點" in md


def test_pick_excludes_firered_staging():
    ts = [{"profile": "accurate", "track": "m", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "local"},
          {"profile": "firered_staging", "track": "m", "start_ms": 0, "end_ms": 1,
           "speaker": "我", "text": "half"}]
    picked = render.pick_transcripts(ts)
    assert [p["text"] for p in picked] == ["local"]   # staging never shown


def test_render_detail_shows_both_versions_with_toggle():
    html = render.render_detail(
        {"id": 7, "title": "週會", "created_at": 1721111111.0},
        [{"profile": "accurate", "track": "mixed", "start_ms": 0, "end_ms": 1,
          "speaker": "我", "text": "本地文字"},
         {"profile": "firered", "track": "mixed", "start_ms": 0, "end_ms": 1,
          "speaker": "我", "text": "校正文字"}],
        [], ["mixed"], [])
    assert "/meetings/7/audio/mixed.m4a" in html
    # both versions rendered; local visible, firered in a hidden toggle block
    assert "本地文字" in html and "校正文字" in html
    assert "tx-toggle" in html and "顯示 FireRed 校正版" in html
    assert "id='tx-firered' style='display:none'" in html   # firered hidden by default


def test_render_detail_no_toggle_without_firered():
    html = render.render_detail(
        {"id": 8, "title": "會", "created_at": 1.0},
        [{"profile": "accurate", "track": "mic", "start_ms": 0, "end_ms": 1,
          "speaker": "我", "text": "只有本地"}], [], ["mic"], [])
    assert "只有本地" in html and "tx-toggle" not in html
