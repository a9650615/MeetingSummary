import re

import app


def test_meeting_page_js_has_no_line_comments():
    """The meeting-page JS is concatenated onto ONE line, so a `//` comment eats the
    rest of the script -> every button dies (real regression, v0.1.12-14). Forbid
    bare // comments in that script (URLs like http:// are stripped first)."""
    html = app._detail_page(
        1, {"title": "t", "status": "finalized", "created_at": 0.0, "lang": "zh-TW"},
        transcripts=[], summaries=[], audio_tracks=(), tags=[])
    js = re.findall(r"<script>(.*?)</script>", html, re.S)[-1]
    cleaned = re.sub(r"\w+://", "", js)   # drop http:// ws:// so only real // remain
    assert "//" not in cleaned, "bare // comment in single-line JS breaks all handlers"
