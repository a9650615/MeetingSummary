from summarize import summarize, build_prompt, _dedup_lines, _ground


def test_ground_replaces_fabricated_owner_and_time():
    transcript = "我: 我們來討論 SIT 的進度\n對方: 好"
    out = "待辦行動:\n1. 負責人: 小米\n2. 舉行時間: 當天"
    g = _ground(out, transcript)
    assert "小米" not in g and "當天" not in g       # neither was in the transcript
    assert "負責人: 未指定" in g and "舉行時間: 未定" in g


def test_ground_keeps_names_actually_said():
    transcript = "我: 這個給 Michael 處理\n對方: 好,期限 週五"
    out = "- 負責人: Michael 事項\n期限: 週五"
    assert _ground(out, transcript) == out          # both present -> untouched


def test_ground_blanks_fabricated_bracket_owner():
    g = _ground("- [小米] 提交報告", "我: 提交報告")
    assert g == "- [未指定] 提交報告"


def test_dedup_collapses_repeated_numbered_loop():
    # The LLM looped one item 4x with incrementing numbers (real report).
    looped = "\n".join(f"{i}. 會議討論: 中午前完成 Bug 區分並提交至 SIT。"
                       for i in range(29, 33)) + "\n34. 會議討論: 中午"
    out = _dedup_lines(looped)
    lines = out.splitlines()
    assert lines == ["29. 會議討論: 中午前完成 Bug 區分並提交至 SIT。",
                     "34. 會議討論: 中午"]  # loop collapsed to first; truncated tail kept


def test_dedup_keeps_distinct_lines():
    txt = "1. 甲做 A\n2. 乙做 B\n3. 丙做 C"
    assert _dedup_lines(txt) == txt  # genuine list untouched


def test_short_transcript_single_pass():
    calls = []
    backend = lambda p: calls.append(p) or "SUMMARY"
    out = summarize("short", kind="minutes", lang="zh-TW",
                    backend=backend, max_chars=1000)
    assert out == "SUMMARY"
    assert len(calls) == 1  # no map-reduce for short input


def test_long_transcript_map_reduce():
    calls = []
    backend = lambda p: calls.append(p) or "chunk-summary"
    text = "\n".join(f"line {i}" for i in range(100))
    summarize(text, kind="bullets", lang="zh-TW", backend=backend, max_chars=50)
    # >1 map call + a reduce call — proves the long path (B3) ran.
    assert len(calls) > 1


def test_prompt_reflects_kind_and_lang():
    assert "決議" in build_prompt("hi", kind="minutes", lang="zh-TW")
    assert "條列" in build_prompt("hi", kind="bullets", lang="zh-TW")
    assert "zh-TW" in build_prompt("hi", kind="minutes", lang="zh-TW")


def test_notes_injected_and_exempt_from_grounding():
    from summarize import build_prompt, _ground
    p = build_prompt("逐字稿", kind="minutes", lang="zh-TW", notes="負責人 Amy；7/3 交稿")
    assert "使用者現場筆記" in p and "Amy" in p
    # a name present only in the notes must survive grounding (it's user truth)
    assert "Amy" in _ground("負責人:Amy", "逐字稿無名\n負責人 Amy")
    # but a name in neither transcript nor notes is still scrubbed
    assert "Amy" not in _ground("負責人:Amy", "逐字稿無名")


def test_ground_scrubs_indented_bracket_owner():
    from summarize import _ground
    # nested/indented "- [name]" bullets must also be grounded (small models emit them)
    assert "未指定" in _ground("  - [小米] 待辦", "逐字稿沒有這個名字")
    assert "小米" not in _ground("    - [小米] 待辦", "逐字稿沒有這個名字")
