from summarize import summarize, build_prompt, _dedup_lines


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
