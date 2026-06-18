from summarize import summarize, build_prompt


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
