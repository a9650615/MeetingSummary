from summarize import (build_correction_prompt, correct_transcript, summarize,
                       build_prompt, _dedup_lines, _drop_empty_deadlines,
                       _drop_meta, _ground, _post, _speakers)


def test_correction_prompt_carries_roster():
    p = build_correction_prompt("我: 找史考特", roster=["Scott", "David"], lang="zh-TW")
    assert "Scott" in p and "David" in p and "校正" in p


def test_correct_transcript_uses_backend_output():
    backend = lambda p: "我: 找 Scott 處理"  # LLM fixed 史考特 -> Scott
    out = correct_transcript("我: 找史考特", roster=["Scott"], lang="zh-TW",
                             backend=backend, max_chars=1000)
    assert out == "我: 找 Scott 處理"


def test_correct_transcript_falls_back_on_backend_error():
    def backend(p):
        raise RuntimeError("model down")
    text = "我: 原文保留"
    # best-effort: a failed correction must not lose the transcript
    assert correct_transcript(text, roster=[], lang="zh-TW", backend=backend) == text


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


def test_ground_scrubs_owner_in_parens_form():
    # meeting 187: the 3B summarizer wrote its own name as an action-item owner.
    g = _ground("2. **測試 DV 環境**: 待辦行動 - Qwen (負責)", "Pei: 我會測試 DV 環境")
    assert "Qwen" not in g and "未指定 (負責)" in g


def test_ground_keeps_paren_owner_actually_said():
    out = "1. 產出 Prompt: 待辦行動 - Nancy (負責)"
    assert _ground(out, "Nancy: 今天會產出 prompt") == out


def test_speakers_from_transcript():
    assert _speakers("Pei: 今天早上\nNancy: 好\nPei: 以上") == {"Pei", "Nancy"}


def test_prompt_enumerates_speaker_roster():
    p = build_prompt("Pei: 我會測試\nNancy: 好", kind="minutes", lang="zh-TW")
    assert "Nancy、Pei" in p and "本次會議的說話者" in p


def test_empty_deadline_field_dropped():
    # user report: "期限：未定" on every line is noise, not information
    src = ("1. **等待 Access Token 批準**: 待辦行動 - Hank (負責)；期限：未定\n"
           "2. 測試延遲 (期限: 未定)\n"
           "3. 上線 期限：2026-08-01")
    out = _drop_empty_deadlines(src)
    assert "未定" not in out
    assert out.splitlines()[0].endswith("(負責)")
    assert out.splitlines()[1] == "2. 測試延遲"
    assert "期限：2026-08-01" in out          # a real deadline survives


def test_empty_deadline_bullet_line_removed():
    assert _drop_empty_deadlines("- 交付報告\n- 期限：未定\n- 下一步") == "- 交付報告\n- 下一步"


def test_meta_disclaimer_stripped():
    src = "**會議重點:**\n- 複製靶站\n\n以上內容均根據提供的逐字稿整理，不涉及杜撰任何人物或資訊。"
    assert _drop_meta(src) == "**會議重點:**\n- 複製靶站"


def test_empty_roll_call_bullet_dropped():
    src = "- Pei: 複製靶站。\n- Hank: 未指定（無實質發言）。\n- Nancy: 測延遲。"
    assert _drop_meta(src) == "- Pei: 複製靶站。\n- Nancy: 測延遲。"
    # a real item that merely starts with 無 stays
    keep = "- Chester: 無法在今天完成訓練"
    assert _drop_meta(keep) == keep


def test_post_applies_deadline_and_meta_cleanup_only_to_summaries():
    src = "- 測試 (期限: 未定)"
    assert _post(src, "zh-TW") == "- 測試"
    assert _post(src, "zh-TW", dedup=False) == src   # transcript correction untouched


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
