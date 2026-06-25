from asr import _collapse_repeats, _degenerate, transcribe


def test_degenerate_loop_detected():
    assert _degenerate("喬" * 50)
    assert _degenerate("ya ya ya ya ya")        # spaces stripped -> few unique chars
    assert not _degenerate("今天我們討論專案進度與下一步")


def test_collapse_keeps_real_doubling():
    assert _collapse_repeats("哈哈哈哈哈哈") == "哈哈哈"        # single char capped to 3
    assert _collapse_repeats("好的好的好的好的") == "好的好的"   # short token capped to 2
    assert _collapse_repeats("這是正常的句子") == "這是正常的句子"


def test_transcribe_drops_loop_segment():
    segs = [{"start": 0, "end": 1, "text": "喬" * 40},
            {"start": 1, "end": 2, "text": "正常內容可以保留"}]
    out = transcribe("x", profile="accurate", track="system", backend=lambda p: segs)
    assert len(out) == 1 and out[0]["text"] == "正常內容可以保留"


def test_transcribe_drops_whisper_silence_hallucinations():
    segs = [{"start": 0, "end": 1, "text": "優優獨播劇場——YoYo Television Series Exclusive"},
            {"start": 1, "end": 2, "text": "謝謝觀看"},      # filler hallucination
            {"start": 2, "end": 3, "text": "今天討論專案進度與時程"}]  # real -> kept
    out = transcribe("x", profile="accurate", track="system", backend=lambda p: segs)
    assert [o["text"] for o in out] == ["今天討論專案進度與時程"]
