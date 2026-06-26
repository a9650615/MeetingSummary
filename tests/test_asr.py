from asr import transcribe


def fake_backend(audio_path):
    # Mimics whisper output: seconds + text, no track/profile knowledge.
    return [
        {"start": 0.0, "end": 1.0, "text": "你好"},
        {"start": 1.0, "end": 2.5, "text": "hello"},
    ]


def test_transcribe_tags_track_profile_and_converts_to_ms():
    segs = transcribe("clip.pcm", profile="accurate", track="mic",
                       backend=fake_backend)
    assert segs == [
        {"start_ms": 0, "end_ms": 1000, "text": "你好",
         "track": "mic", "profile": "accurate"},
        {"start_ms": 1000, "end_ms": 2500, "text": "hello",
         "track": "mic", "profile": "accurate"},
    ]


def test_transcribe_drops_empty_segments():
    backend = lambda p: [{"start": 0.0, "end": 0.5, "text": "  "},
                         {"start": 0.5, "end": 1.0, "text": "ok"}]
    segs = transcribe("c.pcm", profile="live", track="system", backend=backend)
    assert [s["text"] for s in segs] == ["ok"]


def test_sentence_split_breaks_long_blob_keeps_short():
    import asr
    # qwen3/ANE return one big blob per 30s window -> split into sentence lines
    segs = asr.transcribe("x", profile="accurate", track="mic",
        backend=lambda p: [{"start": 0.0, "end": 30.0,
                            "text": "第一句講完了。第二句換個話題。好，謝謝大家。"}])
    assert len(segs) == 3
    assert segs[0]["text"] == "第一句講完了。" and segs[0]["start_ms"] == 0
    assert segs[2]["end_ms"] == 30000 and segs[1]["start_ms"] > 0  # time distributed
    # whisper-style short sentence (no punctuation) stays one line
    one = asr.transcribe("x", profile="accurate", track="mic",
        backend=lambda p: [{"start": 0.0, "end": 2.0, "text": "今天會議討論預算"}])
    assert len(one) == 1
