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
