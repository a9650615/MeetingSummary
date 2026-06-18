import numpy as np

from live import FixedWindowChunker, LiveSession, VadChunker


def tone(ms, sr=16000, amp=8000):
    n = int(sr * ms / 1000)
    return (np.ones(n, dtype=np.int16) * amp).tobytes()


def silence(ms, sr=16000):
    n = int(sr * ms / 1000)
    return np.zeros(n, dtype=np.int16).tobytes()


def test_vad_cuts_at_silence_after_speech():
    ch = VadChunker(frame_ms=30, silence_ms=90, max_window_s=100)
    out = ch.feed(tone(300) + silence(150))  # speech then a 150 ms pause
    assert len(out) == 1
    # leftover (the empty post-cut frames) is silence-only -> flush emits nothing
    assert ch.flush() == []


def test_vad_force_cut_at_max_window():
    ch = VadChunker(frame_ms=30, silence_ms=100000, max_window_s=0.3)
    out = ch.feed(tone(500))  # 500 ms continuous speech, ceiling 300 ms
    assert len(out) >= 1
    assert len(out[0]) == int(0.3 * 16000) * 2  # cut exactly at the ceiling


def test_vad_no_cut_on_pure_silence():
    ch = VadChunker(frame_ms=30, silence_ms=90, max_window_s=100)
    assert ch.feed(silence(500)) == []   # never spoke -> nothing to emit


def test_vad_flush_emits_speech_tail():
    ch = VadChunker(frame_ms=30, silence_ms=90, max_window_s=100)
    ch.feed(tone(200))           # speech, no trailing silence yet
    assert len(ch.flush()) == 1  # tail flushed


def test_live_drops_repetition_hallucination():
    backend = lambda w: [{"start": 0, "end": 1,
                          "text": "segment segment segment segment segment"}]
    s = LiveSession(backend=backend, chunker=FixedWindowChunker(32000))
    assert s.feed(b"\x00" * 32000) == []


def test_vad_drops_nonspeech_force_cut_window():
    # Buffer fills to the ceiling with sub-threshold noise -> never "speech" ->
    # window is dropped, not transcribed (no hallucination feed).
    ch = VadChunker(frame_ms=30, silence_ms=100000, max_window_s=0.3,
                    rms_threshold=500)
    quiet = (np.ones(int(0.5 * 16000), dtype=np.int16) * 100).tobytes()  # rms 100 < 500
    assert ch.feed(quiet) == []


def test_live_session_uses_injected_vad_chunker():
    seen = []
    backend = lambda w: seen.append(len(w)) or [{"start": 0, "end": 1, "text": "ok"}]
    s = LiveSession(backend=backend, chunker=VadChunker(frame_ms=30, silence_ms=90))
    out = s.feed(tone(300) + silence(150))
    assert len(out) == 1 and out[0]["profile"] == "live"


def test_emits_one_segment_per_full_window_with_offset():
    calls = []

    def backend(window_bytes):
        calls.append(len(window_bytes))
        return [{"start": 0.0, "end": 1.0, "text": "片段"}]

    s = LiveSession(backend=backend, sample_rate=16000, window_s=1.0, track="mic")
    out = s.feed(b"\x00" * 32000)  # 1.0 s @ 16 kHz 16-bit = exactly one window
    assert len(out) == 1
    assert out[0]["start_ms"] == 0 and out[0]["profile"] == "live"

    out2 = s.feed(b"\x00" * 32000)  # second window -> +1000 ms offset
    assert out2[0]["start_ms"] == 1000
    assert calls == [32000, 32000]


def test_buffers_partial_until_window_full():
    s = LiveSession(backend=lambda w: [{"start": 0, "end": 0.5, "text": "x"}],
                    sample_rate=16000, window_s=1.0)
    assert s.feed(b"\x00" * 16000) == []   # half window -> nothing yet
    assert len(s.feed(b"\x00" * 16000)) == 1  # completes the window


def test_flush_transcribes_the_tail():
    s = LiveSession(backend=lambda w: [{"start": 0, "end": 0.3, "text": "尾巴"}],
                    sample_rate=16000, window_s=1.0)
    s.feed(b"\x00" * 8000)
    assert s.flush()[0]["text"] == "尾巴"


def test_drops_empty_segments():
    s = LiveSession(backend=lambda w: [{"start": 0, "end": 1, "text": "   "}],
                    sample_rate=16000, window_s=1.0)
    assert s.feed(b"\x00" * 32000) == []
