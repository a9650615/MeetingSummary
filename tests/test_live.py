from live import LiveSession


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
