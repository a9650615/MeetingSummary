import numpy as np

from live import (AdaptiveBackend, FixedWindowChunker, LiveSession,
                  TwoPassSession, VadChunker, preprocess)


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


def test_twopass_finalizes_on_silence():
    s = TwoPassSession(backend=lambda a: [{"start": 0, "end": 1, "text": "完整句"}],
                       frame_ms=30, silence_ms=90, interim_s=100)
    finals = [e for e in s.feed(tone(300) + silence(150)) if e["kind"] == "final"]
    assert len(finals) == 1
    assert finals[0]["text"] == "完整句" and finals[0]["start_ms"] == 0
    # end_ms present (consumer stores it) and after start
    assert finals[0]["end_ms"] > finals[0]["start_ms"]


def _step_clock(step):
    # monotonic-ish fake clock advancing `step` seconds each call
    t = [0.0]
    def c():
        t[0] += step
        return t[0]
    return c


def test_finalize_warns_when_behind_realtime(capsys):
    # Stall diagnosis: when final ASR (+diar) takes longer than the utterance's
    # own realtime, _finalize logs a 'live SLOW' line splitting asr vs diar.
    s = TwoPassSession(backend=lambda a: [{"start": 0, "end": 1, "text": "句"}],
                       frame_ms=30, silence_ms=90, min_speech_ms=30, interim_s=100,
                       clock=_step_clock(5.0))  # 5s "compute" per clock delta
    s.feed(tone(300) + silence(150))            # ~0.3s audio << 5s asr
    err = capsys.readouterr().err
    assert "live SLOW" in err and "asr 5.0s" in err


def test_finalize_silent_when_realtime(capsys):
    # Fast path (compute < realtime) logs nothing — no spam on the common case.
    s = TwoPassSession(backend=lambda a: [{"start": 0, "end": 1, "text": "句"}],
                       frame_ms=30, silence_ms=90, min_speech_ms=30, interim_s=100,
                       clock=_step_clock(0.0))  # 0s compute
    s.feed(tone(300) + silence(150))
    assert "live SLOW" not in capsys.readouterr().err


def test_finalize_skips_diarization_when_behind():
    # want_diarize=False (consume sets this under backlog) must NOT call speaker_fn
    # -> line falls back to side label; post-meeting /diarize relabels later.
    calls = []
    s = TwoPassSession(backend=lambda a: [{"start": 0, "end": 1, "text": "句"}],
                       frame_ms=30, silence_ms=90, min_speech_ms=30, interim_s=100,
                       speaker_fn=lambda a: calls.append(1) or "Ray")
    finals = [e for e in s.feed(tone(300) + silence(150), want_diarize=False)
              if e["kind"] == "final"]
    assert finals and "speaker" not in finals[0]   # unlabeled
    assert calls == []                              # embedding skipped


def test_finalize_diarizes_by_default():
    calls = []
    s = TwoPassSession(backend=lambda a: [{"start": 0, "end": 1, "text": "句"}],
                       frame_ms=30, silence_ms=90, min_speech_ms=30, interim_s=100,
                       speaker_fn=lambda a: calls.append(1) or "Ray")
    finals = [e for e in s.feed(tone(300) + silence(150)) if e["kind"] == "final"]
    assert finals and finals[0].get("speaker") == "Ray" and calls == [1]


def test_twopass_drops_silence_hallucination_by_char_rate():
    # A fluent sentence far too long for the brief speech present = a silence
    # hallucination (whisper confabulating over near-silence, the 對方-track report).
    long_text = "This is meeting tonight she has a great job today that is it well"
    s = TwoPassSession(backend=lambda a: [{"start": 0, "end": 1, "text": long_text}],
                       frame_ms=30, silence_ms=90, min_speech_ms=90, interim_s=100)
    finals = [e for e in s.feed(tone(300) + silence(150)) if e["kind"] == "final"]
    assert finals == []  # char/sec gate zeroed it


def test_twopass_keeps_short_text_matching_speech():
    # A short line plausibly spoken in the window survives the char/sec gate.
    s = TwoPassSession(backend=lambda a: [{"start": 0, "end": 1, "text": "好的"}],
                       frame_ms=30, silence_ms=90, min_speech_ms=90, interim_s=100)
    finals = [e for e in s.feed(tone(300) + silence(150)) if e["kind"] == "final"]
    assert len(finals) == 1 and finals[0]["text"] == "好的"


def test_twopass_emits_interim_while_speaking():
    s = TwoPassSession(backend=lambda a: [{"start": 0, "end": 1, "text": "定"}],
                       interim_backend=lambda a: [{"start": 0, "end": 1, "text": "暫"}],
                       frame_ms=30, silence_ms=100000, interim_s=0.09)
    kinds = [e["kind"] for e in s.feed(tone(300))]  # speaking, no pause
    assert "interim" in kinds and "final" not in kinds


def test_interim_cadence_adapts_to_compute_load():
    # next interim interval = compute_time / duty, clamped — so a slow model backs
    # off and a fast one runs more often, holding ASR duty cycle ~= target.
    s = TwoPassSession(backend=lambda a: [], interim_backend=lambda a: [],
                       interim_s=0.6, interim_duty=0.75,
                       interim_min_s=0.4, interim_max_s=3.0)
    base = s._interim_dyn
    s._adapt_interim(0.2)          # first call = cold-load warmup -> ignored
    assert s._interim_dyn == base
    s._adapt_interim(0.2)          # 0.2/0.75=0.27s -> below min -> clamp up
    assert s._interim_dyn == int(0.4 * 16000) * 2
    s._adapt_interim(1.5)          # 1.5/0.75=2.0s -> in range
    assert s._interim_dyn == int(2.0 * 16000) * 2
    fast = s._interim_dyn
    s._adapt_interim(5.0)          # 5/0.75=6.7s -> clamp to max (slow model backs off)
    assert s._interim_dyn == int(3.0 * 16000) * 2 > fast


def test_twopass_skips_interim_when_not_wanted():
    # want_interim=False (we're behind) -> no interim ASR, finals still happen.
    calls = []
    s = TwoPassSession(
        backend=lambda a: calls.append("final") or [{"start": 0, "end": 1, "text": "句"}],
        interim_backend=lambda a: calls.append("interim") or [{"start": 0, "end": 1, "text": "暫"}],
        frame_ms=30, silence_ms=90, interim_s=0.03)
    kinds = [e["kind"] for e in s.feed(tone(300) + silence(150), want_interim=False)]
    assert "interim" not in kinds and "final" in kinds
    assert "interim" not in calls  # interim model never called


def test_speech_fn_overrides_energy_vad():
    # an injected speech_fn (e.g. silero) replaces the energy decision per frame
    calls = {"n": 0}
    s = TwoPassSession(backend=lambda a: [{"start": 0, "end": 1, "text": "x"}],
                       frame_ms=30, silence_ms=90,
                       speech_fn=lambda fb: calls.__setitem__("n", calls["n"] + 1) or True)
    s.feed(tone(120))
    assert calls["n"] > 0 and s._has_speech


def test_discarded_blip_still_advances_timeline():
    # A too-short blip is dropped from ASR, but its audio is in the saved file,
    # so the committed-byte clock (= timestamp basis) must still advance — else
    # later timestamps drift ahead of the audio (paragraph-mode bug).
    s = TwoPassSession(backend=lambda a: [], frame_ms=30, silence_ms=90,
                       min_speech_ms=100000, interim_s=100)  # nothing ever "enough"
    s.feed(tone(150) + silence(150))  # speech+silence, force-discarded as a blip
    assert s._committed_bytes > 0     # bytes counted despite discard


def test_twopass_no_work_on_silence():
    # Perf: pure silence must not call any backend.
    calls = []
    s = TwoPassSession(backend=lambda a: calls.append(1) or [],
                       interim_backend=lambda a: calls.append(1) or [],
                       frame_ms=30, silence_ms=90, interim_s=0.03)
    s.feed(silence(500))
    assert calls == []


def test_twopass_drops_repetition_final():
    s = TwoPassSession(backend=lambda a: [{"start": 0, "end": 1, "text": "a a a a a"}],
                       frame_ms=30, silence_ms=90)
    assert [e for e in s.feed(tone(300) + silence(150)) if e["kind"] == "final"] == []


def test_twopass_flush_finalizes_tail():
    s = TwoPassSession(backend=lambda a: [{"start": 0, "end": 1, "text": "尾"}])
    s.feed(tone(300))
    fin = s.flush()
    assert fin and fin[0]["kind"] == "final"


def test_twopass_drops_short_blip_without_calling_asr():
    # Cough/breath: brief energy, below min_speech -> no final, ASR NOT called.
    calls = []
    s = TwoPassSession(backend=lambda a: calls.append(1) or [],
                       frame_ms=30, silence_ms=90, min_speech_ms=250)
    finals = [e for e in s.feed(tone(60) + silence(150)) if e["kind"] == "final"]
    assert finals == [] and calls == []


def test_drops_youtube_outro_hallucination():
    from live import _is_hallucination
    assert _is_hallucination("Thank you for your attention guys.")
    assert _is_hallucination("謝謝觀看")
    assert _is_hallucination("Thanks for watching!")
    # a real sentence merely containing "thank you" is kept (phrase doesn't dominate)
    assert not _is_hallucination("thank you everyone, now let's review the Q3 budget numbers")
    assert not _is_hallucination("我們下週一上線新版本")


def test_twopass_final_drops_outro_hallucination():
    s = TwoPassSession(
        backend=lambda a: [{"start": 0, "end": 1, "text": "Thank you for watching."}],
        frame_ms=30, silence_ms=90)
    assert [e for e in s.feed(tone(300) + silence(150)) if e["kind"] == "final"] == []


def test_twopass_drops_filler_word():
    s = TwoPassSession(backend=lambda a: [{"start": 0, "end": 1, "text": "Yeah"}],
                       frame_ms=30, silence_ms=90)
    assert [e for e in s.feed(tone(300) + silence(150)) if e["kind"] == "final"] == []


def test_twopass_adaptive_detects_quiet_speech():
    # rms 250 is below the old fixed 500 threshold — adaptive VAD still catches it.
    s = TwoPassSession(backend=lambda a: [{"start": 0, "end": 1, "text": "輕聲細語"}],
                       frame_ms=30, silence_ms=90)
    finals = [e for e in s.feed(tone(300, amp=250) + silence(150))
              if e["kind"] == "final"]
    assert len(finals) == 1


def test_adaptive_downgrades_when_slow_then_stays():
    # call 1 = warmup (cold-load, ignored); calls 2 & 3 slow -> downgrade.
    ticks = iter([0, 2, 0, 2, 0, 2, 0, 0.1])
    slow = lambda b: [{"start": 0, "end": 1, "text": "x"}]
    fast = lambda b: [{"start": 0, "end": 1, "text": "y"}]
    ab = AdaptiveBackend([slow, fast], ["turbo", "small"], sample_rate=16000,
                         rtf_budget=0.8, patience=2, clock=lambda: next(ticks))
    win = b"\x00" * 32000  # 1.0 s
    ab(win)
    assert ab.current_model == "turbo"        # warmup ignored
    ab(win)
    assert ab.current_model == "turbo"        # 1 overrun
    ab(win)
    assert ab.current_model == "small"        # 2nd overrun -> downgrade
    assert "切換" in ab.pop_notice() and ab.pop_notice() is None
    assert ab(win)[0]["text"] == "y"          # now using the fast tier


def test_adaptive_on_change_fires_with_new_model():
    ticks = iter([0, 2, 0, 2, 0, 2])  # warmup + 2 slow -> downgrade
    seen = []
    ab = AdaptiveBackend([lambda b: [], lambda b: []], ["turbo", "small"],
                         rtf_budget=0.8, patience=2, clock=lambda: next(ticks),
                         on_change=seen.append)
    win = b"\x00" * 32000
    ab(win); ab(win); ab(win)
    assert seen == ["small"]


def test_adaptive_stays_on_fast_backend():
    ticks = iter([0, 0.1, 0, 0.1])
    fast = lambda b: [{"start": 0, "end": 1, "text": "ok"}]
    ab = AdaptiveBackend([fast, fast], ["a", "b"], rtf_budget=0.8, patience=2,
                         clock=lambda: next(ticks))
    win = b"\x00" * 32000
    ab(win); ab(win)
    assert ab.current_model == "a" and ab.pop_notice() is None


def test_preprocess_removes_dc_and_normalizes():
    sig = (np.sin(np.linspace(0, 12, 1000)) * 0.02 + 0.1).astype(np.float32)
    out = preprocess(sig)
    assert abs(float(out.mean())) < 1e-4              # DC offset removed
    assert abs(float(np.max(np.abs(out))) - 0.95) < 1e-3  # peak-normalized


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
