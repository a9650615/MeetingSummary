import backends


def test_route_qwen3_vs_whisper():
    assert backends.route("Qwen/Qwen3-ASR-0.6B") == "qwen3"
    assert backends.route("mlx-community/Qwen3-ASR-1.7B-8bit") == "qwen3mlx"  # MLX-native (mlx-audio)
    assert backends.route("qwen3-asr-0.6b-q4-k-m") == "qwen3cpp"  # femelo GGUF sidecar
    assert backends.route("qwen3-asr-1.7b") == "chatllm"          # chatllm.cpp 1.7B
    assert backends.route("Qwen/Qwen3-ASR-1.7B") == "qwen3"       # transformers, not chatllm
    assert backends.route("ane-qwen3-0.6b") == "ane"          # ANE speech CLI
    assert backends.route("ane-qwen3-0.6b-hybrid") == "ane"
    assert backends._ANE_IDS["ane-qwen3-0.6b"] == ("qwen3-coreml-full", "0.6B")
    assert backends.route("mlx-community/whisper-large-v3-turbo") == "whisper"
    assert backends.route("mlx-community/whisper-small-mlx") == "whisper"


def test_stt_text_empty_is_not_repr():
    class STTOutput:
        def __init__(self, text):
            self.text = text
        def __repr__(self):
            return f"STTOutput(text={self.text!r}, segments=[...])"
    assert backends._stt_text(STTOutput("你好")) == "你好"
    assert backends._stt_text(STTOutput("")) == ""          # silence -> empty, NOT repr
    assert backends._stt_text(STTOutput("  hi ")) == "hi"
    assert backends._stt_text({"text": "嗨"}) == "嗨"
    assert backends._stt_text({"text": ""}) == ""
    assert "STTOutput(" not in backends._stt_text(STTOutput(""))


def test_qwen3_words_to_segments():
    words = [{"start": 0.0, "end": 0.5, "word": "今天"},
             {"start": 0.5, "end": 1.0, "word": "會議"},
             {"start": 1.0, "end": 1.2, "word": "  "}]  # blank dropped
    segs = backends.qwen3_words_to_segments(words, "今天會議")
    assert [s["text"] for s in segs] == ["今天", "會議"]
    assert segs[1]["start"] == 0.5
    # no words -> single fallback segment from text
    assert backends.qwen3_words_to_segments([], "整句") == [
        {"start": 0.0, "end": 0.0, "text": "整句"}]
    assert backends.qwen3_words_to_segments([], "") == []


def test_live_manager_set_model_rebuilds_chain():
    made = []
    mgr = backends.LiveModelManager(
        make=lambda m, lang=None: made.append(m) or (lambda b: [{"start": 0, "end": 1, "text": "x"}]),
        model="turbo", fallback=["small", "base"], rtf_budget=0.8)
    assert mgr.requested == "turbo" and mgr.current == "turbo"
    assert mgr.backend.models == ["turbo", "small", "base"]  # chain = model + fallback
    assert mgr(b"\x00" * 32000)[0]["text"] == "x"  # shim delegates to backend


def test_live_manager_hot_swap():
    mgr = backends.LiveModelManager(
        make=lambda m, lang=None: (lambda b: [{"start": 0, "end": 1, "text": m}]),
        model="turbo", fallback=["small"])
    assert mgr.current == "turbo"
    mgr.set_model("small")  # hot swap, no restart
    assert mgr.requested == "small" and mgr.current == "small"
    assert mgr.backend.models == ["small"]  # fallback dedups the chosen
    assert mgr(b"\x00" * 32000)[0]["text"] == "small"


def test_denoise_file_graceful_without_speech(monkeypatch, tmp_path):
    # no `speech` CLI -> returns src unchanged (never blocks transcription)
    import shutil as _sh
    monkeypatch.setattr(_sh, "which", lambda _x: None)
    monkeypatch.setattr("os.path.exists", lambda _p: False)
    src = str(tmp_path / "a.pcm")
    assert backends.denoise_file(src, raw_pcm=True) == src




def test_clean_firered_strips_sil_and_special_tokens():
    from backends import _clean_firered
    assert _clean_firered("我<sil><sil><sil>好") == "我好"
    assert _clean_firered("<sil><sil>") == ""          # silence-only -> empty (line dropped)
    assert _clean_firered("要打 BP<sil>的時候") == "要打 BP的時候"
    assert _clean_firered("hello <sil> world") == "hello world"
    assert _clean_firered(None) == ""


def test_route_groq():
    assert backends.route("groq-whisper-large-v3") == "groq"
    assert backends.route("groq-whisper-large-v3-turbo") == "groq"


def test_route_breeze():
    assert backends.route("MediaTek-Research/Breeze-ASR-25") == "breeze"


def test_breeze_lang_locked_never_auto():
    # No None/"" entry: auto-detect is exactly the failure mode this model
    # exists to avoid, so every caller-passed language (including unset) must
    # resolve to a real forced language, never fall through to whisper auto-detect.
    assert backends._BREEZE_LANG.get("") == "chinese"
    assert backends._BREEZE_LANG.get(None or "") == "chinese"
    assert backends._BREEZE_LANG.get("zh") == "chinese"
    assert backends._BREEZE_LANG.get("en") == "english"


def test_groq_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import pytest
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        backends.groq_backend("groq-whisper-large-v3")


class _FakeResp:
    def __init__(self, status_code=200, json_data=None, headers=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._json


def test_groq_backend_maps_segments_and_filters_hallucinations(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()
    seen = {}

    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        seen["url"] = url
        seen["data"] = data
        return _FakeResp(json_data={"segments": [
            {"start": 0.0, "end": 1.2, "text": "你好",
             "no_speech_prob": 0.1, "compression_ratio": 1.5, "avg_logprob": -0.3},
            {"start": 1.2, "end": 2.0, "text": "靜音噪音",
             "no_speech_prob": 0.9, "compression_ratio": 1.2, "avg_logprob": -0.1},
        ]})

    monkeypatch.setattr(httpx, "post", fake_post)
    run = backends.groq_backend("groq-whisper-large-v3")
    out = run(b"\x00\x01" * 4000)  # 8000 bytes, > 3200 floor
    assert out == [{"start": 0.0, "end": 1.2, "text": "你好"}]
    assert seen["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert seen["data"]["model"] == "whisper-large-v3"


def test_groq_backend_skips_tiny_audio_without_calling_api(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()

    def fail_post(*a, **k):
        raise AssertionError("should not call the API for tiny audio")

    monkeypatch.setattr(httpx, "post", fail_post)
    run = backends.groq_backend("groq-whisper-large-v3")
    assert run(b"\x00\x01") == []  # 2 bytes, well under the 3200 floor


def test_groq_backend_non_200_returns_empty_not_raise(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: _FakeResp(status_code=429, text="rate limited"))
    run = backends.groq_backend("groq-whisper-large-v3")
    assert run(b"\x00\x01" * 4000) == []


def test_groq_backend_network_error_returns_empty_not_raise(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()

    def raise_timeout(*a, **k):
        raise httpx.ConnectTimeout("boom")

    monkeypatch.setattr(httpx, "post", raise_timeout)
    run = backends.groq_backend("groq-whisper-large-v3")
    assert run(b"\x00\x01" * 4000) == []


def test_groq_rpm_guard_skips_past_the_safe_margin(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()
    calls = {"n": 0}

    def counting_post(*a, **k):
        calls["n"] += 1
        return _FakeResp(json_data={"segments": []})

    monkeypatch.setattr(httpx, "post", counting_post)
    run = backends.groq_backend("groq-whisper-large-v3")
    for _ in range(25):  # well past _GROQ_RPM_SAFE (18)
        run(b"\x00\x01" * 4000)
    assert calls["n"] == backends._GROQ_RPM_SAFE


def test_make_backend_dispatches_groq(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(json_data={"segments": [
        {"start": 0.0, "end": 1.0, "text": "哈囉",
         "no_speech_prob": 0.1, "compression_ratio": 1.5, "avg_logprob": -0.3},
    ]}))
    run = backends.make_backend("groq-whisper-large-v3-turbo")
    assert callable(run)
    out = run(b"\x00\x01" * 4000)
    assert out == [{"start": 0.0, "end": 1.0, "text": "哈囉"}]


def test_groq_backend_tolerates_segment_missing_fields(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(json_data={"segments": [
        {"no_speech_prob": 0.1, "compression_ratio": 1.5, "avg_logprob": -0.3},  # missing start/end/text
    ]}))
    run = backends.groq_backend("groq-whisper-large-v3")
    assert run(b"\x00\x01" * 4000) == [{"start": 0.0, "end": 0.0, "text": ""}]


def test_groq_can_call_wait_false_skips_when_throttled(monkeypatch):
    # Existing live behaviour must be unchanged: wait=False (the default)
    # returns False immediately once the RPM budget is exhausted.
    import time
    backends._GROQ_STATE["calls"].clear()
    model = "wait-false-model"
    for _ in range(backends._GROQ_RPM_SAFE):
        assert backends._groq_can_call(model) is True
    assert backends._groq_can_call(model) is False
    assert backends._groq_can_call(model, wait=False) is False


def test_groq_can_call_wait_true_blocks_until_slot_frees(monkeypatch):
    # Batch must never silently drop a window: wait=True blocks (sleeps) until
    # a slot frees up instead of returning False. Pin the deque + clock entirely
    # (mixing real and fake monotonic times would desync the RPM window), and
    # no-op time.sleep so the "blocking" doesn't actually slow the test down.
    import time
    from collections import deque
    backends._GROQ_STATE["calls"].clear()
    model = "wait-true-model"
    # As if _GROQ_RPM_SAFE calls all happened at t=0 — currently throttled.
    backends._GROQ_STATE["calls"][model] = deque([0.0] * backends._GROQ_RPM_SAFE)
    ticks = iter([0.0, 61.0, 61.0, 61.0])  # first check: still throttled; then 61s later: free
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    assert backends._groq_can_call(model, wait=True) is True  # blocked, then succeeded
    assert slept  # it actually went through the sleep path, not an immediate return


def test_groq_backend_wait_kwarg_forwarded_to_rpm_guard(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()
    seen = {}

    def fake_can_call(model, wait=False):
        seen["wait"] = wait
        return True

    monkeypatch.setattr(backends, "_groq_can_call", fake_can_call)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(json_data={"segments": []}))
    run = backends.groq_backend("groq-whisper-large-v3", wait=True)
    run(b"\x00\x01" * 4000)
    assert seen["wait"] is True


def test_make_backend_forwards_wait_only_to_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    seen = {}

    def fake_groq_backend(model, language=None, wait=False):
        seen["wait"] = wait
        return lambda audio: []

    monkeypatch.setattr(backends, "groq_backend", fake_groq_backend)
    backends.make_backend("groq-whisper-large-v3", wait=True)
    assert seen["wait"] is True


def test_groq_warn_quota_checks_audio_seconds_header(monkeypatch, capsys):
    resp = _FakeResp(headers={"x-ratelimit-remaining-audio-seconds": "10"})
    backends._groq_warn_quota(resp)
    err = capsys.readouterr().err
    assert "audio-seconds" in err


def test_groq_backend_tolerates_non_dict_json_body(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(json_data=[]))
    run = backends.groq_backend("groq-whisper-large-v3")
    assert run(b"\x00\x01" * 4000) == []


def test_groq_backend_tolerates_segments_not_a_list(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(json_data={"segments": None}))
    run = backends.groq_backend("groq-whisper-large-v3")
    assert run(b"\x00\x01" * 4000) == []


def test_groq_backend_warns_on_unrecognized_model_id(monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends.groq_backend("groq-not-a-real-model")
    err = capsys.readouterr().err
    assert "unrecognized model id" in err
    assert "groq-not-a-real-model" in err


def test_groq_backend_known_model_id_warns_nothing(monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends.groq_backend("groq-whisper-large-v3")
    err = capsys.readouterr().err
    assert "unrecognized model id" not in err
