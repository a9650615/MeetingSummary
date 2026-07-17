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
