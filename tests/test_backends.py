import backends


def test_route_qwen3_vs_whisper():
    assert backends.route("Qwen/Qwen3-ASR-0.6B") == "qwen3"
    assert backends.route("mlx-community/Qwen3-ASR-1.7B-8bit") == "qwen3mlx"  # MLX-native (mlx-audio)
    assert backends.route("qwen3-asr-0.6b-q4-k-m") == "qwen3cpp"  # femelo GGUF sidecar
    assert backends.route("qwen3-asr-1.7b") == "chatllm"          # chatllm.cpp 1.7B
    assert backends.route("Qwen/Qwen3-ASR-1.7B") == "qwen3"       # transformers, not chatllm
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
