import backends


def test_route_qwen3_vs_whisper():
    assert backends.route("Qwen/Qwen3-ASR-0.6B") == "qwen3"
    assert backends.route("mlx-community/qwen3-asr-something") == "qwen3"
    assert backends.route("mlx-community/whisper-large-v3-turbo") == "whisper"
    assert backends.route("mlx-community/whisper-small-mlx") == "whisper"


def test_live_manager_set_model_rebuilds_chain():
    made = []
    mgr = backends.LiveModelManager(
        make=lambda m: made.append(m) or (lambda b: [{"start": 0, "end": 1, "text": "x"}]),
        model="turbo", fallback=["small", "base"], rtf_budget=0.8)
    assert mgr.requested == "turbo" and mgr.current == "turbo"
    assert mgr.backend.models == ["turbo", "small", "base"]  # chain = model + fallback
    assert mgr(b"\x00" * 32000)[0]["text"] == "x"  # shim delegates to backend


def test_live_manager_hot_swap():
    mgr = backends.LiveModelManager(
        make=lambda m: (lambda b: [{"start": 0, "end": 1, "text": m}]),
        model="turbo", fallback=["small"])
    assert mgr.current == "turbo"
    mgr.set_model("small")  # hot swap, no restart
    assert mgr.requested == "small" and mgr.current == "small"
    assert mgr.backend.models == ["small"]  # fallback dedups the chosen
    assert mgr(b"\x00" * 32000)[0]["text"] == "small"
