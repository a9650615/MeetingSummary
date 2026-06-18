import modelprofile as mp


def test_16gb_zh_uses_belle():
    rec = mp.recommend({"ram_gb": 16, "arch": "arm64"}, lang="zh-TW")
    assert "belle" in rec["live"] and "belle" in rec["accurate"]
    assert isinstance(rec["fallback"], list) and rec["fallback"]
    assert rec["interim"] and rec["summary"]


def test_16gb_english_uses_vanilla_turbo():
    rec = mp.recommend({"ram_gb": 16}, lang="en")
    assert rec["live"] == "mlx-community/whisper-large-v3-turbo"
    assert "belle" not in rec["accurate"]


def test_smaller_ram_picks_smaller_models():
    big = mp.recommend({"ram_gb": 16}, lang="zh-TW")
    mid = mp.recommend({"ram_gb": 8}, lang="zh-TW")
    small = mp.recommend({"ram_gb": 4}, lang="zh-TW")
    # live model shrinks as RAM drops
    assert big["live"] != mid["live"] != small["live"]
    assert mid["live"] == "mlx-community/whisper-small-mlx"
    assert small["live"] == "mlx-community/whisper-base-mlx"


def test_detect_hardware_shape():
    hw = mp.detect_hardware()
    assert hw["ram_gb"] > 0 and hw["cores"] >= 1 and isinstance(hw["arch"], str)
    assert isinstance(hw["chip"], str)


def test_probe_picks_best_within_rtf_budget():
    # 3 s clip. A: 3 s -> rtf 1.0 (too slow). B: 0.9 s -> rtf 0.3 (ok).
    ticks = iter([0, 3, 0, 0.9])
    chosen = mp.probe_models(["A", "B"], audio_seconds=3.0, run=lambda m: None,
                             clock=lambda: next(ticks), target_rtf=0.5)
    assert chosen == "B"


def test_probe_falls_back_to_smallest_if_all_slow():
    ticks = iter([0, 3, 0, 3])
    chosen = mp.probe_models(["A", "B"], audio_seconds=3.0, run=lambda m: None,
                             clock=lambda: next(ticks), target_rtf=0.5)
    assert chosen == "B"  # smallest/last


def test_chosen_model_roundtrip(tmp_path):
    p = str(tmp_path / "model_profile.json")
    assert mp.load_chosen(p) is None
    mp.save_chosen(p, "mlx-community/whisper-small-mlx")
    assert mp.load_chosen(p) == "mlx-community/whisper-small-mlx"
