import modelprofile as mp


def test_16gb_uses_q4_for_efficiency():
    # q4 (half-RAM) live/interim/fallback to avoid OOM with many resident models.
    rec = mp.recommend({"ram_gb": 16, "arch": "arm64"}, lang="zh-TW")
    assert rec["live"] == "mlx-community/whisper-small-mlx-q4"
    assert "q4" in rec["interim"] and all("q4" in m for m in rec["fallback"])
    assert "belle" not in rec["live"] and "belle" not in rec["accurate"]
    assert "7B" not in rec["summary"]  # 7B was a big OOM driver


def test_smaller_ram_picks_smaller_models():
    big = mp.recommend({"ram_gb": 16}, lang="zh-TW")
    mid = mp.recommend({"ram_gb": 8}, lang="zh-TW")
    small = mp.recommend({"ram_gb": 4}, lang="zh-TW")
    # 16GB + 8GB both default to small-q4; 4GB drops to base-q4
    assert big["live"] == "mlx-community/whisper-small-mlx-q4"
    assert mid["live"] == "mlx-community/whisper-small-mlx-q4"
    assert small["live"] == "mlx-community/whisper-base-mlx-q4"


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


def test_probe_skips_models_that_error():
    # A model that throws on load/run (e.g. belle on an unsupported runtime) is
    # skipped, not chosen — so a broken model never silently kills finals.
    ticks = iter([0, 0, 0.3])

    def run(m):
        if m == "broken":
            raise RuntimeError("incompatible")

    chosen = mp.probe_models(["broken", "good"], audio_seconds=3.0, run=run,
                             clock=lambda: next(ticks), target_rtf=0.5)
    assert chosen == "good"


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
