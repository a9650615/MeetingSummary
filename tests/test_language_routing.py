from backends import _honor_language, route


def test_forced_language_reroutes_femelo_to_whisper():
    femelo = "qwen3-asr-0.6b-q4-k-m"
    assert route(femelo) == "qwen3cpp"
    # auto-detect (no language) keeps the fast femelo path
    assert _honor_language(femelo, None) == femelo
    assert _honor_language(femelo, "") == femelo
    # a FORCED language -> whisper, which actually honors it (femelo ignores it)
    assert route(_honor_language(femelo, "zh")) == "whisper"


def test_forced_language_leaves_other_backends_alone():
    for m in ("mlx-community/whisper-small-mlx-q4", "Qwen/Qwen3-ASR-0.6B",
              "qwen3-asr-1.7b"):
        assert _honor_language(m, "zh") == m
