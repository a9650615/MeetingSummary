"""One factory, one engine per model id.

There used to be make_live_backend and make_batch_backend, and for several ids
they returned DIFFERENT implementations of the same model — which is why
re-transcribing could come back worse than the live captions. make_backend
replaces both: the model id picks the engine, and the callable takes either PCM
bytes (live) or an audio path (re-transcribe), never a different code path.
"""
import backends


def test_only_one_factory_remains():
    assert hasattr(backends, "make_backend")
    assert not hasattr(backends, "make_live_backend")
    assert not hasattr(backends, "make_batch_backend")


def test_pcm_bytes_accepts_bytes_and_a_pcm_path(tmp_path):
    raw = b"\x01\x02\x03\x04"
    assert backends._pcm_bytes(raw) == raw
    p = tmp_path / "a.pcm"
    p.write_bytes(raw)
    assert backends._pcm_bytes(str(p)) == raw


def test_takes_bytes_feeds_a_path_through_as_pcm(tmp_path):
    seen = []
    wrapped = backends._takes_bytes(lambda b: seen.append(b) or [])
    p = tmp_path / "a.pcm"
    p.write_bytes(b"\x09\x09")
    wrapped(str(p))
    wrapped(b"\x09\x09")
    assert seen == [b"\x09\x09", b"\x09\x09"]  # engine only ever sees PCM bytes


def test_takes_path_writes_bytes_to_a_temp_and_cleans_up():
    seen = {}

    def _engine(path):
        with open(path, "rb") as f:
            seen["data"] = f.read()
        seen["path"] = path
        return []

    backends._takes_path(_engine)(b"\xaa\xbb")
    assert seen["data"] == b"\xaa\xbb"
    import os
    assert not os.path.exists(seen["path"])   # temp removed


def test_takes_path_passes_a_path_straight_through():
    got = []
    backends._takes_path(lambda p: got.append(p) or [])("/tmp/x.wav")
    assert got == ["/tmp/x.wav"]


def test_ane_ids_route_to_one_engine(monkeypatch):
    # Both ANE ids used to fork: hybrid -> in-repo helper, plain -> homebrew CLI.
    # Now the helper serves both whenever it is built.
    calls = []
    monkeypatch.setattr(backends, "ane_helper_bin", lambda: "/fake/helper")
    monkeypatch.setattr(backends, "ane_live_backend", lambda: calls.append("helper"))
    monkeypatch.setattr(backends, "ane_speech_backend",
                        lambda *a, **k: calls.append("cli"))
    backends.make_backend("ane-qwen3-0.6b-hybrid")
    backends.make_backend("ane-qwen3-0.6b")
    assert calls == ["helper", "helper"]


def test_ane_falls_back_to_the_cli_when_the_helper_is_not_built(monkeypatch):
    calls = []
    monkeypatch.setattr(backends, "ane_helper_bin", lambda: None)
    monkeypatch.setattr(backends, "ane_live_backend", lambda: calls.append("helper"))
    monkeypatch.setattr(backends, "ane_speech_backend",
                        lambda *a, **k: calls.append("cli"))
    backends.make_backend("ane-qwen3-0.6b-hybrid")
    assert calls == ["cli"]
