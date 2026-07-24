"""Startup-latency guard: the silero VAD onnx session and the sherpa speaker
embedding extractor are loaded ONCE and reused across tracks/sessions, so the
/ws start handshake no longer pays a per-session model load. If someone drops
the cache, these fail (constructor called more than once)."""
import sys
import types


def test_silero_session_loaded_once(monkeypatch):
    import live

    calls = {"n": 0}

    class _FakeSess:
        def run(self, *a, **k):  # never actually run in this test
            return None

    def _fake_ort_session(path, providers=None):
        calls["n"] += 1
        return _FakeSess()

    fake_ort = types.SimpleNamespace(InferenceSession=_fake_ort_session)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(live, "_SILERO_SESS", {}, raising=True)

    a = live.SileroVad("models/silero_vad_v4.onnx")
    b = live.SileroVad("models/silero_vad_v4.onnx")  # second track/session
    assert a._s is b._s          # same shared session object
    assert calls["n"] == 1        # constructed only once


def test_emb_extractor_cached_per_key(monkeypatch):
    import diarize

    calls = {"n": 0}

    class _FakeExt:
        pass

    def _fake_extractor(cfg):
        calls["n"] += 1
        return _FakeExt()

    fake_sherpa = types.SimpleNamespace(
        SpeakerEmbeddingExtractor=_fake_extractor,
        SpeakerEmbeddingExtractorConfig=lambda **k: object())
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa)
    monkeypatch.setattr(diarize, "_resolve_models", lambda *a, **k: (None, "emb.onnx"))
    monkeypatch.setattr(diarize, "_EMB_EXT", {}, raising=True)

    r1 = diarize.embedding_extractor(provider="coreml")
    r2 = diarize.embedding_extractor(provider="coreml")  # second session, same key
    assert callable(r1) and callable(r2)
    assert calls["n"] == 1        # extractor built only once for the key

    diarize.embedding_extractor(provider="cpu")  # different provider -> new build
    assert calls["n"] == 2
