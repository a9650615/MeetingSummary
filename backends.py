"""Modular ASR backend registry + hot-reloadable live model.

route(model) decides which engine a model id belongs to. make_* build the actual
callables (lazy — weights load on first use, so hot-swapping is instant). The
LiveModelManager lets the running server switch the live model with no restart:
TwoPassSession calls `manager.backend` through a thin shim, so set_model() takes
effect on the next utterance — even mid-session."""
from live import AdaptiveBackend, mlx_whisper_live_backend


def route(model):
    """Engine for a model id. Qwen3-ASR runs offline (transformers) for the
    accurate/batch path; whisper(-mlx) is the Apple-native live engine."""
    m = model.lower()
    if "qwen3-asr" in m:
        return "qwen3"
    return "whisper"


def make_live_backend(model):
    """Live = whisper-MLX only (Qwen3-ASR live streaming needs vLLM/CUDA, which
    Apple Silicon can't run). Returns a callable(pcm_bytes) -> segments."""
    return mlx_whisper_live_backend(model)


def make_batch_backend(model):
    """Batch/accurate: route to Qwen3-ASR (best zh, offline transformers) or
    whisper-MLX. Returns a callable(audio_path) -> segments."""
    if route(model) == "qwen3":
        return qwen3_batch_backend(model)
    import asr
    return asr.mlx_whisper_backend(model)


def qwen3_batch_backend(model="Qwen/Qwen3-ASR-0.6B"):
    """Offline Qwen3-ASR via the qwen-asr transformers backend. Apple Silicon:
    torch on MPS/CPU (slow, not realtime) — for post-meeting accuracy only.
    Lazy import so the heavy torch/transformers stack isn't required otherwise."""
    from qwen_asr import Qwen3ASRModel  # noqa: PLC0415

    asr_model = Qwen3ASRModel.from_pretrained(model)

    def _run(audio_path):
        out = asr_model.transcribe(str(audio_path), language=None)  # auto lang
        text = " ".join(t.text for t in out).strip()
        return [{"start": 0.0, "end": 0.0, "text": text}] if text else []

    return _run


class LiveModelManager:
    """Holds the current live AdaptiveBackend and rebuilds it on set_model()."""

    def __init__(self, make, model, fallback=(), rtf_budget=0.8, on_change=None):
        self._make = make
        self._fallback = list(fallback)
        self._rtf_budget = rtf_budget
        self._on_change = on_change
        self.backend = None
        self.requested = None
        self.set_model(model)

    def set_model(self, model):
        chain = [model] + [m for m in self._fallback if m != model]
        self.backend = AdaptiveBackend(
            [self._make(m) for m in chain], chain,
            rtf_budget=self._rtf_budget, on_change=self._on_change)
        self.requested = model
        return model

    @property
    def current(self):
        return self.backend.current_model

    def __call__(self, pcm_bytes):  # shim so a captured ref always hits the live one
        return self.backend(pcm_bytes)
