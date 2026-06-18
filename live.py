"""Live transcription: buffer streamed PCM into fixed windows, transcribe each
with the turbo profile, emit segments with absolute timestamps (spec Phase 2).

Source-agnostic: feed it bytes from browser getUserMedia, the Swift helper, or
a file. Each window is transcribed independently (no cross-window context), so
boundary errors are expected — live is a preview, the accurate re-pass is the
trusted transcript (spec M1)."""


class LiveSession:
    def __init__(self, *, backend, sample_rate=16000, window_s=5.0, track="mic"):
        self.backend = backend          # callable(window_bytes) -> [{start,end,text}]
        self.sample_rate = sample_rate
        self.window_s = window_s
        self.track = track
        self.window_bytes = int(window_s * sample_rate * 2)  # 16-bit mono
        self._buf = bytearray()
        self._windows_done = 0

    def feed(self, pcm_bytes):
        self._buf.extend(pcm_bytes)
        out = []
        while len(self._buf) >= self.window_bytes:
            window = bytes(self._buf[:self.window_bytes])
            del self._buf[:self.window_bytes]
            out.extend(self._transcribe(window))
        return out

    def flush(self):
        if not self._buf:
            return []
        window = bytes(self._buf)
        self._buf.clear()
        return self._transcribe(window)

    def _transcribe(self, window):
        offset_ms = round(self._windows_done * self.window_s * 1000)
        self._windows_done += 1
        out = []
        for s in self.backend(window):
            text = s["text"].strip()
            if not text:
                continue
            out.append({
                "start_ms": round(s["start"] * 1000) + offset_ms,
                "end_ms": round(s["end"] * 1000) + offset_ms,
                "text": text,
                "track": self.track,
                "profile": "live",
            })
        return out


def mlx_whisper_live_backend(model="mlx-community/whisper-large-v3-turbo"):
    """Real live backend — Apple Silicon only, lazy import. Takes int16 PCM
    bytes for one window, returns whisper segments."""
    import mlx_whisper  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    def _run(window_bytes):
        audio = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return mlx_whisper.transcribe(audio, path_or_hf_repo=model)["segments"]

    return _run
