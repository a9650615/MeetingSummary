"""Live transcription: chunk streamed PCM, transcribe each chunk, emit segments
with absolute timestamps (spec Phase 2, profile=live).

Chunking is pluggable:
- FixedWindowChunker: cut every N seconds (deterministic, used in tests).
- VadChunker: cut at silence after speech (or a max-length ceiling) so windows
  hold whole phrases instead of slicing mid-word.

Each chunk is transcribed independently (no cross-chunk context) — live is a
preview; the accurate re-pass is the trusted transcript (spec M1)."""
import numpy as np


def _rms(frame_bytes):
    if not frame_bytes:
        return 0.0
    a = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def _is_repetition(text):
    """Whisper loops on silence/noise, repeating one token (e.g. 'segment
    segment segment...'). Drop those degenerate outputs."""
    parts = text.split()
    if len(parts) >= 4 and len(set(parts)) <= 2:
        return True
    stripped = text.replace(" ", "")
    return len(stripped) >= 8 and len(set(stripped)) <= 3


class FixedWindowChunker:
    def __init__(self, window_bytes):
        self.window_bytes = window_bytes
        self._buf = bytearray()

    def feed(self, pcm):
        self._buf.extend(pcm)
        out = []
        while len(self._buf) >= self.window_bytes:
            out.append(bytes(self._buf[:self.window_bytes]))
            del self._buf[:self.window_bytes]
        return out

    def flush(self):
        if not self._buf:
            return []
        w = bytes(self._buf)
        self._buf.clear()
        return [w]


class VadChunker:
    """Energy VAD: emit a window when a speech run is followed by >= silence_ms
    of quiet, or when the buffer hits max_window_s (latency ceiling).

    ponytail: RMS-threshold heuristic, no deps. Swap for silero-VAD if it
    mis-cuts on noisy input — the upgrade path is the same feed()/flush() API."""

    def __init__(self, sample_rate=16000, frame_ms=30, silence_ms=400,
                 max_window_s=8.0, rms_threshold=500):
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit mono
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.max_bytes = int(max_window_s * sample_rate) * 2
        self.rms_threshold = rms_threshold
        self._buf = bytearray()
        self._scan = 0
        self._silence_run = 0
        self._has_speech = False

    def _reset(self):
        self._scan = 0
        self._silence_run = 0
        self._has_speech = False

    def _find_cut(self):
        n = len(self._buf)
        while self._scan + self.frame_bytes <= n:
            frame = self._buf[self._scan:self._scan + self.frame_bytes]
            self._scan += self.frame_bytes
            if _rms(frame) >= self.rms_threshold:
                self._has_speech = True
                self._silence_run = 0
            else:
                self._silence_run += 1
                if self._has_speech and self._silence_run >= self.silence_frames:
                    return self._scan  # cut after the silence that ended speech
        if n >= self.max_bytes:
            return self.max_bytes
        return None

    def feed(self, pcm):
        self._buf.extend(pcm)
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            had_speech = self._has_speech
            window = bytes(self._buf[:cut])
            del self._buf[:cut]
            self._reset()
            if had_speech:           # drop non-speech windows (force-cut noise)
                out.append(window)
        return out

    def flush(self):
        if not self._buf or not self._has_speech:
            self._buf.clear()
            self._reset()
            return []
        w = bytes(self._buf)
        self._buf.clear()
        self._reset()
        return [w]


class LiveSession:
    def __init__(self, *, backend, chunker=None, sample_rate=16000,
                 window_s=5.0, track="mic"):
        self.backend = backend
        self.sample_rate = sample_rate
        self.track = track
        self.chunker = chunker or FixedWindowChunker(int(window_s * sample_rate * 2))
        self._emitted_bytes = 0  # cumulative -> absolute timestamp offset

    def feed(self, pcm_bytes):
        out = []
        for window in self.chunker.feed(pcm_bytes):
            out.extend(self._transcribe(window))
        return out

    def flush(self):
        out = []
        for window in self.chunker.flush():
            out.extend(self._transcribe(window))
        return out

    def _transcribe(self, window):
        offset_ms = round(self._emitted_bytes / (self.sample_rate * 2) * 1000)
        self._emitted_bytes += len(window)
        out = []
        for s in self.backend(window):
            text = s["text"].strip()
            if not text or _is_repetition(text):
                continue
            out.append({
                "start_ms": round(s["start"] * 1000) + offset_ms,
                "end_ms": round(s["end"] * 1000) + offset_ms,
                "text": text,
                "track": self.track,
                "profile": "live",
            })
        return out


def mlx_whisper_live_backend(model="mlx-community/whisper-small-mlx"):
    """Real live backend — Apple Silicon only, lazy import. Takes int16 PCM
    bytes for one window, returns whisper segments."""
    import mlx_whisper  # noqa: PLC0415

    def _run(window_bytes):
        audio = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segs = mlx_whisper.transcribe(
            audio, path_or_hf_repo=model,
            condition_on_previous_text=False,  # don't propagate a loop forward
        )["segments"]
        # Whisper's standard hallucination guards: drop non-speech / repetitive /
        # low-confidence segments before they reach the subtitle.
        return [s for s in segs
                if s.get("no_speech_prob", 0) < 0.6
                and s.get("compression_ratio", 0) < 2.4
                and s.get("avg_logprob", 0) > -1.0]

    return _run
