"""ASR engine: wraps a pluggable backend (mlx-whisper / SenseVoice).
Backend = callable(audio_path) -> [{start, end, text}] in seconds.
transcribe() normalizes to ms and tags track + profile (spec component 3)."""


def transcribe(audio_path, *, profile, track, backend):
    out = []
    for seg in backend(audio_path):
        text = seg["text"].strip()
        if not text:
            continue
        out.append({
            "start_ms": round(seg["start"] * 1000),
            "end_ms": round(seg["end"] * 1000),
            "text": text,
            "track": track,
            "profile": profile,
        })
    return out


def mlx_whisper_backend(model="mlx-community/whisper-large-v3-mlx"):
    """Real backend — Apple Silicon only. Imported lazily so tests and
    non-Mac dev don't need mlx-whisper installed.
    ponytail: 16 kHz mono PCM assumed; resample upstream in the capture helper."""
    import mlx_whisper  # noqa: PLC0415

    def _run(audio_path):
        result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=model)
        return result["segments"]

    return _run
