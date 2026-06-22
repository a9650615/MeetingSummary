"""ASR engine: wraps a pluggable backend (mlx-whisper / SenseVoice).
Backend = callable(audio_path) -> [{start, end, text}] in seconds.
transcribe() normalizes to ms and tags track + profile (spec component 3)."""
import zhtw


def transcribe(audio_path, *, profile, track, backend):
    out = []
    for seg in backend(audio_path):
        text = seg["text"].strip()
        if not text:
            continue
        out.append({
            "start_ms": round(seg["start"] * 1000),
            "end_ms": round(seg["end"] * 1000),
            "text": zhtw.to_tw(text),  # normalize 簡->繁(台灣)
            "track": track,
            "profile": profile,
        })
    return out


def mlx_whisper_backend(model="mlx-community/whisper-large-v3-mlx"):
    """Real backend — Apple Silicon only. Imported lazily so tests and
    non-Mac dev don't need mlx-whisper installed.

    Containers (m4a/wav/mp3) are passed by path (ffmpeg decodes). Raw headerless
    .pcm (our recorder/segment format, 16 kHz mono s16le) is loaded into an array
    here — ffmpeg can't sniff a format from raw PCM."""
    import mlx_whisper  # noqa: PLC0415

    def _run(audio_path):
        audio_path = str(audio_path)
        if audio_path.endswith(".pcm"):
            import numpy as np  # noqa: PLC0415
            with open(audio_path, "rb") as f:
                audio = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0
            result = mlx_whisper.transcribe(audio, path_or_hf_repo=model)
        else:
            result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model)
        return result["segments"]

    return _run
