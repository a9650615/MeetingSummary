"""ASR engine: wraps a pluggable backend (mlx-whisper / SenseVoice).
Backend = callable(audio_path) -> [{start, end, text}] in seconds.
transcribe() normalizes to ms and tags track + profile (spec component 3)."""
import re

import zhtw

_RUN = re.compile(r"(.)\1{3,}")        # a char repeated 4+ times
_SHORT_LOOP = re.compile(r"(.{2,8}?)\1{3,}")  # a short token repeated 4+ times


def _collapse_repeats(text):
    """Tame degenerate ASR loops without hurting real doubling (好好/哈哈):
    cap a single char to 3 copies, a short repeated token to 2."""
    text = _RUN.sub(lambda m: m.group(1) * 3, text)
    return _SHORT_LOOP.sub(lambda m: m.group(1) * 2, text)


def _degenerate(text):
    """A whole-segment loop (e.g. 喬喬喬…×100) -> drop it entirely."""
    s = text.replace(" ", "")
    return len(s) >= 8 and len(set(s)) <= 3


def transcribe(audio_path, *, profile, track, backend):
    out = []
    for seg in backend(audio_path):
        text = seg["text"].strip()
        if not text or _degenerate(text):   # repetition loop -> skip the segment
            continue
        text = _collapse_repeats(text)
        out.append({
            "start_ms": round(seg["start"] * 1000),
            "end_ms": round(seg["end"] * 1000),
            "text": zhtw.to_tw(text.strip()),  # normalize 簡->繁(台灣)
            "track": track,
            "profile": profile,
        })
    return out


def mlx_whisper_backend(model="mlx-community/whisper-large-v3-mlx", language=None):
    """Real backend — Apple Silicon only. Imported lazily so tests and
    non-Mac dev don't need mlx-whisper installed. language=None -> auto-detect;
    or a whisper code ("zh"/"en"/"ja"...) to force it.

    Containers (m4a/wav/mp3) are passed by path (ffmpeg decodes). Raw headerless
    .pcm (our recorder/segment format, 16 kHz mono s16le) is loaded into an array
    here — ffmpeg can't sniff a format from raw PCM."""
    import mlx_whisper  # noqa: PLC0415
    lang = language or None

    def _run(audio_path):
        audio_path = str(audio_path)
        if audio_path.endswith(".pcm"):
            import numpy as np  # noqa: PLC0415
            with open(audio_path, "rb") as f:
                audio = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0
            src = audio
        else:
            src = audio_path
        # condition_on_previous_text=False stops a loop from cascading forward;
        # compression_ratio_threshold drops the high-repetition (degenerate) windows.
        result = mlx_whisper.transcribe(
            src, path_or_hf_repo=model, language=lang,
            condition_on_previous_text=False, compression_ratio_threshold=2.4)
        return result["segments"]

    return _run
