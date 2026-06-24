"""Sidecar CLI: run Qwen3-ASR (.cpp/GGUF, Metal) in the 3.14 venv and emit JSON.

The main app is Python 3.10; py-qwen3-asr-cpp ships a cp314 native module, so it
lives in .venv-qwen314 and the app shells out here.

Usage:  .venv-qwen314/bin/python qwen3_cpp_cli.py <audio_path>
Output (stdout, last line): {"text": ..., "language": ..., "words": [{start,end,word}]}
"""
import json
import sys

from py_qwen3_asr_cpp.model import Qwen3ASRModel

_ASR = "qwen3-asr-0.6b-q4-k-m"
_ALIGN = "qwen3-forced-aligner-0.6b-q4-k-m"


def main():
    audio = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else ""   # "" -> auto-detect
    model = Qwen3ASRModel(asr_model=_ASR, align_model=_ALIGN, n_threads=4)
    res, al = model.transcribe_and_align(audio, language=lang or "")
    words = [{"start": float(w.start), "end": float(w.end), "word": w.word}
             for w in (al.words if al and al.success else [])]
    # Emit one clean JSON line (model load logs go to stderr); prefix-tagged so the
    # caller can pick it out reliably.
    print("QWEN3JSON:" + json.dumps(
        {"text": res.text, "language": res.language, "words": words},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
