"""Persistent Qwen3-ASR .cpp daemon (runs in .venv-qwen314).

Loads the GGUF model ONCE, then serves many requests so live can use it without
paying the ~1.3s model load per utterance. Protocol (line-based over stdin/stdout):
  - on start, prints `QWEN3READY` once the model is loaded
  - per request: read one line = a wav file path -> print `QWEN3JSON:{...}`
ggml/Metal logs go to stderr; stdout carries only the protocol lines.
"""
import json
import sys

from py_qwen3_asr_cpp.model import Qwen3ASRModel

_ASR = "qwen3-asr-0.6b-q4-k-m"  # text-only (no aligner) -> faster load + per-call


def main():
    model = Qwen3ASRModel(asr_model=_ASR, n_threads=4)
    print("QWEN3READY", flush=True)
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        lang, _, path = line.partition("\t")   # protocol: "lang<TAB>path"
        if not path:                            # back-compat: bare path = auto lang
            path, lang = lang, ""
        text = ""
        try:
            res = model.transcribe(path, language=lang or "")
            res = res[0] if isinstance(res, tuple) else res
            text = getattr(res, "text", "") or ""
        except Exception as e:
            print(f"QWEN3ERR: {e}", file=sys.stderr, flush=True)
        print("QWEN3JSON:" + json.dumps({"text": text}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
