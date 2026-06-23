#!/usr/bin/env bash
# One-click build/install for the optional .cpp ASR runtimes (Metal).
# Triggered by the 模型管理 tab. Idempotent: re-runs are cheap if already built.
#   setup_runtime.sh femelo    -> .venv-qwen314 + py-qwen3-asr-cpp (0.6b GGUF)
#   setup_runtime.sh chatllm   -> clone + cmake build chatllm.cpp (1.7b GGUF)
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  femelo)
    command -v python3.14 >/dev/null || { command -v /opt/homebrew/bin/python3.14 >/dev/null && PY=/opt/homebrew/bin/python3.14; } || { echo "need python3.14 (brew install python@3.14)"; exit 1; }
    PY="${PY:-python3.14}"
    [ -d .venv-qwen314 ] || "$PY" -m venv .venv-qwen314
    .venv-qwen314/bin/pip install -q --upgrade pip
    .venv-qwen314/bin/python -c "import py_qwen3_asr_cpp" 2>/dev/null \
      || .venv-qwen314/bin/pip install "git+https://github.com/femelo/py-qwen3-asr-cpp"
    .venv-qwen314/bin/python -c "from py_qwen3_asr_cpp.model import Qwen3ASRModel; print('femelo OK')"
    ;;
  chatllm)
    command -v cmake >/dev/null || { echo "need cmake (brew install cmake)"; exit 1; }
    [ -d chatllm.cpp ] || git clone --depth 1 https://github.com/foldl/chatllm.cpp
    cd chatllm.cpp
    [ -f bindings/libchatllm.dylib ] || { cmake -B build; cmake --build build -j --config Release; cmake --build build --target libchatllm -j; }
    test -f bindings/libchatllm.dylib && echo "chatllm OK"
    ;;
  *) echo "usage: setup_runtime.sh femelo|chatllm"; exit 2 ;;
esac
