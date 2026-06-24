#!/usr/bin/env bash
# Install the optional .cpp ASR runtimes (Metal). Triggered by 設定 → 加速 runtime.
#   setup_runtime.sh chatllm   -> prebuilt download (no cmake); fallback build
#   setup_runtime.sh femelo    -> .venv-qwen314 (any python>=3.11) + py-qwen3-asr-cpp
# Auto-installs the build env via brew when needed. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"
REPO="${MEETING_REPO:-a9650615/MeetingSummary}"

_brew() { command -v brew >/dev/null && brew install "$1"; }

_prebuilt_chatllm() {
  # Fetch chatllm-runtime-arm64.tar.gz from the latest release -> no cmake needed.
  command -v curl >/dev/null && command -v python3 >/dev/null || return 1
  local url
  url="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null \
    | python3 -c "import sys,json
a=[x['browser_download_url'] for x in json.load(sys.stdin).get('assets',[]) if x['name']=='chatllm-runtime-arm64.tar.gz']
print(a[0] if a else '')" 2>/dev/null)" || return 1
  [ -n "$url" ] || return 1
  echo "下載預編譯 chatllm…"
  curl -fsSL "$url" | tar xzf - || return 1   # -> chatllm.cpp/{bindings,scripts}
  test -f chatllm.cpp/bindings/libchatllm.dylib
}

case "${1:-}" in
  femelo)
    # Needs python>=3.11 (the native ext is version-specific; 3.14 was never required).
    PY=""
    for c in python3.13 python3.12 python3.11 python3.14 python3 \
             /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12; do
      p="$(command -v "$c" 2>/dev/null)" || continue
      "$p" -c 'import sys;sys.exit(0 if sys.version_info[:2]>=(3,11) else 1)' 2>/dev/null \
        && { PY="$p"; break; }
    done
    if [ -z "$PY" ]; then
      echo "找不到 python>=3.11，嘗試 brew 安裝…"
      _brew python@3.13 && PY="$(brew --prefix 2>/dev/null)/bin/python3.13"
    fi
    [ -n "$PY" ] && [ -x "$PY" ] || { echo "need python>=3.11 (brew install python@3.13)"; exit 1; }
    echo "femelo 使用 $("$PY" --version 2>&1)"
    [ -d .venv-qwen314 ] || "$PY" -m venv .venv-qwen314
    .venv-qwen314/bin/pip install -q --upgrade pip
    .venv-qwen314/bin/python -c "import py_qwen3_asr_cpp" 2>/dev/null \
      || .venv-qwen314/bin/pip install py-qwen3-asr-cpp \
      || .venv-qwen314/bin/pip install "git+https://github.com/femelo/py-qwen3-asr-cpp"
    .venv-qwen314/bin/python -c "from py_qwen3_asr_cpp.model import Qwen3ASRModel; print('femelo OK')"
    ;;
  chatllm)
    if [ ! -f chatllm.cpp/bindings/libchatllm.dylib ]; then
      _prebuilt_chatllm && { echo "chatllm OK (prebuilt)"; exit 0; }
      echo "預編譯不可用，改從原始碼編譯…"
      command -v cmake >/dev/null || { echo "找不到 cmake，嘗試 brew 安裝…"; _brew cmake; }
      command -v cmake >/dev/null || { echo "need cmake (brew install cmake)"; exit 1; }
      [ -d chatllm.cpp ] || git clone --depth 1 https://github.com/foldl/chatllm.cpp
      ( cd chatllm.cpp
        cmake -B build
        cmake --build build -j --config Release
        cmake --build build --target libchatllm -j )
    fi
    test -f chatllm.cpp/bindings/libchatllm.dylib && echo "chatllm OK"
    ;;
  *) echo "usage: setup_runtime.sh femelo|chatllm"; exit 2 ;;
esac
