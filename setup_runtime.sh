#!/usr/bin/env bash
# Install the optional .cpp ASR runtimes (Metal). Triggered by 設定 → 加速 runtime.
#   setup_runtime.sh chatllm   -> prebuilt download (no cmake); fallback build
#   setup_runtime.sh femelo    -> .venv-qwen314 (any python>=3.11) + py-qwen3-asr-cpp
# Auto-installs the build env via brew when needed. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"  # server runs non-login shell -> ensure brew is found
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

    # py-qwen3-asr-cpp publishes NO wheels for our (python, arm64) combos, so it's
    # ALWAYS a source build (scikit-build/cmake -> needs cmake + a C/C++ toolchain).
    # Without them the build installs the python files but NOT the native
    # _py_qwen3_asr_cpp.*.so -> "cannot import name '_py_qwen3_asr_cpp'". Verify the
    # REAL import (not just `import py_qwen3_asr_cpp`, which a broken pkg passes
    # half-way), and on failure force a CLEAN rebuild — a half-installed pkg makes a
    # plain `pip install` a no-op ('already satisfied'), so it can never self-heal.
    _femelo_ok() {
      .venv-qwen314/bin/python -c "from py_qwen3_asr_cpp.model import Qwen3ASRModel" 2>/dev/null
    }
    if ! _femelo_ok; then
      command -v cmake >/dev/null || { echo "找不到 cmake，嘗試 brew 安裝…"; _brew cmake; }
      command -v cmake >/dev/null || { echo "need cmake (brew install cmake)"; exit 1; }
      xcode-select -p >/dev/null 2>&1 \
        || echo "警告: 未偵測到 Xcode Command Line Tools，原生模組編譯可能失敗 (xcode-select --install)"
      .venv-qwen314/bin/pip uninstall -y py-qwen3-asr-cpp >/dev/null 2>&1 || true  # drop any broken half-install
      echo "編譯 py-qwen3-asr-cpp (原始碼)…"
      .venv-qwen314/bin/pip install --force-reinstall --no-cache-dir py-qwen3-asr-cpp \
        || .venv-qwen314/bin/pip install --force-reinstall --no-cache-dir \
             "git+https://github.com/femelo/py-qwen3-asr-cpp"
    fi
    if _femelo_ok; then
      echo "femelo OK"
    else
      echo "femelo 安裝失敗: 原生模組 _py_qwen3_asr_cpp 沒編出來。請確認已裝 Xcode Command Line Tools (xcode-select --install) 與 cmake (brew install cmake) 後重試。" >&2
      exit 1
    fi
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
  speech)
    # ANE (Neural Engine) ASR: Qwen3-ASR CoreML via the homebrew `speech` CLI
    # (homebrew-core). Power-efficient transcription, off the GPU.
    command -v brew >/dev/null || { echo "need Homebrew (brew install …)"; exit 1; }
    if command -v speech >/dev/null; then echo "speech already installed"; exit 0; fi
    echo "安裝 speech (Qwen3-ASR ANE)…"
    brew install speech && command -v speech >/dev/null && echo "speech OK"
    ;;
  qwen3-ane)
    # Live ANE helper: download the prebuilt binary + mlx.metallib from the latest
    # release (no Xcode needed); fall back to building from source (needs full Xcode
    # for the Metal toolchain).
    OUT="swift/qwen3-ane/.build/release"
    if [ -f "$OUT/qwen3-ane" ] && [ -f "$OUT/mlx.metallib" ]; then
      echo "qwen3-ane already installed"; exit 0
    fi
    mkdir -p "$OUT"
    url="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null \
      | python3 -c "import sys,json
a=[x['browser_download_url'] for x in json.load(sys.stdin).get('assets',[]) if x['name']=='qwen3-ane-arm64.tar.gz']
print(a[0] if a else '')" 2>/dev/null)" || true
    if [ -n "${url:-}" ]; then
      echo "下載預編譯 qwen3-ane…"
      curl -fsSL "$url" | tar xzf - -C "$OUT" && { echo "qwen3-ane OK (prebuilt)"; exit 0; }
    fi
    echo "預編譯不可用，改從原始碼編譯(需 Xcode 的 metal toolchain)…"
    bash build_qwen3_ane.sh
    ;;
  *) echo "usage: setup_runtime.sh femelo|chatllm|speech|qwen3-ane"; exit 2 ;;
esac
