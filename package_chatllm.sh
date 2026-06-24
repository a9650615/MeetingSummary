#!/usr/bin/env bash
# Build + package a PORTABLE chatllm.cpp runtime for arm64 macOS.
# Output tar contains chatllm.cpp/{bindings,scripts}; the bindings dylib has its
# rpath rewritten to @loader_path with libggml copied beside it, so it loads from
# any extract path with NO cmake on the user's machine (setup_runtime.sh fetches it).
#   ./package_chatllm.sh /abs/path/chatllm-runtime-arm64.tar.gz
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-$PWD/chatllm-runtime-arm64.tar.gz}"

command -v cmake >/dev/null || { echo "need cmake to build chatllm"; exit 1; }
[ -d chatllm.cpp ] || git clone --depth 1 https://github.com/foldl/chatllm.cpp
cd chatllm.cpp
if [ ! -f bindings/libchatllm.dylib ]; then
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  # Build ONLY the shared lib (cmake pulls in its ggml deps). Building the default
  # "all" target compiles every chatllm example/tool too — minutes vs ~40 min.
  cmake --build build --target libchatllm -j --config Release
fi
test -f bindings/libchatllm.dylib

STAGE="$(mktemp -d)/chatllm.cpp"
mkdir -p "$STAGE/bindings" "$STAGE/scripts"
cp bindings/libchatllm.dylib bindings/chatllm.py "$STAGE/bindings/"
cp -R scripts/. "$STAGE/scripts/" 2>/dev/null || true

# libchatllm needs libggml* via @rpath. Copy the real versioned dylibs beside it
# and recreate the soname symlinks so @loader_path resolves them.
for base in libggml libggml-base libggml-cpu libggml-blas libggml-metal; do
  real="$(ls build/lib/${base}.0.*.dylib 2>/dev/null | head -1 || true)"
  [ -n "$real" ] || continue
  cp "$real" "$STAGE/bindings/"
  ( cd "$STAGE/bindings"
    ln -sf "$(basename "$real")" "${base}.0.dylib"
    ln -sf "$(basename "$real")" "${base}.dylib" )
done

# rpath: drop the absolute build/lib path, point at the dylib's own dir.
abs="$(cd build/lib && pwd)"
install_name_tool -delete_rpath "$abs" "$STAGE/bindings/libchatllm.dylib" 2>/dev/null || true
install_name_tool -add_rpath @loader_path "$STAGE/bindings/libchatllm.dylib"
# editing the load commands invalidates the signature -> re-sign ad-hoc so Gatekeeper
# doesn't kill it on other machines.
codesign --force --sign - "$STAGE/bindings/libchatllm.dylib" 2>/dev/null || true

( cd "$(dirname "$STAGE")" && tar czf "$OUT" chatllm.cpp )
echo "packaged -> $OUT ($(du -h "$OUT" | cut -f1))"
