#!/usr/bin/env bash
# Build the persistent ANE live-ASR helper (Swift, speech-swift Qwen3-ASR CoreML).
# Output: swift/qwen3-ane/.build/release/qwen3-ane + mlx.metallib beside it (MLX
# needs its Metal shader lib next to the binary at runtime).
set -euo pipefail
cd "$(dirname "$0")/swift/qwen3-ane"
export PATH="/opt/homebrew/bin:$PATH"

swift build -c release
OUT=".build/release"
test -f "$OUT/qwen3-ane" || { echo "build failed: no binary" >&2; exit 1; }

# mlx.metallib: compile it (needs FULL Xcode's metal toolchain) else copy a prebuilt
# one (the homebrew `speech` bottle or the pip `mlx` package both ship a matching one).
if [ ! -f "$OUT/mlx.metallib" ] && xcrun -f metal >/dev/null 2>&1; then
  S=$(find .build/checkouts/speech-swift/scripts -name build_mlx_metallib.sh 2>/dev/null | head -1)
  [ -n "$S" ] && BUILD_DIR="$PWD/.build" bash "$S" release || true
fi
if [ ! -f "$OUT/mlx.metallib" ]; then
  for c in /opt/homebrew/Cellar/speech/*/libexec/mlx.metallib \
           ../../.venv/lib/python*/site-packages/mlx/lib/mlx.metallib; do
    [ -f "$c" ] && { cp "$c" "$OUT/mlx.metallib"; echo "copied prebuilt metallib: $c"; break; }
  done
fi
test -f "$OUT/mlx.metallib" || { echo "no mlx.metallib (need Xcode metal toolchain, or install speech/mlx)" >&2; exit 1; }
echo "qwen3-ane OK: $OUT/qwen3-ane (+ mlx.metallib)"
