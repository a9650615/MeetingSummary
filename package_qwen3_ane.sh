#!/usr/bin/env bash
# Package the prebuilt ANE live helper (binary + mlx.metallib) for release upload,
# so users get live ANE without a Swift/Xcode build. Extracts back into
# swift/qwen3-ane/.build/release/ (where backends.ane_helper_bin() looks).
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-qwen3-ane-arm64.tar.gz}"
SRC="swift/qwen3-ane/.build/release"
test -f "$SRC/qwen3-ane" && test -f "$SRC/mlx.metallib" \
  || { echo "build first: ./build_qwen3_ane.sh" >&2; exit 1; }
tar czf "$OUT" -C "$SRC" qwen3-ane mlx.metallib
echo "packaged: $OUT ($(du -h "$OUT" | cut -f1))"
