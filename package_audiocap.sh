#!/usr/bin/env bash
# Package the prebuilt ScreenCaptureKit system-audio helper for release upload,
# so users get native system audio without a Swift build. Extracts back into
# swift/audiocap/.build/release/ (where backends.audiocap_bin() looks).
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-audiocap-arm64.tar.gz}"
SRC="swift/audiocap/.build/release"
test -f "$SRC/audiocap" || { echo "build first: ./build_audiocap.sh" >&2; exit 1; }
tar czf "$OUT" -C "$SRC" audiocap
echo "packaged: $OUT ($(du -h "$OUT" | cut -f1))"
