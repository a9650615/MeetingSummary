#!/usr/bin/env bash
# Package the prebuilt floating control panel for release upload. Extracts back
# into swift/floatpanel/.build/release/.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-floatpanel-arm64.tar.gz}"
SRC="swift/floatpanel/.build/release"
test -f "$SRC/floatpanel" || { echo "build first: ./build_floatpanel.sh" >&2; exit 1; }
tar czf "$OUT" -C "$SRC" floatpanel
echo "packaged: $OUT ($(du -h "$OUT" | cut -f1))"
