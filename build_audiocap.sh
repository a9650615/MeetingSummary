#!/usr/bin/env bash
# Build the native ScreenCaptureKit system-audio helper (swift/audiocap).
# Pure Swift + system frameworks — no Metal/metallib needed (unlike qwen3-ane),
# so Command-Line-Tools Swift is enough. Output: swift/audiocap/.build/release/audiocap
set -euo pipefail
cd "$(dirname "$0")/swift/audiocap"
swift build -c release
# Ad-hoc sign with a STABLE designated identifier so macOS TCC can attribute +
# persist the Screen-Recording grant to this binary (an unsigned binary gets a
# fresh identity each build -> the grant never sticks -> capture stays denied).
codesign --force --sign - --identifier io.aift.audiocap ".build/release/audiocap" 2>/dev/null \
  && echo "ad-hoc signed (io.aift.audiocap)" || echo "WARN: codesign failed (TCC may not persist)"
echo "built: $(pwd)/.build/release/audiocap"
echo "NOTE: capturing system audio needs macOS Screen-Recording permission for the"
echo "launching app (System Settings → Privacy & Security → Screen Recording)."
