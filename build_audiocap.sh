#!/usr/bin/env bash
# Build the native ScreenCaptureKit system-audio helper (swift/audiocap).
# Pure Swift + system frameworks — no Metal/metallib needed (unlike qwen3-ane),
# so Command-Line-Tools Swift is enough. Output: swift/audiocap/.build/release/audiocap
set -euo pipefail
cd "$(dirname "$0")/swift/audiocap"
swift build -c release
echo "built: $(pwd)/.build/release/audiocap"
echo "NOTE: capturing system audio needs macOS Screen-Recording permission for the"
echo "launching app (System Settings → Privacy & Security → Screen Recording)."
