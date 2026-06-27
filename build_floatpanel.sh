#!/usr/bin/env bash
# Build the lightweight floating control panel (swift/floatpanel). Pure AppKit,
# no Metal — Command-Line-Tools Swift is enough.
# Output: swift/floatpanel/.build/release/floatpanel
set -euo pipefail
cd "$(dirname "$0")/swift/floatpanel"
swift build -c release
echo "built: $(pwd)/.build/release/floatpanel"
echo "run alongside the server (honors MEETING_PORT, default 8765)."
