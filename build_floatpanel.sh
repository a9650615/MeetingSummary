#!/usr/bin/env bash
# Build the lightweight floating control panel (swift/floatpanel). Pure AppKit,
# no Metal — Command-Line-Tools Swift is enough.
# Output: swift/floatpanel/.build/release/floatpanel
#
# This raw binary is what backends.floatpanel_bin() runs on a dev machine, so it
# MUST be signed the same way build_floatpanel_app.sh signs the bundle. macOS TCC
# keys the Screen & System Audio Recording grant to the code identity, and a plain
# `swift build` is adhoc/linker-signed with a fresh cdhash every time — so every
# rebuild silently revoked the grant, and system audio then recorded as pure
# digital silence with nothing but a permission nag to show for it.
set -euo pipefail
cd "$(dirname "$0")/swift/floatpanel"
swift build -c release
BIN="$(pwd)/.build/release/floatpanel"

SIGN_ID="${FLOATPANEL_SIGN_ID:-MeetingSummary Dev}"
# No -v: a self-signed identity is untrusted so it's excluded from "valid
# identities only", but codesign can still use it (that's all TCC needs).
if security find-identity -p codesigning 2>/dev/null | grep -q "$SIGN_ID"; then
  codesign --force -s "$SIGN_ID" --identifier io.meetingsummary.floatpanel "$BIN"
  echo "signed: $SIGN_ID (TCC grant persists across rebuilds)"
else
  echo "warning: no '$SIGN_ID' signing identity — left adhoc." >&2
  echo "  Screen-recording permission will be LOST on every rebuild." >&2
  echo "  Fix once:  ./setup_signing_cert.sh" >&2
fi

echo "built: $BIN"
echo "run alongside the server (honors MEETING_PORT, default 8765)."
