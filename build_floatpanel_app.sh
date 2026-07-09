#!/usr/bin/env bash
# Wrap the floatpanel binary in a real .app bundle (越原生 app 越好): a Dock icon
# named "MeetingSummary", LaunchServices registration, and a stable bundle-id TCC
# identity (io.meetingsummary.floatpanel). The panel IS the native front-end —
# giving it a proper bundle makes it a first-class Dock app instead of a raw
# terminal binary (which the Dock/menu-bar render unreliably).
#   ./build_floatpanel_app.sh          -> dist/panel/MeetingSummary.app
#   open dist/panel/MeetingSummary.app (honors MEETING_PORT; default 8765)
# Bundle basename IS "MeetingSummary" (the Dock uses the .app filename, not
# CFBundleName) — parked under dist/panel/ so it doesn't clash with the
# launcher's dist/MeetingSummary.app.
set -eu
cd "$(dirname "$0")"
BIN=swift/floatpanel/.build/release/floatpanel
[ -x "$BIN" ] || { echo "build the binary first: (cd swift/floatpanel && swift build -c release)"; exit 1; }
APP=dist/panel/MeetingSummary.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/floatpanel"
cp swift/floatpanel/Info.plist "$APP/Contents/Info.plist"
# Reuse the launcher app's icon if it's been built; harmless if absent.
[ -f dist/MeetingSummary.app/Contents/Resources/icon.icns ] &&
  cp dist/MeetingSummary.app/Contents/Resources/icon.icns "$APP/Contents/Resources/icon.icns"
# Refresh LaunchServices so the Dock picks up the name/icon without a re-login.
LSR=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
[ -x "$LSR" ] && "$LSR" -f "$APP" 2>/dev/null || true
echo "built $APP"
