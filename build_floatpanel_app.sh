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
# Code-sign so the macOS TCC "Screen & System Audio Recording" grant SURVIVES
# rebuilds. TCC keys a grant to the app's code identity; an adhoc / linker-signed
# binary gets a fresh cdhash on every build, so the grant is lost each rebuild and
# the system-audio tap silently records zeros until re-approved. A STABLE
# self-signed identity fixes it. Create one ONCE (no Apple account needed):
#   Keychain Access → Certificate Assistant → Create a Certificate…
#     Name: "MeetingSummary Dev"   Identity Type: Self Signed Root
#     Certificate Type: Code Signing
# then rebuild — the grant then persists across rebuilds.
SIGN_ID="${FLOATPANEL_SIGN_ID:-MeetingSummary Dev}"
if security find-identity -v -p codesigning 2>/dev/null | grep -q "$SIGN_ID"; then
  codesign --force --deep -s "$SIGN_ID" --identifier io.meetingsummary.floatpanel "$APP" \
    && echo "signed: $SIGN_ID (TCC grant persists across rebuilds)"
else
  # Adhoc fallback: at least pin the real bundle id (linker default is "floatpanel");
  # cdhash still changes, so the grant is lost each rebuild.
  codesign --force --deep -s - --identifier io.meetingsummary.floatpanel "$APP" 2>/dev/null || true
  echo "warning: no '$SIGN_ID' signing identity — adhoc signed." >&2
  echo "  Screen-recording permission is LOST on every rebuild. Fix once:" >&2
  echo "  Keychain Access → Certificate Assistant → Create a Certificate:" >&2
  echo "  name '$SIGN_ID', Identity Type 'Self Signed Root', Type 'Code Signing'." >&2
fi

# Refresh LaunchServices so the Dock picks up the name/icon without a re-login.
LSR=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
[ -x "$LSR" ] && "$LSR" -f "$APP" 2>/dev/null || true
echo "built $APP"
