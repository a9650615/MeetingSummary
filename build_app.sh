#!/usr/bin/env bash
# Build MeetingSummary.app — a tiny launcher bundle. Ships ONLY the Python source
# (+ plist + icon); the venv/deps and models download on first launch, so the .app
# itself is ~KB. Capture-safe: it opens the PWA in your browser (WKWebView breaks
# getDisplayMedia), not an embedded webview.
#   build:  ./build_app.sh   ->  ./dist/MeetingSummary.app
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
APP="dist/MeetingSummary.app"
PORT="${MEETING_PORT:-8765}"
REPO_SLUG="${MEETING_REPO:-a9650615/MeetingSummary}"   # GitHub releases auto-update

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/app"

# 1. Python source (small) — everything needed to run + first-run setup.
cp -R *.py requirements.txt requirements-app.txt supervise.sh meeting_watch.py \
      micbusy.swift bootstrap.py stop.sh \
      "$APP/Contents/Resources/app/" 2>/dev/null || true
[ -f models/silero_vad_v4.onnx ] && { mkdir -p "$APP/Contents/Resources/app/models"; \
  cp models/silero_vad_v4.onnx "$APP/Contents/Resources/app/models/"; }

# 2. Info.plist
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>MeetingSummary</string>
  <key>CFBundleDisplayName</key><string>Meeting·Summary</string>
  <key>CFBundleIdentifier</key><string>io.meetingsummary.app</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleExecutable</key><string>launcher</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST

# 3. Launcher (the bundle executable): first-run setup, supervise, open browser.
cat > "$APP/Contents/MacOS/launcher" <<LAUNCH
#!/usr/bin/env bash
# Launch-first UX: bring up the native floating panel (approach B — launched
# directly by this .app so macOS TCC attributes recording to the app, not python).
# bootstrap.py installs deps in the background (skipped if a venv is ready) then
# hands off to the real server; it opens the browser only if the panel isn't
# installed. Reuses the dev project's venv if it exists -> instant.
set -u
PORT="\${MEETING_PORT:-$PORT}"   # runtime MEETING_PORT wins; build-time PORT is only the default
PROJECT="$ROOT"
SRC="\$(cd "\$(dirname "\$0")/../Resources/app" && pwd)"
WD="\$HOME/Library/Application Support/MeetingSummary"
[ -x "\$PROJECT/.venv/bin/python" ] && WD="\$PROJECT"   # dev machine: reuse working venv
FP="\$(dirname "\$0")/../Resources/panel/MeetingSummary.app/Contents/MacOS/floatpanel"
# Native entry: launch the floating panel as a DIRECT child of this .app,
# detached so it survives our exit. Why direct (not via the python server):
# macOS TCC blames the "responsible process" for screen-recording/mic; the
# panel captures mic + system audio ITSELF (in-process), so spawned by the
# .app it attributes to the app, not the detached python server ("python" in
# the Privacy panes). The ( … & ) reparents it to launchd, which preserves
# that attribution. \$FP is the nested panel BUNDLE's binary (Contents/Resources
# /panel/MeetingSummary.app) — not a raw swift build-dir binary, which the
# Dock/menu-bar render unreliably and which carries the wrong TCC identity.
# Returns non-zero if not installed (dev build without swift, or nesting failed).
launch_panel() { [ -x "\$FP" ] && ( MEETING_PORT="\$PORT" "\$FP" >/dev/null 2>&1 & ) ; }

# Already running? just (re)show the panel; browser fallback if the panel isn't installed.
if /usr/bin/curl -fsS -m 1 "http://127.0.0.1:\$PORT/health" >/dev/null 2>&1; then
  launch_panel || open "http://127.0.0.1:\$PORT"
  exit 0
fi
# Cold start: set up the working dir if this isn't a dev checkout.
if [ ! -x "\$PROJECT/.venv/bin/python" ]; then   # distribution: release-update keeps code fresh
  mkdir -p "\$WD"
  /usr/bin/rsync -a --exclude .venv --exclude data "\$SRC/" "\$WD/" 2>/dev/null || cp -R "\$SRC/." "\$WD/"
fi
export MEETING_PORT=\$PORT
export MEETING_REPO="$REPO_SLUG"                # bootstrap polls GitHub releases
export MEETING_APP_BUNDLE="\$(cd "\$(dirname "\$0")/../.." && pwd)"  # updater swaps this whole bundle
cd "\$WD"
# Start the server detached, then exit immediately so the dock icon doesn't bounce.
# bootstrap calls setsid() to survive the .app's process-group reap; it opens the
# browser itself ONLY if the native panel isn't installed (else the panel is the UI).
# Log (not /dev/null) so a bootstrap-phase death before supervise.sh's own log
# takes over still leaves a trace to diagnose — was previously silently discarded.
nohup /usr/bin/python3 bootstrap.py >>"\$WD/launcher.log" 2>&1 &
launch_panel   # server still coming up; the panel polls + connects when it's ready
exit 0
LAUNCH
chmod +x "$APP/Contents/MacOS/launcher"

# 4. Icon (.icns from a generated PNG, best-effort)
if command -v sips >/dev/null && command -v iconutil >/dev/null; then
  python3 - <<'PY' >/tmp/_ms_icon.png 2>/dev/null || true
import struct,zlib,sys
def chunk(t,d):
    c=t+d;return struct.pack(">I",len(d))+c+struct.pack(">I",zlib.crc32(c)&0xffffffff)
s=512;rgb=(0x5b,0x54,0xe6)
ihdr=struct.pack(">IIBBBBB",s,s,8,2,0,0,0);row=b"\x00"+bytes(rgb)*s
sys.stdout.buffer.write(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",ihdr)+chunk(b"IDAT",zlib.compress(row*s,9))+chunk(b"IEND",b""))
PY
  if [ -s /tmp/_ms_icon.png ]; then
    ICONSET=/tmp/_ms.iconset; rm -rf "$ICONSET"; mkdir -p "$ICONSET"
    for sz in 16 32 128 256 512; do
      sips -z $sz $sz /tmp/_ms_icon.png --out "$ICONSET/icon_${sz}x${sz}.png" >/dev/null 2>&1
      sips -z $((sz*2)) $((sz*2)) /tmp/_ms_icon.png --out "$ICONSET/icon_${sz}x${sz}@2x.png" >/dev/null 2>&1
    done
    iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/icon.icns" 2>/dev/null || true
  fi
fi

# 5. Nest the floatpanel bundle at Contents/Resources/panel/MeetingSummary.app so the
# launcher runs a real .app, not a raw binary (Dock/menu-bar render raw binaries
# unreliably, and TCC keys off the bundle identity io.meetingsummary.floatpanel).
# Best-effort like the icon block above: dev machines without a floatpanel release
# build still produce a working (degraded — panel-less, browser-fallback) launcher.
FP_BIN="swift/floatpanel/.build/release/floatpanel"
if [ -x "$FP_BIN" ]; then
  [ -d dist/panel/MeetingSummary.app ] || ./build_floatpanel_app.sh
  mkdir -p "$APP/Contents/Resources/panel"
  cp -R dist/panel/MeetingSummary.app "$APP/Contents/Resources/panel/"
  echo "nested panel: $APP/Contents/Resources/panel/MeetingSummary.app"
else
  echo "warning: $FP_BIN not built — shipping launcher WITHOUT the native panel" >&2
  echo "  (degraded: falls back to opening the browser). To include it:" >&2
  echo "  (cd swift/floatpanel && swift build -c release) && ./build_app.sh" >&2
fi

# 6. Code-sign (launcher + nested panel) with a stable identity so macOS TCC
# grants persist across updates. Uses the self-signed "MeetingSummary Dev"
# identity when present (create once: ./setup_signing_cert.sh), else adhoc but
# pinned to the real bundle id. Self-signed/adhoc still trips Gatekeeper on
# other machines (right-click > Open) — that's expected until Developer ID +
# notarization. --deep signs the nested panel (its own io.meetingsummary.floatpanel
# identifier is preserved from its Info.plist).
SIGN_ID="${FLOATPANEL_SIGN_ID:-MeetingSummary Dev}"
if security find-identity -p codesigning 2>/dev/null | grep -q "$SIGN_ID"; then
  codesign --force --deep -s "$SIGN_ID" --identifier io.meetingsummary.app "$APP" \
    && echo "signed: $SIGN_ID"
else
  codesign --force --deep -s - --identifier io.meetingsummary.app "$APP" 2>/dev/null || true
  echo "note: no '$SIGN_ID' identity — adhoc signed (run ./setup_signing_cert.sh for a stable one)" >&2
fi

echo "built $APP"
echo "run: open '$APP'  (first launch installs deps + downloads models on demand)"
echo "unsigned/self-signed -> first time: right-click the app > Open to bypass Gatekeeper"
