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
# Launch-first UX: open the browser immediately; bootstrap.py serves a progress
# page on the port and installs deps in the background (skipped if a venv is
# ready), then hands off to the real server. Reuses the dev project's venv if it
# exists -> instant; otherwise sets up in Application Support.
set -u
PORT="\${MEETING_PORT:-$PORT}"   # runtime MEETING_PORT wins; build-time PORT is only the default
# Already running? skip bootstrap/rsync entirely — jump straight to the page.
if /usr/bin/curl -fsS -m 1 "http://127.0.0.1:\$PORT/health" >/dev/null 2>&1; then
  open "http://127.0.0.1:\$PORT"
  exit 0
fi
PROJECT="$ROOT"
SRC="\$(cd "\$(dirname "\$0")/../Resources/app" && pwd)"
WD="\$HOME/Library/Application Support/MeetingSummary"
if [ -x "\$PROJECT/.venv/bin/python" ]; then
  WD="\$PROJECT"                       # dev machine: reuse working venv (instant)
else                                   # distribution: bootstrap release-update keeps code fresh
  mkdir -p "\$WD"
  /usr/bin/rsync -a --exclude .venv --exclude data "\$SRC/" "\$WD/" 2>/dev/null || cp -R "\$SRC/." "\$WD/"
fi
export MEETING_PORT=\$PORT
export MEETING_REPO="$REPO_SLUG"                # bootstrap polls GitHub releases
cd "\$WD"
# Start the server detached, then exit immediately so the dock icon doesn't bounce
# waiting on a foreground GUI app. bootstrap calls setsid() to survive into its own
# session (the .app's process group gets reaped when we exit) and opens the browser
# itself once it's serving the progress page.
nohup /usr/bin/python3 bootstrap.py >/dev/null 2>&1 &
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

echo "built $APP"
echo "run: open '$APP'  (first launch installs deps + downloads models on demand)"
echo "unsigned -> first time: right-click the app > Open to bypass Gatekeeper"
