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

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/app"

# 1. Python source (small) — everything needed to run + first-run setup.
cp -R *.py requirements.txt requirements-app.txt supervise.sh meeting_watch.py micbusy.swift \
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
  <key>CFBundleIdentifier</key><string>io.aift.meetingsummary</string>
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
set -u
PORT=$PORT
SRC="\$(cd "\$(dirname "\$0")/../Resources/app" && pwd)"
HOME_DIR="\$HOME/Library/Application Support/MeetingSummary"
mkdir -p "\$HOME_DIR"
# sync code into a writable home (Resources is read-only in a signed app)
/usr/bin/rsync -a --exclude .venv --exclude data "\$SRC/" "\$HOME_DIR/" 2>/dev/null || cp -R "\$SRC/." "\$HOME_DIR/"
cd "\$HOME_DIR"
PY="\$HOME_DIR/.venv/bin/python"
if [ ! -x "\$PY" ]; then
  osascript -e 'display notification "首次啟動：安裝相依套件中（會下載，約數分鐘）" with title "Meeting·Summary"' || true
  /usr/bin/python3 -m venv .venv
  "\$HOME_DIR/.venv/bin/pip" install -q --upgrade pip
  "\$HOME_DIR/.venv/bin/pip" install -q -r requirements-app.txt
fi
[ -f micbusy ] || swiftc micbusy.swift -o micbusy -framework CoreAudio 2>/dev/null || true
export MEETING_PORT=\$PORT
cleanup(){ pkill -9 -f "python -m app" 2>/dev/null; pkill -9 -f meeting_watch.py 2>/dev/null; exit 0; }
trap cleanup INT TERM EXIT
bash supervise.sh >/tmp/meetingsummary.log 2>&1 &
for i in \$(seq 1 90); do
  /usr/bin/curl -sf "http://127.0.0.1:\$PORT/health" >/dev/null && break; sleep 1
done
open "http://127.0.0.1:\$PORT"
wait   # keep the app alive (Dock icon) until quit -> cleanup stops the server
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
