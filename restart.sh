#!/usr/bin/env bash
# Restart MeetingSummary so the LATEST repo code is actually loaded.
#
# Why this exists: clicking the .app icon does NOT reload code. The launcher
# short-circuits when /health already answers ("already running -> just show the
# panel"), so a running server keeps serving stale code no matter how many times
# you relaunch. Only a real stop+start swaps the code. Run this after editing
# any .py to pick the change up:  ./restart.sh
set -u
PORT="${MEETING_PORT:-8765}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> stopping (supervisor + server + watcher + bootstrap)…"
pkill -f "supervise.sh"     2>/dev/null
pkill -f "python -m app"    2>/dev/null
pkill -f "meeting_watch.py" 2>/dev/null
pkill -f "bootstrap.py"     2>/dev/null
lsof -ti "tcp:$PORT" 2>/dev/null | xargs kill -9 2>/dev/null

# wait for the port to actually free (pkill is async)
for i in $(seq 1 10); do
  lsof -ti "tcp:$PORT" >/dev/null 2>&1 || break
  sleep 0.5
done

echo "==> starting supervisor (loads current repo code)…"
cd "$HERE"
MEETING_PORT="$PORT" nohup ./supervise.sh >>"$HERE/launcher.log" 2>&1 &

echo "==> waiting for health on :$PORT …"
for i in $(seq 1 40); do
  code=$(curl -sS -o /dev/null -w "%{http_code}" -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then echo "up: HTTP 200 (${i}x)"; exit 0; fi
  sleep 2
done
echo "still not healthy after ~80s — check launcher.log" >&2
exit 1
