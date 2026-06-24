#!/usr/bin/env bash
# Keep the live server up: restart if the process dies OR if /health stops
# responding (hung / not doing work). Run in a terminal:  ./supervise.sh
# Stop with Ctrl-C. Env (ASR_MODEL, LIVE_MODEL, ...) is passed through.
#
# ponytail: a poll-loop supervisor, no deps. For survive-reboot/login, wrap it in
# the launchd plist below (see README note) — KeepAlive restarts the supervisor.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
PORT=${MEETING_PORT:-8765}
export MEETING_PORT=$PORT   # server (app.py) + health check bind the same port
LOG=/tmp/meeting_server.log
POLL=10          # seconds between health checks
MAX_FAILS=3      # consecutive /health failures -> restart (hang detection)
SRV=""

healthy() {
  "$PY" - <<'PYEOF' 2>/dev/null
import urllib.request, sys, os
port = os.environ.get("MEETING_PORT", "8765")
try:
    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
    sys.exit(0)
except Exception:
    sys.exit(1)
PYEOF
}

start() {
  pkill -9 -f "python -m app" 2>/dev/null
  lsof -ti tcp:$PORT 2>/dev/null | xargs kill -9 2>/dev/null
  sleep 1
  "$PY" -m app >>"$LOG" 2>&1 &
  SRV=$!
  echo "[supervise] started pid $SRV ($(date '+%H:%M:%S'))"
}

cleanup() { echo "[supervise] stopping"; kill -9 "$SRV" 2>/dev/null; exit 0; }
trap cleanup INT TERM

start
fails=0
while true; do
  sleep "$POLL"
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "[supervise] server exited — restarting"; start; fails=0; continue
  fi
  if healthy; then
    fails=0
  else
    fails=$((fails + 1))
    echo "[supervise] /health fail $fails/$MAX_FAILS"
    if [ "$fails" -ge "$MAX_FAILS" ]; then
      echo "[supervise] unresponsive — restarting"; start; fails=0
    fi
  fi
done
