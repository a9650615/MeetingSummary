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

# Single-instance: a 2nd supervisor would fight the 1st (each start() kills the
# port holder) -> endless restart loop -> live dies. Refuse to start a duplicate.
LOCK="/tmp/meeting_supervise.$PORT.pid"
# Atomic create (noclobber) — TOCTOU-safe vs a near-simultaneous 2nd launch.
if ! (set -C; echo $$ > "$LOCK") 2>/dev/null; then
  if kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "[supervise] already running (pid $(cat "$LOCK")) on port $PORT — exiting"
    exit 0
  fi
  echo $$ > "$LOCK"   # stale lock (dead pid) -> take it over
fi
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

WATCH=""
cleanup() { echo "[supervise] stopping"; kill -9 "$SRV" "$WATCH" 2>/dev/null; rm -f "$LOCK"; exit 0; }
trap cleanup INT TERM

start
# Meeting watcher: notifies you to record when the mic goes in use (a real call).
start_watch() { "$PY" meeting_watch.py >>"$LOG" 2>&1 & WATCH=$!;
  echo "[supervise] meeting watcher pid $WATCH"; }
start_watch
fails=0
while true; do
  sleep "$POLL"
  # Heartbeat the single-instance lock. The startup guard is TOCTOU-racy: a 2nd
  # supervisor can slip past during the 1st's restart gap (lock momentarily stale),
  # then its start() pkills the healthy app -> flap war -> app intermittently
  # unreachable. Re-check ownership each loop: if the lock now names a different
  # LIVE pid, I'm the duplicate -> bow out (don't kill its app, just leave).
  owner=$(cat "$LOCK" 2>/dev/null || true)
  if [ -n "$owner" ] && [ "$owner" != "$$" ] && kill -0 "$owner" 2>/dev/null; then
    echo "[supervise] another instance (pid $owner) owns port $PORT — exiting"
    kill -9 "$SRV" "$WATCH" 2>/dev/null; exit 0
  fi
  [ "$owner" = "$$" ] || echo $$ > "$LOCK"   # missing/stale -> reclaim
  # keep the watcher alive too — it had no respawn, so a single crash killed
  # meeting detection for the whole session.
  kill -0 "$WATCH" 2>/dev/null || { echo "[supervise] watcher died — restarting"; start_watch; }
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
