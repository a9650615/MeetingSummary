#!/usr/bin/env bash
# Keep the live server up: restart if the process dies OR if /health stops
# responding (hung / not doing work). Run in a terminal:  ./supervise.sh
# Stop with Ctrl-C. Env (ASR_MODEL, LIVE_MODEL, ...) is passed through.
#
# ponytail: a poll-loop supervisor, no deps. For survive-reboot/login, wrap it in
# the launchd plist below (see README note) — KeepAlive restarts the supervisor.
set -u
cd "$(dirname "$0")"
source ./lifecycle.sh   # THE process list + lock + stop, shared with stop.sh/restart.sh//shutdown
PY=.venv/bin/python
PORT="$MS_PORT"
export MEETING_PORT=$PORT   # server (app.py) + health check bind the same port
LOG=/tmp/meeting_server.log

# Single-instance. The old guard trusted `kill -0 $(cat lock)`, so once a
# SIGKILLed run leaked the pid file and macOS recycled that PID onto an unrelated
# process, every launch printed "already running" and the app could never start
# again. ms_take_lock verifies the PID really is a supervisor before believing it.
if ! owner="$(ms_take_lock)"; then
  echo "[supervise] already running (pid $owner) on port $PORT — exiting"
  exit 0
fi
ms_clear_quit               # a fresh run means we are no longer quitting
POLL=10          # seconds between health checks
MAX_FAILS=3      # consecutive /health failures -> restart (hang detection)
SRV=""
WATCH=""

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
  # Kill only OUR server, by PID. This used to be `pkill -9 -f "python -m app"`,
  # which is machine-wide: two supervisors each killed the other's healthy child,
  # restarted it, and flapped forever. TERM first, so app.py's exit handler gets
  # to reap the ANE helper and the qwen3cpp daemon instead of orphaning them.
  [ -n "$SRV" ] && _ms_kill_pids "$SRV"
  # Anything still holding the port goes by PID too, never by name.
  _ms_kill_pids "$(lsof -ti "tcp:$PORT" 2>/dev/null | tr '\n' ' ')"
  "$PY" -m app >>"$LOG" 2>&1 &
  SRV=$!
  echo "[supervise] started pid $SRV ($(date '+%H:%M:%S'))"
}

# EXIT as well as the signals: any normal return (including the duplicate-instance
# bow-out below) has to drop the lock, or the next launch inherits a stale one.
cleanup() {
  echo "[supervise] stopping"
  _ms_kill_pids "$SRV $WATCH"
  ms_release_lock
}
trap cleanup EXIT INT TERM HUP

start
# Meeting watcher: notifies you to record when the mic goes in use (a real call).
start_watch() { "$PY" meeting_watch.py >>"$LOG" 2>&1 & WATCH=$!;
  echo "[supervise] meeting watcher pid $WATCH"; }
start_watch
fails=0
while true; do
  sleep "$POLL"
  # Heartbeat the lock. If it now names a DIFFERENT live supervisor, I'm the
  # duplicate -> bow out. The EXIT trap reaps my own children; I never touch its.
  owner="$(ms_lock_owner || true)"
  if [ -n "$owner" ] && [ "$owner" != "$$" ]; then
    echo "[supervise] another instance (pid $owner) owns port $PORT — exiting"
    exit 0
  fi
  [ "$owner" = "$$" ] || echo $$ >"$MS_LOCK"   # missing/stale -> reclaim
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
