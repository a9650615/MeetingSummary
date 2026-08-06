#!/usr/bin/env bash
# THE process-lifecycle contract for MeetingSummary. Sourced by supervise.sh,
# stop.sh and restart.sh, and shelled out to by app.py's /shutdown — so there is
# exactly ONE definition of "what this app consists of" and "how to stop it".
#
# Why this file exists: the kill list used to be hand-rolled in three places
# (stop.sh, restart.sh, /shutdown) and all three were incomplete in the same way
# — they missed the floatpanel, the ANE helper, the qwen3cpp daemon and the lock
# file. Quitting therefore left orphans holding CoreML models, and the surviving
# panel relaunched the whole tree ~7.5s later, so "quit" never actually quit.
#
# ponytail: plain bash, no deps. Kills are by PID where we have one and by
# PATH-SCOPED pattern otherwise, never by a bare program name — `pkill -f
# "python -m app"` was machine-wide and is how two supervisors killed each
# other's healthy servers.
set -u

# ${BASH_SOURCE[0]:-$0}: this file is sourced by bash scripts, but also gets
# sourced from an interactive zsh, where BASH_SOURCE is unset and `set -u` would
# abort the whole thing on line 1.
MS_HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
MS_PORT="${MEETING_PORT:-8765}"
MS_LOCK="/tmp/meeting_supervise.$MS_PORT.pid"
# Presence of this file means "a human asked us to quit". The floatpanel checks
# it before relaunching the backend, which is what breaks the resurrection cycle
# panel -> launcher -> bootstrap -> supervise -> server -> panel.
MS_QUIT="/tmp/meetingsummary-quit.$MS_PORT"

# --- lock -------------------------------------------------------------------

# Echo the PID owning the lock, but ONLY if that PID is really a supervisor.
# `kill -0` alone was the bug: once macOS recycled a leaked lock's PID onto an
# unrelated process, every launch printed "already running" and the app could
# never start again.
ms_lock_owner() {
  local pid
  pid="$(cat "$MS_LOCK" 2>/dev/null || true)"
  [ -n "${pid:-}" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  ps -o command= -p "$pid" 2>/dev/null | grep -q "supervise.sh" || return 1
  echo "$pid"
}

# Take the lock for $$. Returns 1 (and echoes the owner) if a REAL supervisor
# already holds it. Atomic on the happy path; the stale-takeover path re-checks
# under the same noclobber guard instead of a bare `echo >`, which used to let
# two launches both "take over" a dead lock.
ms_take_lock() {
  if (set -C; echo $$ >"$MS_LOCK") 2>/dev/null; then
    return 0
  fi
  local owner
  if owner="$(ms_lock_owner)"; then
    echo "$owner"
    return 1
  fi
  rm -f "$MS_LOCK"                       # provably stale -> drop and retry once
  (set -C; echo $$ >"$MS_LOCK") 2>/dev/null || { owner="$(ms_lock_owner)" && echo "$owner"; return 1; }
  return 0
}

ms_release_lock() { rm -f "$MS_LOCK"; }

# --- stop -------------------------------------------------------------------

_ms_kill_pids() {  # SIGTERM, brief grace, then SIGKILL — the server needs TERM
  local pids="$1" i                      # to run its handler and reap its own children
  [ -n "$pids" ] || return 0
  kill $pids 2>/dev/null
  for i in 1 2 3 4 5 6; do
    sleep 0.25
    kill -0 $pids 2>/dev/null || return 0
  done
  kill -9 $pids 2>/dev/null
}

ms_stop_all() {
  touch "$MS_QUIT"                       # tell the panel this is intentional

  # 1. The panel first: it is the one process that can resurrect everything else.
  _ms_kill_pids "$(pgrep -f 'floatpanel' 2>/dev/null | tr '\n' ' ')"

  # 2. Supervisor before its children, so it can't restart them mid-teardown.
  _ms_kill_pids "$(pgrep -f "$MS_HERE/supervise.sh|supervise\.sh" 2>/dev/null | tr '\n' ' ')"

  # 3. Server + watcher + bootstrap. TERM first: app.py's handler reaps the ANE
  #    helper and the qwen3cpp daemon, which nothing used to clean up.
  _ms_kill_pids "$(pgrep -f 'python -m app|meeting_watch\.py|bootstrap\.py' 2>/dev/null | tr '\n' ' ')"

  # 4. Model helpers, path-scoped so a second checkout is never touched. These
  #    survive a SIGKILLed parent by reparenting to launchd, each holding a
  #    loaded CoreML/GGUF model.
  _ms_kill_pids "$(pgrep -f "$MS_HERE/swift/qwen3-ane" 2>/dev/null | tr '\n' ' ')"
  _ms_kill_pids "$(pgrep -f "$MS_HERE/qwen3_cpp_daemon.py" 2>/dev/null | tr '\n' ' ')"

  # 5. Whatever still holds the port, by PID (not by pattern).
  _ms_kill_pids "$(lsof -ti "tcp:$MS_PORT" 2>/dev/null | tr '\n' ' ')"

  ms_release_lock
  rm -f "/tmp/meetingsummary-floatpanel-$MS_PORT.lock"
}

# Everything is gone and the port is free?
ms_is_stopped() {
  [ -z "$(lsof -ti "tcp:$MS_PORT" 2>/dev/null)" ] &&
    [ -z "$(pgrep -f 'python -m app|supervise\.sh' 2>/dev/null)" ]
}

# Called by supervise.sh at startup: a new run means we're not quitting anymore.
ms_clear_quit() { rm -f "$MS_QUIT"; }
