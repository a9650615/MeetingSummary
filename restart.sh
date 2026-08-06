#!/usr/bin/env bash
# Restart MeetingSummary so the LATEST repo code is actually loaded.
#
# Why this exists: clicking the .app icon does NOT reload code. The launcher
# short-circuits when /health already answers ("already running -> just show the
# panel"), so a running server keeps serving stale code no matter how many times
# you relaunch. Only a real stop+start swaps the code. Run this after editing
# any .py to pick the change up:  ./restart.sh
#
# The stop half is ms_stop_all (lifecycle.sh) — the same one stop.sh and
# /shutdown use. Restart used to hand-roll a four-pattern kill list that missed
# the floatpanel and the model helpers, so "restarting" repeatedly left a live
# supervisor behind and the new one bowed out with "already running — exiting".
set -u
cd "$(dirname "$0")"
source ./lifecycle.sh
HERE="$MS_HERE"
PORT="$MS_PORT"

echo "==> stopping everything…"
ms_stop_all
for i in $(seq 1 20); do            # ms_stop_all is TERM-then-KILL; let it settle
  ms_is_stopped && break
  sleep 0.5
done
ms_is_stopped || { echo "still not fully stopped — check 'pgrep -fl floatpanel|python -m app'" >&2; exit 1; }

echo "==> starting supervisor (loads current repo code)…"
MEETING_PORT="$PORT" nohup ./supervise.sh >>"$HERE/launcher.log" 2>&1 &

echo "==> waiting for health on :$PORT …"
for i in $(seq 1 40); do
  code=$(curl -sS -o /dev/null -w "%{http_code}" -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then echo "up: HTTP 200 (${i}x)"; exit 0; fi
  sleep 2
done
echo "still not healthy after ~80s — check launcher.log" >&2
exit 1
