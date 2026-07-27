#!/usr/bin/env bash
# Stop MeetingSummary — the WHOLE app, not just the server.
#   ./stop.sh
# The kill list lives in lifecycle.sh (ms_stop_all) so stop.sh, restart.sh and
# app.py's /shutdown can no longer drift apart: all three used to hand-roll it,
# and all three missed the floatpanel, the ANE helper and the qwen3cpp daemon —
# so "stop" left orphans holding models, and the surviving panel relaunched the
# whole tree ~7.5s later.
set -u
cd "$(dirname "$0")"
source ./lifecycle.sh
ms_stop_all
echo "stopped — port $MS_PORT freed"
