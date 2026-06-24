#!/usr/bin/env bash
# Stop MeetingSummary cleanly: supervisor + server + meeting watcher, free the port.
#   ./stop.sh
PORT="${MEETING_PORT:-8765}"
pkill -f "supervise.sh" 2>/dev/null
pkill -f "python -m app" 2>/dev/null
pkill -f "meeting_watch.py" 2>/dev/null
pkill -f "bootstrap.py" 2>/dev/null
lsof -ti "tcp:$PORT" 2>/dev/null | xargs kill -9 2>/dev/null
echo "stopped — port $PORT freed"
