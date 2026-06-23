#!/usr/bin/env python3
"""Meeting watcher — Notion/Granola-style. Polls for known meeting-app processes;
when one starts and you're not already recording, fires a macOS notification
linking to the live page. Run in the background:  python meeting_watch.py &

ponytail: process-based detection (Zoom/Teams/Webex native — the common case).
Browser-tab meetings (Meet) and raw mic-open need a CoreAudio/Swift helper — add
MEETING_WATCH_APPS to extend the list, or wire that helper later.
"""
import json
import os
import subprocess
import time
import urllib.request

URL = "http://127.0.0.1:8000/live"
HEALTH = "http://127.0.0.1:8000/health"
APPS = (os.environ.get("MEETING_WATCH_APPS")
        or "zoom.us,MSTeams,Microsoft Teams,Webex,RingCentral,Around,Gather,Discord").split(",")
POLL_S = int(os.environ.get("MEETING_WATCH_POLL_S", "12"))


def meeting_app_running():
    for app in APPS:
        app = app.strip()
        if app and subprocess.run(["pgrep", "-i", "-f", app],
                                  capture_output=True).returncode == 0:
            return app
    return None


def is_recording():
    try:
        with urllib.request.urlopen(HEALTH, timeout=3) as r:
            return json.load(r).get("recording", False)
    except Exception:
        return False  # server down -> definitely not recording


def notify(app):
    title = "📝 偵測到會議"
    msg = f"{app} 正在進行 — 開始錄製? {URL}"
    if subprocess.run(["which", "terminal-notifier"], capture_output=True).returncode == 0:
        subprocess.run(["terminal-notifier", "-title", title, "-message", msg,
                        "-open", URL, "-sound", "Glass"])
    else:
        subprocess.run(["osascript", "-e",
                        f'display notification "{msg}" with title "{title}" sound name "Glass"'])


def main():
    notified = False
    while True:
        app = meeting_app_running()
        if app and not is_recording():
            if not notified:
                notify(app)
                notified = True
        elif not app:
            notified = False  # meeting ended -> re-arm for the next one
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
