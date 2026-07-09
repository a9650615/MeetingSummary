"""First-run bootstrap (stdlib only — runs on system python3, no deps needed).

Launch-first UX: binds the port immediately and serves a self-refreshing
"準備中…" page, installs the venv + deps in a background thread (skipped if a
working .venv already exists -> instant), then hands the port to the real server
via exec. The browser, opened straight away, shows progress then flips to the app.
"""
import http.server
import json
import os
import subprocess
import sys
import threading

PORT = int(os.environ.get("MEETING_PORT", "8765"))
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "bin", "python")
PIP = os.path.join(HERE, ".venv", "bin", "pip")
REQ = "requirements-app.txt" if os.path.exists(os.path.join(HERE, "requirements-app.txt")) else "requirements.txt"
LOG = os.path.join(HERE, "setup.log")
state = {"step": "啟動中…", "line": "", "done": False, "error": ""}


def _floatpanel_installed():
    # Mirrors backends.floatpanel_bin() without importing it (bootstrap runs on
    # system python3, pre-venv). When installed, the .app's own launcher already
    # spawns the native panel directly (build_app.sh's launch_panel — so TCC
    # attributes recording to the app, not python) so we skip opening the
    # browser here.
    p = os.environ.get("FLOATPANEL_BIN")
    if p and os.path.exists(p):
        return True
    return os.path.exists(os.path.join(HERE, "swift", "floatpanel", ".build", "release", "floatpanel"))


def _is_apple_silicon_hw():
    # True even under Rosetta — checks the HARDWARE, not the process arch.
    try:
        return subprocess.run(["sysctl", "-n", "hw.optional.arm64"],
                              capture_output=True, text=True).stdout.strip() == "1"
    except Exception:
        return False


def _fetch_pydist():
    """Fetch the prebuilt Python (python-build-standalone, every dep incl. mlx
    already installed) from the latest release and lay it down as `.venv` — the
    exact layout (bin/python, bin/pip) every other script already assumes, so
    nothing downstream needs to know this isn't a real venv. This is what makes
    a clean Mac work with ZERO system Python: no Xcode CLT prompt from Apple's
    /usr/bin/python3 stub, no version mismatch, no pip wait at first launch.
    Apple Silicon only (that's the only prebuilt asset). Never raises — returns
    False on any failure so the caller falls back to create-venv-and-pip-install."""
    if not _is_apple_silicon_hw():
        return False
    try:
        import io
        import json as _json
        import shutil
        import tarfile
        import urllib.request
        repo = os.environ.get("MEETING_REPO", "a9650615/MeetingSummary")
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/releases/latest",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "MeetingSummary"})
        rel = _json.load(urllib.request.urlopen(req, timeout=10))
        url = next((a["browser_download_url"] for a in rel.get("assets", [])
                    if a.get("name") == "pydist-arm64.tar.gz"), None)
        if not url:
            return False
        state["step"] = "下載預裝 Python 執行環境…"
        data = urllib.request.urlopen(url, timeout=600).read()
        tmp = os.path.join(HERE, ".venv.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(tmp)
        extracted = os.path.join(tmp, "python")
        if not os.path.isfile(os.path.join(extracted, "bin", "python3")):
            shutil.rmtree(tmp, ignore_errors=True)
            return False
        venv = os.path.join(HERE, ".venv")
        shutil.rmtree(venv, ignore_errors=True)
        shutil.move(extracted, venv)
        shutil.rmtree(tmp, ignore_errors=True)
        link = os.path.join(venv, "bin", "python")
        if not os.path.exists(link):
            os.symlink("python3", link)
        try:  # baked-in already -> skip the redundant pip-install-core step below
            import hashlib
            with open(os.path.join(HERE, REQ), "rb") as rf:
                h = hashlib.md5(rf.read()).hexdigest()
            with open(os.path.join(venv, ".reqhash"), "w") as mf:
                mf.write(h)
        except Exception:
            pass
        return True
    except Exception as e:
        state["line"] = "pydist: " + str(e)[:120]
        return False


def _pip(args, log=False):
    """Run pip; stream to the page+log if log=True. Returns returncode (never raises)."""
    try:
        if not log:
            return subprocess.run([PIP, *args], cwd=HERE, capture_output=True).returncode
        with open(LOG, "a") as f:
            p = subprocess.Popen([PIP, *args], cwd=HERE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                f.write(line); f.flush()
                if line.strip():
                    state["line"] = line.strip()[:140]
            p.wait()
            return p.returncode
    except Exception as e:
        state["line"] = str(e)[:140]
        return 1


def _reqs_changed():
    # Re-run pip when requirements-app.txt changed since the last successful install.
    try:
        import hashlib
        with open(os.path.join(HERE, REQ), "rb") as rf:
            h = hashlib.md5(rf.read()).hexdigest()
        marker = os.path.join(HERE, ".venv", ".reqhash")
        if os.path.exists(marker):
            with open(marker) as mf:
                if mf.read().strip() == h:
                    return False, h
        return True, h
    except Exception:
        return False, ""


def setup():
    # Best-effort throughout — never block. Install what works, skip what doesn't;
    # the server degrades gracefully (missing mlx -> live falls back / disabled).
    try:
        # NO auto-update on launch — updating is a manual decision (模型管理 →
        # 檢查更新 → 更新並重啟). Launch always runs the current code as-is.
        if not os.path.exists(PY):
            state["step"] = "建立 Python 環境…"
            if not _fetch_pydist():
                subprocess.run([sys.executable, "-m", "venv", ".venv"], cwd=HERE,
                               capture_output=True)
                _pip(["install", "-q", "--upgrade", "pip"])
        core_missing = subprocess.run([PY, "-c", "import fastapi,uvicorn"], cwd=HERE,
                                      capture_output=True).returncode != 0
        changed, h = _reqs_changed()        # requirements updated since last install?
        if core_missing or changed:
            state["step"] = "安裝核心套件(首次約數分鐘)…"
            if _pip(["install", "--no-input", "-r", REQ], log=True) == 0 and h:
                try:
                    with open(os.path.join(HERE, ".venv", ".reqhash"), "w") as _f:
                        _f.write(h)
                except Exception:
                    pass
        # mlx = Apple-Silicon Metal ASR; best-effort. Failure is fine — skip it.
        if _is_apple_silicon_hw() and subprocess.run(
                [PY, "-c", "import mlx_whisper"], cwd=HERE, capture_output=True).returncode != 0:
            state["step"] = "安裝 mlx 加速(可選,失敗會略過)…"
            # mlx-audio = MLX-native Qwen3-ASR (Metal, ~4x faster than the chatllm GGUF)
            _pip(["install", "--no-input", "mlx-whisper", "mlx-lm", "mlx-audio"], log=True)
        if not os.path.exists(os.path.join(HERE, "micbusy")):
            subprocess.run(["swiftc", "micbusy.swift", "-o", "micbusy",
                            "-framework", "CoreAudio"], cwd=HERE, capture_output=True)
        # pyobjc = the native floating meeting-detection HUD (Notion-style).
        # best-effort: without it meeting_watch.py falls back to notifications.
        if _is_apple_silicon_hw() and subprocess.run(
                [PY, "-c", "import AppKit"], cwd=HERE, capture_output=True).returncode != 0:
            _pip(["install", "--no-input", "pyobjc-framework-Cocoa"], log=True)
    except Exception as e:
        state["line"] = "setup: " + str(e)[:120]   # note it, but still start
    state["step"] = "啟動服務…"
    state["done"] = True


_PAGE = ("<!doctype html><meta charset=utf-8><title>準備中</title>"
         "<body style='font:16px -apple-system,sans-serif;padding:48px;max-width:560px;margin:auto'>"
         "<h2>📝 Meeting·Summary</h2><p id=s>啟動中…</p>"
         "<div id=bar style='height:6px;background:#eee;border-radius:3px;overflow:hidden'>"
         "<div style='height:100%;width:40%;background:#5b54e6;animation:i 1.2s linear infinite'></div></div>"
         "<pre id=l style='color:#888;font-size:12px;white-space:pre-wrap;margin-top:14px'></pre>"
         "<style>@keyframes i{0%{margin-left:-40%}100%{margin-left:100%}}</style>"
         "<script>const S=document.getElementById('s'),L=document.getElementById('l');"
         "setInterval(async()=>{"
         # ALWAYS probe /health first: the moment the real server has taken over the
         # port (handoff complete) it answers JSON -> reload. The bootstrap server
         # serves HTML for /health (.json() throws), so this only fires post-handoff.
         # (Don't gate this on having seen /_setup's done flag — once the bootstrap
         # server is shut down, /_setup fails and we'd never get here = stuck on 啟動中.)
         "try{let j=await(await fetch('/health')).json();if(j&&j.status==='ok'){location.reload();return;}}catch(e){}"
         # still booting -> show setup progress (best-effort; fails after handoff).
         "try{let r=await(await fetch('/_setup')).json();"
         "if(r.error){S.innerHTML='<b style=color:#e5484d>啟動失敗</b>';"
         "document.getElementById('bar').style.display='none';"
         "L.style.color='#e5484d';L.textContent=r.error;return;}"
         "S.textContent=r.done?'啟動服務中…':r.step;L.textContent=r.line||'';"
         "}catch(e){S.textContent='啟動服務中…';}"
         "},800)</script>")


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/_setup":
            b = json.dumps(state).encode()
            self.send_response(200); self.send_header("content-type", "application/json")
            self.end_headers(); self.wfile.write(b)
            return
        self.send_response(200); self.send_header("content-type", "text/html;charset=utf-8")
        self.end_headers(); self.wfile.write(_PAGE.encode())

    def log_message(self, *a):
        pass


def _health_ok():
    try:
        import urllib.request
        r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
        return json.load(r).get("status") == "ok"
    except Exception:
        return False


def _bind():
    """Bind the port, reclaiming it from a stale/stuck holder. We only get here when
    the launcher's /health probe failed, so a bound port means a dead-but-bound or
    hung previous server — clicking the icon must still open the app, not crash on
    'Address already in use'. If a concurrent launch already brought up a HEALTHY
    server (race), defer to it (open browser + exit)."""
    import time
    try:
        return http.server.HTTPServer(("127.0.0.1", PORT), H)
    except OSError:
        if _health_ok():                       # someone else won the race + is healthy
            subprocess.run(["open", f"http://127.0.0.1:{PORT}"])
            sys.exit(0)
        subprocess.run(f"lsof -ti tcp:{PORT} | xargs kill -9", shell=True,
                       capture_output=True)    # stale holder -> reclaim
        time.sleep(0.6)
        return http.server.HTTPServer(("127.0.0.1", PORT), H)


def main():
    import platform
    # On Apple Silicon but launched under Rosetta (x86_64)? Re-exec natively arm64
    # BEFORE binding — else the venv/mlx install would be x86 and fail.
    if _is_apple_silicon_hw() and platform.machine() == "x86_64":
        os.execvp("arch", ["arch", "-arm64", sys.executable, os.path.abspath(__file__)])
    # Detach into our own session so the server survives the .app launcher exiting.
    # The launcher backgrounds us then exits; LaunchServices then reaps the .app's
    # process group — which would kill this server (nohup blocks SIGHUP, not the
    # group reap). setsid() puts us in a fresh session, immune to that. (No-op /
    # harmless if we're already a session leader, e.g. run straight from a shell.)
    try:
        os.setsid()
    except OSError:
        pass
    srv = _bind()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    # Open the browser ourselves (we're in the surviving session) — the launcher
    # has already exited, so it can't. The progress page shows now, then reloads to
    # the app once the real server takes over. Skip it when the native panel is
    # installed: the .app's launcher already spawned it directly (launch_panel,
    # called right alongside this bootstrap), so opening the browser too would
    # double the UI. Panel absent -> browser is both the progress page and the
    # entry point.
    if not _floatpanel_installed():
        subprocess.run(["open", f"http://127.0.0.1:{PORT}"], capture_output=True)
    setup()
    # Can the real server even import? If yes, hand off. If not, it's a genuine
    # failure -> show the log on the page (don't exec a doomed server / hang silently).
    chk = subprocess.run([PY, "-c", "import app"], cwd=HERE, capture_output=True, text=True)
    if chk.returncode == 0:
        srv.shutdown()
        os.chdir(HERE)
        os.execv("/bin/bash", ["/bin/bash", "supervise.sh"])
    else:
        tail = ""
        try:
            with open(LOG) as lf:
                tail = lf.read()[-2000:]
        except Exception:
            pass
        state["error"] = ("啟動失敗 — 無法載入伺服器。\n\n=== import 錯誤 ===\n"
                          + (chk.stderr or "")[-1000:]
                          + ("\n\n=== setup.log ===\n" + tail if tail else ""))
        import time
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
