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


def setup():
    import platform
    try:
        if platform.machine() != "arm64":
            state["error"] = f"需 Apple Silicon（此機為 {platform.machine()}）；mlx 無法安裝。"
            return
        if not os.path.exists(PY):
            state["step"] = "建立 Python 環境…"
            r = subprocess.run(["python3", "-m", "venv", ".venv"], cwd=HERE,
                               capture_output=True, text=True)
            if r.returncode != 0:
                state["error"] = "建立 venv 失敗：" + (r.stderr or "")[-200:]
                return
            subprocess.run([PIP, "install", "-q", "--upgrade", "pip"], cwd=HERE)
        ready = subprocess.run([PY, "-c", "import fastapi,uvicorn,mlx_whisper"],
                               cwd=HERE, capture_output=True).returncode == 0
        if not ready:
            state["step"] = "下載並安裝相依套件(首次約數分鐘)…"
            with open(LOG, "w") as logf:
                p = subprocess.Popen([PIP, "install", "--no-input", "-r", REQ],
                                     cwd=HERE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:        # stream pip output -> page + log
                    logf.write(line); logf.flush()
                    line = line.strip()
                    if line:
                        state["line"] = line[:140]
                p.wait()
            if p.returncode != 0:
                state["error"] = f"安裝失敗（returncode {p.returncode}），詳見 {LOG}"
                return
        if not os.path.exists(os.path.join(HERE, "micbusy")):
            subprocess.run(["swiftc", "micbusy.swift", "-o", "micbusy",
                            "-framework", "CoreAudio"], cwd=HERE)
    except Exception as e:
        state["error"] = str(e)
        return
    state["step"] = "啟動服務…"
    state["done"] = True


_PAGE = ("<!doctype html><meta charset=utf-8><title>準備中</title>"
         "<body style='font:16px -apple-system,sans-serif;padding:48px;max-width:560px;margin:auto'>"
         "<h2>📝 Meeting·Summary</h2><p id=s>啟動中…</p>"
         "<div id=bar style='height:6px;background:#eee;border-radius:3px;overflow:hidden'>"
         "<div style='height:100%;width:40%;background:#5b54e6;animation:i 1.2s linear infinite'></div></div>"
         "<pre id=l style='color:#888;font-size:12px;white-space:pre-wrap;margin-top:14px'></pre>"
         "<style>@keyframes i{0%{margin-left:-40%}100%{margin-left:100%}}</style>"
         "<script>setInterval(async()=>{try{let r=await(await fetch('/_setup')).json();"
         "if(r.error){document.getElementById('s').innerHTML='<b style=color:#e5484d>'+r.error+'</b>';"
         "document.getElementById('bar').style.display='none';return;}"
         "document.getElementById('s').textContent=r.step;"
         "document.getElementById('l').textContent=r.line||'';"
         "if(r.done)setTimeout(()=>location.reload(),1500)}catch(e){location.reload()}},1000)</script>")


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


def main():
    srv = http.server.HTTPServer(("127.0.0.1", PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    setup()
    if state["error"]:
        # keep serving the error page so the user sees what failed (don't exec a
        # broken server). They fix it (e.g. wrong arch / see setup.log) and relaunch.
        import time
        while True:
            time.sleep(3600)
    srv.shutdown()  # free the port for the real server
    os.chdir(HERE)
    # hand off: replace this process with the supervised real server on the same port
    os.execv("/bin/bash", ["/bin/bash", "supervise.sh"])


if __name__ == "__main__":
    main()
