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


def _is_apple_silicon_hw():
    # True even under Rosetta — checks the HARDWARE, not the process arch.
    try:
        return subprocess.run(["sysctl", "-n", "hw.optional.arm64"],
                              capture_output=True, text=True).stdout.strip() == "1"
    except Exception:
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


def _local_version():
    try:
        return open(os.path.join(HERE, "VERSION")).read().strip()
    except Exception:
        return "0.0.0"


def _vt(s):
    import re
    return tuple(int(x) for x in re.findall(r"\d+", s)[:3]) or (0,)


def _release_update():
    # Preferred online update: poll GitHub Releases; if the latest tag > local
    # VERSION, download that release's source tarball and overwrite the code
    # (venv/models/data preserved). Versioned + controlled; no git needed.
    repo = os.environ.get("MEETING_REPO", "")  # "owner/name"; unset -> skip
    if not repo:
        return
    try:
        import io
        import shutil
        import tarfile
        import tempfile
        import urllib.request
        state["step"] = "檢查更新…"
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/releases/latest",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "MeetingSummary"})
        rel = json.load(urllib.request.urlopen(req, timeout=10))
        tag = rel.get("tag_name", "")
        if _vt(tag) <= _vt(_local_version()):
            return
        state["step"] = f"下載更新 {tag}…"
        data = urllib.request.urlopen(rel["tarball_url"], timeout=120).read()
        tmp = tempfile.mkdtemp()
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(tmp)
        top = os.path.join(tmp, os.listdir(tmp)[0])  # owner-repo-<sha>/
        keep = {".venv", "data", "models", ".git", "setup.log", "dist"}
        for item in os.listdir(top):
            if item in keep:
                continue
            src, dst = os.path.join(top, item), os.path.join(HERE, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass  # offline / API limit / no release -> run the current code


def _git_update():
    # Online update: if the workdir is a git clone with a remote, fast-forward to
    # the latest pushed code (pristine deploy -> reset --hard, no conflicts). Code
    # is tiny; venv/models/data are gitignored so they're untouched. Offline -> skip.
    if not os.path.isdir(os.path.join(HERE, ".git")):
        return
    try:
        has_remote = subprocess.run(["git", "-C", HERE, "remote"],
                                    capture_output=True, text=True).stdout.strip()
        if not has_remote:
            return
        state["step"] = "檢查更新…"
        if subprocess.run(["git", "-C", HERE, "fetch", "--quiet"],
                          capture_output=True, timeout=30).returncode == 0:
            subprocess.run(["git", "-C", HERE, "reset", "--hard", "--quiet", "@{u}"],
                           capture_output=True, timeout=15)
    except Exception:
        pass


def _reqs_changed():
    # Re-run pip when requirements-app.txt changed since the last successful install.
    try:
        import hashlib
        h = hashlib.md5(open(os.path.join(HERE, REQ), "rb").read()).hexdigest()
        marker = os.path.join(HERE, ".venv", ".reqhash")
        if os.path.exists(marker) and open(marker).read().strip() == h:
            return False, h
        return True, h
    except Exception:
        return False, ""


def setup():
    # Best-effort throughout — never block. Install what works, skip what doesn't;
    # the server degrades gracefully (missing mlx -> live falls back / disabled).
    try:
        _release_update()   # preferred: GitHub Releases (versioned)
        _git_update()       # dev convenience: pull if workdir is a git clone
        if not os.path.exists(PY):
            state["step"] = "建立 Python 環境…"
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
                    open(os.path.join(HERE, ".venv", ".reqhash"), "w").write(h)
                except Exception:
                    pass
        # mlx = Apple-Silicon Metal ASR; best-effort. Failure is fine — skip it.
        if _is_apple_silicon_hw() and subprocess.run(
                [PY, "-c", "import mlx_whisper"], cwd=HERE, capture_output=True).returncode != 0:
            state["step"] = "安裝 mlx 加速(可選,失敗會略過)…"
            _pip(["install", "--no-input", "mlx-whisper", "mlx-lm"], log=True)
        if not os.path.exists(os.path.join(HERE, "micbusy")):
            subprocess.run(["swiftc", "micbusy.swift", "-o", "micbusy",
                            "-framework", "CoreAudio"], cwd=HERE, capture_output=True)
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
         "<script>let _done=false;const S=document.getElementById('s'),L=document.getElementById('l');"
         "setInterval(async()=>{"
         "if(!_done){try{let r=await(await fetch('/_setup')).json();"
         "if(r.error){S.innerHTML='<b style=color:#e5484d>啟動失敗</b>';"
         "document.getElementById('bar').style.display='none';"
         "L.style.color='#e5484d';L.textContent=r.error;return;}"
         "S.textContent=r.step;L.textContent=r.line||'';if(r.done)_done=true;"
         "}catch(e){}return;}"
         # done -> wait for the REAL server to actually answer /health (JSON) before
         # reloading, so we never reload into the dead port during handoff.
         "S.textContent='啟動服務中…';"
         "try{let j=await(await fetch('/health')).json();if(j&&j.status==='ok')location.reload();}catch(e){}"
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


def main():
    import platform
    # On Apple Silicon but launched under Rosetta (x86_64)? Re-exec natively arm64
    # BEFORE binding — else the venv/mlx install would be x86 and fail.
    if _is_apple_silicon_hw() and platform.machine() == "x86_64":
        os.execvp("arch", ["arch", "-arm64", sys.executable, os.path.abspath(__file__)])
    srv = http.server.HTTPServer(("127.0.0.1", PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
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
            tail = open(LOG).read()[-2000:]
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
