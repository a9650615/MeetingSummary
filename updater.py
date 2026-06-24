"""Online-update helpers (stdlib only — bootstrap uses these pre-venv too).

GitHub Releases: compare the latest tag to the local VERSION; download that
release's source tarball and overwrite the code (venv/models/data preserved).
Used by bootstrap.py (on launch) and app.py (manual 檢查更新 button)."""
import json
import os
import urllib.request

_KEEP = {".venv", "data", "models", ".git", "setup.log", "dist", ".venv-qwen314"}


def current_version(here):
    try:
        return open(os.path.join(here, "VERSION")).read().strip()
    except Exception:
        return "0.0.0"


def _vt(s):
    import re
    return tuple(int(x) for x in re.findall(r"\d+", s)[:3]) or (0,)


def check(repo, here):
    """{current, latest, has_update, tarball, error?} — never raises."""
    cur = current_version(here)
    if not repo:
        return {"current": cur, "latest": None, "has_update": False, "error": "no repo"}
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/releases/latest",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "MeetingSummary"})
        rel = json.load(urllib.request.urlopen(req, timeout=10))
        tag = rel.get("tag_name", "")
        return {"current": cur, "latest": tag, "has_update": _vt(tag) > _vt(cur),
                "tarball": rel.get("tarball_url")}
    except Exception as e:
        return {"current": cur, "latest": None, "has_update": False, "error": str(e)[:140]}


def apply(repo, here):
    """Download + extract the latest release over `here` (code only). Returns the
    check() dict + {applied: bool}. Never raises."""
    info = check(repo, here)
    if not info.get("has_update") or not info.get("tarball"):
        info["applied"] = False
        return info
    try:
        import io
        import shutil
        import tarfile
        import tempfile
        data = urllib.request.urlopen(info["tarball"], timeout=180).read()
        tmp = tempfile.mkdtemp()
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(tmp)
        top = os.path.join(tmp, os.listdir(tmp)[0])  # owner-repo-<sha>/
        for item in os.listdir(top):
            if item in _KEEP:
                continue
            src, dst = os.path.join(top, item), os.path.join(here, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        shutil.rmtree(tmp, ignore_errors=True)
        info["applied"] = True
    except Exception as e:
        info["applied"] = False
        info["error"] = str(e)[:140]
    return info
