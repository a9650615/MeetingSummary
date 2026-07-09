"""Online-update helpers (stdlib only — bootstrap uses these pre-venv too).

GitHub Releases: compare the latest tag to the local VERSION. If we're running
inside a real .app bundle (bundle_path given), download that release's
MeetingSummary-*.zip asset — the full signed bundle, not source — and swap the
whole bundle in place, then relaunch it: the fresh launcher's own cold-start
rsync refreshes the Python source, nested floatpanel, everything, so we don't
need to know about any of that here. Without a bundle_path (dev checkout with
no .app) fall back to patching just the Python source from the release's
source tarball, same as before.
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
    """{current, latest, has_update, tarball, asset_url, error?} — never raises."""
    cur = current_version(here)
    if not repo:
        return {"current": cur, "latest": None, "has_update": False, "error": "no repo"}
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/releases/latest",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "MeetingSummary"})
        rel = json.load(urllib.request.urlopen(req, timeout=10))
        tag = rel.get("tag_name", "")
        asset_url = next((a.get("browser_download_url") for a in rel.get("assets", [])
                           if a.get("name", "").startswith("MeetingSummary-")
                           and a.get("name", "").endswith(".zip")), None)
        return {"current": cur, "latest": tag, "has_update": _vt(tag) > _vt(cur),
                "tarball": rel.get("tarball_url"), "asset_url": asset_url}
    except Exception as e:
        return {"current": cur, "latest": None, "has_update": False, "error": str(e)[:140]}


def _apply_bundle(info, bundle_path):
    """Swap the whole .app for the downloaded release zip. Extracts with `ditto`
    (not zipfile) to preserve the code signature's resource fork data. Renames the
    old bundle aside first so a failed move can roll back instead of leaving a
    half-replaced .app."""
    import shutil
    import subprocess
    import tempfile
    data = urllib.request.urlopen(info["asset_url"], timeout=180).read()
    zip_path = os.path.join(tempfile.mkdtemp(), "release.zip")
    with open(zip_path, "wb") as f:
        f.write(data)
    extract_dir = tempfile.mkdtemp()
    subprocess.run(["/usr/bin/ditto", "-x", "-k", zip_path, extract_dir], check=True)
    new_apps = [n for n in os.listdir(extract_dir) if n.endswith(".app")]
    if not new_apps:
        raise RuntimeError("release zip had no .app inside")
    new_app = os.path.join(extract_dir, new_apps[0])
    if not os.path.isfile(os.path.join(new_app, "Contents", "MacOS", "launcher")):
        raise RuntimeError("downloaded bundle missing launcher")
    backup = bundle_path.rstrip("/") + ".old"
    shutil.rmtree(backup, ignore_errors=True)
    os.rename(bundle_path, backup)
    try:
        shutil.move(new_app, bundle_path)
    except Exception:
        os.rename(backup, bundle_path)   # roll back — never leave bundle_path missing/half-written
        raise
    shutil.rmtree(backup, ignore_errors=True)
    info["relaunching"] = True
    info["bundle_path"] = bundle_path


def apply(repo, here, bundle_path=None):
    """Apply the latest release (whole-bundle swap when possible, else source-only
    patch). Returns the check() dict + {applied: bool}. Never raises."""
    info = check(repo, here)
    if not info.get("has_update"):
        info["applied"] = False
        return info
    if bundle_path and info.get("asset_url") and os.path.isfile(
            os.path.join(bundle_path, "Contents", "MacOS", "launcher")):
        try:
            _apply_bundle(info, bundle_path)
            info["applied"] = True
        except Exception as e:
            info["applied"] = False
            info["error"] = str(e)[:140]
        return info
    if not info.get("tarball"):
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
