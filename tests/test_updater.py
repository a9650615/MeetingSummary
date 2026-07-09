import io
import json
import os
import zipfile

import updater


def test_vt_compares_numerically_not_lexically():
    assert updater._vt("0.1.9") < updater._vt("0.11.0")
    assert updater._vt("v0.11.0") == (0, 11, 0)


def test_check_picks_zip_asset_not_tarball_or_other_assets(monkeypatch, tmp_path):
    (tmp_path / "VERSION").write_text("0.1.0")
    release = {
        "tag_name": "v0.2.0",
        "tarball_url": "https://api.github.com/repos/x/x/tarball/v0.2.0",
        "assets": [
            {"name": "chatllm-runtime-arm64.tar.gz", "browser_download_url": "https://x/chatllm.tar.gz"},
            {"name": "MeetingSummary-v0.2.0.zip", "browser_download_url": "https://x/MeetingSummary-v0.2.0.zip"},
        ],
    }

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=10):
        return _Resp(json.dumps(release).encode())

    monkeypatch.setattr(updater.urllib.request, "urlopen", fake_urlopen)
    info = updater.check("x/x", str(tmp_path))
    assert info["has_update"] is True
    assert info["asset_url"] == "https://x/MeetingSummary-v0.2.0.zip"


def test_apply_without_bundle_path_falls_back_to_source_patch(monkeypatch, tmp_path):
    (tmp_path / "VERSION").write_text("0.1.0")
    monkeypatch.setattr(updater, "check", lambda repo, here: {
        "current": "0.1.0", "latest": "v0.2.0", "has_update": True,
        "tarball": None, "asset_url": "https://x/MeetingSummary-v0.2.0.zip",
    })
    info = updater.apply("x/x", str(tmp_path), bundle_path=None)
    assert info["applied"] is False  # no tarball either -> nothing to patch with
    assert "relaunching" not in info


def _make_release_zip(tmp_path):
    zpath = tmp_path / "release.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("MeetingSummary.app/Contents/MacOS/launcher", "#!/bin/sh\necho new\n")
        zf.writestr("MeetingSummary.app/Contents/Info.plist", "<plist/>")
    return zpath.read_bytes()


def test_apply_swaps_whole_bundle_and_marks_relaunching(monkeypatch, tmp_path):
    here = tmp_path / "wd"
    here.mkdir()
    (here / "VERSION").write_text("0.1.0")

    bundle = tmp_path / "MeetingSummary.app"
    (bundle / "Contents" / "MacOS").mkdir(parents=True)
    (bundle / "Contents" / "MacOS" / "launcher").write_text("#!/bin/sh\necho old\n")

    zip_bytes = _make_release_zip(tmp_path)

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url_or_req, timeout=10):
        return _Resp(zip_bytes)

    monkeypatch.setattr(updater, "check", lambda repo, here_: {
        "current": "0.1.0", "latest": "v0.2.0", "has_update": True,
        "tarball": None, "asset_url": "https://x/MeetingSummary-v0.2.0.zip",
    })
    monkeypatch.setattr(updater.urllib.request, "urlopen", fake_urlopen)

    info = updater.apply("x/x", str(here), bundle_path=str(bundle))

    assert info["applied"] is True
    assert info["relaunching"] is True
    assert info["bundle_path"] == str(bundle)
    new_launcher = bundle / "Contents" / "MacOS" / "launcher"
    assert "new" in new_launcher.read_text()
    assert not os.path.exists(str(bundle) + ".old")


def test_apply_rolls_back_bundle_on_move_failure(monkeypatch, tmp_path):
    here = tmp_path / "wd"
    here.mkdir()
    (here / "VERSION").write_text("0.1.0")

    bundle = tmp_path / "MeetingSummary.app"
    (bundle / "Contents" / "MacOS").mkdir(parents=True)
    (bundle / "Contents" / "MacOS" / "launcher").write_text("#!/bin/sh\necho old\n")

    monkeypatch.setattr(updater, "check", lambda repo, here_: {
        "current": "0.1.0", "latest": "v0.2.0", "has_update": True,
        "tarball": None, "asset_url": "https://x/MeetingSummary-v0.2.0.zip",
    })

    def boom(*a, **k):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(updater.urllib.request, "urlopen", boom)

    info = updater.apply("x/x", str(here), bundle_path=str(bundle))

    assert info["applied"] is False
    assert "error" in info
    # bundle must still be exactly where it was — never left missing/half-written
    assert (bundle / "Contents" / "MacOS" / "launcher").read_text() == "#!/bin/sh\necho old\n"
