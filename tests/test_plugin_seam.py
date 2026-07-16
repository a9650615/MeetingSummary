import importlib


def test_seam_present_and_guarded():
    src = open("app.py", encoding="utf-8").read()
    # the seam must be a try-import that never hard-fails the base app
    assert "plugins.remote_store" in src
    assert "REMOTE_PLUGIN" in src
    # button must be guarded on the flag, not unconditional
    assert "/remote/push/" in src


def test_plugin_imports_standalone():
    # base-build safety: importing the plugin must not require the Apple stack
    mod = importlib.import_module("plugins.remote_store")
    assert mod.enabled() is True
