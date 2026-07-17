import importlib


def test_seam_present_and_guarded():
    src = open("app.py", encoding="utf-8").read()
    # the seam must be a try-import that never hard-fails the base app
    assert "plugins.remote_store" in src
    assert "REMOTE_PLUGIN" in src
    # feature must be gated on enabled(), not registered unconditionally
    assert "_remote_plugin.enabled()" in src
    assert "/remote/push/" in src


def test_plugin_imports_standalone():
    # base-build safety: importing the plugin must not require the Apple stack
    mod = importlib.import_module("plugins.remote_store")
    assert hasattr(mod, "enabled")


def test_plugin_disabled_by_default(monkeypatch):
    # a normal release: no flag -> feature OFF
    monkeypatch.delenv("REMOTE_STORE", raising=False)
    monkeypatch.delenv("REMOTE_STORE_URL_ENABLE", raising=False)
    mod = importlib.import_module("plugins.remote_store")
    assert mod.enabled() is False


def test_plugin_enabled_by_flag(monkeypatch):
    monkeypatch.setenv("REMOTE_STORE", "1")
    mod = importlib.import_module("plugins.remote_store")
    assert mod.enabled() is True
