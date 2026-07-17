import importlib


def test_seam_present_and_guarded():
    src = open("app.py", encoding="utf-8").read()
    # the seam must be a try-import that never hard-fails the base app
    assert "plugins.remote_store" in src
    assert "REMOTE_PLUGIN" in src
    assert "/remote/push/" in src
    # button gated on the render-time remote_store setting, not unconditional
    assert "remote_enabled" in src
    assert 'get_setting("remote_store"' in src


def test_plugin_imports_standalone():
    # base-build safety: importing the plugin must not require the Apple stack
    mod = importlib.import_module("plugins.remote_store")
    assert hasattr(mod, "enabled") and hasattr(mod, "is_on")


def test_plugin_disabled_by_default(monkeypatch):
    # a normal release: no env flag -> enabled() OFF
    monkeypatch.delenv("REMOTE_STORE", raising=False)
    monkeypatch.delenv("REMOTE_STORE_URL_ENABLE", raising=False)
    mod = importlib.import_module("plugins.remote_store")
    assert mod.enabled() is False


def test_is_on_reads_setting(monkeypatch, tmp_path):
    # the UI toggle: is_on() true when the remote_store setting is "1"
    monkeypatch.delenv("REMOTE_STORE", raising=False)
    from store import Store
    mod = importlib.import_module("plugins.remote_store")
    s = Store(tmp_path / "s.db")
    assert mod.is_on(s) is False
    s.set_setting("remote_store", "1")
    assert mod.is_on(s) is True


def test_plugin_enabled_by_env_flag(monkeypatch):
    monkeypatch.setenv("REMOTE_STORE", "1")
    mod = importlib.import_module("plugins.remote_store")
    assert mod.enabled() is True
