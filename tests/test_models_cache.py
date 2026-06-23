import os

from app import _scan_model_cache, _safe_model_path


def test_scan_lists_entries_with_size(tmp_path):
    root = tmp_path / "hub"
    (root / "models--a").mkdir(parents=True)
    (root / "models--a" / "f.bin").write_bytes(b"\x00" * 2048)
    (root / "m.gguf").write_bytes(b"\x00" * 1024)
    got = _scan_model_cache([str(root)])
    names = {e["name"]: e for e in got}
    assert "models--a" in names and "m.gguf" in names
    assert names["models--a"]["size_mb"] >= 0  # dir size summed
    assert all(e["path"].startswith(str(root)) for e in got)


def test_safe_model_path_blocks_traversal(tmp_path):
    root = tmp_path / "hub"
    (root / "models--x").mkdir(parents=True)
    assert _safe_model_path(str(root / "models--x"), [str(root)])
    assert not _safe_model_path(str(root), [str(root)])          # the root itself
    assert not _safe_model_path("/etc/passwd", [str(root)])      # outside
    assert not _safe_model_path(str(root / ".." / "evil"), [str(root)])  # traversal
