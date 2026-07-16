"""Guard: backends.py must not import `live` at module top — live imports MLX
(Apple-only), and `import backends` must stay clean on the CPU-only Linux VM
that runs the FireRed worker."""
import os


def test_no_module_top_live_import():
    path = os.path.join(os.path.dirname(__file__), "..", "backends.py")
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines:
        assert not line.startswith("from live import"), \
            f"module-top 'from live import' found: {line!r}"
