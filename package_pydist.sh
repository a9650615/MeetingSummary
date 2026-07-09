#!/usr/bin/env bash
# Build a fully self-contained, portable Python runtime with every server
# dependency pre-installed (core + mlx Apple-Silicon accel) — a python-build-
# standalone interpreter, not a venv, so it needs NO system Python at all on the
# user's machine: no Xcode CLT prompt from Apple's /usr/bin/python3 stub, no
# version mismatch, no network wait for pip at first launch. bootstrap.py
# extracts this straight into `.venv` (same bin/python + bin/pip layout every
# other script already expects) and skips venv-creation + pip-install entirely.
#   ./package_pydist.sh /abs/path/pydist-arm64.tar.gz
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-$PWD/pydist-arm64.tar.gz}"
PYDIST_URL="${PYDIST_URL:-https://github.com/astral-sh/python-build-standalone/releases/download/20260623/cpython-3.12.13%2B20260623-aarch64-apple-darwin-install_only.tar.gz}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
echo "下載 python-build-standalone…"
curl -fsSL -o "$WORK/pydist.tar.gz" "$PYDIST_URL"
tar xzf "$WORK/pydist.tar.gz" -C "$WORK"
PY="$WORK/python/bin/python3"
ln -sf python3 "$WORK/python/bin/python"

"$PY" -m pip install -q --upgrade pip
echo "安裝核心套件…"
"$PY" -m pip install -q --no-input -r requirements-app.txt
echo "安裝 mlx 加速…"
"$PY" -m pip install -q --no-input mlx-whisper mlx-lm mlx-audio \
  || echo "warning: mlx install failed — pydist 仍會打包，但缺 ANE/Metal 加速" >&2
"$PY" -m pip install -q --no-input pyobjc-framework-Cocoa || true

find "$WORK/python" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

( cd "$WORK" && tar czf "$OUT" python )
echo "packaged -> $OUT ($(du -h "$OUT" | cut -f1))"
