#!/usr/bin/env bash
# Deploy the remote-store server to the Azure VM over `az ssh`. rsync the repo
# subset it needs (store.py + viewer/ + server/), build + run docker compose.
# Own container + port, isolated from acp's on :5555. Data/model volumes live
# only on the VM and survive redeploys.
set -euo pipefail
VM_IP="${VM_IP:-10.102.0.7}"
HOST_PORT="${HOST_PORT:-5556}"
REMOTE_DIR="${REMOTE_DIR:-meeting_store}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_CFG="/tmp/az_ssh_cfg_${VM_IP}"
COMPOSE="docker compose -f server/docker/docker-compose.yml"

echo "==> az ssh config ($VM_IP)"
az ssh config --ip "$VM_IP" --file "$SSH_CFG" --overwrite >/dev/null
run() { az ssh vm --ip "$VM_IP" -- "$@"; }

echo "==> rsync (store.py + viewer/ + server/) -> $VM_IP:$REMOTE_DIR/"
rsync -az --delete -e "ssh -F $SSH_CFG" \
  --exclude '__pycache__' --exclude '*.pyc' \
  --exclude data --exclude models \
  "$REPO_ROOT/store.py" "$REPO_ROOT/viewer" "$REPO_ROOT/server" \
  "$VM_IP:$REMOTE_DIR/"

echo "==> build + recreate (HOST_PORT=$HOST_PORT)"
run 'cd '"$REMOTE_DIR"' && HOST_PORT='"$HOST_PORT"' '"$COMPOSE"' up -d --build'

echo "==> health"
run 'cd '"$REMOTE_DIR"' && '"$COMPOSE"' ps | tail -2
    for i in $(seq 1 15); do
      code=$(curl -sSL -o /dev/null -w "%{http_code}" http://localhost:'"$HOST_PORT"'/health 2>/dev/null || echo 000)
      [ "$code" = "200" ] && { echo "HTTP 200"; break; }
      sleep 2
    done
    [ "$code" = "200" ] || echo "HTTP $code (not healthy after 30s)"'
echo "==> done: http://$VM_IP:$HOST_PORT"
