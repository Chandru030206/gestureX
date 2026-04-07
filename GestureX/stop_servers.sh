#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Stopping servers listening on 8000 and 3000..."
lsof -ti :8000 | xargs -r kill -9 || true
lsof -ti :3000 | xargs -r kill -9 || true

for f in "$ROOT_DIR/backend/uvicorn.pid" "$ROOT_DIR/frontend/http.pid"; do
  if [ -f "$f" ]; then
    pid=$(cat "$f" 2>/dev/null || true)
    if [ -n "$pid" ]; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$f"
  fi
done

echo "Stopped. Confirm with: lsof -i :8000 || true and lsof -i :3000 || true"
