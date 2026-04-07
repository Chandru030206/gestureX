#!/usr/bin/env bash
set -euo pipefail

# Start both backend (uvicorn) and frontend (http.server) for GestureX.
# Uses the project's venv python discovered at runtime if possible.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer explicit venv python if present
VENV_PY="/Users/chandrurajinikanth/Desktop/gesture project/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  VENV_PY="$(command -v python3 || command -v python)"
fi

echo "Using python: $VENV_PY"

echo "Stopping any processes on ports 8000 and 3000..."
lsof -ti :8000 | xargs -r kill -9 || true
lsof -ti :3000 | xargs -r kill -9 || true

echo "Starting backend (uvicorn) from $ROOT_DIR/backend ..."
cd "$ROOT_DIR/backend"
nohup "$VENV_PY" -m uvicorn app:app --reload --host 0.0.0.0 --port 8000 > "$ROOT_DIR/backend/uvicorn.out" 2>&1 &
echo $! > "$ROOT_DIR/backend/uvicorn.pid"

echo "Starting frontend static server from $ROOT_DIR/frontend on port 3000..."
cd "$ROOT_DIR/frontend"
nohup "$VENV_PY" -m http.server 3000 > "$ROOT_DIR/frontend/http.out" 2>&1 &
echo $! > "$ROOT_DIR/frontend/http.pid"

sleep 1

echo "Checking backend health..."
if curl -sS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "Backend is up: http://127.0.0.1:8000/health"
  curl -sS http://127.0.0.1:8000/health | sed -n '1,200p'
else
  echo "Backend did not respond on http://127.0.0.1:8000/health. See $ROOT_DIR/backend/uvicorn.out"
  tail -n 200 "$ROOT_DIR/backend/uvicorn.out" || true
fi

echo "Frontend should be available at http://localhost:3000"
echo "Logs: $ROOT_DIR/backend/uvicorn.out and $ROOT_DIR/frontend/http.out"

echo "Done."
