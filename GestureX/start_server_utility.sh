#!/bin/bash
# start_server_utility.sh
# Activates the environment, starts the server, and verifies health.

# --- CONFIGURE THIS ---
VENV_PATH="/Users/chandrurajinikanth/Desktop/gesture project/.venv/bin/activate"
BACKEND_DIR="backend"
MAIN_FILE="app.py"
PORT=8000
# ----------------------

echo "🚀 Starting GestureX Backend..."

# 1. Activate Environment
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
else
    echo "⚠️  Venv not found at $VENV_PATH. Using system python."
fi

# 2. Start Server in Background
cd "$BACKEND_DIR"
nohup python3 -m uvicorn app:app --reload --host 0.0.0.0 --port 8000 > uvicorn.out 2>&1 &
SERVER_PID=$!
cd ..

echo "⏳ Waiting for initialization (10s)..."
sleep 10

# 3. Health Check
RESPONSE=$(curl -s http://127.0.0.1:$PORT/health)

if [[ $RESPONSE == *"healthy"* ]]; then
    echo "✅ Server is live! (PID: $SERVER_PID)"
    echo "Response: $RESPONSE"
else
    echo "❌ Server failed to start or health endpoint unreachable."
    echo "Check backend/uvicorn.out for error details."
    tail -n 10 backend/uvicorn.out
fi
