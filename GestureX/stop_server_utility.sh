#!/bin/bash
# stop_server_utility.sh
# Safely kills the backend process and purges stale Python cache
# to ensure the environment is pristine for the new model weights.

# --- CONFIGURE THIS ---
PORT=8000
# ----------------------

echo "🛑 Stopping GestureX Backend on port $PORT..."

# 1. Find and kill process on PORT
PID=$(lsof -ti :$PORT)

if [ -z "$PID" ]; then
    echo "⚠️  No process found running on port $PORT."
else
    echo "🔪 Killing process $PID..."
    kill -9 $PID
    sleep 2
fi

# 2. Confirm port is free
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "❌ ERROR: Port $PORT is still occupied."
    exit 1
else
    echo "✅ Port $PORT is now free."
fi

# 3. Clear Bytecode Cache
echo "🧹 Clearing __pycache__ files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

echo "✨ Server stopped and cache cleared."
