#!/usr/bin/env bash
# Sanctum — run the full demo (server + frontend)
set -euo pipefail

echo "=== Sanctum Demo ==="

# Ensure we're in the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Start the FastAPI server in the background
echo "Starting server on :8000..."
uvicorn src.server.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Start the frontend dev server
echo "Starting frontend on :5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Cleanup on exit
cleanup() {
    echo "Shutting down..."
    kill $SERVER_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
}
trap cleanup EXIT

echo ""
echo "=== Sanctum is running ==="
echo "  Server:   http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo "  Health:   http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop."

wait
