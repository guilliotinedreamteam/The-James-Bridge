#!/bin/bash
echo "Starting NeuroBridge System..."

# Start Backend
echo "Starting FastAPI Backend on port 8000..."
uvicorn neurobridge.api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start Frontend
echo "Starting React Frontend..."
npm run dev &
FRONTEND_PID=$!

echo "NeuroBridge is running."
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:5173" # Vite default

# Trap to kill both on exit
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT

wait
