#!/bin/bash

# Start backend
echo "Starting NeuroBridge Backend..."
uvicorn neurobridge.api:app --host 0.0.0.0 --port 8000 --reload &

# Start frontend
echo "Starting NeuroBridge Frontend..."
npm run dev
