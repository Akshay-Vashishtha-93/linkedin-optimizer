#!/bin/bash
# LinkedIn Optimizer — Start Dashboard
# Runs server.py which serves dashboard + handles Apify sync

cd "$(dirname "$0")"

PORT="${PORT:-8080}"

if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "Port $PORT is already in use."
  echo "Start on another port with: PORT=8081 ./start.sh"
  exit 1
fi

echo "Starting LinkedIn Optimizer..."
echo "Dashboard: http://localhost:$PORT/dashboard.html"
echo "Press Ctrl+C to stop"
echo ""

PORT="$PORT" python3 server.py
