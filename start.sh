#!/bin/bash
export PORT=${PORT:-8001}
echo "Starting Server on port $PORT..."
python server.py &
echo "Starting Agent..."
exec python agent.py dev
