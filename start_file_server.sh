#!/bin/bash

# =============================================================================
# START FILE SERVER SCRIPT
# =============================================================================
# Dedicated file server for serving analysis output files (videos, images)
# This reduces load on the main analysis API
# =============================================================================

echo "📁 Starting LineVision File Server..."

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads
mkdir -p outputs

# Set environment variables (optional - can override defaults)
export FILE_SERVER_HOST="${FILE_SERVER_HOST:-0.0.0.0}"
export FILE_SERVER_PORT="${FILE_SERVER_PORT:-2804}"

echo "🌐 File Server URL: http://${FILE_SERVER_HOST}:${FILE_SERVER_PORT}"
echo "🔗 Public URL: https://download-linevision.ngrok.app"

# Start File Server with Gunicorn
echo "🔥 Starting Gunicorn file server..."
gunicorn --bind ${FILE_SERVER_HOST}:${FILE_SERVER_PORT} \
         --workers 4 \
         --threads 2 \
         --access-logfile - \
         --error-logfile - \
         file_server:app
