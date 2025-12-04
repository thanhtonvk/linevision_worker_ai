#!/bin/bash

# =============================================================================
# START SERVER SCRIPT
# =============================================================================

echo "🚀 Starting LineVision Worker AI Server..."

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads
mkdir -p output

# Start Gunicorn
echo "🔥 Starting Gunicorn server..."
gunicorn --config gunicorn_config.py app:app
