#!/bin/bash

# =============================================================================
# START SERVER SCRIPT
# =============================================================================

echo "🚀 Starting LineVision download Server..."


# Start Gunicorn
echo "🔥 Starting Gunicorn server..."
gunicorn --config gunicorn_config.py file_server:app

