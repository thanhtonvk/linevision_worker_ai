# =============================================================================
# GUNICORN CONFIGURATION
# =============================================================================

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('API_PORT', '2803')}"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1  # Recommended formula
worker_class = "sync"
worker_connections = 1000
timeout = 3000  # 50 minutes - increased timeout for long video processing
keepalive = 2

# Use memory-based tmp directory for better performance
worker_tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"  # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "linevision_worker_ai"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (nếu cần)
# keyfile = None
# certfile = None

# Preload app để tiết kiệm memory
preload_app = True

# Restart workers sau N requests để tránh memory leak
max_requests = 1000
max_requests_jitter = 50

# Graceful timeout - allow more time for cleanup
graceful_timeout = 120

# Limit request line size
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
