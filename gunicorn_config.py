# =============================================================================
# GUNICORN CONFIGURATION
# =============================================================================
# IMPORTANT: Chỉ dùng 1 worker để đảm bảo GPU queue hoạt động đúng
# GPU tasks cần được xử lý tuần tự để tránh OOM

import os

# Server socket
bind = f"0.0.0.0:{os.getenv('API_PORT', '2803')}"
backlog = 2048

# Worker processes
# QUAN TRỌNG: Chỉ dùng 1 worker vì:
# 1. GPU queue cần được share giữa các requests
# 2. Chỉ có 16GB GPU RAM, không thể chạy nhiều tasks đồng thời
# 3. Dùng threads thay vì processes để xử lý concurrent requests
workers = 1
threads = 4  # Xử lý 4 requests đồng thời trong cùng 1 process
worker_class = "gthread"  # Threaded worker để hỗ trợ concurrent requests
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
# Với 1 worker, preload không cần thiết nhưng vẫn giữ để load models 1 lần
preload_app = False  # Tắt preload để đảm bảo threads được khởi tạo đúng

# Restart workers sau N requests để tránh memory leak
max_requests = 1000
max_requests_jitter = 50

# Graceful timeout - allow more time for cleanup
graceful_timeout = 120

# Limit request line size
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
