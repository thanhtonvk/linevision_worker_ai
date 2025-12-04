# Deployment Guide - LineVision Worker AI

## Chạy với Gunicorn (Production)

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy với Gunicorn

#### Cách 1: Sử dụng script (Khuyến nghị)

```bash
chmod +x start_server.sh
./start_server.sh
```

#### Cách 2: Chạy trực tiếp

```bash
gunicorn --config gunicorn_config.py app:app
```

#### Cách 3: Chạy với custom settings

```bash
# Chạy với 4 workers
gunicorn -w 4 -b 0.0.0.0:2803 --timeout 300 app:app

# Chạy với nhiều workers hơn
gunicorn -w 8 -b 0.0.0.0:2803 --timeout 300 --worker-class sync app:app
```

### 3. Chạy trong background với nohup

```bash
nohup gunicorn --config gunicorn_config.py app:app > server.log 2>&1 &
```

### 4. Chạy với systemd (Linux)

Tạo file `/etc/systemd/system/linevision.service`:

```ini
[Unit]
Description=LineVision Worker AI
After=network.target

[Service]
User=your_username
WorkingDirectory=/path/to/linevision_worker_ai
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn --config gunicorn_config.py app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Sau đó:

```bash
sudo systemctl daemon-reload
sudo systemctl start linevision
sudo systemctl enable linevision
sudo systemctl status linevision
```

## Chạy với Flask Development Server

⚠️ **CHỈ dùng cho development, KHÔNG dùng cho production!**

```bash
python app.py
```

## Configuration

### Environment Variables

Tạo file `.env`:

```bash
API_HOST=0.0.0.0
API_PORT=2803
DEBUG=False
CLEANUP_HOURS=24
MAX_CONTENT_LENGTH=524288000
```

### Gunicorn Configuration

File `gunicorn_config.py` đã được cấu hình với:

- **Workers**: Tự động tính theo CPU cores (cpu_count \* 2 + 1)
- **Timeout**: 300 giây (5 phút) cho video processing
- **Max requests**: 1000 requests/worker để tránh memory leak
- **Preload app**: True để tiết kiệm memory

## Monitoring

### Xem logs

```bash
# Nếu chạy với systemd
sudo journalctl -u linevision -f

# Nếu chạy với nohup
tail -f server.log
```

### Kiểm tra status

```bash
# Check process
ps aux | grep gunicorn

# Check port
lsof -i :2803
```

### Stop server

```bash
# Nếu chạy với systemd
sudo systemctl stop linevision

# Nếu chạy với nohup
pkill -f gunicorn
```

## Performance Tips

1. **Tăng số workers** nếu server có nhiều CPU cores
2. **Tăng timeout** nếu xử lý video lớn
3. **Sử dụng nginx** làm reverse proxy phía trước
4. **Enable caching** cho static files
5. **Monitor memory usage** và restart workers định kỳ

## Nginx Reverse Proxy (Optional)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 500M;

    location / {
        proxy_pass http://127.0.0.1:2803;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

## Troubleshooting

### Port already in use

```bash
# Tìm process đang dùng port
lsof -i :2803

# Kill process
kill -9 <PID>
```

### Workers timeout

- Tăng `timeout` trong `gunicorn_config.py`
- Giảm số `workers` nếu thiếu RAM

### Out of memory

- Giảm số `workers`
- Tăng `max_requests` để restart workers thường xuyên hơn
- Kiểm tra memory leak trong code
