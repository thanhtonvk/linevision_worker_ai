# 🚀 Quick Start Guide - Gunicorn

## ✅ Server đã chạy thành công!

Gunicorn đã được cấu hình và test thành công. Dưới đây là các lệnh để chạy server:

## Cách chạy

### 1️⃣ Chạy đơn giản (Khuyến nghị cho development)

```bash
gunicorn app:app --bind 0.0.0.0:2803 --workers 1 --timeout 300
```

### 2️⃣ Chạy với nhiều workers (Production)

```bash
gunicorn app:app --bind 0.0.0.0:2803 --workers 4 --timeout 300
```

### 3️⃣ Chạy với config file

```bash
gunicorn --config gunicorn_config.py app:app
```

### 4️⃣ Chạy với script

```bash
./start_server.sh
```

### 5️⃣ Chạy trong background

```bash
nohup gunicorn app:app --bind 0.0.0.0:2803 --workers 4 --timeout 300 > server.log 2>&1 &
```

## Kiểm tra server

### Test API

```bash
# Health check
curl http://localhost:2803/api/health

# Hoặc mở browser
open http://localhost:2803
```

### Xem logs (nếu chạy background)

```bash
tail -f server.log
```

### Dừng server

```bash
# Tìm process
ps aux | grep gunicorn

# Kill process
pkill -f gunicorn

# Hoặc kill theo PID
kill -9 <PID>
```

## Thông số quan trọng

- **Port**: 2803 (đã cấu hình trong settings.py)
- **Workers**: Số lượng process xử lý requests
  - 1 worker: Cho development/testing
  - 4-8 workers: Cho production (tùy CPU cores)
- **Timeout**: 300 giây (5 phút) - đủ cho video processing

## Lưu ý

⚠️ **Lỗi bạn gặp trước đó** có thể do:

1. Chưa chỉ định app module: `app:app`
2. Chưa có bind address: `--bind 0.0.0.0:2803`
3. Config file có vấn đề

✅ **Giải pháp**: Dùng lệnh đơn giản ở trên là chạy được ngay!

## Test API check_var

```bash
curl -X POST http://localhost:2803/api/check_var \
  -F "video=@path/to/video.mp4"
```
