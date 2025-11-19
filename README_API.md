# Tennis Analysis Flask API

API Flask để phân tích video tennis với khả năng trả về kết quả dưới dạng JSON và links để truy cập hình ảnh/video qua trình duyệt.

## 🚀 Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Đảm bảo có các model files
- `ball_best.pt`
- `yolov8m.pt`
- `yolov8n-pose.pt`

## 📖 Sử dụng

### Khởi động server

```bash
python flask_api.py
```

Server sẽ chạy tại: `http://localhost:5000`

## 🔌 API Endpoints

### 1. Health Check
```
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Tennis Analysis API",
  "timestamp": "2025-11-19T22:45:00"
}
```

### 2. Analyze Video (Endpoint chính)
```
POST /api/analyze
```

**Parameters (form-data):**
- `video` (file, required): Video file cần phân tích
- `ball_conf` (float, optional): Confidence threshold cho ball detection (default: 0.7)
- `person_conf` (float, optional): Confidence threshold cho person detection (default: 0.6)
- `angle_threshold` (float, optional): Ngưỡng góc (default: 50)
- `intersection_threshold` (float, optional): Ngưỡng giao điểm (default: 100)
- `court_bounds` (string, optional): Tọa độ sân tennis "x1,y1,x2,y2" (default: "100,100,400,500")

**Response:**
```json
{
  "success": true,
  "data": {
    "request_id": "abc123...",
    "timestamp": "2025-11-19T22:45:00",
    "highest_speed_info": {
      "frame": 150,
      "time_seconds": 5.0,
      "velocity": 450.5,
      "person_id": 1,
      "shoulder_angle": 135.5,
      "knee_bend_angle": 45.2,
      "cropped_image_url": "http://localhost:5000/files/abc123/highest_speed_player_xyz.jpg"
    },
    "best_players": [
      {
        "rank": 1,
        "player_id": 1,
        "score": 85.5,
        "in_court_ratio": 0.95,
        "avg_ball_speed": 420.3,
        "avg_shoulder_angle": 130.2,
        "avg_knee_bend_angle": 42.5,
        "total_hits": 25,
        "cropped_image_url": "http://localhost:5000/files/abc123/player_1_rank_1_crop_xyz.jpg"
      }
    ],
    "match_statistics": {
      "rally_ratio": 0.85,
      "in_court_ratio": 0.92,
      "out_court_ratio": 0.08,
      "total_hits": 50,
      "total_in_court": 46,
      "total_out_court": 4
    },
    "visualization_video_url": "http://localhost:5000/files/abc123/visualization_abc123.mp4"
  }
}
```

### 3. Serve Files
```
GET /files/<folder>/<filename>
```

Endpoint này phục vụ các file hình ảnh và video. Bạn có thể mở trực tiếp trong trình duyệt.

**Ví dụ:**
```
http://localhost:5000/files/abc123/highest_speed_player_xyz.jpg
http://localhost:5000/files/abc123/visualization_abc123.mp4
```

### 4. Get All Results
```
GET /api/results/<request_id>
```

Lấy danh sách tất cả các files của một request.

**Response:**
```json
{
  "request_id": "abc123",
  "files": {
    "highest_speed_player_xyz.jpg": "http://localhost:5000/files/abc123/highest_speed_player_xyz.jpg",
    "player_1_rank_1_crop_xyz.jpg": "http://localhost:5000/files/abc123/player_1_rank_1_crop_xyz.jpg",
    "visualization_abc123.mp4": "http://localhost:5000/files/abc123/visualization_abc123.mp4"
  }
}
```

## 🧪 Test API

### Sử dụng test script

```bash
python test_api.py
```

### Sử dụng cURL

```bash
# Health check
curl http://localhost:5000/api/health

# Analyze video
curl -X POST http://localhost:5000/api/analyze \
  -F "video=@crop_video/part_000.mp4" \
  -F "ball_conf=0.7" \
  -F "person_conf=0.6" \
  -F "angle_threshold=50" \
  -F "intersection_threshold=100" \
  -F "court_bounds=100,100,400,500"
```

### Sử dụng Postman

1. Tạo POST request đến `http://localhost:5000/api/analyze`
2. Chọn Body → form-data
3. Thêm key `video` với type `File` và chọn video file
4. Thêm các parameters khác (optional)
5. Send request

## 📁 Cấu trúc thư mục

```
LineVision/Research/
├── flask_api.py              # Flask API server
├── test_api.py               # Test script
├── requirements_api.txt      # Dependencies
├── README_API.md            # Documentation (file này)
├── tennis_analysis_module.py # Module phân tích
├── uploads/                  # Thư mục lưu video upload
└── outputs/                  # Thư mục lưu kết quả
    └── <request_id>/        # Mỗi request có folder riêng
        ├── *.jpg            # Hình ảnh crop
        └── *.mp4            # Video visualization
```

## 🔒 Lưu ý

1. **File size limit**: Mặc định là 500MB, có thể thay đổi trong `flask_api.py`
2. **Supported formats**: mp4, avi, mov, mkv
3. **Storage**: Files được lưu trong thư mục `outputs/` theo request_id
4. **URLs**: Tất cả URLs trả về đều có thể mở trực tiếp trong trình duyệt
5. **CORS**: Nếu cần gọi từ frontend khác domain, thêm Flask-CORS

## 🌐 Deploy lên Production

### Sử dụng Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 flask_api:app
```

### Sử dụng Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements_api.txt .
RUN pip install -r requirements_api.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "flask_api:app"]
```

## 📞 Support

Nếu có vấn đề, hãy kiểm tra:
1. Server đã chạy chưa (`python flask_api.py`)
2. Model files có tồn tại không
3. Video path có đúng không
4. Port 5000 có bị chiếm không

## 📝 License

MIT License
