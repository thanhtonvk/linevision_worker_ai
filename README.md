# LineVision Worker AI - Tennis Analysis API

Hệ thống phân tích video tennis sử dụng AI với khả năng tracking bóng, phát hiện người chơi, và phân tích kỹ thuật.

## 🌟 Tính năng

- **Ball Detection & Tracking**: Phát hiện và theo dõi bóng tennis trong video
- **Person Tracking**: Tracking người chơi qua các frame
- **Pose Estimation**: Phân tích tư thế và kỹ thuật đánh bóng
- **Technical Analysis**: Phân tích góc mở vai, góc khụy gối, độ chính xác cú đánh
- **Match Statistics**: Thống kê tỉ lệ đối kháng, bóng trong/ngoài sân
- **Visualization**: Tạo video visualization với annotations đầy đủ

## 📁 Cấu trúc dự án

```
linevision_worker_ai/
├── src/                          # Source code
│   ├── api/                      # API endpoints
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/                     # Business logic
│   │   ├── __init__.py
│   │   ├── ball_detector.py
│   │   ├── person_tracker.py
│   │   ├── tennis_analyzer.py
│   │   └── tennis_analysis_module.py
│   ├── visualization/            # Visualization
│   │   ├── __init__.py
│   │   └── visualizer.py
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── helpers.py
│       └── calib.py
├── config/                       # Configuration
│   ├── __init__.py
│   └── settings.py
├── models/                       # AI models (.pt files)
│   ├── ball_best.pt
│   ├── yolov8m.pt
│   ├── yolov8m-pose.pt
│   └── yolov8n-pose.pt
├── tests/                        # Tests
│   ├── __init__.py
│   ├── test_api.py
│   └── test_api_simple.py
├── examples/                     # Examples
│   └── example_usage.py
├── uploads/                      # Uploaded videos (auto-created)
├── outputs/                      # Analysis results (auto-created)
├── app.py                        # Main application
├── requirements.txt              # Dependencies
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd linevision_worker_ai
```

### 2. Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình environment variables

```bash
cp .env.example .env
# Chỉnh sửa .env nếu cần
```

### 5. Đảm bảo có model files

Đặt các file model (.pt) vào thư mục `models/`:
- `ball_best.pt` - Model phát hiện bóng
- `yolov8m.pt` - Model phát hiện người
- `yolov8n-pose.pt` - Model pose estimation

## 📖 Sử dụng

### Khởi động API server

```bash
python app.py
```

Server sẽ chạy tại `http://localhost:5000`

### API Endpoints

#### 1. Health Check

```bash
GET /api/health
```

Response:
```json
{
  "status": "healthy",
  "service": "Tennis Analysis API",
  "timestamp": "2025-12-03T23:00:00"
}
```

#### 2. Analyze Video

```bash
POST /api/analyze
Content-Type: multipart/form-data

Parameters:
- video (file, required): Video file
- ball_conf (float, optional): Ball detection confidence (default: 0.7)
- person_conf (float, optional): Person detection confidence (default: 0.6)
- angle_threshold (float, optional): Angle threshold (default: 50)
- intersection_threshold (float, optional): Intersection threshold (default: 100)
- court_bounds (string, optional): Court bounds as "x1,y1,x2,y2" (default: "100,100,400,500")
```

Response:
```json
{
  "request_id": "abc123...",
  "timestamp": "2025-12-03T23:00:00",
  "expires_at": "2025-12-04T23:00:00",
  "highest_speed_info": {
    "frame": 150,
    "time_seconds": 5.0,
    "velocity": 85.5,
    "person_id": 1,
    "shoulder_angle": 75.2,
    "knee_bend_angle": 145.8,
    "cropped_image_url": "http://localhost:5000/files/abc123.../highest_speed_player_xyz.jpg"
  },
  "best_players": [
    {
      "rank": 1,
      "player_id": 1,
      "score": 85.5,
      "in_court_ratio": 0.85,
      "avg_ball_speed": 75.2,
      "avg_shoulder_angle": 80.5,
      "avg_knee_bend_angle": 140.2,
      "total_hits": 25,
      "cropped_image_url": "http://localhost:5000/files/abc123.../player_1_rank_1_crop_xyz.jpg"
    }
  ],
  "match_statistics": {
    "rally_ratio": 0.45,
    "in_court_ratio": 0.82,
    "out_court_ratio": 0.18,
    "total_hits": 50,
    "total_in_court": 41,
    "total_out_court": 9
  },
  "visualization_video_url": "http://localhost:5000/files/abc123.../visualization_abc123.mp4"
}
```

#### 3. Get Results

```bash
GET /api/results/<request_id>
```

Response:
```json
{
  "request_id": "abc123...",
  "files": {
    "highest_speed_player_xyz.jpg": "http://localhost:5000/files/abc123.../highest_speed_player_xyz.jpg",
    "player_1_rank_1_crop_xyz.jpg": "http://localhost:5000/files/abc123.../player_1_rank_1_crop_xyz.jpg",
    "visualization_abc123.mp4": "http://localhost:5000/files/abc123.../visualization_abc123.mp4"
  }
}
```

#### 4. Serve Files

```bash
GET /files/<folder>/<filename>
```

### Example với curl

```bash
# Analyze video
curl -X POST http://localhost:5000/api/analyze \
  -F "video=@tennis_match.mp4" \
  -F "ball_conf=0.7" \
  -F "person_conf=0.6"
```

### Example với Python

```python
import requests

url = "http://localhost:5000/api/analyze"
files = {"video": open("tennis_match.mp4", "rb")}
data = {
    "ball_conf": 0.7,
    "person_conf": 0.6,
    "angle_threshold": 50
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(result)
```

## ⚠️ Lưu ý quan trọng

- **Auto Cleanup**: Files (images và videos) sẽ tự động bị xóa sau 24 giờ để tiết kiệm dung lượng
- **Download Files**: Hãy download kết quả quan trọng trong vòng 24 giờ
- **Cleanup Schedule**: Cleanup chạy mỗi 1 giờ trong background
- **Expiration Time**: Mỗi response có field `expires_at` cho biết khi nào files sẽ bị xóa

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/ -v

# Test API manually
python tests/test_api_simple.py
```

## 📝 Configuration

Chỉnh sửa `config/settings.py` hoặc `.env` để thay đổi cấu hình:

- Model paths
- Upload/Output folders
- API host và port
- Default parameters
- Cleanup settings

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

[Add your license here]

## 👥 Authors

[Add authors here]

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV
- Flask
