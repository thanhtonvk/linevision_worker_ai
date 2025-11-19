# Tennis Analysis System

Hệ thống phân tích tennis với tracking người và pose estimation sử dụng YOLO.

## 🎯 Tính năng chính

- **Ball Detection**: Phát hiện và tracking bóng tennis
- **Person Tracking**: Tracking người chơi qua các frame
- **Pose Estimation**: Phân tích tư thế đánh bóng
- **Technique Analysis**: Phát hiện lỗi kỹ thuật tennis
- **Court Accuracy**: Thống kê độ chính xác cú đánh (trong sân/ngoài sân)
- **Velocity Analysis**: Phân tích vận tốc bóng
- **Visualization**: Tạo video và biểu đồ chi tiết

## 📁 Cấu trúc project

```
tennis_analysis/
├── main.py                 # Script chính để chạy
├── tennis_analyzer.py      # Class chính TennisAnalyzer
├── ball_detector.py        # Class BallDetector
├── person_tracker.py       # Class PersonTracker
├── visualization.py        # Class TennisVisualizer
├── requirements.txt        # Dependencies
└── README.md              # Hướng dẫn sử dụng
```

## 🚀 Cài đặt

1. **Clone repository:**
```bash
git clone <repository-url>
cd tennis_analysis
```

2. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

3. **Tải models:**
- `ball_best.pt` - Model detect bóng tennis
- `yolov8m.pt` - Model detect người
- `yolov8n-pose.pt` - Model pose estimation

## 🎮 Sử dụng

### Chạy phân tích cơ bản:

```bash
python main.py
```

### Sử dụng trong code:

```python
from tennis_analyzer import TennisAnalyzer

# Khởi tạo analyzer
analyzer = TennisAnalyzer(
    ball_model_path="ball_best.pt",
    person_model_path="yolov8m.pt", 
    pose_model_path="yolov8n-pose.pt"
)

# Chạy phân tích
results = analyzer.analyze_tennis_match(
    video_path="path/to/video.mp4",
    ball_conf=0.7,
    person_conf=0.6,
    angle_threshold=50,
    intersection_threshold=100,
    court_bounds=(100, 100, 400, 500)
)

# Tạo visualizations
analyzer.create_visualizations(results, "output_prefix")
```

## 📊 Output files

Sau khi chạy, hệ thống sẽ tạo các file:

- `tennis_analysis_pose_analysis.mp4` - Video với pose tracking
- `tennis_analysis_technique_analysis.png` - Biểu đồ phân tích kỹ thuật
- `tennis_analysis_court_accuracy.png` - Biểu đồ độ chính xác cú đánh
- `tennis_analysis_detailed_report.txt` - Báo cáo chi tiết

## ⚙️ Cấu hình

### Tham số chính:

- `video_path`: Đường dẫn video input
- `ball_conf`: Confidence threshold cho ball detection (0.0-1.0)
- `person_conf`: Confidence threshold cho person detection (0.0-1.0)
- `angle_threshold`: Góc threshold cho direction change (độ)
- `intersection_threshold`: Threshold cho ball-person intersection (pixels)
- `court_bounds`: Ranh giới sân tennis (x1, y1, x2, y2)

### Models:

- **Ball Detection**: YOLO model được train riêng cho bóng tennis
- **Person Detection**: YOLOv8m (COCO dataset)
- **Pose Estimation**: YOLOv8n-pose (COCO keypoints)

## 📈 Thống kê được tạo

### Tổng quan:
- Tổng số frames và thời gian video
- Số người được track
- Tổng cú đánh, cú đánh bởi người, cú đánh chạm đất

### Từng người chơi:
- Số frames xuất hiện
- Tổng cú đánh và tỷ lệ đánh bóng
- Cú đánh trong sân vs ngoài sân
- Tỷ lệ chính xác
- Chi tiết từng cú đánh (frame, vị trí, trạng thái)

### Phân tích kỹ thuật:
- Lỗi khụy gối không đủ sâu
- Lỗi dẫm vạch khi đánh bóng
- Lỗi tư thế sau khi đánh bóng
- Góc vai, góc khụy gối, vị trí vợt

### Vận tốc bóng:
- Vận tốc trung bình, tối đa, tối thiểu
- Vận tốc tại từng cú đánh

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **"Video file not found"**
   - Kiểm tra đường dẫn video trong `main.py`
   - Đảm bảo file video tồn tại

2. **"Model file not found"**
   - Tải các model files cần thiết
   - Kiểm tra đường dẫn model trong code

3. **"No person detected"**
   - Giảm `person_conf` threshold
   - Kiểm tra chất lượng video

4. **"Memory error"**
   - Giảm `batch_size` trong BallDetector
   - Xử lý video ngắn hơn

## 📝 Lưu ý

- Video nên có chất lượng tốt và đủ ánh sáng
- Sân tennis nên rõ ràng và có contrast tốt
- Người chơi nên di chuyển trong tầm nhìn của camera
- Hệ thống hoạt động tốt nhất với video 30fps

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Hãy tạo issue hoặc pull request.

## 📄 License

MIT License