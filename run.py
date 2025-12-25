# =============================================================================
# DETECT PERSON & BALL - VẼ LÊN VIDEO
# =============================================================================

import cv2
import os
from ultralytics import YOLO

# ====== CẤU HÌNH ======
VIDEO_PATH = "c.mp4"
OUTPUT_PATH = "c_detected.mp4"
BALL_MODEL_PATH = "models/ball_best.pt"
PERSON_MODEL_PATH = "yolo11m.pt"
BALL_CONF = 0.3
PERSON_CONF = 0.3

# ====== LOAD MODELS ======
print("Loading models...")
ball_model = YOLO(BALL_MODEL_PATH)
person_model = YOLO(PERSON_MODEL_PATH)
print("Models loaded!")

# ====== ĐỌC VIDEO ======
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {VIDEO_PATH}")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")

# ====== TẠO VIDEO WRITER (H.264 để giảm dung lượng) ======
# Sử dụng H.264 codec (avc1) thay vì mp4v để giảm dung lượng đáng kể
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Fallback nếu avc1 không hoạt động
if not out.isOpened():
    print("avc1 codec không khả dụng, thử XVID...")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    OUTPUT_PATH = OUTPUT_PATH.replace('.mp4', '.avi')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ====== XỬ LÝ TỪNG FRAME ======
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ball
    ball_results = ball_model.predict(frame, conf=BALL_CONF, verbose=False)

    # Detect person (class 0)
    person_results = person_model.predict(frame, conf=PERSON_CONF, verbose=False, classes=[0])

    # Vẽ ball detections (màu xanh lá)
    for result in ball_results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf.cpu().numpy()[0])

                # Vẽ circle tại center của ball
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max(10, (x2 - x1) // 2)
                cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                cv2.putText(frame, f"Ball {conf:.2f}", (cx - 30, cy - radius - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Vẽ person detections (màu đỏ)
    for result in person_results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf.cpu().numpy()[0])

                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Vẽ frame info
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Ghi frame
    out.write(frame)

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"Processed {frame_idx}/{total_frames} frames...")

# ====== CLEANUP ======
cap.release()
out.release()

print(f"\nDone! Output saved to: {OUTPUT_PATH}")
