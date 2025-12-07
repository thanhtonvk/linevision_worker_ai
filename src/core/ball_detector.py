# =============================================================================
# BALL DETECTOR CLASS - PHÁT HIỆN VÀ TRACKING BÓNG TENNIS
# =============================================================================

import cv2
import math
import numpy as np
from ultralytics import YOLO
from itertools import combinations

class BallDetector:
    """
    Class để detect và track bóng tennis trong video
    """
    
    def __init__(self, model_path="src/models/ball_best.pt", person_model_path="src/models/yolov8m.pt"):
        self.model = YOLO(model_path)
        self.person_model = YOLO(person_model_path)
        self.batch_size = 8
        self.conf = 0.7
        
    def read_video(self, video_path):
        """Đọc video và trả về danh sách frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def batch_frames(self, frames):
        """Chia frames thành các batch để xử lý"""
        return [frames[i:i+self.batch_size] for i in range(0, len(frames), self.batch_size)]

    def detect_positions(self, frames):
        """Detect vị trí bóng trong các frames"""
        batches = self.batch_frames(frames)
        positions = []
        for batch in batches:
            results = self.model.predict(batch, batch=self.batch_size, verbose=False, conf=self.conf)
            for res in results:
                if res.boxes is not None and len(res.boxes) > 0:
                    best_idx = res.boxes.conf.argmax()
                    x, y, w, h = res.boxes.xywh[best_idx].cpu().numpy()
                    positions.append((x, y))
                else:
                    positions.append((-1, -1))
        return positions

    def correct_positions(self, positions):
        """Sửa các vị trí bóng bất thường"""
        corrected_positions = []
        current_group = []

        def process_group(group):
            if len(group) < 2:
                return group
            mean_x = sum(x for x, _ in group) / len(group)
            mean_y = sum(y for _, y in group) / len(group)
            mean_point = (mean_x, mean_y)
            distances = [math.dist(p1, p2) for p1, p2 in combinations(group, 2)]
            avg_dist = sum(distances) / len(distances)
            corrected = []
            for (x, y) in group:
                d = math.dist((x, y), mean_point)
                if d > avg_dist:
                    dx, dy = x - mean_x, y - mean_y
                    length = math.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        dx, dy = dx/length, dy/length
                        corrected.append((mean_x + dx*avg_dist, mean_y + dy*avg_dist))
                    else:
                        corrected.append((mean_x, mean_y))
                else:
                    corrected.append((x, y))
            return corrected

        for i, pos in enumerate(positions):
            if pos == (-1, -1):
                if current_group:
                    corrected_positions.extend(process_group(current_group))
                    current_group = []
                corrected_positions.append(pos)
            else:
                current_group.append(pos)
        
        if current_group:
            corrected_positions.extend(process_group(current_group))
        
        return corrected_positions

    def smooth_positions(self, positions, window_size=5):
        """Làm mượt vị trí bóng"""
        if len(positions) < window_size:
            return positions
        
        smoothed = []
        for i in range(len(positions)):
            if positions[i] == (-1, -1):
                smoothed.append(positions[i])
                continue
            
            start = max(0, i - window_size // 2)
            end = min(len(positions), i + window_size // 2 + 1)
            window = [pos for pos in positions[start:end] if pos != (-1, -1)]
            
            if len(window) > 0:
                avg_x = sum(x for x, y in window) / len(window)
                avg_y = sum(y for x, y in window) / len(window)
                smoothed.append((avg_x, avg_y))
            else:
                smoothed.append(positions[i])
        
        return smoothed

    def interpolate_missing_positions(self, positions, max_gap=10):
        """Nội suy vị trí bóng bị thiếu"""
        interpolated = []
        i = 0
        
        while i < len(positions):
            if positions[i] != (-1, -1):
                interpolated.append(positions[i])
                i += 1
            else:
                # Tìm vị trí bắt đầu và kết thúc của gap
                start_idx = i
                while i < len(positions) and positions[i] == (-1, -1):
                    i += 1
                end_idx = i
                gap_size = end_idx - start_idx
                
                if gap_size <= max_gap and start_idx > 0 and end_idx < len(positions):
                    # Nội suy
                    prev_pos = interpolated[-1]
                    next_pos = positions[end_idx] if end_idx < len(positions) else prev_pos
                    
                    for j in range(gap_size):
                        ratio = (j + 1) / (gap_size + 1)
                        x = prev_pos[0] + ratio * (next_pos[0] - prev_pos[0])
                        y = prev_pos[1] + ratio * (next_pos[1] - prev_pos[1])
                        interpolated.append((x, y))
                else:
                    # Không nội suy, giữ nguyên
                    for j in range(gap_size):
                        interpolated.append((-1, -1))
        
        return interpolated

    def detect_persons(self, frames, person_conf=0.5):
        """Detect người trong frames"""
        batches = self.batch_frames(frames)
        person_detections = []
        
        for batch in batches:
            results = self.person_model.predict(batch, batch=self.batch_size, verbose=False, conf=person_conf)
            for res in results:
                frame_persons = []
                if res.boxes is not None and len(res.boxes) > 0:
                    for box in res.boxes:
                        if int(box.cls) == 0:  # person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf.cpu().numpy()[0]
                            if conf >= person_conf:
                                frame_persons.append({
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'conf': conf
                                })
                person_detections.append(frame_persons)
        
        return person_detections

    def check_ball_person_intersection(self, ball_pos, person_bboxes, threshold=80, dynamic_threshold=True):
        """Kiểm tra bóng có chạm người không"""
        if ball_pos == (-1, -1) or not person_bboxes:
            return False
        
        ball_x, ball_y = ball_pos
        
        for person in person_bboxes:
            x1, y1, x2, y2 = person['bbox']
            conf = person['conf']
            
            # Calculate person bbox dimensions
            person_width = x2 - x1
            person_height = y2 - y1
            person_size = max(person_width, person_height)
            
            # Dynamic threshold based on person size and confidence
            if dynamic_threshold:
                size_factor = min(2.0, max(0.5, person_size / 100))
                conf_factor = min(1.5, max(0.8, conf))
                dynamic_thresh = int(threshold * size_factor * conf_factor)
            else:
                dynamic_thresh = threshold
            
            # Check if ball is within dynamic threshold distance of person bbox
            if (x1 - dynamic_thresh <= ball_x <= x2 + dynamic_thresh and 
                y1 - dynamic_thresh <= ball_y <= y2 + dynamic_thresh):
                return True
        return False

    def get_enhanced_direction_change_flags(self, frames, positions, angle_threshold=45, person_conf=0.5, intersection_threshold=80):
        """Phân loại thay đổi hướng: bóng chạm người vs chạm đất"""
        print("Đang detect người trong video...")
        person_detections = self.detect_persons(frames, person_conf=person_conf)
        
        print("Đang phân tích thay đổi hướng...")
        change_points = set(self._detect_direction_changes(positions, angle_threshold))
        
        direction_flags = []
        person_hit_count = 0
        ground_hit_count = 0
        
        for i in range(len(positions)):
            if i in change_points:
                ball_pos = positions[i]
                persons = person_detections[i] if i < len(person_detections) else []
                
                if self.check_ball_person_intersection(ball_pos, persons, threshold=intersection_threshold):
                    direction_flags.append(2)  # Bóng được đánh bởi người
                    person_hit_count += 1
                else:
                    direction_flags.append(1)  # Bóng chạm đất
                    ground_hit_count += 1
            else:
                direction_flags.append(0)  # Không thay đổi hướng
        
        print(f"Phân loại hoàn tất:")
        print(f"- Bóng chạm người: {person_hit_count} lần")
        print(f"- Bóng chạm đất: {ground_hit_count} lần")
        print(f"- Tổng thay đổi hướng: {person_hit_count + ground_hit_count} lần")
                
        return direction_flags, person_detections

    def _detect_direction_changes(self, positions, angle_threshold=45, min_distance=20, min_velocity=10, min_frames_gap=5):
        """Phát hiện các điểm thay đổi hướng"""
        change_points = []
        clean_positions = [(i, p) for i, p in enumerate(positions) if p != (-1, -1)]

        if len(clean_positions) < 3:
            return change_points

        for i in range(2, len(clean_positions)):
            idx1, p1 = clean_positions[i-2]
            idx2, p2 = clean_positions[i-1]
            idx3, p3 = clean_positions[i]

            v1 = (p2[0]-p1[0], p2[1]-p1[1])
            v2 = (p3[0]-p2[0], p3[1]-p2[1])

            len1 = math.hypot(*v1)
            len2 = math.hypot(*v2)
            
            if len1 < min_velocity or len2 < min_velocity:
                continue
                
            dist1 = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            dist2 = math.hypot(p3[0]-p2[0], p3[1]-p2[1])
            if dist1 < min_distance or dist2 < min_distance:
                continue

            dot = v1[0]*v2[0] + v1[1]*v2[1]
            cos_theta = max(-1, min(1, dot / (len1 * len2)))
            angle = math.degrees(math.acos(cos_theta))

            if angle > angle_threshold:
                if len(change_points) == 0 or idx2 - change_points[-1] >= min_frames_gap:
                    change_points.append(idx2)

        return change_points

    def save_cropped_ball_video(self, frames, positions, output_path, crop_size=50, fps=30):
        """Lưu video với vùng crop quanh bóng"""
        cropped_frames = []
        prev_crop = None

        for i, frame in enumerate(frames):
            pos = positions[i] if i < len(positions) else (-1, -1)

            if pos == (-1, -1):
                if prev_crop is not None:
                    cropped_frames.append(prev_crop)
                continue

            x, y = int(pos[0]), int(pos[1])
            h, w = frame.shape[:2]
            x1 = max(0, x - crop_size)
            y1 = max(0, y - crop_size)
            x2 = min(w, x + crop_size)
            y2 = min(h, y + crop_size)

            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                if prev_crop is not None:
                    cropped_frames.append(prev_crop)
                continue

            prev_crop = cropped.copy()
            cropped_frames.append(cropped)

        if not cropped_frames:
            print("Không có frame nào để lưu!")
            return

        height, width = cropped_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in cropped_frames:
            out.write(frame)

        out.release()
        print(f"Đã lưu video: {output_path}")
