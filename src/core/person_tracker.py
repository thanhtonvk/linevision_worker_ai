# =============================================================================
# PERSON TRACKER CLASS - TRACKING NGƯỜI VÀ POSE ESTIMATION
# =============================================================================

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math
import gc
from config.settings import settings


class PersonTracker:
    """
    Class để tracking người và phân tích pose trong tennis
    Optimized for 12GB GPU with batch inference
    """

    def __init__(
        self,
        pose_model_path="src/models/yolov8n-pose.pt",
        batch_size=16,  # Optimized for 12GB GPU
    ):
        # Chỉ sử dụng pose model - nó trả về cả bbox và keypoints
        # Không cần person_model riêng, giảm bộ nhớ GPU
        self.pose_model = YOLO(pose_model_path)
        self.batch_size = batch_size
        self.tracked_persons = {}  # {person_id: person_data}
        self.next_person_id = 1
        self.ball_hits_by_person = defaultdict(list)  # {person_id: [hit_data]}
        self.player_positions = defaultdict(list)  # {player_id: [(frame, x, y), ...]}
        self.max_frame_height = settings.max_frame_height
        self.enable_frame_resize = settings.enable_frame_resize

        # Move model to GPU if available
        import torch
        if torch.cuda.is_available():
            self.pose_model.to('cuda')

    def _resize_frame(self, frame):
        """
        Resize frame to reduce memory usage while maintaining aspect ratio
        Returns: (resized_frame, scale_factor)
        """
        if not self.enable_frame_resize:
            return frame, 1.0

        height, width = frame.shape[:2]
        if height <= self.max_frame_height:
            return frame, 1.0

        scale = self.max_frame_height / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        return resized, scale

    def _batch_frames(self, frames):
        """Chia frames thành các batch để xử lý"""
        return [
            frames[i : i + self.batch_size]
            for i in range(0, len(frames), self.batch_size)
        ]

    def _batch_pose_detection(self, frames, conf_threshold=0.5):
        """Batch pose detection cho tất cả frames
        Trả về cả person bboxes và keypoints từ pose model (gộp 2 trong 1)

        Args:
            frames: List of video frames
            conf_threshold: Confidence threshold for detection

        Returns:
            Tuple of (all_pose_detections, all_person_detections)
            - all_pose_detections: List of pose data for each frame
            - all_person_detections: List of person bboxes for each frame (compatible with ball_detector)
        """
        import torch

        batches = self._batch_frames(frames)
        all_pose_detections = []
        all_person_detections = []

        for batch_idx, batch in enumerate(batches):
            resized_batch = []
            scales = []
            for frame in batch:
                resized_frame, scale = self._resize_frame(frame)
                resized_batch.append(resized_frame)
                scales.append(scale)

            pose_results = self.pose_model.predict(
                resized_batch,
                batch=len(resized_batch),
                verbose=False,
                conf=conf_threshold,
                half=True
            )

            for res_idx, (pose_result, scale) in enumerate(zip(pose_results, scales)):
                frame_poses = []
                frame_persons = []

                # Pose model trả về cả boxes và keypoints
                if pose_result.boxes is not None and pose_result.keypoints is not None:
                    boxes = pose_result.boxes
                    keypoints_list = pose_result.keypoints

                    for i in range(len(boxes)):
                        # Extract bbox
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        box_conf = float(boxes.conf[i].cpu().numpy())

                        # Scale back if resized
                        if scale != 1.0:
                            x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale

                        # Extract keypoints
                        kpts = keypoints_list.xy[i].cpu().numpy()
                        kpts_conf = keypoints_list.conf[i].cpu().numpy()

                        if scale != 1.0:
                            kpts = kpts / scale

                        frame_poses.append({
                            "keypoints": kpts,
                            "conf": kpts_conf,
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "box_conf": box_conf
                        })

                        # Person detection format (compatible with ball_detector)
                        frame_persons.append({
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "conf": box_conf
                        })

                all_pose_detections.append(frame_poses)
                all_person_detections.append(frame_persons)

            if batch_idx % 20 == 0 and batch_idx > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return all_pose_detections, all_person_detections

    def detect_and_track_persons(
        self, frames, ball_positions, direction_flags, conf_threshold=0.5
    ):
        """Detect và track người qua các frame với batch inference
        Sử dụng pose model duy nhất cho cả person detection và pose estimation

        Args:
            frames: List of video frames
            ball_positions: Ball positions for each frame
            direction_flags: Direction change flags
            conf_threshold: Confidence threshold for detection

        Returns:
            Tuple of (tracked_person_detections, pose_detections, raw_person_detections)
            - tracked_person_detections: List of tracked person data per frame
            - pose_detections: List of pose data per frame
            - raw_person_detections: List of person bboxes per frame (for ball_detector)
        """
        # Chạy pose detection 1 lần duy nhất - lấy cả pose và person bboxes
        pose_detections, raw_person_detections = self._batch_pose_detection(frames, conf_threshold)

        tracked_person_detections = []

        for frame_idx in range(len(frames)):
            frame_poses = pose_detections[frame_idx]

            # Pose model đã có sẵn bbox trong mỗi pose, sử dụng trực tiếp
            tracked_frame_data = self._track_persons_from_poses(frame_poses, frame_idx)

            tracked_person_detections.append(tracked_frame_data)

            if frame_idx < len(ball_positions) and ball_positions[frame_idx] != (-1, -1):
                self._check_ball_person_hits(
                    ball_positions[frame_idx],
                    tracked_frame_data,
                    direction_flags[frame_idx] if frame_idx < len(direction_flags) else 0,
                    frame_idx,
                )

            if frame_idx % 50 == 0 and frame_idx > 0:
                gc.collect()

        return tracked_person_detections, pose_detections, raw_person_detections

    def _track_persons_from_poses(self, frame_poses, frame_idx):
        """Track người từ pose detections (đã có sẵn bbox)

        Args:
            frame_poses: List of pose data với bbox đã có sẵn
            frame_idx: Frame index

        Returns:
            List of tracked person data
        """
        tracked_data = []

        for pose_data in frame_poses:
            bbox = pose_data.get("bbox")
            if bbox is None:
                continue

            person = {
                "bbox": bbox,
                "conf": pose_data.get("box_conf", 0.5)
            }
            pose = {
                "keypoints": pose_data.get("keypoints"),
                "conf": pose_data.get("conf")
            }

            # Tìm person gần nhất trong frame trước
            best_match_id = None
            best_iou = 0

            for person_id, person_data in self.tracked_persons.items():
                if person_data["last_seen"] == frame_idx - 1:
                    iou = self._calculate_iou(person["bbox"], person_data["bbox"])
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_match_id = person_id

            if best_match_id is not None:
                person_id = best_match_id
                self.tracked_persons[person_id].update({
                    "bbox": person["bbox"],
                    "conf": person["conf"],
                    "pose": pose,
                    "last_seen": frame_idx,
                    "frame_count": self.tracked_persons[person_id]["frame_count"] + 1,
                })
            else:
                person_id = self.next_person_id
                self.next_person_id += 1
                self.tracked_persons[person_id] = {
                    "bbox": person["bbox"],
                    "conf": person["conf"],
                    "pose": pose,
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "frame_count": 1,
                }

            tracked_data.append({
                "person_id": person_id,
                "person": person,
                "pose": pose
            })

            # Lưu vị trí người chơi cho heatmap
            x1, y1, x2, y2 = person["bbox"]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            self.player_positions[person_id].append((frame_idx, center_x, center_y))

        return tracked_data

    def _match_persons_with_poses(self, persons, poses):
        """Match persons với poses dựa trên vị trí bbox"""
        matched_data = []

        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            person_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            best_pose = None
            min_distance = float("inf")

            for pose in poses:
                valid_keypoints = pose["keypoints"][pose["conf"] > 0.5]
                if len(valid_keypoints) > 0:
                    pose_center = np.mean(valid_keypoints, axis=0)
                    distance = np.linalg.norm(np.array(person_center) - pose_center)

                    if distance < min_distance:
                        min_distance = distance
                        best_pose = pose

            matched_data.append(
                {"person": person, "pose": best_pose, "distance": min_distance}
            )

        return matched_data

    def _track_persons_across_frames(self, matched_data, frame_idx):
        """Track người qua các frame sử dụng IoU và distance"""
        tracked_data = []

        for data in matched_data:
            person = data["person"]
            pose = data["pose"]

            # Tìm person gần nhất trong frame trước
            best_match_id = None
            best_iou = 0

            for person_id, person_data in self.tracked_persons.items():
                if person_data["last_seen"] == frame_idx - 1:  # Chỉ check frame trước
                    iou = self._calculate_iou(person["bbox"], person_data["bbox"])
                    if iou > best_iou and iou > 0.3:  # Threshold IoU
                        best_iou = iou
                        best_match_id = person_id

            if best_match_id is not None:
                # Update existing person
                person_id = best_match_id
                self.tracked_persons[person_id].update(
                    {
                        "bbox": person["bbox"],
                        "conf": person["conf"],
                        "pose": pose,
                        "last_seen": frame_idx,
                        "frame_count": self.tracked_persons[person_id]["frame_count"]
                        + 1,
                    }
                )
            else:
                # Create new person
                person_id = self.next_person_id
                self.next_person_id += 1
                self.tracked_persons[person_id] = {
                    "bbox": person["bbox"],
                    "conf": person["conf"],
                    "pose": pose,
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "frame_count": 1,
                }

            tracked_data.append(
                {"person_id": person_id, "person": person, "pose": pose}
            )

            # Lưu vị trí người chơi cho heatmap - NEW
            x1, y1, x2, y2 = person["bbox"]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            self.player_positions[person_id].append((frame_idx, center_x, center_y))

        return tracked_data

    def _calculate_iou(self, bbox1, bbox2):
        """Tính IoU giữa 2 bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Tính intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _check_ball_person_hits(
        self, ball_pos, tracked_persons, direction_flag, frame_idx
    ):
        """Kiểm tra bóng có chạm người không và lưu thông tin hit"""
        if direction_flag == 2:  # Bóng được đánh bởi người
            ball_x, ball_y = ball_pos

            for person_data in tracked_persons:
                person_id = person_data["person_id"]
                x1, y1, x2, y2 = person_data["person"]["bbox"]

                # Kiểm tra bóng có trong vùng người không
                if x1 <= ball_x <= x2 and y1 <= ball_y <= y2:
                    hit_data = {
                        "frame": frame_idx,
                        "ball_pos": ball_pos,
                        "person_bbox": person_data["person"]["bbox"],
                        "pose": person_data["pose"],
                        "person_id": person_id,
                    }
                    self.ball_hits_by_person[person_id].append(hit_data)
                    break

    def analyze_tennis_technique(self, person_detections, court_bounds):
        """Phân tích kỹ thuật tennis dựa trên pose estimation"""
        print("Đang phân tích kỹ thuật tennis...")

        technique_analysis = {
            "person_stats": {},
            "technique_errors": [],
            "pose_analysis": {},
            "court_accuracy": {},
        }

        for person_id, person_data in self.tracked_persons.items():
            hits = self.ball_hits_by_person[person_id]

            person_stats = {
                "total_hits": len(hits),
                "hits_in_court": 0,
                "hits_out_court": 0,
                "technique_errors": [],
                "hit_details": [],  # Chi tiết từng cú đánh
            }

            # Phân tích từng cú đánh
            for hit in hits:
                pose = hit["pose"]
                ball_pos = hit["ball_pos"]
                frame_idx = hit["frame"]

                # Kiểm tra trong/ngoài sân
                is_in_court = self._is_ball_in_court(ball_pos, court_bounds)

                hit_detail = {
                    "frame": frame_idx,
                    "ball_pos": ball_pos,
                    "is_in_court": is_in_court,
                    "pose_analysis": {},
                    "technique_errors": [],
                }

                if pose is not None:
                    # Phân tích pose
                    pose_analysis = self._analyze_pose(pose, ball_pos, court_bounds)
                    hit_detail["pose_analysis"] = pose_analysis

                    # Kiểm tra lỗi kỹ thuật
                    errors = self._detect_technique_errors(pose, ball_pos, court_bounds)
                    person_stats["technique_errors"].extend(errors)
                    hit_detail["technique_errors"] = errors

                    # Đếm trong/ngoài sân
                    if is_in_court:
                        person_stats["hits_in_court"] += 1
                    else:
                        person_stats["hits_out_court"] += 1
                else:
                    # Nếu không có pose, vẫn đếm trong/ngoài sân
                    if is_in_court:
                        person_stats["hits_in_court"] += 1
                    else:
                        person_stats["hits_out_court"] += 1

                person_stats["hit_details"].append(hit_detail)

            # Tính tỷ lệ chính xác
            total_hits = person_stats["hits_in_court"] + person_stats["hits_out_court"]
            if total_hits > 0:
                person_stats["accuracy_percentage"] = (
                    person_stats["hits_in_court"] / total_hits
                ) * 100
            else:
                person_stats["accuracy_percentage"] = 0

            technique_analysis["person_stats"][person_id] = person_stats

        # Tính thống kê tổng hợp
        technique_analysis["court_accuracy"] = self._calculate_court_accuracy_summary(
            technique_analysis["person_stats"]
        )

        return technique_analysis

    def _calculate_court_accuracy_summary(self, person_stats):
        """Tính thống kê tổng hợp về độ chính xác"""
        # Lọc bỏ những người không có cú đánh nào
        active_persons = {
            pid: stats for pid, stats in person_stats.items() if stats["total_hits"] > 0
        }

        total_in_court = sum(
            stats["hits_in_court"] for stats in active_persons.values()
        )
        total_out_court = sum(
            stats["hits_out_court"] for stats in active_persons.values()
        )
        total_hits = total_in_court + total_out_court

        summary = {
            "total_in_court": total_in_court,
            "total_out_court": total_out_court,
            "total_hits": total_hits,
            "overall_accuracy": (
                (total_in_court / total_hits * 100) if total_hits > 0 else 0
            ),
            "by_person": {},
            "active_persons_count": len(active_persons),
            "total_persons_count": len(person_stats),
        }

        for person_id, stats in active_persons.items():
            summary["by_person"][person_id] = {
                "hits_in_court": stats["hits_in_court"],
                "hits_out_court": stats["hits_out_court"],
                "total_hits": stats["total_hits"],
                "accuracy_percentage": stats["accuracy_percentage"],
            }

        return summary

    def _analyze_pose(self, pose, ball_pos, court_bounds):
        """Phân tích pose để xác định tư thế đánh bóng"""
        if pose is None:
            return {}

        keypoints = pose["keypoints"]
        conf = pose["conf"]

        analysis = {}
        valid_keypoints = conf > 0.5

        if valid_keypoints[5] and valid_keypoints[6]:  # Shoulders
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            shoulder_angle = self._calculate_angle(
                left_shoulder, right_shoulder, ball_pos
            )
            analysis["shoulder_angle"] = shoulder_angle

        if valid_keypoints[13] and valid_keypoints[14]:  # Knees
            left_knee = keypoints[13]
            right_knee = keypoints[14]
            knee_bend = self._calculate_knee_bend(left_knee, right_knee)
            analysis["knee_bend"] = knee_bend

        if valid_keypoints[9] and valid_keypoints[10]:  # Wrists
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]
            racket_position = self._estimate_racket_position(
                left_wrist, right_wrist, ball_pos
            )
            analysis["racket_position"] = racket_position

        return analysis

    def _detect_technique_errors(self, pose, ball_pos, court_bounds):
        """Phát hiện lỗi kỹ thuật tennis"""
        errors = []

        if pose is None:
            return errors

        keypoints = pose["keypoints"]
        conf = pose["conf"]

        # 1. Kiểm tra khụy gối (knee bend)
        if conf[13] > 0.5 and conf[14] > 0.5:  # Knees
            left_knee = keypoints[13]
            right_knee = keypoints[14]
            knee_bend = self._calculate_knee_bend(left_knee, right_knee)

            if knee_bend < 120:  # Góc khụy gối quá nhỏ
                errors.append(
                    {
                        "type": "insufficient_knee_bend",
                        "description": "Khụy gối không đủ sâu",
                        "severity": "medium",
                    }
                )

        # 2. Kiểm tra vị trí chân (foot position)
        if conf[15] > 0.5 and conf[16] > 0.5:  # Ankles
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]

            # Kiểm tra dẫm vạch
            if self._is_stepping_on_line(left_ankle, right_ankle, court_bounds):
                errors.append(
                    {
                        "type": "stepping_on_line",
                        "description": "Dẫm vạch khi đánh bóng",
                        "severity": "high",
                    }
                )

        # 3. Kiểm tra tư thế sau khi đánh bóng
        if conf[5] > 0.5 and conf[6] > 0.5:  # Shoulders
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]

            # Kiểm tra follow-through
            follow_through = self._check_follow_through(
                left_shoulder, right_shoulder, ball_pos
            )
            if not follow_through:
                errors.append(
                    {
                        "type": "poor_follow_through",
                        "description": "Tư thế sau khi đánh bóng không tốt",
                        "severity": "low",
                    }
                )

        return errors

    def _calculate_angle(self, p1, p2, p3):
        """Tính góc giữa 3 điểm"""
        v1 = p1 - p3
        v2 = p2 - p3

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return np.arccos(cos_angle) * 180 / np.pi

    def _calculate_knee_bend(self, left_knee, right_knee):
        """Tính góc khụy gối"""
        return self._calculate_angle(
            left_knee, right_knee, (left_knee + right_knee) / 2
        )

    def _estimate_racket_position(self, left_wrist, right_wrist, ball_pos):
        """Ước tính vị trí vợt dựa trên vị trí cổ tay"""
        wrist_center = (left_wrist + right_wrist) / 2
        distance_to_ball = np.linalg.norm(wrist_center - ball_pos)
        return distance_to_ball

    def _is_ball_in_court(self, ball_pos, court_bounds):
        """Kiểm tra bóng có trong sân không"""
        x, y = ball_pos
        x1, y1, x2, y2 = court_bounds
        return x1 <= x <= x2 and y1 <= y <= y2

    def _is_stepping_on_line(self, left_ankle, right_ankle, court_bounds):
        """Kiểm tra có dẫm vạch không"""
        x1, y1, x2, y2 = court_bounds

        # Kiểm tra vạch ngang (net)
        net_y = (y1 + y2) / 2
        if abs(left_ankle[1] - net_y) < 20 or abs(right_ankle[1] - net_y) < 20:
            return True

        # Kiểm tra vạch dọc
        if abs(left_ankle[0] - x1) < 20 or abs(left_ankle[0] - x2) < 20:
            return True
        if abs(right_ankle[0] - x1) < 20 or abs(right_ankle[0] - x2) < 20:
            return True

        return False

    def _check_follow_through(self, left_shoulder, right_shoulder, ball_pos):
        """Kiểm tra follow-through sau khi đánh bóng"""
        shoulder_center = (left_shoulder + right_shoulder) / 2
        distance = np.linalg.norm(shoulder_center - ball_pos)
        return distance < 100  # Threshold

    def get_person_statistics(self):
        """Lấy thống kê tổng hợp về người chơi"""
        stats = {}

        for person_id, person_data in self.tracked_persons.items():
            hits = self.ball_hits_by_person[person_id]

            stats[person_id] = {
                "total_frames": person_data["frame_count"],
                "total_hits": len(hits),
                "hit_rate": (
                    len(hits) / person_data["frame_count"]
                    if person_data["frame_count"] > 0
                    else 0
                ),
                "first_seen": person_data["first_seen"],
                "last_seen": person_data["last_seen"],
            }

        return stats

    def get_player_positions(self):
        """Lấy vị trí người chơi theo thời gian cho heatmap - NEW"""
        return dict(self.player_positions)
