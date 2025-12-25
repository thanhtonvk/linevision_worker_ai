# =============================================================================
# PERSON TRACKER CLASS - TRACKING NGƯỜI CHƠI TENNIS
# =============================================================================

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math
import gc
import os
from config.settings import settings


class PersonTracker:
    """
    Class để tracking người chơi tennis
    Sử dụng YOLO11m cho person detection (không dùng pose)
    Optimized for 12GB GPU with batch inference
    """

    def __init__(
        self,
        person_model_path="yolo11m.pt",
        batch_size=16,  # Optimized for 12GB GPU
    ):
        self.person_model = YOLO(person_model_path)
        self.batch_size = batch_size
        self.tracked_persons = {}  # {person_id: person_data}
        self.next_person_id = 1
        self.ball_hits_by_person = defaultdict(list)  # {person_id: [hit_data]}
        self.player_positions = defaultdict(list)  # {player_id: [(frame, x, y), ...]}
        self.player_frames = defaultdict(list)  # {player_id: [frame_indices]} - NEW for highlight
        self.max_frame_height = settings.max_frame_height
        self.enable_frame_resize = settings.enable_frame_resize


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

    def _batch_person_detection(self, frames, conf_threshold=0.5):
        """Batch person detection cho tất cả frames

        Args:
            frames: List of video frames
            conf_threshold: Confidence threshold for detection

        Returns:
            List of person detections for each frame
        """
        import torch

        batches = self._batch_frames(frames)
        all_person_detections = []

        for batch_idx, batch in enumerate(batches):
            resized_batch = []
            scales = []
            for frame in batch:
                resized_frame, scale = self._resize_frame(frame)
                resized_batch.append(resized_frame)
                scales.append(scale)

            results = self.person_model.predict(
                resized_batch,
                batch=len(resized_batch),
                verbose=False,
                conf=conf_threshold,
                half=True
            )

            for res_idx, (result, scale) in enumerate(zip(results, scales)):
                frame_persons = []

                if result.boxes is not None:
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        # Chỉ lấy class 0 (person)
                        if int(boxes.cls[i]) != 0:
                            continue

                        # Extract bbox
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        box_conf = float(boxes.conf[i].cpu().numpy())

                        # Scale back if resized
                        if scale != 1.0:
                            x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale

                        frame_persons.append({
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "conf": box_conf
                        })

                all_person_detections.append(frame_persons)

            if batch_idx % 20 == 0 and batch_idx > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return all_person_detections

    def detect_and_track_persons(
        self, frames, ball_positions, direction_flags, conf_threshold=0.5
    ):
        """Detect và track người qua các frame với batch inference

        Args:
            frames: List of video frames
            ball_positions: Ball positions for each frame
            direction_flags: Direction change flags
            conf_threshold: Confidence threshold for detection

        Returns:
            Tuple of (tracked_person_detections, raw_person_detections)
        """
        # Chạy person detection
        raw_person_detections = self._batch_person_detection(frames, conf_threshold)

        tracked_person_detections = []

        for frame_idx in range(len(frames)):
            frame_persons = raw_person_detections[frame_idx]

            # Track người qua các frame
            tracked_frame_data = self._track_persons_across_frames(frame_persons, frame_idx)

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

        return tracked_person_detections, raw_person_detections

    def _track_persons_across_frames(self, frame_persons, frame_idx):
        """Track người qua các frame sử dụng IoU"""
        tracked_data = []

        for person in frame_persons:
            bbox = person["bbox"]
            conf = person["conf"]

            # Tìm person gần nhất trong frame trước
            best_match_id = None
            best_iou = 0

            for person_id, person_data in self.tracked_persons.items():
                # Cho phép gap tối đa 5 frames
                if frame_idx - person_data["last_seen"] <= 5:
                    iou = self._calculate_iou(bbox, person_data["bbox"])
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_match_id = person_id

            if best_match_id is not None:
                person_id = best_match_id
                self.tracked_persons[person_id].update({
                    "bbox": bbox,
                    "conf": conf,
                    "last_seen": frame_idx,
                    "frame_count": self.tracked_persons[person_id]["frame_count"] + 1,
                })
            else:
                person_id = self.next_person_id
                self.next_person_id += 1
                self.tracked_persons[person_id] = {
                    "bbox": bbox,
                    "conf": conf,
                    "first_seen": frame_idx,
                    "last_seen": frame_idx,
                    "frame_count": 1,
                }

            tracked_data.append({
                "person_id": person_id,
                "person": person,
            })

            # Lưu vị trí người chơi cho heatmap
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            self.player_positions[person_id].append((frame_idx, center_x, center_y))

            # Lưu frame index cho highlight video
            self.player_frames[person_id].append(frame_idx)

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

            # Mở rộng vùng kiểm tra
            threshold = 100

            for person_data in tracked_persons:
                person_id = person_data["person_id"]
                x1, y1, x2, y2 = person_data["person"]["bbox"]

                # Kiểm tra bóng có gần người không
                if (x1 - threshold <= ball_x <= x2 + threshold and
                    y1 - threshold <= ball_y <= y2 + threshold):
                    hit_data = {
                        "frame": frame_idx,
                        "ball_pos": ball_pos,
                        "person_bbox": person_data["person"]["bbox"],
                        "person_id": person_id,
                    }
                    self.ball_hits_by_person[person_id].append(hit_data)
                    break

    def get_player_positions(self):
        """Lấy vị trí người chơi theo thời gian cho heatmap"""
        return dict(self.player_positions)

    def get_player_frames(self):
        """Lấy danh sách frame indices cho mỗi player"""
        return dict(self.player_frames)

    def create_player_highlight_videos(
        self,
        frames,
        output_folder,
        fps=30,
        min_frames=30,
        padding_frames=15,
        base_url=""
    ):
        """Tạo video highlight cho từng player

        Args:
            frames: List of all video frames
            output_folder: Thư mục lưu output
            fps: FPS của video output
            min_frames: Số frame tối thiểu để tạo highlight
            padding_frames: Số frame thêm trước/sau mỗi hit
            base_url: URL cơ sở cho file paths

        Returns:
            Dict {player_id: {"video_path": path, "hit_count": count, ...}}
        """
        os.makedirs(output_folder, exist_ok=True)
        highlight_results = {}

        for person_id, hits in self.ball_hits_by_person.items():
            if len(hits) == 0:
                continue

            # Thu thập tất cả frame indices cần cho highlight
            highlight_frames_set = set()

            for hit in hits:
                hit_frame = hit["frame"]
                # Thêm frames xung quanh mỗi hit
                start_frame = max(0, hit_frame - padding_frames)
                end_frame = min(len(frames), hit_frame + padding_frames + 1)

                for f in range(start_frame, end_frame):
                    highlight_frames_set.add(f)

            # Chuyển thành list và sort
            highlight_frame_indices = sorted(list(highlight_frames_set))

            if len(highlight_frame_indices) < min_frames:
                continue

            # Tạo video highlight
            output_path = os.path.join(output_folder, f"player_{person_id}_highlight.mp4")

            # Lấy kích thước từ frame đầu tiên
            first_frame = frames[highlight_frame_indices[0]]
            height, width = first_frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            try:
                for frame_idx in highlight_frame_indices:
                    frame = frames[frame_idx].copy()

                    # Vẽ bbox của player lên frame
                    for pos_data in self.player_positions.get(person_id, []):
                        if pos_data[0] == frame_idx:
                            # Tìm bbox tại frame này
                            if person_id in self.tracked_persons:
                                bbox = None
                                # Tìm trong hits
                                for hit in hits:
                                    if hit["frame"] == frame_idx:
                                        bbox = hit["person_bbox"]
                                        break

                                if bbox:
                                    x1, y1, x2, y2 = bbox
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(
                                        frame,
                                        f"Player {person_id}",
                                        (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 255, 0),
                                        2
                                    )
                            break

                    # Đánh dấu frame hit
                    for hit in hits:
                        if hit["frame"] == frame_idx:
                            # Vẽ circle tại vị trí bóng
                            ball_pos = hit["ball_pos"]
                            cv2.circle(
                                frame,
                                (int(ball_pos[0]), int(ball_pos[1])),
                                15,
                                (0, 0, 255),
                                3
                            )
                            cv2.putText(
                                frame,
                                "HIT!",
                                (int(ball_pos[0]) + 20, int(ball_pos[1])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2
                            )
                            break

                    out.write(frame)

            finally:
                out.release()

            # Thông tin highlight
            highlight_results[person_id] = {
                "video_path": f"{base_url}/player_{person_id}_highlight.mp4" if base_url else output_path,
                "hit_count": len(hits),
                "total_frames": len(highlight_frame_indices),
                "duration_seconds": round(len(highlight_frame_indices) / fps, 2),
                "hits": [
                    {
                        "frame": h["frame"],
                        "ball_pos": h["ball_pos"]
                    }
                    for h in hits
                ]
            }

        return highlight_results

    def create_combined_highlight_video(
        self,
        frames,
        ball_positions,
        output_path,
        fps=30,
        padding_frames=30
    ):
        """Tạo video highlight tổng hợp tất cả các cú đánh

        Args:
            frames: List of all video frames
            ball_positions: Ball positions for each frame
            output_path: Đường dẫn file output
            fps: FPS của video output
            padding_frames: Số frame thêm trước/sau mỗi hit

        Returns:
            Dict với thông tin video
        """
        # Thu thập tất cả hits từ tất cả players
        all_hits = []
        for person_id, hits in self.ball_hits_by_person.items():
            for hit in hits:
                all_hits.append({
                    **hit,
                    "person_id": person_id
                })

        # Sort theo frame
        all_hits.sort(key=lambda x: x["frame"])

        if len(all_hits) == 0:
            return None

        # Thu thập frame indices
        highlight_frames_set = set()
        for hit in all_hits:
            hit_frame = hit["frame"]
            start_frame = max(0, hit_frame - padding_frames)
            end_frame = min(len(frames), hit_frame + padding_frames + 1)
            for f in range(start_frame, end_frame):
                highlight_frames_set.add(f)

        highlight_frame_indices = sorted(list(highlight_frames_set))

        # Tạo video
        first_frame = frames[highlight_frame_indices[0]]
        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            for frame_idx in highlight_frame_indices:
                frame = frames[frame_idx].copy()

                # Vẽ bóng
                if frame_idx < len(ball_positions) and ball_positions[frame_idx] != (-1, -1):
                    ball_pos = ball_positions[frame_idx]
                    cv2.circle(
                        frame,
                        (int(ball_pos[0]), int(ball_pos[1])),
                        8,
                        (0, 255, 255),
                        -1
                    )

                # Đánh dấu hits
                for hit in all_hits:
                    if hit["frame"] == frame_idx:
                        ball_pos = hit["ball_pos"]
                        person_id = hit["person_id"]

                        # Vẽ marker hit
                        cv2.circle(
                            frame,
                            (int(ball_pos[0]), int(ball_pos[1])),
                            20,
                            (0, 0, 255),
                            3
                        )
                        cv2.putText(
                            frame,
                            f"P{person_id} HIT!",
                            (int(ball_pos[0]) + 25, int(ball_pos[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                out.write(frame)

        finally:
            out.release()

        return {
            "total_hits": len(all_hits),
            "total_frames": len(highlight_frame_indices),
            "duration_seconds": round(len(highlight_frame_indices) / fps, 2),
            "players_involved": list(set(h["person_id"] for h in all_hits))
        }

    def get_person_stats(self):
        """Lấy thống kê tracking người"""
        stats = {}
        for person_id, person_data in self.tracked_persons.items():
            stats[person_id] = {
                "first_seen": person_data["first_seen"],
                "last_seen": person_data["last_seen"],
                "frame_count": person_data["frame_count"],
                "total_hits": len(self.ball_hits_by_person[person_id]),
            }
        return stats
