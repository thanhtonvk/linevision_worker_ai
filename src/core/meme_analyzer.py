# =============================================================================
# MEME ANALYZER - PHÂN TÍCH VÀ GÁN MEME CHO CÁC CÚ ĐÁNH
# =============================================================================

import json
import os
import math
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from typing import List, Dict, Tuple, Optional


class MemeAnalyzer:
    """
    Phân tích các cú đánh và gán meme phù hợp dựa trên các chỉ số:
    - Cú đánh xoáy nhất (góc thay đổi lớn)
    - Cú đánh mạnh nhất (tốc độ cao)
    - Cú đánh cắm nhất (góc đi xuống dốc)
    - Cú đánh bay lên trời (góc đi lên cao)
    - Tỉ lệ bóng trong/ngoài sân
    - Di chuyển nhanh/rộng nhất
    """

    def __init__(self, meme_json_path: str = "data/meme.json"):
        """
        Khởi tạo MemeAnalyzer

        Args:
            meme_json_path: Đường dẫn file meme.json
        """
        self.memes = self._load_memes(meme_json_path)
        self.meme_by_category = self._group_by_category()
        self.meme_cache = {}  # Cache downloaded meme images

    def _load_memes(self, path: str) -> List[Dict]:
        """Load memes từ file JSON"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[MEME] Không thể load meme.json: {e}")
            return []

    def _group_by_category(self) -> Dict[str, List[Dict]]:
        """Nhóm memes theo category"""
        grouped = {}
        for meme in self.memes:
            category = meme.get("name_category", "")
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(meme)
        return grouped

    def analyze_shots(
        self,
        ball_positions: List[Tuple[float, float]],
        ball_hits_by_person: Dict[int, List[Dict]],
        player_stats: Dict[int, Dict],
        player_positions: Dict[int, List],
        fps: float = 30.0
    ) -> Dict[int, Dict]:
        """
        Phân tích các cú đánh và gán meme cho mỗi player

        Args:
            ball_positions: Danh sách vị trí bóng theo frame
            ball_hits_by_person: Dict {player_id: [hits]}
            player_stats: Dict {player_id: stats} từ PlayerStatsAnalyzer
            player_positions: Dict {player_id: [(frame, x, y), ...]}
            fps: Frame rate của video

        Returns:
            Dict {player_id: {"memes": [...], "shot_analysis": {...}}}
        """
        results = {}

        for player_id, hits in ball_hits_by_person.items():
            if len(hits) == 0:
                continue

            player_result = {
                "memes": [],
                "shot_analysis": {},
                "best_shots": {}
            }

            # Phân tích từng cú đánh
            shot_metrics = self._analyze_shot_metrics(hits, ball_positions, fps)
            player_result["shot_analysis"] = shot_metrics

            # Lấy stats của player
            stats = player_stats.get(player_id, {})

            # Gán meme dựa trên phân tích
            assigned_memes = self._assign_memes(
                shot_metrics,
                stats,
                player_positions.get(player_id, []),
                fps
            )
            player_result["memes"] = assigned_memes

            # Lưu best shots cho từng category
            player_result["best_shots"] = self._get_best_shots(shot_metrics, hits)

            results[player_id] = player_result

        return results

    def _analyze_shot_metrics(
        self,
        hits: List[Dict],
        ball_positions: List[Tuple[float, float]],
        fps: float
    ) -> Dict:
        """
        Phân tích metrics của các cú đánh

        Returns:
            Dict với các metrics: max_speed, max_spin, max_steep_angle, max_up_angle, etc.
        """
        metrics = {
            "speeds": [],
            "spin_angles": [],  # Góc thay đổi hướng
            "trajectory_angles": [],  # Góc bay của bóng
            "max_speed": 0,
            "max_spin": 0,
            "max_steep_angle": 0,  # Góc cắm xuống
            "max_up_angle": 0,  # Góc bay lên
            "best_speed_hit": None,
            "best_spin_hit": None,
            "best_steep_hit": None,
            "best_up_hit": None
        }

        for i, hit in enumerate(hits):
            hit_frame = hit["frame"]
            ball_pos = hit["ball_pos"]

            # Tính tốc độ bóng (pixels/frame)
            speed = self._calculate_ball_speed(
                ball_positions, hit_frame, window=5
            )
            metrics["speeds"].append(speed)

            if speed > metrics["max_speed"]:
                metrics["max_speed"] = speed
                metrics["best_speed_hit"] = hit

            # Tính góc thay đổi hướng (spin/xoáy)
            spin_angle = self._calculate_direction_change(
                ball_positions, hit_frame, window=10
            )
            metrics["spin_angles"].append(spin_angle)

            if spin_angle > metrics["max_spin"]:
                metrics["max_spin"] = spin_angle
                metrics["best_spin_hit"] = hit

            # Tính góc bay của bóng sau khi đánh
            trajectory_angle = self._calculate_trajectory_angle(
                ball_positions, hit_frame, window=10
            )
            metrics["trajectory_angles"].append(trajectory_angle)

            # Góc âm = đi xuống (cắm), góc dương = đi lên
            if trajectory_angle < metrics["max_steep_angle"]:
                metrics["max_steep_angle"] = trajectory_angle
                metrics["best_steep_hit"] = hit

            if trajectory_angle > metrics["max_up_angle"]:
                metrics["max_up_angle"] = trajectory_angle
                metrics["best_up_hit"] = hit

        # Tính average
        if metrics["speeds"]:
            metrics["avg_speed"] = sum(metrics["speeds"]) / len(metrics["speeds"])
        if metrics["spin_angles"]:
            metrics["avg_spin"] = sum(metrics["spin_angles"]) / len(metrics["spin_angles"])

        return metrics

    def _calculate_ball_speed(
        self,
        ball_positions: List[Tuple[float, float]],
        frame_idx: int,
        window: int = 5
    ) -> float:
        """Tính tốc độ bóng tại frame (pixels/frame)"""
        start_idx = max(0, frame_idx - window)
        end_idx = min(len(ball_positions), frame_idx + window)

        total_distance = 0
        valid_frames = 0

        for i in range(start_idx, end_idx - 1):
            pos1 = ball_positions[i]
            pos2 = ball_positions[i + 1]

            if pos1 != (-1, -1) and pos2 != (-1, -1):
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                distance = math.sqrt(dx * dx + dy * dy)
                total_distance += distance
                valid_frames += 1

        if valid_frames > 0:
            return total_distance / valid_frames
        return 0

    def _calculate_direction_change(
        self,
        ball_positions: List[Tuple[float, float]],
        frame_idx: int,
        window: int = 10
    ) -> float:
        """Tính góc thay đổi hướng (độ xoáy) tại frame"""
        # Lấy hướng trước và sau hit
        before_start = max(0, frame_idx - window)
        after_end = min(len(ball_positions), frame_idx + window)

        # Vector hướng trước hit
        before_vec = self._get_direction_vector(ball_positions, before_start, frame_idx)
        # Vector hướng sau hit
        after_vec = self._get_direction_vector(ball_positions, frame_idx, after_end)

        if before_vec is None or after_vec is None:
            return 0

        # Tính góc giữa 2 vectors
        angle = self._angle_between_vectors(before_vec, after_vec)
        return angle

    def _calculate_trajectory_angle(
        self,
        ball_positions: List[Tuple[float, float]],
        frame_idx: int,
        window: int = 10
    ) -> float:
        """
        Tính góc bay của bóng sau khi đánh
        Góc dương = đi lên, góc âm = đi xuống (cắm)
        """
        end_idx = min(len(ball_positions), frame_idx + window)

        # Lấy vị trí bóng sau hit
        valid_positions = []
        for i in range(frame_idx, end_idx):
            if ball_positions[i] != (-1, -1):
                valid_positions.append(ball_positions[i])

        if len(valid_positions) < 2:
            return 0

        # Tính hướng di chuyển trung bình
        dx = valid_positions[-1][0] - valid_positions[0][0]
        dy = valid_positions[-1][1] - valid_positions[0][1]

        if abs(dx) < 1:
            dx = 1

        # Góc với trục ngang (dương = lên, âm = xuống)
        # Chú ý: trong hệ tọa độ ảnh, y tăng khi đi xuống
        angle = math.degrees(math.atan2(-dy, abs(dx)))
        return angle

    def _get_direction_vector(
        self,
        positions: List[Tuple[float, float]],
        start_idx: int,
        end_idx: int
    ) -> Optional[Tuple[float, float]]:
        """Lấy vector hướng di chuyển"""
        valid_start = None
        valid_end = None

        for i in range(start_idx, end_idx):
            if positions[i] != (-1, -1):
                if valid_start is None:
                    valid_start = positions[i]
                valid_end = positions[i]

        if valid_start is None or valid_end is None or valid_start == valid_end:
            return None

        return (valid_end[0] - valid_start[0], valid_end[1] - valid_start[1])

    def _angle_between_vectors(
        self,
        v1: Tuple[float, float],
        v2: Tuple[float, float]
    ) -> float:
        """Tính góc giữa 2 vectors (độ)"""
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 < 0.001 or mag2 < 0.001:
            return 0

        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        angle = math.degrees(math.acos(cos_angle))
        return angle

    def _assign_memes(
        self,
        shot_metrics: Dict,
        player_stats: Dict,
        player_positions: List,
        fps: float
    ) -> List[Dict]:
        """Gán memes dựa trên phân tích"""
        assigned = []

        # 1. Cú đánh mạnh nhất (tốc độ cao)
        if shot_metrics["max_speed"] > 20:  # threshold
            meme = self._get_meme_for_category("Cú đánh mạnh nhất")
            if meme:
                assigned.append({
                    **meme,
                    "reason": f"Tốc độ bóng: {shot_metrics['max_speed']:.1f} px/frame",
                    "hit_frame": shot_metrics["best_speed_hit"]["frame"] if shot_metrics["best_speed_hit"] else None
                })

        # 2. Cú đánh xoáy nhất (góc thay đổi lớn)
        if shot_metrics["max_spin"] > 60:  # threshold 60 độ
            meme = self._get_meme_for_category("Cú đánh xoáy nhất")
            if meme:
                assigned.append({
                    **meme,
                    "reason": f"Góc xoáy: {shot_metrics['max_spin']:.1f}°",
                    "hit_frame": shot_metrics["best_spin_hit"]["frame"] if shot_metrics["best_spin_hit"] else None
                })

        # 3. Cú đánh cắm nhất (góc xuống dốc)
        if shot_metrics["max_steep_angle"] < -30:  # góc âm = xuống
            meme = self._get_meme_for_category("Cú đánh cắm nhất")
            if meme:
                assigned.append({
                    **meme,
                    "reason": f"Góc cắm: {abs(shot_metrics['max_steep_angle']):.1f}°",
                    "hit_frame": shot_metrics["best_steep_hit"]["frame"] if shot_metrics["best_steep_hit"] else None
                })

        # 4. Cú đánh bay lên trời
        if shot_metrics["max_up_angle"] > 30:  # góc dương = lên
            meme = self._get_meme_for_category("Cú đánh bay lên trời")
            if meme:
                assigned.append({
                    **meme,
                    "reason": f"Góc bay: {shot_metrics['max_up_angle']:.1f}°",
                    "hit_frame": shot_metrics["best_up_hit"]["frame"] if shot_metrics["best_up_hit"] else None
                })

        # 5. Tỉ lệ bóng trong sân cao nhất
        accuracy = player_stats.get("accuracy", {})
        total_hits = accuracy.get("total_hits", 0)
        in_court = accuracy.get("in_court", 0)

        if total_hits > 0:
            in_court_ratio = in_court / total_hits
            if in_court_ratio > 0.7:  # > 70% trong sân
                meme = self._get_meme_for_category("Tỉ lệ bóng trong sân cao nhất")
                if meme:
                    assigned.append({
                        **meme,
                        "reason": f"Tỉ lệ trong sân: {in_court_ratio * 100:.1f}%",
                        "hit_frame": None
                    })
            elif in_court_ratio < 0.3:  # < 30% trong sân
                meme = self._get_meme_for_category("Tỉ lệ bóng ngoài sân cao nhất")
                if meme:
                    assigned.append({
                        **meme,
                        "reason": f"Tỉ lệ ngoài sân: {(1 - in_court_ratio) * 100:.1f}%",
                        "hit_frame": None
                    })

        # 6. Người di chuyển nhanh/rộng nhất (dựa trên player_positions)
        if player_positions:
            movement_stats = self._analyze_movement(player_positions, fps)

            if movement_stats["avg_speed"] > 5:  # threshold
                meme = self._get_meme_for_category("Người di chuyển nhanh nhất")
                if meme:
                    assigned.append({
                        **meme,
                        "reason": f"Tốc độ di chuyển: {movement_stats['avg_speed']:.1f} px/frame",
                        "hit_frame": None
                    })

            if movement_stats["coverage_area"] > 50000:  # threshold
                meme = self._get_meme_for_category("Người di chuyển rộng nhất")
                if meme:
                    assigned.append({
                        **meme,
                        "reason": f"Vùng bao phủ: {movement_stats['coverage_area']:.0f} px²",
                        "hit_frame": None
                    })

        return assigned

    def _get_meme_for_category(self, category: str) -> Optional[Dict]:
        """Lấy meme cho category (random nếu có nhiều)"""
        memes = self.meme_by_category.get(category, [])
        if memes:
            import random
            return random.choice(memes).copy()
        return None

    def _get_best_shots(self, shot_metrics: Dict, hits: List[Dict]) -> Dict:
        """Lấy các cú đánh tốt nhất cho mỗi category"""
        best = {}

        if shot_metrics["best_speed_hit"]:
            best["fastest"] = {
                "frame": shot_metrics["best_speed_hit"]["frame"],
                "speed": shot_metrics["max_speed"],
                "ball_pos": shot_metrics["best_speed_hit"]["ball_pos"]
            }

        if shot_metrics["best_spin_hit"]:
            best["most_spin"] = {
                "frame": shot_metrics["best_spin_hit"]["frame"],
                "spin_angle": shot_metrics["max_spin"],
                "ball_pos": shot_metrics["best_spin_hit"]["ball_pos"]
            }

        if shot_metrics["best_steep_hit"]:
            best["steepest"] = {
                "frame": shot_metrics["best_steep_hit"]["frame"],
                "angle": shot_metrics["max_steep_angle"],
                "ball_pos": shot_metrics["best_steep_hit"]["ball_pos"]
            }

        if shot_metrics["best_up_hit"]:
            best["highest"] = {
                "frame": shot_metrics["best_up_hit"]["frame"],
                "angle": shot_metrics["max_up_angle"],
                "ball_pos": shot_metrics["best_up_hit"]["ball_pos"]
            }

        return best

    def _analyze_movement(self, positions: List, fps: float) -> Dict:
        """Phân tích chuyển động của player"""
        if not positions:
            return {"avg_speed": 0, "coverage_area": 0}

        speeds = []
        x_coords = []
        y_coords = []

        for i in range(len(positions)):
            _, x, y = positions[i]
            x_coords.append(x)
            y_coords.append(y)

            if i > 0:
                _, prev_x, prev_y = positions[i - 1]
                dx = x - prev_x
                dy = y - prev_y
                speed = math.sqrt(dx * dx + dy * dy)
                speeds.append(speed)

        avg_speed = sum(speeds) / len(speeds) if speeds else 0

        # Tính vùng bao phủ (bounding box area)
        if x_coords and y_coords:
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            coverage_area = width * height
        else:
            coverage_area = 0

        return {
            "avg_speed": avg_speed,
            "coverage_area": coverage_area
        }

    def download_meme_image(self, image_url: str) -> Optional[np.ndarray]:
        """Download meme image từ URL và convert sang OpenCV format"""
        if image_url in self.meme_cache:
            return self.meme_cache[image_url]

        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                # Load image với PIL (hỗ trợ GIF)
                img = Image.open(BytesIO(response.content))

                # Nếu là GIF, lấy frame đầu tiên
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    img.seek(0)

                # Convert sang RGB nếu cần
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Convert sang numpy array (BGR for OpenCV)
                img_array = np.array(img)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                self.meme_cache[image_url] = img_bgr
                return img_bgr

        except Exception as e:
            print(f"[MEME] Không thể download meme {image_url}: {e}")

        return None

    def overlay_meme_on_frame(
        self,
        frame: np.ndarray,
        meme_image: np.ndarray,
        position: str = "top-right",
        scale: float = 0.2,
        opacity: float = 0.9
    ) -> np.ndarray:
        """
        Chèn meme lên frame

        Args:
            frame: Frame gốc
            meme_image: Ảnh meme
            position: Vị trí (top-left, top-right, bottom-left, bottom-right, center)
            scale: Tỉ lệ kích thước meme so với frame (0.0 - 1.0)
            opacity: Độ trong suốt (0.0 - 1.0)

        Returns:
            Frame với meme overlay
        """
        h, w = frame.shape[:2]
        mh, mw = meme_image.shape[:2]

        # Resize meme
        new_w = int(w * scale)
        new_h = int(mh * new_w / mw)
        meme_resized = cv2.resize(meme_image, (new_w, new_h))

        # Tính vị trí
        padding = 20
        if position == "top-left":
            x, y = padding, padding
        elif position == "top-right":
            x, y = w - new_w - padding, padding
        elif position == "bottom-left":
            x, y = padding, h - new_h - padding
        elif position == "bottom-right":
            x, y = w - new_w - padding, h - new_h - padding
        elif position == "center":
            x, y = (w - new_w) // 2, (h - new_h) // 2
        else:
            x, y = padding, padding

        # Đảm bảo trong bounds
        x = max(0, min(x, w - new_w))
        y = max(0, min(y, h - new_h))

        # Overlay với opacity
        result = frame.copy()
        roi = result[y:y + new_h, x:x + new_w]

        if roi.shape[:2] == meme_resized.shape[:2]:
            blended = cv2.addWeighted(roi, 1 - opacity, meme_resized, opacity, 0)
            result[y:y + new_h, x:x + new_w] = blended

        return result
