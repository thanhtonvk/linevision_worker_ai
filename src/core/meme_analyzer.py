# =============================================================================
# MEME ANALYZER - PHÂN TÍCH VÀ GÁN MEME CHO CÁC CÚ ĐÁNH
# =============================================================================

import json
import math
from typing import List, Dict, Tuple, Optional


class MemeAnalyzer:
    """
    Phân tích các cú đánh và gán meme phù hợp.
    Chỉ trả về thông tin meme (name, image_url), không tự động chèn vào video.
    """

    def __init__(self, meme_json_path: str = "data/meme.json"):
        self.memes = self._load_memes(meme_json_path)
        self.meme_by_category = self._group_by_category()

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

        Returns:
            Dict {player_id: {"memes": [{"name": ..., "image_url": ...}, ...]}}
        """
        results = {}

        for player_id, hits in ball_hits_by_person.items():
            if len(hits) == 0:
                continue

            # Phân tích từng cú đánh
            shot_metrics = self._analyze_shot_metrics(hits, ball_positions, fps)

            # Lấy stats của player
            stats = player_stats.get(player_id, {})

            # Gán meme dựa trên phân tích
            assigned_memes = self._assign_memes(
                shot_metrics,
                stats,
                player_positions.get(player_id, []),
                fps
            )

            results[player_id] = {"memes": assigned_memes}

        return results

    def _analyze_shot_metrics(
        self,
        hits: List[Dict],
        ball_positions: List[Tuple[float, float]],
        fps: float
    ) -> Dict:
        """Phân tích metrics của các cú đánh"""
        metrics = {
            "max_speed": 0,
            "max_spin": 0,
            "max_steep_angle": 0,
            "max_up_angle": 0,
        }

        for hit in hits:
            hit_frame = hit["frame"]

            # Tính tốc độ bóng
            speed = self._calculate_ball_speed(ball_positions, hit_frame, window=5)
            if speed > metrics["max_speed"]:
                metrics["max_speed"] = speed

            # Tính góc thay đổi hướng (spin/xoáy)
            spin_angle = self._calculate_direction_change(ball_positions, hit_frame, window=10)
            if spin_angle > metrics["max_spin"]:
                metrics["max_spin"] = spin_angle

            # Tính góc bay của bóng sau khi đánh
            trajectory_angle = self._calculate_trajectory_angle(ball_positions, hit_frame, window=10)
            if trajectory_angle < metrics["max_steep_angle"]:
                metrics["max_steep_angle"] = trajectory_angle
            if trajectory_angle > metrics["max_up_angle"]:
                metrics["max_up_angle"] = trajectory_angle

        return metrics

    def _calculate_ball_speed(self, ball_positions, frame_idx, window=5):
        """Tính tốc độ bóng tại frame"""
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
                total_distance += math.sqrt(dx * dx + dy * dy)
                valid_frames += 1

        return total_distance / valid_frames if valid_frames > 0 else 0

    def _calculate_direction_change(self, ball_positions, frame_idx, window=10):
        """Tính góc thay đổi hướng"""
        before_start = max(0, frame_idx - window)
        after_end = min(len(ball_positions), frame_idx + window)

        before_vec = self._get_direction_vector(ball_positions, before_start, frame_idx)
        after_vec = self._get_direction_vector(ball_positions, frame_idx, after_end)

        if before_vec is None or after_vec is None:
            return 0

        return self._angle_between_vectors(before_vec, after_vec)

    def _calculate_trajectory_angle(self, ball_positions, frame_idx, window=10):
        """Tính góc bay của bóng sau khi đánh"""
        end_idx = min(len(ball_positions), frame_idx + window)

        valid_positions = []
        for i in range(frame_idx, end_idx):
            if ball_positions[i] != (-1, -1):
                valid_positions.append(ball_positions[i])

        if len(valid_positions) < 2:
            return 0

        dx = valid_positions[-1][0] - valid_positions[0][0]
        dy = valid_positions[-1][1] - valid_positions[0][1]

        if abs(dx) < 1:
            dx = 1

        return math.degrees(math.atan2(-dy, abs(dx)))

    def _get_direction_vector(self, positions, start_idx, end_idx):
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

    def _angle_between_vectors(self, v1, v2):
        """Tính góc giữa 2 vectors (độ)"""
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 < 0.001 or mag2 < 0.001:
            return 0

        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    def _assign_memes(self, shot_metrics, player_stats, player_positions, fps):
        """
        Gán memes dựa trên phân tích.
        Chỉ trả về name và image_url.
        """
        assigned = []

        # 1. Cú đánh mạnh nhất
        if shot_metrics["max_speed"] > 20:
            meme = self._get_meme_for_category("Cú đánh mạnh nhất")
            if meme:
                assigned.append({"name": meme.get("name", ""), "image_url": meme.get("image_url", "")})

        # 2. Cú đánh xoáy nhất
        if shot_metrics["max_spin"] > 60:
            meme = self._get_meme_for_category("Cú đánh xoáy nhất")
            if meme:
                assigned.append({"name": meme.get("name", ""), "image_url": meme.get("image_url", "")})

        # 3. Cú đánh cắm nhất
        if shot_metrics["max_steep_angle"] < -30:
            meme = self._get_meme_for_category("Cú đánh cắm nhất")
            if meme:
                assigned.append({"name": meme.get("name", ""), "image_url": meme.get("image_url", "")})

        # 4. Cú đánh bay lên trời
        if shot_metrics["max_up_angle"] > 30:
            meme = self._get_meme_for_category("Cú đánh bay lên trời")
            if meme:
                assigned.append({"name": meme.get("name", ""), "image_url": meme.get("image_url", "")})

        # 5. Tỉ lệ bóng trong/ngoài sân
        accuracy = player_stats.get("accuracy", {})
        total_hits = accuracy.get("total_hits", 0)
        in_court = accuracy.get("in_court", 0)

        if total_hits > 0:
            in_court_ratio = in_court / total_hits
            if in_court_ratio > 0.7:
                meme = self._get_meme_for_category("Tỉ lệ bóng trong sân cao nhất")
                if meme:
                    assigned.append({"name": meme.get("name", ""), "image_url": meme.get("image_url", "")})
            elif in_court_ratio < 0.3:
                meme = self._get_meme_for_category("Tỉ lệ bóng ngoài sân cao nhất")
                if meme:
                    assigned.append({"name": meme.get("name", ""), "image_url": meme.get("image_url", "")})

        # 6. Người di chuyển nhanh/rộng nhất
        if player_positions:
            movement_stats = self._analyze_movement(player_positions, fps)

            if movement_stats["avg_speed"] > 5:
                meme = self._get_meme_for_category("Người di chuyển nhanh nhất")
                if meme:
                    assigned.append({"name": meme.get("name", ""), "image_url": meme.get("image_url", "")})

            if movement_stats["coverage_area"] > 50000:
                meme = self._get_meme_for_category("Người di chuyển rộng nhất")
                if meme:
                    assigned.append({"name": meme.get("name", ""), "image_url": meme.get("image_url", "")})

        return assigned

    def _get_meme_for_category(self, category: str) -> Optional[Dict]:
        """Lấy meme cho category (random nếu có nhiều)"""
        memes = self.meme_by_category.get(category, [])
        if memes:
            import random
            return random.choice(memes).copy()
        return None

    def _analyze_movement(self, positions, fps):
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
                speeds.append(math.sqrt(dx * dx + dy * dy))

        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        coverage_area = 0
        if x_coords and y_coords:
            coverage_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))

        return {"avg_speed": avg_speed, "coverage_area": coverage_area}
