# =============================================================================
# COURT GEOMETRY - TÍNH TOÁN HÌNH HỌC SÂN TENNIS
# =============================================================================

import numpy as np
import math
from typing import List, Tuple, Optional


class CourtGeometry:
    """
    Class xử lý tính toán hình học cho sân tennis
    Hỗ trợ:
    - Kiểm tra điểm trong sân (point-in-polygon)
    - Kiểm tra bóng qua lưới
    - Xác định vùng người chơi
    - Tính khoảng cách và phân loại cú đánh
    """

    def __init__(self, court_points: List[Tuple[int, int]], net_start_idx: int = 2, net_end_idx: int = 8):
        """
        Khởi tạo CourtGeometry

        Args:
            court_points: Danh sách 12 điểm tọa độ sân [(x, y), ...]
            net_start_idx: Index điểm bắt đầu lưới (mặc định = 2, tức điểm thứ 3)
            net_end_idx: Index điểm kết thúc lưới (mặc định = 8, tức điểm thứ 9)
        """
        self.court_points = court_points
        self.court_polygon = np.array(court_points, dtype=np.float32)

        # Đường lưới (net line)
        self.net_start = court_points[net_start_idx]  # Điểm 3
        self.net_end = court_points[net_end_idx]      # Điểm 9
        self.net_y = (self.net_start[1] + self.net_end[1]) / 2

        # Tính các vùng sân
        self._calculate_court_zones()

        # Tính chiều dài sân để phân loại cú đánh
        self._calculate_court_dimensions()

    def _calculate_court_zones(self):
        """Tính toán các vùng sân (baseline, midcourt)"""
        all_y = [p[1] for p in self.court_points]
        self.min_y = min(all_y)
        self.max_y = max(all_y)

        # Baseline thresholds
        court_height = self.max_y - self.min_y
        self.baseline_top_threshold = self.min_y + court_height * 0.25
        self.baseline_bottom_threshold = self.max_y - court_height * 0.25

    def _calculate_court_dimensions(self):
        """Tính kích thước sân để phân loại độ dài cú đánh"""
        all_x = [p[0] for p in self.court_points]
        all_y = [p[1] for p in self.court_points]

        self.court_width = max(all_x) - min(all_x)
        self.court_height = max(all_y) - min(all_y)

        # Diagonal length để làm chuẩn
        self.court_diagonal = math.sqrt(self.court_width**2 + self.court_height**2)

        # Thresholds cho short/medium/long shots
        self.short_threshold = self.court_diagonal * 0.33
        self.long_threshold = self.court_diagonal * 0.66

    def is_point_in_court(self, point: Tuple[float, float]) -> bool:
        """
        Kiểm tra điểm có nằm trong sân không (point-in-polygon)

        Args:
            point: Tọa độ (x, y)

        Returns:
            True nếu điểm trong sân, False nếu ngoài
        """
        if point == (-1, -1) or point is None:
            return False

        x, y = point
        n = len(self.court_polygon)
        inside = False

        p1x, p1y = self.court_polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = self.court_polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def is_ball_crossed_net(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float]) -> bool:
        """
        Kiểm tra bóng có vượt qua lưới không

        Args:
            start_pos: Vị trí bắt đầu của bóng
            end_pos: Vị trí kết thúc của bóng

        Returns:
            True nếu bóng đã vượt qua lưới
        """
        if start_pos == (-1, -1) or end_pos == (-1, -1):
            return False
        if start_pos is None or end_pos is None:
            return False

        start_y = start_pos[1]
        end_y = end_pos[1]

        # Nếu bắt đầu và kết thúc ở khác phía của net -> đã qua lưới
        return (start_y < self.net_y) != (end_y < self.net_y)

    def get_player_zone(self, player_pos: Tuple[float, float]) -> str:
        """
        Xác định vùng người chơi đang đứng

        Args:
            player_pos: Vị trí (x, y) của người chơi

        Returns:
            "baseline_top", "baseline_bottom", hoặc "midcourt"
        """
        if player_pos is None or player_pos == (-1, -1):
            return "unknown"

        y = player_pos[1]

        if y < self.baseline_top_threshold:
            return "baseline_top"
        elif y > self.baseline_bottom_threshold:
            return "baseline_bottom"
        else:
            return "midcourt"

    def is_serve_position(self, player_pos: Tuple[float, float]) -> bool:
        """
        Kiểm tra người chơi có ở vị trí giao bóng không

        Args:
            player_pos: Vị trí người chơi

        Returns:
            True nếu ở vị trí có thể giao bóng (baseline)
        """
        zone = self.get_player_zone(player_pos)
        return zone in ["baseline_top", "baseline_bottom"]

    def calculate_shot_distance(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """
        Tính khoảng cách di chuyển của bóng

        Args:
            start: Vị trí bắt đầu
            end: Vị trí kết thúc

        Returns:
            Khoảng cách (pixels)
        """
        if start == (-1, -1) or end == (-1, -1):
            return 0.0
        if start is None or end is None:
            return 0.0

        return math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

    def classify_shot_length(self, distance: float) -> str:
        """
        Phân loại độ dài cú đánh

        Args:
            distance: Khoảng cách di chuyển của bóng

        Returns:
            "short", "medium", hoặc "long"
        """
        if distance < self.short_threshold:
            return "short"
        elif distance < self.long_threshold:
            return "medium"
        else:
            return "long"

    def get_shot_trajectory_info(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float]
    ) -> dict:
        """
        Lấy thông tin đầy đủ về quỹ đạo cú đánh

        Args:
            start_pos: Vị trí bắt đầu
            end_pos: Vị trí kết thúc

        Returns:
            Dict chứa thông tin: crossed_net, in_court, distance, length_type
        """
        crossed_net = self.is_ball_crossed_net(start_pos, end_pos)
        in_court = self.is_point_in_court(end_pos)
        distance = self.calculate_shot_distance(start_pos, end_pos)
        length_type = self.classify_shot_length(distance)

        return {
            "crossed_net": crossed_net,
            "in_court": in_court,
            "distance": distance,
            "length_type": length_type,
            "start_pos": start_pos,
            "end_pos": end_pos
        }

    def get_player_side(self, player_pos: Tuple[float, float]) -> str:
        """
        Xác định người chơi ở bên nào của sân (so với lưới)

        Args:
            player_pos: Vị trí người chơi

        Returns:
            "top" hoặc "bottom"
        """
        if player_pos is None or player_pos == (-1, -1):
            return "unknown"

        return "top" if player_pos[1] < self.net_y else "bottom"

    def get_court_bounds(self) -> Tuple[int, int, int, int]:
        """
        Lấy bounding box của sân

        Returns:
            (x_min, y_min, x_max, y_max)
        """
        all_x = [p[0] for p in self.court_points]
        all_y = [p[1] for p in self.court_points]
        return (min(all_x), min(all_y), max(all_x), max(all_y))
