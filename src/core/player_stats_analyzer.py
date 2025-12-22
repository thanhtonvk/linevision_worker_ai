# =============================================================================
# PLAYER STATS ANALYZER - PHÂN TÍCH CHỈ SỐ NGƯỜI CHƠI TENNIS
# =============================================================================

import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from ..utils.court_geometry import CourtGeometry


class PlayerStatsAnalyzer:
    """
    Class phân tích chỉ số chi tiết cho từng người chơi tennis
    Bao gồm 8 chỉ số chính:
    1. Độ chính xác (Accuracy)
    2. Giao bóng (Serve)
    3. Trả bóng (Return)
    4. Tốc độ giao bóng (Serve Speed)
    5. Tốc độ Drive (Drive Speed)
    6. Mật độ bóng (Shot Density)
    7. Phạm vi bao quát sân (Coverage Heatmap)
    8. Bảng xếp hạng (Ranking)
    """

    def __init__(
        self,
        court_points: List[Tuple[int, int]],
        net_start_idx: int = 2,
        net_end_idx: int = 8,
        fps: float = 30.0
    ):
        """
        Khởi tạo PlayerStatsAnalyzer

        Args:
            court_points: 12 điểm tọa độ sân
            net_start_idx: Index điểm bắt đầu lưới (mặc định = 2)
            net_end_idx: Index điểm kết thúc lưới (mặc định = 8)
            fps: Frame rate của video
        """
        self.court = CourtGeometry(court_points, net_start_idx, net_end_idx)
        self.fps = fps

        # Dữ liệu theo dõi
        self.player_shots = defaultdict(list)  # {player_id: [shot_data, ...]}
        self.player_positions = defaultdict(list)  # {player_id: [(frame, x, y), ...]}
        self.ball_positions = []  # [(x, y), ...] theo frame
        self.rallies = []  # Danh sách các rally

    def set_data(
        self,
        ball_positions: List[Tuple[float, float]],
        ball_hits_by_person: Dict[int, List[dict]],
        player_positions: Dict[int, List[Tuple[int, float, float]]],
        direction_flags: List[int]
    ):
        """
        Thiết lập dữ liệu từ các module tracking

        Args:
            ball_positions: Vị trí bóng theo frame
            ball_hits_by_person: Các cú đánh theo người chơi
            player_positions: Vị trí người chơi theo frame
            direction_flags: Cờ thay đổi hướng bóng
        """
        self.ball_positions = ball_positions
        self.player_positions = player_positions
        self.direction_flags = direction_flags

        # Xử lý các cú đánh
        self._process_shots(ball_hits_by_person)

        # Phân tích rally
        self._analyze_rallies()

    def _process_shots(self, ball_hits_by_person: Dict[int, List[dict]]):
        """Xử lý và phân loại các cú đánh"""
        for player_id, hits in ball_hits_by_person.items():
            for hit in hits:
                frame_idx = hit["frame"]
                ball_pos = hit["ball_pos"]
                player_bbox = hit.get("person_bbox", None)

                # Tính vị trí người chơi (center của bbox)
                player_pos = None
                if player_bbox:
                    x1, y1, x2, y2 = player_bbox
                    player_pos = ((x1 + x2) / 2, (y1 + y2) / 2)

                # Tính vị trí bóng trước và sau cú đánh
                prev_ball_pos = self._get_ball_pos_before(frame_idx)
                next_ball_pos = self._get_ball_pos_after(frame_idx)

                # Phân loại cú đánh
                shot_info = self._classify_shot(
                    player_pos, ball_pos, prev_ball_pos, next_ball_pos, frame_idx
                )

                shot_data = {
                    "frame": frame_idx,
                    "ball_pos": ball_pos,
                    "player_pos": player_pos,
                    "prev_ball_pos": prev_ball_pos,
                    "next_ball_pos": next_ball_pos,
                    **shot_info
                }

                self.player_shots[player_id].append(shot_data)

    def _get_ball_pos_before(self, frame_idx: int, lookback: int = 5) -> Optional[Tuple[float, float]]:
        """Lấy vị trí bóng trước cú đánh"""
        for i in range(frame_idx - 1, max(0, frame_idx - lookback) - 1, -1):
            if i < len(self.ball_positions) and self.ball_positions[i] != (-1, -1):
                return self.ball_positions[i]
        return None

    def _get_ball_pos_after(self, frame_idx: int, lookahead: int = 10) -> Optional[Tuple[float, float]]:
        """Lấy vị trí bóng sau cú đánh"""
        for i in range(frame_idx + 1, min(len(self.ball_positions), frame_idx + lookahead + 1)):
            if self.ball_positions[i] != (-1, -1):
                return self.ball_positions[i]
        return None

    def _classify_shot(
        self,
        player_pos: Optional[Tuple[float, float]],
        ball_pos: Tuple[float, float],
        prev_pos: Optional[Tuple[float, float]],
        next_pos: Optional[Tuple[float, float]],
        frame_idx: int
    ) -> dict:
        """Phân loại cú đánh (serve, drive, return)"""
        is_serve = False
        is_return = False
        crossed_net = False
        in_court = False
        distance = 0.0
        speed = 0.0

        # Kiểm tra vị trí giao bóng
        if player_pos:
            is_serve = self.court.is_serve_position(player_pos)

        # Kiểm tra bóng qua lưới
        if prev_pos and next_pos:
            crossed_net = self.court.is_ball_crossed_net(ball_pos, next_pos)

        # Kiểm tra điểm rơi trong sân
        if next_pos:
            in_court = self.court.is_point_in_court(next_pos)
            distance = self.court.calculate_shot_distance(ball_pos, next_pos)

        # Tính tốc độ bóng (pixels/frame)
        if prev_pos and next_pos:
            total_distance = self.court.calculate_shot_distance(prev_pos, next_pos)
            speed = total_distance / 2  # Chia cho 2 vì qua 2 frames

        # Phân loại độ dài cú đánh
        length_type = self.court.classify_shot_length(distance)

        return {
            "is_serve": is_serve,
            "is_return": is_return,
            "crossed_net": crossed_net,
            "in_court": in_court,
            "distance": distance,
            "speed": speed,
            "length_type": length_type
        }

    def _analyze_rallies(self):
        """Phân tích rally để xác định serve và return"""
        # Tìm các điểm bắt đầu rally (khi bóng từ baseline)
        current_rally = []
        rally_start = True

        for player_id, shots in self.player_shots.items():
            for i, shot in enumerate(sorted(shots, key=lambda x: x["frame"])):
                if rally_start and shot["is_serve"]:
                    shot["shot_type"] = "serve"
                    rally_start = False
                elif not rally_start and len(current_rally) == 1:
                    shot["shot_type"] = "return"
                    shot["is_return"] = True
                else:
                    shot["shot_type"] = "drive"

                current_rally.append(shot)

    # =========================================================================
    # 1. ĐỘ CHÍNH XÁC (ACCURACY)
    # =========================================================================
    def calculate_accuracy(self, player_id: int) -> dict:
        """
        Tính độ chính xác cho người chơi

        Returns:
            {
                "total_hits": int,
                "in_court": int,
                "out_court": int,
                "not_over_net": int
            }
        """
        shots = self.player_shots.get(player_id, [])

        total_hits = len(shots)
        in_court = sum(1 for s in shots if s["in_court"] and s["crossed_net"])
        out_court = sum(1 for s in shots if not s["in_court"] and s["crossed_net"])
        not_over_net = sum(1 for s in shots if not s["crossed_net"])

        return {
            "total_hits": total_hits,
            "in_court": in_court,
            "out_court": out_court,
            "not_over_net": not_over_net,
            "accuracy_pct": (in_court / total_hits * 100) if total_hits > 0 else 0.0
        }

    # =========================================================================
    # 2. GIAO BÓNG (SERVE)
    # =========================================================================
    def analyze_serves(self, player_id: int) -> dict:
        """
        Phân tích các cú giao bóng

        Returns:
            {
                "total": int,
                "in_court": int,
                "out_court": int,
                "not_over_net": int
            }
        """
        shots = self.player_shots.get(player_id, [])
        serves = [s for s in shots if s.get("is_serve", False) or s.get("shot_type") == "serve"]

        total = len(serves)
        in_court = sum(1 for s in serves if s["in_court"] and s["crossed_net"])
        out_court = sum(1 for s in serves if not s["in_court"] and s["crossed_net"])
        not_over_net = sum(1 for s in serves if not s["crossed_net"])

        return {
            "total": total,
            "in_court": in_court,
            "out_court": out_court,
            "not_over_net": not_over_net,
            "success_rate": (in_court / total * 100) if total > 0 else 0.0
        }

    # =========================================================================
    # 3. TRẢ BÓNG (RETURN)
    # =========================================================================
    def analyze_returns(self, player_id: int) -> dict:
        """
        Phân tích các cú trả bóng

        Returns:
            {
                "total": int,
                "in_court": int,
                "out_court": int,
                "not_over_net": int
            }
        """
        shots = self.player_shots.get(player_id, [])
        returns = [s for s in shots if s.get("is_return", False) or s.get("shot_type") == "return"]

        total = len(returns)
        in_court = sum(1 for s in returns if s["in_court"] and s["crossed_net"])
        out_court = sum(1 for s in returns if not s["in_court"] and s["crossed_net"])
        not_over_net = sum(1 for s in returns if not s["crossed_net"])

        return {
            "total": total,
            "in_court": in_court,
            "out_court": out_court,
            "not_over_net": not_over_net,
            "success_rate": (in_court / total * 100) if total > 0 else 0.0
        }

    # =========================================================================
    # 4. TỐC ĐỘ GIAO BÓNG (SERVE SPEED)
    # =========================================================================
    def calculate_serve_speed(self, player_id: int) -> dict:
        """
        Tính tốc độ giao bóng (pixels/frame)

        Returns:
            {
                "avg_speed": float,
                "max_speed": float,
                "speeds": list
            }
        """
        shots = self.player_shots.get(player_id, [])
        serves = [s for s in shots if s.get("is_serve", False) or s.get("shot_type") == "serve"]

        speeds = [s["speed"] for s in serves if s["speed"] > 0]

        return {
            "avg_speed": np.mean(speeds) if speeds else 0.0,
            "max_speed": max(speeds) if speeds else 0.0,
            "speeds": speeds
        }

    # =========================================================================
    # 5. TỐC ĐỘ DRIVE (DRIVE SPEED)
    # =========================================================================
    def calculate_drive_speed(self, player_id: int) -> dict:
        """
        Tính tốc độ drive (pixels/frame)

        Returns:
            {
                "avg_speed": float,
                "max_speed": float,
                "speeds": list
            }
        """
        shots = self.player_shots.get(player_id, [])
        drives = [s for s in shots if s.get("shot_type") == "drive" or (not s.get("is_serve", False) and not s.get("is_return", False))]

        speeds = [s["speed"] for s in drives if s["speed"] > 0]

        return {
            "avg_speed": np.mean(speeds) if speeds else 0.0,
            "max_speed": max(speeds) if speeds else 0.0,
            "speeds": speeds
        }

    # =========================================================================
    # 6. MẬT ĐỘ BÓNG (SHOT DENSITY)
    # =========================================================================
    def analyze_shot_length(self, player_id: int) -> dict:
        """
        Phân tích mật độ/độ dài cú đánh

        Returns:
            {
                "long_pct": float,
                "medium_pct": float,
                "short_pct": float,
                "counts": {"long": int, "medium": int, "short": int}
            }
        """
        shots = self.player_shots.get(player_id, [])

        counts = {"long": 0, "medium": 0, "short": 0}
        for shot in shots:
            length_type = shot.get("length_type", "medium")
            counts[length_type] = counts.get(length_type, 0) + 1

        total = sum(counts.values())

        return {
            "long_pct": (counts["long"] / total * 100) if total > 0 else 0.0,
            "medium_pct": (counts["medium"] / total * 100) if total > 0 else 0.0,
            "short_pct": (counts["short"] / total * 100) if total > 0 else 0.0,
            "counts": counts
        }

    # =========================================================================
    # 7. PHẠM VI BAO QUÁT SÂN (HEATMAP DATA)
    # =========================================================================
    def get_heatmap_data(self, player_id: int) -> List[Tuple[float, float]]:
        """
        Lấy dữ liệu vị trí để vẽ heatmap

        Returns:
            List các tọa độ (x, y) của người chơi qua các frame
        """
        positions = self.player_positions.get(player_id, [])
        return [(x, y) for frame, x, y in positions if x != -1 and y != -1]

    # =========================================================================
    # 8. BẢNG XẾP HẠNG (RANKING)
    # =========================================================================
    def calculate_player_ranking(self) -> List[dict]:
        """
        Tính bảng xếp hạng người chơi (chỉ tính người có cú đánh)

        Returns:
            List sorted by score: [
                {
                    "player_id": int,
                    "rank": int,
                    "score": float,
                    "in_out_ratio": float,
                    "avg_speed": float,
                    "total_hits": int
                },
                ...
            ]
        """
        rankings = []

        all_speeds = []
        all_hits = []

        # Lọc player có cú đánh
        active_players = [
            pid for pid in self.player_shots.keys()
            if len(self.player_shots[pid]) > 0
        ]

        # Thu thập dữ liệu để normalize
        for player_id in active_players:
            accuracy = self.calculate_accuracy(player_id)
            serve_speed = self.calculate_serve_speed(player_id)
            drive_speed = self.calculate_drive_speed(player_id)

            all_speeds.extend(serve_speed["speeds"])
            all_speeds.extend(drive_speed["speeds"])
            all_hits.append(accuracy["total_hits"])

        max_speed = max(all_speeds) if all_speeds else 1
        max_hits = max(all_hits) if all_hits else 1

        # Tính điểm cho từng người (chỉ active players)
        for player_id in active_players:
            accuracy = self.calculate_accuracy(player_id)
            serve_speed = self.calculate_serve_speed(player_id)
            drive_speed = self.calculate_drive_speed(player_id)
            shot_density = self.analyze_shot_length(player_id)

            # Tỷ lệ trong sân
            total_hits = accuracy["total_hits"]
            in_out_ratio = accuracy["in_court"] / total_hits if total_hits > 0 else 0

            # Tốc độ trung bình (kết hợp serve và drive)
            avg_speed = (serve_speed["avg_speed"] + drive_speed["avg_speed"]) / 2

            # Normalized scores
            normalized_speed = avg_speed / max_speed if max_speed > 0 else 0
            normalized_hits = total_hits / max_hits if max_hits > 0 else 0

            # Đa dạng cú đánh (entropy-like)
            density = shot_density
            variety_score = 1 - abs(density["long_pct"] - density["short_pct"]) / 100

            # Tổng điểm
            score = (
                0.4 * in_out_ratio +
                0.3 * normalized_speed +
                0.2 * normalized_hits +
                0.1 * variety_score
            )

            rankings.append({
                "player_id": player_id,
                "score": score,
                "in_out_ratio": in_out_ratio,
                "avg_speed": avg_speed,
                "total_hits": total_hits,
                "accuracy_pct": accuracy["accuracy_pct"]
            })

        # Sắp xếp theo điểm giảm dần
        rankings.sort(key=lambda x: x["score"], reverse=True)

        # Thêm rank
        for i, r in enumerate(rankings):
            r["rank"] = i + 1

        return rankings

    # =========================================================================
    # TỔNG HỢP TẤT CẢ CHỈ SỐ
    # =========================================================================
    def get_full_player_stats(self, player_id: int) -> dict:
        """
        Lấy tất cả chỉ số cho một người chơi

        Returns:
            Dict chứa đầy đủ 8 chỉ số
        """
        return {
            "player_id": player_id,
            "accuracy": self.calculate_accuracy(player_id),
            "serve": {
                **self.analyze_serves(player_id),
                **self.calculate_serve_speed(player_id)
            },
            "return": self.analyze_returns(player_id),
            "drive": self.calculate_drive_speed(player_id),
            "shot_density": self.analyze_shot_length(player_id),
            "heatmap_positions": self.get_heatmap_data(player_id)
        }

    def get_all_players_stats(self) -> dict:
        """
        Lấy tất cả chỉ số cho tất cả người chơi có cú đánh

        Returns:
            {player_id: stats_dict, ...}
        """
        all_stats = {}

        # Chỉ trả về player có cú đánh (hits > 0)
        for player_id in self.player_shots.keys():
            shots = self.player_shots[player_id]
            if len(shots) > 0:  # Chỉ tính player có cú đánh
                all_stats[player_id] = self.get_full_player_stats(player_id)

        # Thêm ranking
        rankings = self.calculate_player_ranking()
        for r in rankings:
            pid = r["player_id"]
            if pid in all_stats:
                all_stats[pid]["ranking"] = r

        print(f"📊 Thống kê: {len(all_stats)} người chơi có cú đánh")
        return all_stats
