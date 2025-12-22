# =============================================================================
# PLAYER ANALYSIS SERVICE - DỊCH VỤ PHÂN TÍCH NGƯỜI CHƠI TENNIS
# =============================================================================

import cv2
import numpy as np
import os
import uuid
import time
from typing import List, Tuple, Dict, Optional
from datetime import datetime

from .ball_detector import BallDetector
from .person_tracker import PersonTracker
from .player_stats_analyzer import PlayerStatsAnalyzer
from ..visualization.stats_visualizer import StatsVisualizer


class PlayerAnalysisService:
    """
    Service phân tích người chơi tennis với 8 chỉ số
    Trả về kết quả dạng JSON với đường dẫn file
    Optimized for 12GB GPU with batch inference
    """

    def __init__(
        self,
        ball_model_path: str = "models/ball_best.pt",
        person_model_path: str = "models/yolov8n.pt",
        pose_model_path: str = "models/yolov8n-pose.pt",
        batch_size: int = 16
    ):
        """
        Khởi tạo service

        Args:
            ball_model_path: Đường dẫn model detect bóng
            person_model_path: Đường dẫn model detect người
            pose_model_path: Đường dẫn model detect pose
            batch_size: Batch size for inference (default 16 for 12GB GPU)
        """
        self.batch_size = batch_size
        self.ball_detector = BallDetector(
            model_path=ball_model_path,
            person_model_path=person_model_path,
            batch_size=batch_size
        )
        # PersonTracker chỉ sử dụng pose model (đã bao gồm cả bbox và keypoints)
        self.person_tracker = PersonTracker(
            pose_model_path=pose_model_path,
            batch_size=batch_size
        )

    def analyze(
        self,
        video_path: str,
        court_points: List[Tuple[int, int]],
        output_folder: str,
        net_start_idx: int = 2,
        net_end_idx: int = 8,
        ball_conf: float = 0.7,
        person_conf: float = 0.6,
        angle_threshold: float = 50,
        intersection_threshold: float = 100,
        base_url: str = ""
    ) -> dict:
        """
        Phân tích video và trả về kết quả JSON

        Args:
            video_path: Đường dẫn video
            court_points: 12 điểm tọa độ sân
            output_folder: Thư mục lưu output
            net_start_idx: Index điểm bắt đầu lưới
            net_end_idx: Index điểm kết thúc lưới
            ball_conf: Confidence detect bóng
            person_conf: Confidence detect người
            angle_threshold: Ngưỡng góc thay đổi hướng
            intersection_threshold: Ngưỡng giao cắt bóng-người
            base_url: URL cơ sở cho file paths (vd: "outputs/abc123")

        Returns:
            Dict kết quả phân tích
        """
        os.makedirs(output_folder, exist_ok=True)

        timings = {}
        total_start = time.time()

        # 1. Đọc video
        step_start = time.time()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        timings["read_video"] = time.time() - step_start

        # 2. Detect bóng
        step_start = time.time()
        ball_positions = self.ball_detector.detect_positions(frames)
        timings["ball_detection"] = time.time() - step_start

        # 3. Person & Pose Detection (chỉ chạy 1 lần với pose model)
        # Pose model trả về cả bbox và keypoints - tối ưu không cần chạy 2 model
        step_start = time.time()
        self.person_tracker.tracked_persons = {}
        self.person_tracker.next_person_id = 1
        self.person_tracker.ball_hits_by_person.clear()
        self.person_tracker.player_positions.clear()

        # Chạy pose detection trước để lấy person_detections
        pose_detections, raw_person_detections = self.person_tracker._batch_pose_detection(
            frames, conf_threshold=person_conf
        )
        timings["person_pose_detection"] = time.time() - step_start

        # 4. Phân tích thay đổi hướng (sử dụng cached person_detections)
        step_start = time.time()
        direction_flags, _ = self.ball_detector.get_enhanced_direction_change_flags(
            frames,
            ball_positions,
            angle_threshold=angle_threshold,
            person_conf=person_conf,
            intersection_threshold=intersection_threshold,
            cached_person_detections=raw_person_detections
        )
        timings["direction_analysis"] = time.time() - step_start

        # 5. Tracking người (sử dụng pose detections đã có)
        step_start = time.time()
        person_detections_tracked = []
        for frame_idx in range(len(frames)):
            frame_poses = pose_detections[frame_idx]
            tracked_frame_data = self.person_tracker._track_persons_from_poses(frame_poses, frame_idx)
            person_detections_tracked.append(tracked_frame_data)

            if frame_idx < len(ball_positions) and ball_positions[frame_idx] != (-1, -1):
                self.person_tracker._check_ball_person_hits(
                    ball_positions[frame_idx],
                    tracked_frame_data,
                    direction_flags[frame_idx] if frame_idx < len(direction_flags) else 0,
                    frame_idx,
                )

        player_positions = self.person_tracker.get_player_positions()
        timings["person_tracking"] = time.time() - step_start

        # 6. Phân tích chỉ số
        step_start = time.time()
        stats_analyzer = PlayerStatsAnalyzer(
            court_points=court_points,
            net_start_idx=net_start_idx,
            net_end_idx=net_end_idx,
            fps=fps
        )

        stats_analyzer.set_data(
            ball_positions=ball_positions,
            ball_hits_by_person=dict(self.person_tracker.ball_hits_by_person),
            player_positions=player_positions,
            direction_flags=direction_flags
        )

        all_stats = stats_analyzer.get_all_players_stats()
        rankings = stats_analyzer.calculate_player_ranking()
        timings["stats_analysis"] = time.time() - step_start

        # 7. Tạo visualization
        step_start = time.time()
        visualizer = StatsVisualizer(court_points=court_points)
        court_image = frames[0].copy() if frames else None

        output_files = {}

        # Ranking board
        ranking_path = os.path.join(output_folder, "ranking_board.png")
        visualizer.create_ranking_board(rankings, output_path=ranking_path)
        output_files["ranking_board"] = f"{base_url}/ranking_board.png"

        # Speed comparison
        serve_speeds = {}
        drive_speeds = {}
        for player_id, stats in all_stats.items():
            serve = stats.get("serve", {})
            drive = stats.get("drive", {})
            serve_speeds[player_id] = serve.get("speeds", [])
            drive_speeds[player_id] = drive.get("speeds", [])

        speed_path = os.path.join(output_folder, "speed_comparison.png")
        visualizer.create_speed_chart(serve_speeds, drive_speeds, output_path=speed_path)
        output_files["speed_comparison"] = f"{base_url}/speed_comparison.png"

        # Per-player outputs
        player_images = {}
        for player_id, stats in all_stats.items():
            player_output = {}

            # Stats table
            stats_path = os.path.join(output_folder, f"player_{player_id}_stats.png")
            visualizer.create_stats_table(stats, output_path=stats_path)
            player_output["stats_table"] = f"{base_url}/player_{player_id}_stats.png"

            # Heatmap
            positions = stats.get("heatmap_positions", [])
            if positions:
                heatmap_path = os.path.join(output_folder, f"player_{player_id}_heatmap.png")
                visualizer.create_heatmap(
                    positions,
                    court_image=court_image,
                    output_path=heatmap_path,
                    player_id=player_id
                )
                player_output["heatmap"] = f"{base_url}/player_{player_id}_heatmap.png"

            # Shot density pie
            density = stats.get("shot_density", {})
            if density and density.get("counts"):
                density_path = os.path.join(output_folder, f"player_{player_id}_density.png")
                visualizer.create_shot_density_pie(density, player_id, output_path=density_path)
                player_output["shot_density_chart"] = f"{base_url}/player_{player_id}_density.png"

            # Player crop image
            if player_id in self.person_tracker.tracked_persons:
                person_data = self.person_tracker.tracked_persons[player_id]
                last_frame_idx = person_data.get("last_seen", 0)
                if last_frame_idx < len(frames):
                    bbox = person_data.get("bbox")
                    if bbox:
                        crop = visualizer.crop_player_image(frames[last_frame_idx], bbox)
                        if crop is not None and crop.size > 0:
                            crop_path = os.path.join(output_folder, f"player_{player_id}_image.jpg")
                            cv2.imwrite(crop_path, crop)
                            player_output["player_image"] = f"{base_url}/player_{player_id}_image.jpg"

            player_images[player_id] = player_output

        timings["visualization"] = time.time() - step_start

        # 7. Xây dựng kết quả JSON
        result = {
            "video_info": {
                "path": video_path,
                "fps": fps,
                "total_frames": len(frames),
                "duration_seconds": round(len(frames) / fps, 2) if fps > 0 else 0
            },
            "court_info": {
                "points": court_points,
                "net_line": {
                    "start": court_points[net_start_idx],
                    "end": court_points[net_end_idx]
                }
            },
            "summary": {
                "total_players": len(all_stats),
                "total_hits": sum(s.get("accuracy", {}).get("total_hits", 0) for s in all_stats.values()),
                "total_in_court": sum(s.get("accuracy", {}).get("in_court", 0) for s in all_stats.values()),
                "total_out_court": sum(s.get("accuracy", {}).get("out_court", 0) for s in all_stats.values()),
                "total_not_over_net": sum(s.get("accuracy", {}).get("not_over_net", 0) for s in all_stats.values())
            },
            "players": {},
            "rankings": rankings,
            "visualizations": {
                **output_files,
                "per_player": player_images
            },
            "timestamp": datetime.now().isoformat()
        }

        # Chi tiết từng người chơi
        for player_id, stats in all_stats.items():
            player_data = {
                "player_id": player_id,
                "accuracy": stats.get("accuracy", {}),
                "serve": {
                    "total": stats.get("serve", {}).get("total", 0),
                    "in_court": stats.get("serve", {}).get("in_court", 0),
                    "out_court": stats.get("serve", {}).get("out_court", 0),
                    "not_over_net": stats.get("serve", {}).get("not_over_net", 0),
                    "avg_speed": round(stats.get("serve", {}).get("avg_speed", 0), 2),
                    "max_speed": round(stats.get("serve", {}).get("max_speed", 0), 2)
                },
                "return": {
                    "total": stats.get("return", {}).get("total", 0),
                    "in_court": stats.get("return", {}).get("in_court", 0),
                    "out_court": stats.get("return", {}).get("out_court", 0),
                    "not_over_net": stats.get("return", {}).get("not_over_net", 0)
                },
                "drive": {
                    "avg_speed": round(stats.get("drive", {}).get("avg_speed", 0), 2),
                    "max_speed": round(stats.get("drive", {}).get("max_speed", 0), 2)
                },
                "shot_density": {
                    "long_pct": round(stats.get("shot_density", {}).get("long_pct", 0), 1),
                    "medium_pct": round(stats.get("shot_density", {}).get("medium_pct", 0), 1),
                    "short_pct": round(stats.get("shot_density", {}).get("short_pct", 0), 1)
                },
                "ranking": stats.get("ranking", {}),
                "images": player_images.get(player_id, {})
            }
            result["players"][f"player_{player_id}"] = player_data

        # Calculate total time
        total_time = time.time() - total_start
        timings["total"] = total_time

        # Add timing info to result
        result["performance"] = {
            "total_time_seconds": round(total_time, 2),
            "average_fps": round(len(frames) / total_time, 1) if total_time > 0 else 0,
            "batch_size": self.batch_size,
            "timings": {k: round(v, 2) for k, v in timings.items()}
        }

        return result
