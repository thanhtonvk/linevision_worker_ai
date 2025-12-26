# =============================================================================
# PLAYER ANALYSIS SERVICE - DỊCH VỤ PHÂN TÍCH NGƯỜI CHƠI TENNIS
# =============================================================================
# Memory-optimized version: Xử lý video theo batch, không load toàn bộ vào RAM

import cv2
import os
import time
from typing import List, Tuple
from datetime import datetime

from .ball_detector import BallDetector
from .person_tracker import PersonTracker
from .player_stats_analyzer import PlayerStatsAnalyzer
from .meme_analyzer import MemeAnalyzer
from ..visualization.stats_visualizer import StatsVisualizer


class PlayerAnalysisService:
    """
    Service phân tích người chơi tennis với 8 chỉ số
    Trả về kết quả dạng JSON với đường dẫn file
    Memory-optimized: Xử lý video theo batch, không load toàn bộ frames vào RAM
    """

    def __init__(
        self,
        ball_model_path: str = "models/ball_best.pt",
        person_model_path: str = "yolo11m.pt",
        batch_size: int = 16
    ):
        """
        Khởi tạo service

        Args:
            ball_model_path: Đường dẫn model detect bóng
            person_model_path: Đường dẫn model detect người
            batch_size: Batch size for inference (default 16 for 12GB GPU)
        """
        self.batch_size = batch_size
        self.ball_detector = BallDetector(
            model_path=ball_model_path,
            person_model_path=person_model_path,
            batch_size=batch_size
        )
        # PersonTracker sử dụng yolo11m cho person detection
        self.person_tracker = PersonTracker(
            person_model_path=person_model_path,
            batch_size=batch_size
        )
        # MemeAnalyzer để phân tích và gán meme
        self.meme_analyzer = MemeAnalyzer(meme_json_path="data/meme.json")

    def _read_frames_batch(self, cap, batch_size):
        """Đọc một batch frames từ video capture

        Args:
            cap: cv2.VideoCapture object
            batch_size: Số frames cần đọc

        Returns:
            List of frames (có thể ít hơn batch_size nếu hết video)
        """
        frames = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def _get_video_info(self, video_path):
        """Lấy thông tin video mà không load frames

        Returns:
            dict với fps, total_frames, width, height
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height
        }

    def _read_specific_frames(self, video_path, frame_indices):
        """Đọc các frames cụ thể từ video

        Args:
            video_path: Đường dẫn video
            frame_indices: List hoặc set các frame index cần đọc

        Returns:
            Dict mapping frame_idx -> frame
        """
        frame_indices = sorted(set(frame_indices))
        frames_dict = {}

        cap = cv2.VideoCapture(video_path)
        current_idx = 0
        target_idx = 0

        while target_idx < len(frame_indices):
            target_frame = frame_indices[target_idx]

            # Skip đến frame cần đọc
            if current_idx < target_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                current_idx = target_frame

            ret, frame = cap.read()
            if not ret:
                break

            if current_idx == target_frame:
                frames_dict[current_idx] = frame
                target_idx += 1

            current_idx += 1

        cap.release()
        return frames_dict

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
        base_url: str = "",
        create_highlights: bool = True
    ) -> dict:
        """
        Phân tích video và trả về kết quả JSON
        Memory-optimized: Xử lý theo batch, không load toàn bộ video vào RAM

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
            create_highlights: Tạo video highlight cho từng player

        Returns:
            Dict kết quả phân tích
        """
        os.makedirs(output_folder, exist_ok=True)

        timings = {}
        total_start = time.time()

        # 1. Lấy thông tin video
        step_start = time.time()
        video_info = self._get_video_info(video_path)
        fps = video_info["fps"]
        total_frames = video_info["total_frames"]
        timings["get_video_info"] = time.time() - step_start

        print(f"[INFO] Video: {total_frames} frames, {fps} FPS, {video_info['width']}x{video_info['height']}")

        # 2. PASS 1: Detect ball và person theo batch (không giữ frames trong RAM)
        step_start = time.time()

        # Reset trackers
        self.person_tracker.tracked_persons = {}
        self.person_tracker.next_person_id = 1
        self.person_tracker.ball_hits_by_person.clear()
        self.person_tracker.player_positions.clear()
        self.person_tracker.player_frames.clear()

        # Lists để lưu kết quả detection
        all_ball_positions = []
        all_person_detections = []
        first_frame = None
        player_last_frames = {}  # player_id -> (frame_idx, bbox)

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while True:
            # Đọc batch frames
            batch_frames = self._read_frames_batch(cap, self.batch_size)
            if not batch_frames:
                break

            # Lưu frame đầu tiên cho visualization
            if first_frame is None:
                first_frame = batch_frames[0].copy()

            # Detect ball trong batch
            batch_ball_positions = self.ball_detector.detect_positions(batch_frames)
            all_ball_positions.extend(batch_ball_positions)

            # Detect person trong batch
            batch_person_detections = self.person_tracker._batch_person_detection(
                batch_frames, conf_threshold=person_conf
            )
            all_person_detections.extend(batch_person_detections)

            # Tracking và check hits cho từng frame trong batch
            for i, frame_persons in enumerate(batch_person_detections):
                current_frame_idx = frame_idx + i
                tracked_frame_data = self.person_tracker._track_persons_across_frames(
                    frame_persons, current_frame_idx
                )

                # Lưu frame cuối của mỗi player để crop ảnh sau
                for person_id, person_data in self.person_tracker.tracked_persons.items():
                    if person_data.get("last_seen") == current_frame_idx:
                        player_last_frames[person_id] = (current_frame_idx, person_data.get("bbox"))

                # Check ball-person hits
                ball_idx = current_frame_idx
                if ball_idx < len(all_ball_positions) and all_ball_positions[ball_idx] != (-1, -1):
                    self.person_tracker._check_ball_person_hits(
                        all_ball_positions[ball_idx],
                        tracked_frame_data,
                        0,  # direction_flag sẽ tính sau
                        current_frame_idx,
                    )

            frame_idx += len(batch_frames)

            # Giải phóng batch frames
            del batch_frames

            if frame_idx % 500 == 0:
                print(f"[PROGRESS] Processed {frame_idx}/{total_frames} frames...")

        cap.release()
        timings["detection_pass"] = time.time() - step_start

        print(f"[INFO] Detection complete: {len(all_ball_positions)} ball positions, {len(all_person_detections)} person detections")

        # 3. Post-process ball positions (nội suy, smooth)
        step_start = time.time()

        # Đếm số frame bị miss trước khi nội suy
        missed_before = sum(1 for p in all_ball_positions if p == (-1, -1))

        # Nội suy với parabol để fill các gaps (max 15 frames)
        ball_positions = self.ball_detector.interpolate_missing_positions(
            all_ball_positions, max_gap=15, use_parabolic=True
        )

        # Smooth để giảm nhiễu
        ball_positions = self.ball_detector.smooth_positions(ball_positions, window_size=5)

        # Đếm số frame bị miss sau khi nội suy
        missed_after = sum(1 for p in ball_positions if p == (-1, -1))

        timings["ball_postprocess"] = time.time() - step_start

        # 4. Phân tích thay đổi hướng (không cần frames, chỉ cần ball_positions và person_detections)
        step_start = time.time()
        direction_flags = self._calculate_direction_flags(
            ball_positions,
            all_person_detections,
            angle_threshold,
            intersection_threshold
        )
        timings["direction_analysis"] = time.time() - step_start

        # 5. Cập nhật lại hits với direction flags
        step_start = time.time()
        self.person_tracker.ball_hits_by_person.clear()
        for frame_idx in range(len(ball_positions)):
            if ball_positions[frame_idx] != (-1, -1) and frame_idx < len(all_person_detections):
                tracked_data = []
                for det in all_person_detections[frame_idx]:
                    # Tìm tracked person tương ứng
                    for person_id, person_data in self.person_tracker.tracked_persons.items():
                        if person_data.get("first_seen", 0) <= frame_idx <= person_data.get("last_seen", 0):
                            tracked_data.append({
                                "person_id": person_id,
                                "bbox": det["bbox"],
                                "conf": det["conf"]
                            })
                            break

                self.person_tracker._check_ball_person_hits(
                    ball_positions[frame_idx],
                    tracked_data,
                    direction_flags[frame_idx] if frame_idx < len(direction_flags) else 0,
                    frame_idx,
                )

        player_positions = self.person_tracker.get_player_positions()
        timings["hits_update"] = time.time() - step_start

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

        # 6.5. Phân tích meme cho các cú đánh
        step_start = time.time()
        meme_analysis = self.meme_analyzer.analyze_shots(
            ball_positions=ball_positions,
            ball_hits_by_person=dict(self.person_tracker.ball_hits_by_person),
            player_stats=all_stats,
            player_positions=player_positions,
            fps=fps
        )
        timings["meme_analysis"] = time.time() - step_start

        # 7. Tạo visualization (chỉ heatmap và player image)
        step_start = time.time()
        visualizer = StatsVisualizer(court_points=court_points)
        court_image = first_frame

        output_files = {}

        # Per-player outputs (chỉ heatmap và player image)
        player_images = {}

        # Thu thập các frame cần đọc để crop player images
        frames_to_read = set()
        for player_id in all_stats.keys():
            if player_id in player_last_frames:
                frame_idx, _ = player_last_frames[player_id]
                frames_to_read.add(frame_idx)

        # Đọc các frames cần thiết
        player_frame_images = self._read_specific_frames(video_path, frames_to_read) if frames_to_read else {}

        for player_id, stats in all_stats.items():
            player_output = {}

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

            # Player crop image
            if player_id in player_last_frames:
                frame_idx, bbox = player_last_frames[player_id]
                if frame_idx in player_frame_images and bbox:
                    frame = player_frame_images[frame_idx]
                    crop = visualizer.crop_player_image(frame, bbox)
                    if crop is not None and crop.size > 0:
                        crop_path = os.path.join(output_folder, f"player_{player_id}_image.jpg")
                        cv2.imwrite(crop_path, crop)
                        player_output["player_image"] = f"{base_url}/player_{player_id}_image.jpg"

            player_images[player_id] = player_output

        # Giải phóng frames đã đọc
        del player_frame_images

        timings["visualization"] = time.time() - step_start

        # 8. Tạo video highlight cho từng player
        highlight_videos = {}
        if create_highlights:
            step_start = time.time()

            # Video highlight cho từng player (có chèn meme)
            highlight_videos = self._create_highlights_streaming(
                video_path=video_path,
                output_folder=output_folder,
                fps=fps,
                ball_positions=ball_positions,
                base_url=base_url,
                meme_analysis=meme_analysis
            )

            # Thêm highlight clips vào player_images
            for player_id, highlight_info in highlight_videos.items():
                if player_id in player_images:
                    player_images[player_id]["highlight_clips"] = highlight_info.get("highlights", [])
                    player_images[player_id]["highlight_summary"] = {
                        "total_clips": highlight_info.get("total_clips", 0),
                        "total_hits": highlight_info.get("total_hits", 0),
                        "total_duration_seconds": highlight_info.get("total_duration_seconds", 0)
                    }

            timings["highlight_videos"] = time.time() - step_start

        # 9. Xây dựng kết quả JSON
        result = {
            "video_info": {
                "path": video_path,
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": round(total_frames / fps, 2) if fps > 0 else 0
            },
            "ball_tracking": {
                "total_frames": total_frames,
                "detected_frames": total_frames - missed_before,
                "missed_frames_before_interpolation": missed_before,
                "missed_frames_after_interpolation": missed_after,
                "interpolated_frames": missed_before - missed_after,
                "detection_rate": round((total_frames - missed_before) / total_frames * 100, 1) if total_frames > 0 else 0,
                "coverage_after_interpolation": round((total_frames - missed_after) / total_frames * 100, 1) if total_frames > 0 else 0
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
            "highlight_videos": highlight_videos,
            "timestamp": datetime.now().isoformat()
        }

        # Chi tiết từng người chơi
        for player_id, stats in all_stats.items():
            # Lấy meme info cho player này
            player_meme = meme_analysis.get(player_id, {})

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
                "images": player_images.get(player_id, {}),
                "highlight": highlight_videos.get(player_id, {}),
                "memes": [
                    {"name": m.get("name", ""), "description": m.get("description", "")}
                    for m in player_meme.get("memes", [])
                ]
            }
            result["players"][f"player_{player_id}"] = player_data

        # Calculate total time
        total_time = time.time() - total_start
        timings["total"] = total_time

        # Add timing info to result
        result["performance"] = {
            "total_time_seconds": round(total_time, 2),
            "average_fps": round(total_frames / total_time, 1) if total_time > 0 else 0,
            "batch_size": self.batch_size,
            "timings": {k: round(v, 2) for k, v in timings.items()}
        }

        return result

    def _calculate_direction_flags(self, ball_positions, person_detections, angle_threshold, intersection_threshold):
        """Tính direction flags dựa trên ball positions và person detections

        Không cần frames, chỉ dựa vào vị trí bóng và người đã detect
        """
        import math

        direction_flags = [0] * len(ball_positions)

        for i in range(2, len(ball_positions) - 1):
            if ball_positions[i] == (-1, -1):
                continue
            if ball_positions[i-1] == (-1, -1) or ball_positions[i+1] == (-1, -1):
                continue

            # Tính vector trước và sau
            prev_x, prev_y = ball_positions[i-1]
            curr_x, curr_y = ball_positions[i]
            next_x, next_y = ball_positions[i+1]

            vec1 = (curr_x - prev_x, curr_y - prev_y)
            vec2 = (next_x - curr_x, next_y - curr_y)

            # Tính góc giữa 2 vectors
            len1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
            len2 = math.sqrt(vec2[0]**2 + vec2[1]**2)

            if len1 < 1 or len2 < 1:
                continue

            dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]
            cos_angle = max(-1, min(1, dot / (len1 * len2)))
            angle = math.degrees(math.acos(cos_angle))

            if angle > angle_threshold:
                # Check xem có người gần bóng không
                if i < len(person_detections):
                    for person in person_detections[i]:
                        bbox = person.get("bbox", [0, 0, 0, 0])
                        px = (bbox[0] + bbox[2]) / 2
                        py = (bbox[1] + bbox[3]) / 2

                        dist = math.sqrt((curr_x - px)**2 + (curr_y - py)**2)
                        if dist < intersection_threshold:
                            direction_flags[i] = 1
                            break

        return direction_flags

    def _create_highlights_streaming(self, video_path, output_folder, fps, ball_positions, base_url, meme_analysis, padding_frames=15):
        """Tạo highlight videos bằng cách đọc video streaming

        Không load toàn bộ video vào RAM
        """
        highlight_results = {}

        # Thu thập thông tin hits cho từng player
        for person_id, hits in self.person_tracker.ball_hits_by_person.items():
            if not hits:
                continue

            # Thu thập các frame ranges cần cho highlight
            highlight_ranges = []
            for hit in hits:
                hit_frame = hit["frame"]
                start_frame = max(0, hit_frame - padding_frames)
                end_frame = min(len(ball_positions), hit_frame + padding_frames + 1)
                highlight_ranges.append((start_frame, end_frame, hit))

            if not highlight_ranges:
                continue

            # Merge overlapping ranges
            highlight_ranges.sort(key=lambda x: x[0])
            merged_ranges = []
            current_start, current_end, current_hits = highlight_ranges[0][0], highlight_ranges[0][1], [highlight_ranges[0][2]]

            for start, end, hit in highlight_ranges[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                    current_hits.append(hit)
                else:
                    merged_ranges.append((current_start, current_end, current_hits))
                    current_start, current_end, current_hits = start, end, [hit]
            merged_ranges.append((current_start, current_end, current_hits))

            # Tạo video cho từng clip
            clips_info = []
            clip_idx = 0

            for start_frame, end_frame, clip_hits in merged_ranges:
                clip_idx += 1
                output_path = os.path.join(output_folder, f"player_{person_id}_clip_{clip_idx}.mp4")

                # Đọc và ghi video clip
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                if not out.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                frame_count = 0
                for frame_idx in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Vẽ ball position
                    if frame_idx < len(ball_positions) and ball_positions[frame_idx] != (-1, -1):
                        bx, by = ball_positions[frame_idx]
                        cv2.circle(frame, (int(bx), int(by)), 8, (0, 255, 255), -1)

                    # Đánh dấu hit frames
                    for hit in clip_hits:
                        if hit["frame"] == frame_idx:
                            bx, by = hit["ball_pos"]
                            cv2.circle(frame, (int(bx), int(by)), 20, (0, 0, 255), 3)
                            cv2.putText(frame, "HIT!", (int(bx) + 25, int(by)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    out.write(frame)
                    frame_count += 1

                out.release()
                cap.release()

                clips_info.append({
                    "clip_path": f"{base_url}/player_{person_id}_clip_{clip_idx}.mp4",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "hits_count": len(clip_hits),
                    "duration_seconds": round(frame_count / fps, 2)
                })

            highlight_results[person_id] = {
                "highlights": clips_info,
                "total_clips": len(clips_info),
                "total_hits": len(hits),
                "total_duration_seconds": sum(c["duration_seconds"] for c in clips_info)
            }

        return highlight_results
