# =============================================================================
# TENNIS ANALYSIS MODULE - MODULE PHÂN TÍCH TENNIS HOÀN CHỈNH
# =============================================================================

import cv2
import numpy as np
from .ball_detector import BallDetector
from .person_tracker import PersonTracker
from ..visualization.visualizer import TennisVisualizer
import math
from collections import defaultdict


class TennisAnalysisModule:
    """
    Module phân tích tennis với đầy đủ thông tin:
    1. Ảnh crop người tốc độ bóng cao nhất
    2. Danh sách người chơi hay nhất
    3. Tỉ lệ đối kháng, tỉ lệ bóng trong/ngoài sân
    """

    def __init__(
        self,
        ball_model_path="ball_best.pt",
        person_model_path="yolov8m.pt",
        pose_model_path="yolov8n-pose.pt",
    ):
        self.ball_detector = BallDetector(ball_model_path, person_model_path)
        self.person_tracker = PersonTracker(pose_model_path, person_model_path)
        self.visualizer = TennisVisualizer()

    def analyze_video(
        self,
        video_path,
        ball_conf=0.7,
        person_conf=0.6,
        angle_threshold=50,
        intersection_threshold=100,
        court_bounds=(100, 100, 400, 500),
    ):
        """
        Phân tích video tennis và trả về kết quả đầy đủ

        Args:
            video_path: Đường dẫn đến video
            ball_conf: Confidence threshold cho ball detection
            person_conf: Confidence threshold cho person detection
            angle_threshold: Ngưỡng góc để phát hiện thay đổi hướng
            intersection_threshold: Ngưỡng khoảng cách để phát hiện bóng chạm người
            court_bounds: (x1, y1, x2, y2) - giới hạn sân tennis

        Returns:
            dict: Kết quả phân tích gồm:
                - highest_speed_info: Thông tin cú đánh tốc độ cao nhất
                - best_players: Danh sách người chơi hay nhất
                - match_statistics: Thống kê trận đấu (rally ratio, in-court ratio, out-court ratio)
                - visualization_video_path: Đường dẫn video visualization
        """
        print("=" * 80)
        print("           TENNIS ANALYSIS MODULE - PHÂN TÍCH VIDEO")
        print("=" * 80)

        # 1. Load video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 Video: {video_path}")
        print(f"🎬 FPS: {fps}, Total frames: {total_frames}")

        # 2. Process video
        frames = []
        print("Đang đọc video...")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Đã đọc {frame_count}/{total_frames} frames...")

        cap.release()
        print(f"✅ Đã đọc xong {len(frames)} frames")

        # 3. Detect ball
        print("Đang detect bóng...")
        ball_positions = self.ball_detector.detect_positions(frames)

        # 4. Detect direction changes
        print("Đang phân tích thay đổi hướng...")
        direction_flags, person_detections_old = (
            self.ball_detector.get_enhanced_direction_change_flags(
                frames,
                ball_positions,
                angle_threshold=angle_threshold,
                person_conf=person_conf,
                intersection_threshold=intersection_threshold,
            )
        )

        # 5. Person tracking và pose estimation
        print("Đang tracking người và phân tích pose...")
        person_detections, pose_detections = (
            self.person_tracker.detect_and_track_persons(
                frames, ball_positions, direction_flags
            )
        )

        # 6. Phân tích kỹ thuật tennis
        technique_analysis = self.person_tracker.analyze_tennis_technique(
            person_detections, court_bounds
        )

        # 7. Tính toán các metrics
        print("Đang tính toán các metrics...")

        # 7.1. Tìm cú đánh tốc độ cao nhất
        highest_speed_info = self._find_highest_speed_hit(
            frames,
            ball_positions,
            person_detections,
            direction_flags,
            fps,
            court_bounds,
        )

        # 7.2. Tính toán danh sách người chơi hay nhất
        best_players = self._calculate_best_players(
            frames,
            ball_positions,
            person_detections,
            technique_analysis,
            fps,
            court_bounds,
        )

        # 7.3. Tính toán thống kê trận đấu
        match_statistics = self._calculate_match_statistics(
            ball_positions, direction_flags, technique_analysis, fps
        )

        # 8. Tạo video visualization
        print("Đang tạo video visualization...")
        visualization_video_path = self._create_visualization_video(
            frames,
            ball_positions,
            direction_flags,
            person_detections,
            pose_detections,
            highest_speed_info,
            best_players,
            match_statistics,
            fps,
        )

        # 9. Tổng hợp kết quả
        results = {
            "highest_speed_info": highest_speed_info,
            "best_players": best_players,
            "match_statistics": match_statistics,
            "visualization_video_path": visualization_video_path,
        }

        print("\n✅ HOÀN THÀNH PHÂN TÍCH!")
        return results

    def _find_highest_speed_hit(
        self,
        frames,
        ball_positions,
        person_detections,
        direction_flags,
        fps,
        court_bounds,
    ):
        """
        Tìm cú đánh có tốc độ bóng cao nhất và trả về thông tin đầy đủ
        """
        max_velocity = 0
        best_hit = None
        best_frame_idx = -1

        # Duyệt qua tất cả các cú đánh (direction_flag == 2)
        for frame_idx in range(len(direction_flags)):
            if direction_flags[frame_idx] == 2:  # Bóng được đánh bởi người
                # Tính vận tốc tại frame này
                velocity = self._calculate_ball_velocity(ball_positions, frame_idx, fps)

                if velocity > max_velocity:
                    max_velocity = velocity
                    best_frame_idx = frame_idx

                    # Tìm người chơi đánh bóng tại frame này
                    ball_pos = ball_positions[frame_idx]
                    person_info = None

                    if frame_idx < len(person_detections):
                        for person_data in person_detections[frame_idx]:
                            person_bbox = person_data["person"]["bbox"]
                            x1, y1, x2, y2 = person_bbox
                            ball_x, ball_y = ball_pos

                            # Kiểm tra bóng có trong vùng người không
                            if x1 <= ball_x <= x2 and y1 <= ball_y <= y2:
                                person_info = person_data
                                break

                    if person_info:
                        person_id = person_info["person_id"]

                        # Crop ảnh người chơi
                        person_bbox = person_info["person"]["bbox"]
                        x1, y1, x2, y2 = person_bbox
                        # Mở rộng bbox một chút
                        padding = 20
                        h, w = frames[frame_idx].shape[:2]
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(w, x2 + padding)
                        y2 = min(h, y2 + padding)

                        cropped_image = frames[frame_idx][y1:y2, x1:x2].copy()

                        # Tính góc mở vai và góc khụy gối trung bình cho người chơi này
                        # (tính trung bình trên tất cả các cú đánh của người này)
                        shoulder_angles = []
                        knee_bend_angles = []

                        # Lấy tất cả các cú đánh của người chơi này
                        person_hits = self.person_tracker.ball_hits_by_person.get(
                            person_id, []
                        )

                        for hit in person_hits:
                            pose = hit.get("pose")
                            if pose is not None:
                                keypoints = pose["keypoints"]
                                conf = pose["conf"]

                                # Tính góc mở vai
                                if (
                                    conf[5] > 0.5 and conf[6] > 0.5
                                ):  # Left and right shoulder
                                    left_shoulder = keypoints[5]
                                    right_shoulder = keypoints[6]
                                    dx = right_shoulder[0] - left_shoulder[0]
                                    dy = right_shoulder[1] - left_shoulder[1]
                                    shoulder_angle = abs(
                                        math.degrees(math.atan2(abs(dy), abs(dx)))
                                    )
                                    shoulder_angles.append(shoulder_angle)

                                # Tính góc khụy gối
                                if (
                                    conf[11] > 0.5 and conf[13] > 0.5 and conf[15] > 0.5
                                ):  # Left side
                                    left_hip = keypoints[11]
                                    left_knee = keypoints[13]
                                    left_ankle = keypoints[15]
                                    left_knee_angle = self._calculate_angle_3points(
                                        left_hip, left_knee, left_ankle
                                    )

                                    if (
                                        conf[12] > 0.5
                                        and conf[14] > 0.5
                                        and conf[16] > 0.5
                                    ):  # Right side
                                        right_hip = keypoints[12]
                                        right_knee = keypoints[14]
                                        right_ankle = keypoints[16]
                                        right_knee_angle = (
                                            self._calculate_angle_3points(
                                                right_hip, right_knee, right_ankle
                                            )
                                        )
                                        knee_bend_angle = (
                                            left_knee_angle + right_knee_angle
                                        ) / 2
                                    else:
                                        knee_bend_angle = left_knee_angle

                                    knee_bend_angles.append(knee_bend_angle)

                        avg_shoulder_angle = (
                            np.mean(shoulder_angles) if shoulder_angles else 0
                        )
                        avg_knee_bend_angle = (
                            np.mean(knee_bend_angles) if knee_bend_angles else 0
                        )

                        best_hit = {
                            "frame": frame_idx,
                            "time_seconds": frame_idx / fps,
                            "velocity": max_velocity,
                            "person_id": person_id,
                            "cropped_image": cropped_image,
                            "shoulder_angle": avg_shoulder_angle,
                            "knee_bend_angle": avg_knee_bend_angle,
                        }

        if best_hit is None:
            return {
                "frame": -1,
                "time_seconds": 0,
                "velocity": 0,
                "person_id": -1,
                "cropped_image": None,
                "shoulder_angle": 0,
                "knee_bend_angle": 0,
            }

        return best_hit

    def _calculate_best_players(
        self,
        frames,
        ball_positions,
        person_detections,
        technique_analysis,
        fps,
        court_bounds,
    ):
        """
        Tính toán danh sách người chơi hay nhất với đầy đủ thông tin
        """
        person_stats = technique_analysis["person_stats"]
        ball_hits_by_person = self.person_tracker.ball_hits_by_person

        players_data = []

        for person_id, stats in person_stats.items():
            if stats["total_hits"] == 0:
                continue

            hits = ball_hits_by_person.get(person_id, [])

            # Tính tỉ lệ bóng trong sân
            in_court_ratio = (
                stats["hits_in_court"] / stats["total_hits"]
                if stats["total_hits"] > 0
                else 0
            )

            # Tính tốc độ bóng trung bình
            velocities = []
            for hit in hits:
                frame_idx = hit["frame"]
                velocity = self._calculate_ball_velocity(ball_positions, frame_idx, fps)
                if velocity > 0:
                    velocities.append(velocity)

            avg_ball_speed = np.mean(velocities) if velocities else 0

            # Tính góc mở vai trung bình và góc khụy gối trung bình
            shoulder_angles = []
            knee_bend_angles = []

            for hit in hits:
                pose = hit.get("pose")
                if pose is not None:
                    keypoints = pose["keypoints"]
                    conf = pose["conf"]

                    # Tính góc mở vai
                    if conf[5] > 0.5 and conf[6] > 0.5:  # Left and right shoulder
                        left_shoulder = keypoints[5]
                        right_shoulder = keypoints[6]
                        dx = right_shoulder[0] - left_shoulder[0]
                        dy = right_shoulder[1] - left_shoulder[1]
                        shoulder_angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
                        shoulder_angles.append(shoulder_angle)

                    # Tính góc khụy gối
                    if (
                        conf[11] > 0.5 and conf[13] > 0.5 and conf[15] > 0.5
                    ):  # Left side
                        left_hip = keypoints[11]
                        left_knee = keypoints[13]
                        left_ankle = keypoints[15]
                        left_knee_angle = self._calculate_angle_3points(
                            left_hip, left_knee, left_ankle
                        )

                        if (
                            conf[12] > 0.5 and conf[14] > 0.5 and conf[16] > 0.5
                        ):  # Right side
                            right_hip = keypoints[12]
                            right_knee = keypoints[14]
                            right_ankle = keypoints[16]
                            right_knee_angle = self._calculate_angle_3points(
                                right_hip, right_knee, right_ankle
                            )
                            knee_bend_angle = (left_knee_angle + right_knee_angle) / 2
                        else:
                            knee_bend_angle = left_knee_angle

                        knee_bend_angles.append(knee_bend_angle)

            avg_shoulder_angle = np.mean(shoulder_angles) if shoulder_angles else 0
            avg_knee_bend_angle = np.mean(knee_bend_angles) if knee_bend_angles else 0

            # Tìm ảnh crop đại diện (từ cú đánh đầu tiên)
            cropped_image = None
            if hits and len(hits) > 0:
                first_hit = hits[0]
                frame_idx = first_hit["frame"]

                if frame_idx < len(person_detections):
                    for person_data in person_detections[frame_idx]:
                        if person_data["person_id"] == person_id:
                            person_bbox = person_data["person"]["bbox"]
                            x1, y1, x2, y2 = person_bbox
                            padding = 20
                            h, w = frames[frame_idx].shape[:2]
                            x1 = max(0, x1 - padding)
                            y1 = max(0, y1 - padding)
                            x2 = min(w, x2 + padding)
                            y2 = min(h, y2 + padding)

                            cropped_image = frames[frame_idx][y1:y2, x1:x2].copy()
                            break

            # Tính điểm số (score) dựa trên nhiều yếu tố
            score = self._calculate_player_score(
                in_court_ratio,
                avg_ball_speed,
                stats["total_hits"],
                avg_shoulder_angle,
                avg_knee_bend_angle,
            )

            players_data.append(
                {
                    "player_id": person_id,
                    "score": score,
                    "cropped_image": cropped_image,
                    "in_court_ratio": in_court_ratio,
                    "avg_ball_speed": avg_ball_speed,
                    "avg_shoulder_angle": avg_shoulder_angle,
                    "avg_knee_bend_angle": avg_knee_bend_angle,
                    "total_hits": stats["total_hits"],
                }
            )

        # Sắp xếp theo điểm số giảm dần
        players_data.sort(key=lambda x: x["score"], reverse=True)

        return players_data

    def _calculate_player_score(
        self,
        in_court_ratio,
        avg_ball_speed,
        total_hits,
        avg_shoulder_angle,
        avg_knee_bend_angle,
    ):
        """
        Tính điểm số cho người chơi dựa trên nhiều yếu tố
        """
        # Normalize các giá trị
        # Tỉ lệ trong sân: 0-1 -> điểm 0-40
        in_court_score = in_court_ratio * 40

        # Tốc độ bóng: normalize về 0-30 (giả sử tốc độ tối đa là 100 pixels/frame)
        speed_score = min(30, (avg_ball_speed / 100) * 30) if avg_ball_speed > 0 else 0

        # Số cú đánh: normalize về 0-20 (giả sử tối đa 50 cú đánh)
        hits_score = min(20, (total_hits / 50) * 20)

        # Góc mở vai: góc tốt thường là 60-120 độ -> điểm 0-5
        if 60 <= avg_shoulder_angle <= 120:
            shoulder_score = 5
        elif 40 <= avg_shoulder_angle < 60 or 120 < avg_shoulder_angle <= 140:
            shoulder_score = 3
        else:
            shoulder_score = 1

        # Góc khụy gối: góc tốt thường là 120-160 độ -> điểm 0-5
        if 120 <= avg_knee_bend_angle <= 160:
            knee_score = 5
        elif 100 <= avg_knee_bend_angle < 120 or 160 < avg_knee_bend_angle <= 180:
            knee_score = 3
        else:
            knee_score = 1

        total_score = (
            in_court_score + speed_score + hits_score + shoulder_score + knee_score
        )

        return total_score

    def _calculate_match_statistics(
        self, ball_positions, direction_flags, technique_analysis, fps
    ):
        """
        Tính toán thống kê trận đấu:
        - Tỉ lệ đối kháng (rally ratio)
        - Tỉ lệ bóng trong sân
        - Tỉ lệ bóng ngoài sân
        """
        # Tính tỉ lệ bóng trong/ngoài sân
        court_accuracy = technique_analysis["court_accuracy"]
        total_hits = court_accuracy["total_hits"]
        total_in_court = court_accuracy["total_in_court"]
        total_out_court = court_accuracy["total_out_court"]

        in_court_ratio = total_in_court / total_hits if total_hits > 0 else 0
        out_court_ratio = total_out_court / total_hits if total_hits > 0 else 0

        # Tính tỉ lệ đối kháng (rally ratio)
        # Rally là chuỗi các cú đánh liên tục giữa các người chơi (direction_flag == 2)
        rally_ratio = self._calculate_rally_ratio(direction_flags, fps)

        return {
            "rally_ratio": rally_ratio,
            "in_court_ratio": in_court_ratio,
            "out_court_ratio": out_court_ratio,
            "total_hits": total_hits,
            "total_in_court": total_in_court,
            "total_out_court": total_out_court,
        }

    def _calculate_rally_ratio(self, direction_flags, fps):
        """
        Tính tỉ lệ đối kháng (rally ratio)
        Rally là thời gian bóng đánh qua lại liên tục giữa các người chơi
        """
        rally_frames = 0
        total_frames = len(direction_flags)

        # Tìm các chuỗi liên tục các cú đánh (direction_flag == 2)
        in_rally = False
        rally_start = -1

        for i, flag in enumerate(direction_flags):
            if flag == 2:  # Bóng được đánh bởi người
                if not in_rally:
                    in_rally = True
                    rally_start = i
            else:
                if in_rally:
                    # Kết thúc rally
                    rally_duration = i - rally_start
                    # Chỉ tính rally nếu có ít nhất 2 cú đánh liên tục
                    if rally_duration >= 2:
                        rally_frames += rally_duration
                    in_rally = False

        # Xử lý trường hợp rally kéo dài đến cuối video
        if in_rally:
            rally_duration = total_frames - rally_start
            if rally_duration >= 2:
                rally_frames += rally_duration

        rally_ratio = rally_frames / total_frames if total_frames > 0 else 0

        return rally_ratio

    def _calculate_ball_velocity(self, ball_positions, frame_idx, fps, window=5):
        """
        Tính vận tốc bóng tại frame cụ thể
        """
        if frame_idx < window or frame_idx >= len(ball_positions) - window:
            return 0.0

        # Lấy vị trí trong window
        positions_window = []
        for i in range(frame_idx - window, frame_idx + window + 1):
            if i < len(ball_positions) and ball_positions[i] != (-1, -1):
                positions_window.append(ball_positions[i])

        if len(positions_window) < 2:
            return 0.0

        # Tính vận tốc trung bình
        total_distance = 0.0
        valid_pairs = 0

        for i in range(1, len(positions_window)):
            p1 = positions_window[i - 1]
            p2 = positions_window[i]
            if p1 != (-1, -1) and p2 != (-1, -1):
                distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                total_distance += distance
                valid_pairs += 1

        if valid_pairs == 0:
            return 0.0

        # Vận tốc trung bình (pixels per frame)
        avg_velocity = total_distance / valid_pairs

        # Chuyển đổi sang pixels per second
        velocity_per_second = avg_velocity * fps

        return velocity_per_second

    def _create_visualization_video(
        self,
        frames,
        ball_positions,
        direction_flags,
        person_detections,
        pose_detections,
        highest_speed_info,
        best_players,
        match_statistics,
        fps,
    ):
        """
        Tạo video visualization với annotations đầy đủ
        """
        output_path = "tennis_analysis_visualization.mp4"

        if not frames:
            print("Không có frames để tạo video!")
            return None

        # Tạo video writer
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # COCO keypoint connections
        skeleton = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],  # Head
            [5, 6],
            [5, 7],
            [7, 9],
            [6, 8],
            [8, 10],  # Arms
            [5, 11],
            [6, 12],
            [11, 12],  # Torso
            [11, 13],
            [12, 14],
            [13, 15],
            [14, 16],  # Legs
        ]

        # Colors for different persons
        person_colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        for frame_idx, frame in enumerate(frames):
            vis_frame = frame.copy()

            # Vẽ thông tin thống kê ở góc trên
            stats_text = [
                f"Rally Ratio: {match_statistics['rally_ratio']:.2%}",
                f"In Court: {match_statistics['in_court_ratio']:.2%}",
                f"Out Court: {match_statistics['out_court_ratio']:.2%}",
            ]

            y_offset = 30
            for i, text in enumerate(stats_text):
                cv2.putText(
                    vis_frame,
                    text,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            # Vẽ bóng
            if frame_idx < len(ball_positions) and ball_positions[frame_idx] != (
                -1,
                -1,
            ):
                x, y = int(ball_positions[frame_idx][0]), int(
                    ball_positions[frame_idx][1]
                )

                # Chọn màu theo loại thay đổi hướng
                if frame_idx < len(direction_flags):
                    if direction_flags[frame_idx] == 1:  # Bóng chạm đất
                        color = (0, 0, 255)  # Đỏ
                    elif direction_flags[frame_idx] == 2:  # Bóng được đánh bởi người
                        color = (0, 255, 0)  # Xanh lá
                    else:
                        color = (255, 0, 0)  # Xanh dương
                else:
                    color = (255, 0, 0)

                cv2.circle(vis_frame, (x, y), 8, color, -1)
                cv2.circle(vis_frame, (x, y), 6, (255, 255, 255), 2)

                # Đánh dấu frame có tốc độ cao nhất
                if highest_speed_info["frame"] == frame_idx:
                    cv2.rectangle(
                        vis_frame, (x - 15, y - 15), (x + 15, y + 15), (0, 255, 255), 3
                    )
                    cv2.putText(
                        vis_frame,
                        "MAX SPEED",
                        (x + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

            # Vẽ person detections và pose
            if frame_idx < len(person_detections):
                for person_data in person_detections[frame_idx]:
                    person_id = person_data["person_id"]
                    bbox = person_data["person"]["bbox"]
                    pose = person_data["pose"]

                    # Color for this person
                    color = person_colors[person_id % len(person_colors)]

                    # Draw bounding box
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

                    # Tìm thứ hạng của người chơi này
                    player_rank = -1
                    for rank, player in enumerate(best_players, 1):
                        if player["player_id"] == person_id:
                            player_rank = rank
                            break

                    label = f"Player {person_id}"
                    if player_rank > 0:
                        label += f" (Rank #{player_rank})"

                    cv2.putText(
                        vis_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    # Draw pose keypoints
                    if pose is not None:
                        keypoints = pose["keypoints"]
                        conf = pose["conf"]

                        # Draw keypoints
                        for i, (x, y) in enumerate(keypoints):
                            if conf[i] > 0.5:  # Only draw confident keypoints
                                cv2.circle(vis_frame, (int(x), int(y)), 3, color, -1)

                        # Draw skeleton
                        for connection in skeleton:
                            pt1_idx, pt2_idx = connection
                            if (
                                pt1_idx < len(keypoints)
                                and pt2_idx < len(keypoints)
                                and conf[pt1_idx] > 0.5
                                and conf[pt2_idx] > 0.5
                            ):

                                pt1 = (
                                    int(keypoints[pt1_idx][0]),
                                    int(keypoints[pt1_idx][1]),
                                )
                                pt2 = (
                                    int(keypoints[pt2_idx][0]),
                                    int(keypoints[pt2_idx][1]),
                                )
                                cv2.line(vis_frame, pt1, pt2, color, 2)

            # Add frame info
            cv2.putText(
                vis_frame,
                f"Frame: {frame_idx}",
                (width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            out.write(vis_frame)

            if frame_idx % 100 == 0:
                print(f"Đã xử lý {frame_idx}/{len(frames)} frames...")

        out.release()
        print(f"✅ Đã tạo video visualization: {output_path}")

        return output_path

    def _calculate_angle_3points(self, p1, p2, p3):
        """
        Tính góc tại điểm p2 giữa 3 điểm p1-p2-p3
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return np.arccos(cos_angle) * 180 / np.pi
