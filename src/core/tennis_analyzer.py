# =============================================================================
# TENNIS ANALYZER - CLASS CHÍNH ĐỂ PHÂN TÍCH TENNIS
# =============================================================================

import cv2
import numpy as np
from .ball_detector import BallDetector
from .person_tracker import PersonTracker
from ..visualization.visualizer import TennisVisualizer


class TennisAnalyzer:
    """
    Class chính để phân tích tennis với tracking người và pose estimation
    """

    def __init__(
        self,
        ball_model_path="src/models/ball_best.pt",
        person_model_path="src/models/yolov8m.pt",
        pose_model_path="src/models/yolov8n-pose.pt",
    ):
        self.ball_detector = BallDetector(ball_model_path, person_model_path)
        self.person_tracker = PersonTracker(pose_model_path, person_model_path)
        self.visualizer = TennisVisualizer()

    def analyze_tennis_match(
        self,
        video_path,
        ball_conf=0.7,
        person_conf=0.6,
        angle_threshold=50,
        intersection_threshold=100,
        court_bounds=(100, 100, 400, 500),
    ):
        """
        Phân tích tennis match với tracking người và pose estimation
        """
        print("=" * 80)
        print("           TENNIS ANALYSIS WITH PERSON TRACKING & POSE ESTIMATION")
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

        # 7. Tính toán thống kê
        person_stats = self.person_tracker.get_person_statistics()

        # 8. Tạo báo cáo
        self._print_analysis_report(
            technique_analysis,
            person_stats,
            direction_flags,
            ball_positions,
            frames,
            fps,
        )

        return {
            "person_tracker": self.person_tracker,
            "technique_analysis": technique_analysis,
            "person_stats": person_stats,
            "ball_positions": ball_positions,
            "direction_flags": direction_flags,
            "frames": frames,
            "fps": fps,
        }

    def _print_analysis_report(
        self,
        technique_analysis,
        person_stats,
        direction_flags,
        ball_positions,
        frames,
        fps,
    ):
        """In báo cáo phân tích"""
        print("\n" + "=" * 80)
        print("=== BÁO CÁO PHÂN TÍCH TENNIS VỚI PERSON TRACKING ===")
        print("=" * 80)

        print(f"\n📊 THỐNG KÊ TỔNG QUAN:")
        print(f"- Tổng số frames: {len(frames)}")
        print(f"- Tổng thời gian: {len(frames)/fps:.2f} giây")
        print(f"- Số người được track: {len(self.person_tracker.tracked_persons)}")

        print(f"\n🎾 THỐNG KÊ BÓNG:")
        ball_hits = sum(1 for flag in direction_flags if flag > 0)
        person_hits = sum(1 for flag in direction_flags if flag == 2)
        ground_hits = sum(1 for flag in direction_flags if flag == 1)

        print(f"- Tổng cú đánh: {ball_hits}")
        print(f"- Cú đánh bởi người: {person_hits}")
        print(f"- Cú đánh chạm đất: {ground_hits}")

        print(f"\n👥 THỐNG KÊ NGƯỜI CHƠI:")
        for person_id, stats in person_stats.items():
            print(f"\nNgười chơi {person_id}:")
            print(f"  - Tổng frames xuất hiện: {stats['total_frames']}")
            print(f"  - Tổng cú đánh: {stats['total_hits']}")
            print(f"  - Tỷ lệ đánh bóng: {stats['hit_rate']:.2%}")
            print(f"  - Xuất hiện từ frame: {stats['first_seen']}")
            print(f"  - Xuất hiện đến frame: {stats['last_seen']}")

        print(f"\n🏆 PHÂN TÍCH KỸ THUẬT VÀ ĐỘ CHÍNH XÁC:")

        # Thống kê tổng hợp
        court_accuracy = technique_analysis["court_accuracy"]
        print(f"\n📊 THỐNG KÊ TỔNG HỢP:")
        print(f"  - Tổng cú đánh: {court_accuracy['total_hits']}")
        print(f"  - Cú đánh trong sân: {court_accuracy['total_in_court']}")
        print(f"  - Cú đánh ngoài sân: {court_accuracy['total_out_court']}")
        print(f"  - Tỷ lệ chính xác tổng: {court_accuracy['overall_accuracy']:.1f}%")

        # Thống kê từng người chơi (chỉ hiển thị những người có cú đánh)
        print(f"\n👥 THỐNG KÊ TỪNG NGƯỜI CHƠI:")
        print(f"📊 Tổng số người được track: {court_accuracy['total_persons_count']}")
        print(f"🎾 Số người có cú đánh: {court_accuracy['active_persons_count']}")

        # Lọc chỉ hiển thị những người có cú đánh
        active_persons = {
            pid: stats
            for pid, stats in technique_analysis["person_stats"].items()
            if stats["total_hits"] > 0
        }

        if not active_persons:
            print("❌ Không có người chơi nào có cú đánh!")
        else:
            for person_id, person_data in active_persons.items():
                print(f"\n🎾 NGƯỜI CHƠI {person_id}:")
                print(f"  📈 Tổng cú đánh: {person_data['total_hits']}")
                print(f"  ✅ Cú đánh trong sân: {person_data['hits_in_court']}")
                print(f"  ❌ Cú đánh ngoài sân: {person_data['hits_out_court']}")
                print(
                    f"  🎯 Tỷ lệ chính xác: {person_data['accuracy_percentage']:.1f}%"
                )

                # Chi tiết từng cú đánh
                if person_data["hit_details"]:
                    print(f"  📝 CHI TIẾT CÁC CÚ ĐÁNH:")
                    for i, hit_detail in enumerate(person_data["hit_details"], 1):
                        status = (
                            "✅ TRONG SÂN"
                            if hit_detail["is_in_court"]
                            else "❌ NGOÀI SÂN"
                        )
                        print(
                            f"    Cú {i}: Frame {hit_detail['frame']} - {status} - Vị trí: {hit_detail['ball_pos']}"
                        )

                # Lỗi kỹ thuật
                if person_data["technique_errors"]:
                    print(
                        f"  ⚠️  Lỗi kỹ thuật phát hiện: {len(person_data['technique_errors'])}"
                    )
                    error_types = {}
                    for error in person_data["technique_errors"]:
                        error_type = error["type"]
                        error_types[error_type] = error_types.get(error_type, 0) + 1

                    for error_type, count in error_types.items():
                        print(f"    + {error_type}: {count} lần")
                else:
                    print(f"  ✅ Không có lỗi kỹ thuật phát hiện")

        # Tính vận tốc bóng tại các vị trí đánh
        print(f"\n⚡ PHÂN TÍCH VẬN TỐC BÓNG:")
        velocities = []
        for person_id, hits in self.person_tracker.ball_hits_by_person.items():
            for hit in hits:
                frame_idx = hit["frame"]
                if frame_idx > 0 and frame_idx < len(ball_positions) - 1:
                    # Tính vận tốc dựa trên vị trí trước và sau
                    prev_pos = (
                        ball_positions[frame_idx - 1]
                        if ball_positions[frame_idx - 1] != (-1, -1)
                        else None
                    )
                    next_pos = (
                        ball_positions[frame_idx + 1]
                        if ball_positions[frame_idx + 1] != (-1, -1)
                        else None
                    )

                    if prev_pos and next_pos:
                        # Tính khoảng cách di chuyển
                        distance = np.sqrt(
                            (next_pos[0] - prev_pos[0]) ** 2
                            + (next_pos[1] - prev_pos[1]) ** 2
                        )
                        # Vận tốc (pixels per frame)
                        velocity = distance / 2  # 2 frames
                        velocities.append(velocity)

                        print(
                            f"  - Người {person_id}, Frame {frame_idx}: Vận tốc = {velocity:.2f} pixels/frame"
                        )

        if velocities:
            print(f"\n📈 THỐNG KÊ VẬN TỐC:")
            print(f"  - Vận tốc trung bình: {np.mean(velocities):.2f} pixels/frame")
            print(f"  - Vận tốc tối đa: {np.max(velocities):.2f} pixels/frame")
            print(f"  - Vận tốc tối thiểu: {np.min(velocities):.2f} pixels/frame")

    def create_visualizations(self, results, output_prefix="tennis_analysis"):
        """Tạo tất cả các visualization"""
        print("\n🎯 TẠO VISUALIZATION VÀ BÁO CÁO...")

        person_tracker = results["person_tracker"]
        technique_analysis = results["technique_analysis"]
        frames = results["frames"]

        # Tạo video visualization
        self.visualizer.create_pose_visualization(
            frames,
            (
                person_tracker.person_detections
                if hasattr(person_tracker, "person_detections")
                else []
            ),
            (
                person_tracker.pose_detections
                if hasattr(person_tracker, "pose_detections")
                else []
            ),
            f"{output_prefix}_pose_analysis.mp4",
        )

        # Tạo biểu đồ phân tích kỹ thuật
        self.visualizer.create_technique_analysis_plot(
            technique_analysis, f"{output_prefix}_technique_analysis.png"
        )

        # Tạo biểu đồ độ chính xác cú đánh
        self.visualizer.create_court_accuracy_visualization(
            technique_analysis, f"{output_prefix}_court_accuracy.png"
        )

        # Tạo báo cáo chi tiết
        self.visualizer.create_detailed_technique_report(
            person_tracker, technique_analysis, f"{output_prefix}_detailed_report.txt"
        )

        print(f"\n✅ HOÀN THÀNH TẠO VISUALIZATION!")
        print("📁 Các file đã tạo:")
        print(f"  - {output_prefix}_pose_analysis.mp4 (video với pose tracking)")
        print(
            f"  - {output_prefix}_technique_analysis.png (biểu đồ phân tích kỹ thuật)"
        )
        print(f"  - {output_prefix}_court_accuracy.png (biểu đồ độ chính xác cú đánh)")
        print(f"  - {output_prefix}_detailed_report.txt (báo cáo chi tiết)")
