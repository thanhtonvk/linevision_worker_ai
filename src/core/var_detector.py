import cv2
import math
from itertools import combinations
from ultralytics import YOLO
import numpy as np
import uuid
import torch
import gc


def detect_direction_changes(positions, angle_threshold=45):
    """
    Phát hiện các chỉ số frame mà bóng đổi hướng mạnh.
    - angle_threshold: góc (độ) thay đổi vận tốc để coi là đổi hướng.
    """
    change_points = []
    # Bỏ qua (-1,-1)
    clean_positions = [(i, p) for i, p in enumerate(positions) if p != (-1, -1)]

    for i in range(2, len(clean_positions)):
        idx1, p1 = clean_positions[i - 2]
        idx2, p2 = clean_positions[i - 1]
        idx3, p3 = clean_positions[i]

        # Vector vận tốc
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Độ dài vector
        len1 = math.hypot(*v1)
        len2 = math.hypot(*v2)
        if len1 == 0 or len2 == 0:
            continue

        # Tính cos góc
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_theta = max(-1, min(1, dot / (len1 * len2)))
        angle = math.degrees(math.acos(cos_theta))

        if angle > angle_threshold:
            change_points.append(idx2)  # đánh dấu điểm giữa (p2)

    return change_points


class VarDetector:
    def __init__(self, model_path, conf=0.8, batch_size=4):
        self.model = YOLO(model_path)
        self.conf = conf
        self.batch_size = batch_size
        self.person_model = YOLO("yolo12s.pt")

    def detect_segment_track_people(self, frames, batch_size=4):
        """
        Detect + track người (classes=[0]) theo batch để tăng tốc và tránh GPU leak.
        """

        # Reset tracker correctly to prevent memory build-up
        if hasattr(self.person_model, "tracker"):
            self.person_model.tracker = None
        if hasattr(self.person_model, "predictor"):
            self.person_model.predictor = None

        id_colors = {}

        def get_color(tid):
            if tid not in id_colors:
                id_colors[tid] = tuple(np.random.randint(0, 255, 3).tolist())
            return id_colors[tid]

        all_tracks = []

        # Avoid creating computation graph (VERY IMPORTANT)
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):

                batch = frames[i : i + batch_size]

                # TRACK (persist=True keeps tracking IDs)
                results = self.person_model.track(
                    batch, persist=False, classes=[0], verbose=False, conf=0.5  # human
                )

                # Process batch results
                for res in results:
                    frame_tracks = []
                    if res.boxes is not None and len(res.boxes) > 0:

                        boxes = res.boxes.xyxy.detach().cpu().numpy()
                        ids = (
                            res.boxes.id.detach().cpu().numpy()
                            if res.boxes.id is not None
                            else []
                        )
                        confs = res.boxes.conf.detach().cpu().numpy()

                        for box, tid, conf in zip(boxes, ids, confs):
                            x1, y1, x2, y2 = map(int, box)
                            frame_tracks.append(
                                {
                                    "id": int(tid),
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": float(conf),
                                    "color": get_color(int(tid)),
                                }
                            )

                    all_tracks.append(frame_tracks)

                # CLEAN GPU MEMORY after each batch
                del results
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        return all_tracks

    # --- Bước 1: Đọc video và chia batch ---
    def read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()
        return frames

    def batch_frames(self, frames):
        return [
            frames[i : i + self.batch_size]
            for i in range(0, len(frames), self.batch_size)
        ]

    # --- Bước 2: Dò bóng YOLO ---
    def detect_positions(self, frames):
        batches = self.batch_frames(frames)
        positions = []
        with torch.no_grad():
            for batch in batches:
                results = self.model.predict(
                    batch, batch=self.batch_size, verbose=False, conf=self.conf
                )
                for res in results:
                    if res.boxes is not None and len(res.boxes) > 0:
                        best_idx = res.boxes.conf.argmax()
                        x, y, w, h = res.boxes.xywh[best_idx].cpu().numpy()
                        positions.append((x, y))
                    else:
                        positions.append((-1, -1))
        del results
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return positions

    # --- Bước 3: Loại bỏ điểm bất thường trong nhóm ---
    def correct_positions(self, positions):
        corrected_positions = []
        current_group = []

        def process_group(group):
            if len(group) < 2:
                return group
            mean_x = sum(x for x, _ in group) / len(group)
            mean_y = sum(y for _, y in group) / len(group)
            mean_point = (mean_x, mean_y)
            distances = [math.dist(p1, p2) for p1, p2 in combinations(group, 2)]
            avg_dist = sum(distances) / len(distances)
            corrected = []
            for x, y in group:
                d = math.dist((x, y), mean_point)
                if d > avg_dist:
                    dx, dy = x - mean_x, y - mean_y
                    length = math.hypot(dx, dy)
                    if length == 0:
                        corrected.append((x, y))
                    else:
                        new_x = mean_x + dx / length * avg_dist
                        new_y = mean_y + dy / length * avg_dist
                        corrected.append((new_x, new_y))
                else:
                    corrected.append((x, y))
            return corrected

        for p in positions:
            if p == (-1, -1):
                if current_group:
                    corrected_positions.extend(process_group(current_group))
                    current_group = []
                corrected_positions.append(p)
            else:
                current_group.append(p)

        if current_group:
            corrected_positions.extend(process_group(current_group))
        return corrected_positions

    # --- Bước 4: Làm mượt bằng khoảng cách trung bình động ---
    def smooth_positions(self, positions, threshold_factor=2.5):
        clean_positions = []
        avg_dist = None
        pending = []

        def dist(a, b):
            return math.dist(a, b)

        for p in positions:
            if p == (-1, -1):
                clean_positions.append(p)
                continue

            if not clean_positions or clean_positions[-1] == (-1, -1):
                clean_positions.append(p)
                continue

            last_valid = clean_positions[-1]
            d = dist(last_valid, p)

            if avg_dist is None:
                avg_dist = d
            else:
                avg_dist = avg_dist * 0.9 + d * 0.1

            if d > threshold_factor * avg_dist:
                pending.append((len(clean_positions), p))
                clean_positions.append((-1, -1))
            else:
                if pending:
                    for idx, bad_p in pending:
                        x = (last_valid[0] + p[0]) / 2
                        y = (last_valid[1] + p[1]) / 2
                        clean_positions[idx] = (x, y)
                    pending.clear()
                clean_positions.append(p)
        return clean_positions

    # --- Bước 5: Nội suy các khoảng trống ---
    def interpolate_positions(self, positions, step=3):
        result = positions.copy()
        n = len(result)
        i = 0
        while i < n:
            if result[i] == (-1, -1):
                start = i - 1
                while start >= 0 and result[start] == (-1, -1):
                    start -= 1

                end = i
                while end < n and result[end] == (-1, -1):
                    end += 1

                gap = end - start - 1
                if (
                    start >= 0
                    and end < n
                    and result[start] != (-1, -1)
                    and result[end] != (-1, -1)
                ):
                    if gap <= step:
                        x1, y1 = result[start]
                        x2, y2 = result[end]
                        for k in range(1, gap + 1):
                            t = k / (gap + 1)
                            xi = x1 + (x2 - x1) * t
                            yi = y1 + (y2 - y1) * t
                            result[start + k] = (xi, yi)
                i = end
            else:
                i += 1
        return result

    def show_frames_with_positions(self, frames, positions):
        for i, frame in enumerate(frames):
            pos = positions[i] if i < len(positions) else (-1, -1)
            if pos != (-1, -1):
                x, y = int(pos[0]), int(pos[1])
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"({x},{y})",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            cv2.imshow("Video with Positions", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    def save_video_with_positions(
        self,
        frames,
        positions,
        output_path,
        fps=30,
        max_points=10,
        clear_after_frames=30,
    ):
        """
        Ghi video với quỹ đạo bóng.
        - max_points: Số điểm tối đa giữ lại trên đường bóng.
        - clear_after_frames: Sau bao nhiêu khung hình kể từ lần cuối phát hiện bóng thì xoá toàn bộ quỹ đạo.
        """
        if not frames:
            print("Không có frame để ghi.")
            return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        trajectory = []
        frames_since_detect = 0  # Đếm số frame từ lần phát hiện bóng cuối cùng

        try:
            for i, frame in enumerate(frames):
                pos = positions[i] if i < len(positions) else (-1, -1)
                frame_copy = frame.copy()

                if pos != (-1, -1):
                    # Có bóng: reset counter và thêm điểm mới
                    x, y = int(pos[0]), int(pos[1])
                    trajectory.append((x, y))
                    frames_since_detect = 0

                    # Giới hạn số điểm trên đường bóng
                    if len(trajectory) > max_points:
                        trajectory.pop(0)

                    # Vẽ điểm bóng hiện tại
                    cv2.circle(frame_copy, (x, y), 6, (0, 0, 255), -1)
                else:
                    # Không phát hiện bóng: tăng counter
                    frames_since_detect += 1
                    # Nếu quá giới hạn clear_after_frames → xoá đường bóng
                    if frames_since_detect >= clear_after_frames:
                        trajectory.clear()
                        frames_since_detect = 0

                # Vẽ đường nối giữa các điểm quỹ đạo
                if len(trajectory) > 1:
                    for j in range(1, len(trajectory)):
                        cv2.line(
                            frame_copy, trajectory[j - 1], trajectory[j], (0, 255, 0), 2
                        )

                out.write(frame_copy)
        finally:
            out.release()
        print(
            f"Đã ghi video kết quả với đường bóng (giữ {max_points} điểm, clear sau {clear_after_frames} frame): {output_path}"
        )

    def save_video_with_direction_changes(
        self,
        frames,
        positions,
        output_path,
        fps=30,
        angle_threshold=45,
        max_traj_len=15,
        change_point_lifespan=60,
        people_tracks=None,
    ):
        """
        Lưu video, vẽ bóng + điểm đổi hướng + tracking người.
        people_tracks: list các track người cho từng frame.
        """
        if not frames:
            print("Không có frame để ghi.")
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        try:
            change_points = set(detect_direction_changes(positions, angle_threshold))
            fixed_points = []  # [x,y,age]
            trajectory = []

            for i, frame in enumerate(frames):
                pos = positions[i] if i < len(positions) else (-1, -1)
                frame_copy = frame.copy()

                # -------- Vẽ người được detect & track --------
                if people_tracks and i < len(people_tracks):
                    for person in people_tracks[i]:
                        x1, y1, x2, y2 = person["bbox"]
                        color = person["color"]
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame_copy,
                            f"ID:{person['id']}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

                # -------- Vẽ bóng & quỹ đạo như code cũ --------
                for fp in fixed_points:  # tăng tuổi
                    fp[2] += 1
                fixed_points = [
                    fp for fp in fixed_points if fp[2] <= change_point_lifespan
                ]

                if pos != (-1, -1):
                    x, y = int(pos[0]), int(pos[1])
                    if i in change_points:
                        fixed_points.append([x, y, 0])
                    else:
                        trajectory.append((x, y))
                        if len(trajectory) > max_traj_len:
                            trajectory.pop(0)

                    cv2.circle(frame_copy, (x, y), 8, (255, 255, 255), -1)
                    cv2.circle(frame_copy, (x, y), 6, (0, 0, 255), -1)

                for fx, fy, age in fixed_points:
                    fade = max(0.3, 1 - age / change_point_lifespan)
                    color = (0, int(255 * fade), int(255 * fade))
                    cv2.circle(frame_copy, (int(fx), int(fy)), 12, color, 2)

                if len(trajectory) > 1:
                    for j in range(1, len(trajectory)):
                        alpha = j / len(trajectory)
                        color = (0, int(255 * alpha), 0)
                        thickness = max(1, int(3 * alpha))
                        cv2.line(
                            frame_copy,
                            trajectory[j - 1],
                            trajectory[j],
                            color,
                            thickness,
                        )

                out.write(frame_copy)
        finally:
            out.release()
        print(f"Đã ghi video đánh dấu đổi hướng: {output_path}")

    def save_cropped_ball_video(
        self, frames, positions, output_path, crop_size=50, fps=30
    ):
        cropped_frames = []
        prev_crop = None  # Lưu crop trước đó

        for i, frame in enumerate(frames):
            pos = positions[i] if i < len(positions) else (-1, -1)

            if pos == (-1, -1):
                # Nếu không có vị trí nhưng có crop trước đó, dùng lại
                if prev_crop is not None:
                    cropped_frames.append(prev_crop)
                continue

            # Lấy vị trí và crop
            x, y = int(pos[0]), int(pos[1])
            h, w = frame.shape[:2]
            x1 = max(0, x - crop_size)
            y1 = max(0, y - crop_size)
            x2 = min(w, x + crop_size)
            y2 = min(h, y + crop_size)

            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                # Nếu crop rỗng, dùng lại crop trước
                if prev_crop is not None:
                    cropped_frames.append(prev_crop)
                continue

            # Lưu crop hiện tại và thêm vào danh sách
            prev_crop = cropped.copy()
            cropped_frames.append(cropped)

        if not cropped_frames:
            print("Không tìm thấy frame nào để crop.")
            return

        # Lấy kích thước chung
        ch, cw = cropped_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (cw, ch))

        try:
            for crop in cropped_frames:
                crop_resized = cv2.resize(crop, (cw, ch))
                out.write(crop_resized)
        finally:
            out.release()
        print(f"Đã ghi video crop bóng: {output_path}")

    def detect_video(self, video_path: str, output_folder: str = "output"):
        import os

        frames = []
        cropped_path = None
        masked_path = None

        try:
            # Đọc và xử lý dữ liệu
            frames = self.read_video(video_path)
            positions = self.detect_positions(frames)
            corrected = self.correct_positions(positions)
            smoothed = self.smooth_positions(corrected, threshold_factor=2.5)
            final_positions = self.interpolate_positions(smoothed, step=5)
            # people_tracks = self.detect_segment_track_people(frames)

            # Tạo tên file duy nhất bằng uuid4
            uid = uuid.uuid4().hex

            # Sử dụng output_folder được truyền vào
            os.makedirs(output_folder, exist_ok=True)

            cropped_path = os.path.join(output_folder, f"output_cropped_{uid}.mp4")
            masked_path = os.path.join(output_folder, f"output_mask_{uid}.mp4")

            # Lưu video tạm
            self.save_cropped_ball_video(
                frames, final_positions, cropped_path, crop_size=100, fps=30
            )
            self.save_video_with_direction_changes(
                frames=frames,
                positions=final_positions,
                output_path=masked_path,
                fps=30,
                angle_threshold=45,
                max_traj_len=15,
                change_point_lifespan=60,
                # people_tracks=people_tracks,
            )
            del positions, corrected, smoothed, final_positions, frames
            gc.collect()
            torch.cuda.empty_cache()

            return {
                "crop": cropped_path,
                "mask": masked_path,
                "origin": video_path,
            }

        except Exception as e:
            # Cleanup on failure
            if cropped_path and os.path.exists(cropped_path):
                try:
                    os.remove(cropped_path)
                except:
                    pass
            if masked_path and os.path.exists(masked_path):
                try:
                    os.remove(masked_path)
                except:
                    pass
            raise e

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    video_path = "crop_video/part_008.mp4"
    model_path = "ball_best.pt"

    detector = BallDetector(model_path, batch_size=32)
    import time

    # Đo thời gian đọc video
    start = time.perf_counter()
    frames = detector.read_video(video_path)
    t_read = time.perf_counter() - start
    print(f"Đọc video: {t_read:.3f}s")

    # Đo thời gian detect
    start = time.perf_counter()
    positions = detector.detect_positions(frames)
    t_detect = time.perf_counter() - start
    print(f"Phát hiện bóng: {t_detect:.3f}s")

    # Đo thời gian sửa vị trí
    start = time.perf_counter()
    corrected = detector.correct_positions(positions)
    t_correct = time.perf_counter() - start
    print(f"Sửa vị trí: {t_correct:.3f}s")

    # Đo thời gian làm mượt
    start = time.perf_counter()
    smoothed = detector.smooth_positions(corrected, threshold_factor=2.5)
    t_smooth = time.perf_counter() - start
    print(f"Làm mượt: {t_smooth:.3f}s")

    # Đo thời gian nội suy
    start = time.perf_counter()
    final_positions = detector.interpolate_positions(smoothed, step=5)
    t_interp = time.perf_counter() - start
    print(f"Nội suy: {t_interp:.3f}s")

    # # Ghi video có vẽ bóng
    # detector.save_video_with_positions(frames, final_positions, "output_marked.mp4", fps=30)

    # Ghi video crop bóng
    detector.save_cropped_ball_video(
        frames, final_positions, "output_cropped.mp4", crop_size=100, fps=30
    )

    # Sau khi có frames và positions
    # change_points = detect_direction_changes(positions, angle_threshold=45)
    detector.save_video_with_direction_changes(
        frames, positions, "output_change.mp4", fps=30, angle_threshold=45
    )
