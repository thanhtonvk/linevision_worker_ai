# =============================================================================
# POSE ESTIMATOR - 2-STEP POSE ESTIMATION
# =============================================================================
# Step 1: Detect persons using YOLO
# Step 2: Run pose estimation on detected persons
# =============================================================================

import cv2
import numpy as np
from ultralytics import YOLO
import math
import gc
import torch
from typing import List, Dict, Tuple, Optional
from config.settings import settings


class PoseEstimator:
    """
    2-step pose estimation:
    1. Detect persons using YOLO (yolo11m.pt)
    2. Run pose estimation on detected person regions (yolo11m-pose.pt)

    Calculates:
    - Shoulder angle (angle of shoulder line vs horizontal)
    - Knee bend angle (hip-knee-ankle angle)
    """

    # COCO Keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def __init__(
        self,
        person_model_path: str = "yolo11m.pt",
        pose_model_path: str = "yolo11m-pose.pt",
        batch_size: int = 8,
        conf_threshold: float = 0.5
    ):
        """
        Initialize PoseEstimator

        Args:
            person_model_path: Path to YOLO person detection model
            pose_model_path: Path to YOLO pose estimation model
            batch_size: Batch size for inference
            conf_threshold: Confidence threshold for keypoint validity
        """
        self.person_model = YOLO(person_model_path)
        self.pose_model = YOLO(pose_model_path)
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.max_frame_height = settings.max_frame_height
        self.enable_frame_resize = settings.enable_frame_resize

    def _resize_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize frame to reduce memory usage while maintaining aspect ratio

        Returns:
            Tuple of (resized_frame, scale_factor)
        """
        if not self.enable_frame_resize:
            return frame, 1.0

        height, width = frame.shape[:2]
        if height <= self.max_frame_height:
            return frame, 1.0

        scale = self.max_frame_height / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _batch_frames(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Split frames into batches"""
        return [
            frames[i:i + self.batch_size]
            for i in range(0, len(frames), self.batch_size)
        ]

    def detect_persons_with_pose(
        self,
        frames: List[np.ndarray],
        person_conf: float = 0.6
    ) -> List[List[Dict]]:
        """
        Detect persons and get pose keypoints for each

        Args:
            frames: List of video frames
            person_conf: Confidence threshold for person detection

        Returns:
            List per frame, each containing list of person detections:
            {
                "bbox": (x1, y1, x2, y2),
                "person_conf": float,
                "keypoints": np.ndarray shape (17, 2),
                "keypoint_conf": np.ndarray shape (17,),
                "shoulder_angle": float,
                "knee_bend_angle": float
            }
        """
        all_detections = []
        batches = self._batch_frames(frames)

        for batch_idx, batch in enumerate(batches):
            batch_detections = self._process_batch(batch, person_conf)
            all_detections.extend(batch_detections)

            # Clear GPU memory periodically
            if batch_idx % 10 == 0 and batch_idx > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return all_detections

    def _process_batch(
        self,
        batch: List[np.ndarray],
        person_conf: float
    ) -> List[List[Dict]]:
        """
        Process a batch of frames

        Args:
            batch: List of frames to process
            person_conf: Person detection confidence threshold

        Returns:
            List of detections for each frame in batch
        """
        batch_detections = []

        # Step 1: Resize frames
        resized_batch = []
        scales = []
        for frame in batch:
            resized_frame, scale = self._resize_frame(frame)
            resized_batch.append(resized_frame)
            scales.append(scale)

        # Step 2: Run pose model directly (it detects persons and estimates pose)
        # yolo11m-pose.pt can do both detection and pose in one pass
        try:
            pose_results = self.pose_model.predict(
                resized_batch,
                batch=len(resized_batch),
                verbose=False,
                conf=person_conf,
                half=True
            )
        except Exception as e:
            print(f"[PoseEstimator] Error in pose detection: {e}")
            # Return empty detections for this batch
            return [[] for _ in batch]

        # Step 3: Process results
        for frame_idx, (result, scale) in enumerate(zip(pose_results, scales)):
            frame_persons = []

            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes
                keypoints_data = result.keypoints

                for i in range(len(boxes)):
                    # Check if it's a person (class 0)
                    if int(boxes.cls[i]) != 0:
                        continue

                    # Extract bbox
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    box_conf = float(boxes.conf[i].cpu().numpy())

                    # Scale back if resized
                    if scale != 1.0:
                        x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale

                    # Extract keypoints
                    kpts = keypoints_data.xy[i].cpu().numpy()  # Shape: (17, 2)
                    kpts_conf = keypoints_data.conf[i].cpu().numpy()  # Shape: (17,)

                    # Scale keypoints back
                    if scale != 1.0:
                        kpts = kpts / scale

                    # Calculate angles
                    shoulder_angle = self._calculate_shoulder_angle(kpts, kpts_conf)
                    knee_bend_angle = self._calculate_knee_bend_angle(kpts, kpts_conf)

                    frame_persons.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "person_conf": box_conf,
                        "keypoints": kpts,
                        "keypoint_conf": kpts_conf,
                        "shoulder_angle": shoulder_angle,
                        "knee_bend_angle": knee_bend_angle
                    })

            batch_detections.append(frame_persons)

        return batch_detections

    def _calculate_shoulder_angle(
        self,
        keypoints: np.ndarray,
        conf: np.ndarray
    ) -> float:
        """
        Calculate angle of shoulder line vs horizontal

        Args:
            keypoints: (17, 2) array of keypoint coordinates
            conf: (17,) array of keypoint confidences

        Returns:
            Shoulder angle in degrees (0 = horizontal)
        """
        if conf[self.LEFT_SHOULDER] < self.conf_threshold or \
           conf[self.RIGHT_SHOULDER] < self.conf_threshold:
            return 0.0

        left = keypoints[self.LEFT_SHOULDER]
        right = keypoints[self.RIGHT_SHOULDER]

        dx = right[0] - left[0]
        dy = right[1] - left[1]

        # Angle from horizontal
        angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
        return angle

    def _calculate_knee_bend_angle(
        self,
        keypoints: np.ndarray,
        conf: np.ndarray
    ) -> float:
        """
        Calculate knee bend angle (hip-knee-ankle)

        Args:
            keypoints: (17, 2) array of keypoint coordinates
            conf: (17,) array of keypoint confidences

        Returns:
            Average knee bend angle in degrees (180 = straight leg)
        """
        angles = []

        # Left leg
        if conf[self.LEFT_HIP] > self.conf_threshold and \
           conf[self.LEFT_KNEE] > self.conf_threshold and \
           conf[self.LEFT_ANKLE] > self.conf_threshold:
            angle = self._angle_3points(
                keypoints[self.LEFT_HIP],
                keypoints[self.LEFT_KNEE],  # vertex
                keypoints[self.LEFT_ANKLE]
            )
            angles.append(angle)

        # Right leg
        if conf[self.RIGHT_HIP] > self.conf_threshold and \
           conf[self.RIGHT_KNEE] > self.conf_threshold and \
           conf[self.RIGHT_ANKLE] > self.conf_threshold:
            angle = self._angle_3points(
                keypoints[self.RIGHT_HIP],
                keypoints[self.RIGHT_KNEE],  # vertex
                keypoints[self.RIGHT_ANKLE]
            )
            angles.append(angle)

        return np.mean(angles) if angles else 0.0

    def _angle_3points(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray
    ) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3

        Args:
            p1: First point (e.g., hip)
            p2: Vertex point (e.g., knee)
            p3: Third point (e.g., ankle)

        Returns:
            Angle in degrees
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return np.degrees(np.arccos(cos_angle))

    def get_pose_for_person(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: int = 20
    ) -> Optional[Dict]:
        """
        Get pose for a specific person region (for second-pass processing)

        Args:
            frame: Full frame
            bbox: Person bounding box (x1, y1, x2, y2)
            padding: Padding around bbox for pose estimation

        Returns:
            Dict with keypoints and angles, or None if detection fails
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # Crop person region
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return None

        try:
            results = self.pose_model.predict(
                person_crop,
                verbose=False,
                conf=0.5,
                half=True
            )
        except Exception as e:
            print(f"[PoseEstimator] Error in single pose detection: {e}")
            return None

        if len(results) == 0 or results[0].keypoints is None:
            return None

        result = results[0]
        if len(result.keypoints.xy) == 0:
            return None

        # Get first detected person
        kpts = result.keypoints.xy[0].cpu().numpy()
        kpts_conf = result.keypoints.conf[0].cpu().numpy()

        # Adjust keypoints to full frame coordinates
        kpts[:, 0] += x1
        kpts[:, 1] += y1

        shoulder_angle = self._calculate_shoulder_angle(kpts, kpts_conf)
        knee_bend_angle = self._calculate_knee_bend_angle(kpts, kpts_conf)

        return {
            "keypoints": kpts,
            "keypoint_conf": kpts_conf,
            "shoulder_angle": shoulder_angle,
            "knee_bend_angle": knee_bend_angle
        }
