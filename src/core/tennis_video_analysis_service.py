# =============================================================================
# TENNIS VIDEO ANALYSIS SERVICE
# =============================================================================
# Main service for tennis video analysis with:
# - Ball tracking and speed calculation
# - Person tracking with pose estimation
# - Player ranking based on performance
# - Memory-optimized streaming processing
# =============================================================================

import cv2
import numpy as np
import math
import gc
import torch
import os
import uuid
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime

from .ball_detector import BallDetector
from .pose_estimator import PoseEstimator
from ..utils.court_geometry import CourtGeometry
from config.settings import settings


class TennisVideoAnalysisService:
    """
    Service for tennis video analysis providing:
    1. Highest speed player detection
    2. Average shoulder and knee angles
    3. Player rankings with detailed stats
    4. Match statistics (rally ratio, in/out court ratios)
    """

    def __init__(
        self,
        ball_model_path: str = None,
        person_model_path: str = None,
        pose_model_path: str = None,
        batch_size: int = 8
    ):
        """
        Initialize TennisVideoAnalysisService

        Args:
            ball_model_path: Path to ball detection model
            person_model_path: Path to person detection model
            pose_model_path: Path to pose estimation model
            batch_size: Batch size for processing
        """
        ball_model_path = ball_model_path or settings.ball_model_path
        person_model_path = person_model_path or settings.person_model_path
        pose_model_path = pose_model_path or settings.pose_model_path

        self.ball_detector = BallDetector(
            model_path=ball_model_path,
            person_model_path=person_model_path,
            batch_size=batch_size
        )
        self.pose_estimator = PoseEstimator(
            person_model_path=person_model_path,
            pose_model_path=pose_model_path,
            batch_size=batch_size
        )
        self.batch_size = batch_size

        # Runtime data (reset on each analyze call)
        self._reset_runtime_data()

    def _reset_runtime_data(self):
        """Reset all runtime tracking data"""
        self.tracked_players = {}  # player_id -> player data
        self.player_hits = defaultdict(list)  # player_id -> list of hit events
        self.player_poses = defaultdict(list)  # player_id -> list of pose data
        self.player_images = {}  # player_id -> cropped image
        self.next_player_id = 1

    def analyze(
        self,
        video_path: str,
        court_bounds: List[Tuple[int, int]],
        output_folder: str,
        ball_conf: float = 0.7,
        person_conf: float = 0.6,
        pixels_per_meter: Optional[float] = None
    ) -> Dict:
        """
        Main analysis entry point

        Args:
            video_path: Path to video file
            court_bounds: List of 4 corner points defining the court
            output_folder: Folder to save output images
            ball_conf: Ball detection confidence threshold
            person_conf: Person detection confidence threshold
            pixels_per_meter: Optional conversion factor for real-world speed

        Returns:
            Dict containing analysis results
        """
        print("=" * 80)
        print("       TENNIS VIDEO ANALYSIS SERVICE - STARTING ANALYSIS")
        print("=" * 80)

        self._reset_runtime_data()

        # Create CourtGeometry for 4-point court
        court_geometry = self._create_court_geometry(court_bounds)

        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        duration_seconds = total_frames / fps if fps > 0 else 0
        print(f"Video: {video_path}")
        print(f"FPS: {fps}, Frames: {total_frames}, Duration: {duration_seconds:.1f}s")
        print(f"Resolution: {frame_width}x{frame_height}")

        # Process video in streaming mode
        print("\n[1/4] Processing video (streaming mode)...")
        ball_positions, person_detections, direction_flags = self._process_video_streaming(
            video_path, ball_conf, person_conf, fps
        )

        # Post-process ball positions
        print("\n[2/4] Post-processing ball trajectory...")
        ball_positions = self.ball_detector.correct_positions(ball_positions)

        # Associate hits with players
        print("\n[3/4] Analyzing player hits and poses...")
        self._associate_hits_with_players(
            ball_positions, person_detections, direction_flags, fps, court_geometry
        )

        # Extract player images (second pass - read specific frames)
        print("\n[4/4] Extracting player images...")
        self._extract_player_images(video_path, output_folder)

        # Calculate all statistics
        print("\nCalculating statistics...")

        # 1. Highest speed player
        highest_speed_player = self._find_highest_speed_player(output_folder)

        # 2. Average stats
        average_stats = self._calculate_average_stats()

        # 3. Player rankings
        player_rankings = self._calculate_player_rankings(court_geometry, output_folder)

        # 4. Match statistics
        match_statistics = self._calculate_match_statistics(
            ball_positions, direction_flags, court_geometry
        )

        print("\nAnalysis complete!")

        return {
            "highest_speed_player": highest_speed_player,
            "average_stats": average_stats,
            "player_rankings": player_rankings,
            "match_statistics": match_statistics
        }

    def _create_court_geometry(self, court_bounds: List[Tuple[int, int]]) -> CourtGeometry:
        """
        Create CourtGeometry from 4 corner points

        Args:
            court_bounds: 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

        Returns:
            CourtGeometry instance
        """
        # For 4-point court, we use a simpler polygon
        # The points should be in order: top-left, top-right, bottom-right, bottom-left
        # We'll create a 4-point polygon and estimate net position

        if len(court_bounds) == 4:
            # Simple 4-point court - create minimal court geometry
            # Estimate net as midpoint between top and bottom
            top_y = min(court_bounds[0][1], court_bounds[1][1])
            bottom_y = max(court_bounds[2][1], court_bounds[3][1])

            # Create 4-point polygon for CourtGeometry
            # CourtGeometry expects 12 points, but we can adapt it
            # For simplicity, replicate points to match expected format
            court_points = []
            for i in range(12):
                idx = i % 4
                court_points.append(court_bounds[idx])

            return CourtGeometry(court_points, net_start_idx=1, net_end_idx=2)
        else:
            # Use provided 12 points
            return CourtGeometry(court_bounds)

    def _process_video_streaming(
        self,
        video_path: str,
        ball_conf: float,
        person_conf: float,
        fps: float
    ) -> Tuple[List, List, List]:
        """
        Process video in streaming mode to avoid loading all frames into memory

        Returns:
            Tuple of (ball_positions, person_detections, direction_flags)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_ball_positions = []
        all_person_detections = []
        all_direction_flags = []

        frame_idx = 0
        batch_count = 0

        while True:
            # Read batch of frames
            batch_frames = []
            batch_frame_indices = []

            for _ in range(self.batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
                batch_frame_indices.append(frame_idx)
                frame_idx += 1

            if not batch_frames:
                break

            # Process batch - Ball detection
            self.ball_detector.conf = ball_conf
            batch_ball_positions = self.ball_detector.detect_positions(batch_frames)
            all_ball_positions.extend(batch_ball_positions)

            # Process batch - Person + Pose detection
            batch_person_detections = self.pose_estimator.detect_persons_with_pose(
                batch_frames, person_conf
            )
            all_person_detections.extend(batch_person_detections)

            # Calculate direction flags for this batch
            for i in range(len(batch_ball_positions)):
                global_idx = batch_frame_indices[i]
                flag = self._calculate_direction_flag(
                    all_ball_positions, global_idx, batch_person_detections[i]
                )
                all_direction_flags.append(flag)

            # Progress
            batch_count += 1
            if batch_count % 10 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"  Processing: {progress:.1f}% ({frame_idx}/{total_frames} frames)")

            # Clear memory periodically
            if batch_count % 20 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Release batch frames
            del batch_frames

        cap.release()
        print(f"  Completed: {len(all_ball_positions)} frames processed")

        return all_ball_positions, all_person_detections, all_direction_flags

    def _calculate_direction_flag(
        self,
        ball_positions: List,
        frame_idx: int,
        persons: List[Dict]
    ) -> int:
        """
        Calculate direction change flag for a frame

        Returns:
            0: No change
            1: Direction changed (bounce)
            2: Direction changed + hit by person
        """
        if frame_idx < 2:
            return 0

        curr_pos = ball_positions[frame_idx] if frame_idx < len(ball_positions) else (-1, -1)
        prev_pos = ball_positions[frame_idx - 1] if frame_idx - 1 < len(ball_positions) else (-1, -1)
        prev2_pos = ball_positions[frame_idx - 2] if frame_idx - 2 < len(ball_positions) else (-1, -1)

        if curr_pos == (-1, -1) or prev_pos == (-1, -1) or prev2_pos == (-1, -1):
            return 0

        # Calculate direction change
        v1 = (prev_pos[0] - prev2_pos[0], prev_pos[1] - prev2_pos[1])
        v2 = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])

        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if mag1 < 1 or mag2 < 1:
            return 0

        # Angle between vectors
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))
        angle = math.degrees(math.acos(cos_angle))

        if angle < 30:  # Threshold for direction change
            return 0

        # Check if ball is near any person
        for person in persons:
            bbox = person["bbox"]
            x1, y1, x2, y2 = bbox

            # Expand bbox for hit detection
            padding = 50
            if (x1 - padding <= curr_pos[0] <= x2 + padding and
                y1 - padding <= curr_pos[1] <= y2 + padding):
                return 2  # Hit by person

        return 1  # Bounce

    def _associate_hits_with_players(
        self,
        ball_positions: List,
        person_detections: List,
        direction_flags: List,
        fps: float,
        court_geometry: CourtGeometry
    ):
        """
        Associate ball hits with players and track their poses
        """
        for frame_idx, flag in enumerate(direction_flags):
            if flag != 2:  # Only process player hits
                continue

            if frame_idx >= len(person_detections):
                continue

            ball_pos = ball_positions[frame_idx] if frame_idx < len(ball_positions) else (-1, -1)
            if ball_pos == (-1, -1):
                continue

            persons = person_detections[frame_idx]

            # Find the closest person to the ball
            best_person = None
            min_dist = float('inf')

            for person in persons:
                bbox = person["bbox"]
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                dist = math.sqrt((center_x - ball_pos[0])**2 + (center_y - ball_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_person = person

            if best_person is None:
                continue

            # Get or create player ID
            player_id = self._get_or_create_player_id(best_person, frame_idx)

            # Calculate ball speed
            speed = self._calculate_ball_speed(ball_positions, frame_idx, fps)

            # Find landing position (next direction change)
            landing_pos = self._find_landing_position(ball_positions, direction_flags, frame_idx)

            # Check if in court
            in_court = court_geometry.is_point_in_court(landing_pos) if landing_pos else False

            # Store hit data
            hit_data = {
                "frame": frame_idx,
                "time_seconds": frame_idx / fps,
                "speed": speed,
                "ball_pos": ball_pos,
                "landing_pos": landing_pos,
                "in_court": in_court,
                "bbox": best_person["bbox"]
            }
            self.player_hits[player_id].append(hit_data)

            # Store pose data
            pose_data = {
                "frame": frame_idx,
                "shoulder_angle": best_person.get("shoulder_angle", 0),
                "knee_bend_angle": best_person.get("knee_bend_angle", 0),
                "keypoints": best_person.get("keypoints"),
                "keypoint_conf": best_person.get("keypoint_conf")
            }
            self.player_poses[player_id].append(pose_data)

            # Store frame info for image extraction
            if player_id not in self.tracked_players:
                self.tracked_players[player_id] = {
                    "first_hit_frame": frame_idx,
                    "bbox_at_first_hit": best_person["bbox"],
                    "total_hits": 0
                }
            self.tracked_players[player_id]["total_hits"] += 1

    def _get_or_create_player_id(self, person: Dict, frame_idx: int) -> int:
        """
        Get existing player ID or create new one based on IoU matching
        """
        bbox = person["bbox"]

        # Try to match with existing players
        best_match_id = None
        best_iou = 0

        for player_id, player_data in self.tracked_players.items():
            # Check recent frames
            recent_hits = [h for h in self.player_hits[player_id] if frame_idx - h["frame"] <= 30]
            if not recent_hits:
                continue

            last_hit = recent_hits[-1]
            last_bbox = last_hit["bbox"]

            iou = self._calculate_iou(bbox, last_bbox)
            if iou > best_iou and iou > 0.2:
                best_iou = iou
                best_match_id = player_id

        if best_match_id is not None:
            return best_match_id

        # Create new player
        player_id = self.next_player_id
        self.next_player_id += 1
        return player_id

    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def _calculate_ball_speed(
        self,
        ball_positions: List,
        frame_idx: int,
        fps: float,
        window: int = 5
    ) -> float:
        """
        Calculate ball speed at a specific frame

        Returns:
            Speed in pixels per second
        """
        if frame_idx < window or frame_idx >= len(ball_positions) - window:
            return 0.0

        valid_positions = []
        for i in range(frame_idx - window, frame_idx + window + 1):
            if i < len(ball_positions) and ball_positions[i] != (-1, -1):
                valid_positions.append((i, ball_positions[i]))

        if len(valid_positions) < 2:
            return 0.0

        total_dist = 0.0
        for i in range(1, len(valid_positions)):
            p1 = valid_positions[i-1][1]
            p2 = valid_positions[i][1]
            total_dist += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        frame_span = valid_positions[-1][0] - valid_positions[0][0]
        time_seconds = frame_span / fps if fps > 0 else 1

        return total_dist / time_seconds if time_seconds > 0 else 0.0

    def _find_landing_position(
        self,
        ball_positions: List,
        direction_flags: List,
        start_frame: int,
        max_frames: int = 60
    ) -> Optional[Tuple[float, float]]:
        """
        Find where the ball lands after a hit
        """
        for i in range(start_frame + 1, min(start_frame + max_frames, len(direction_flags))):
            if direction_flags[i] in [1, 2]:  # Bounce or another hit
                if i < len(ball_positions) and ball_positions[i] != (-1, -1):
                    return ball_positions[i]
        return None

    def _extract_player_images(self, video_path: str, output_folder: str):
        """
        Extract player images from video (second pass)
        """
        if not self.tracked_players:
            return

        # Collect frames we need
        frames_to_read = {}
        for player_id, player_data in self.tracked_players.items():
            frame_idx = player_data["first_hit_frame"]
            if frame_idx not in frames_to_read:
                frames_to_read[frame_idx] = []
            frames_to_read[frame_idx].append({
                "player_id": player_id,
                "bbox": player_data["bbox_at_first_hit"]
            })

        # Read specific frames
        cap = cv2.VideoCapture(video_path)
        frame_indices = sorted(frames_to_read.keys())

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            for player_info in frames_to_read[frame_idx]:
                player_id = player_info["player_id"]
                bbox = player_info["bbox"]

                # Crop and save image
                x1, y1, x2, y2 = bbox
                h, w = frame.shape[:2]

                # Add padding
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                cropped = frame[y1:y2, x1:x2]

                if cropped.size > 0:
                    filename = f"player_{player_id}_{uuid.uuid4().hex[:8]}.jpg"
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, cropped)
                    self.player_images[player_id] = f"outputs/{os.path.basename(output_folder)}/{filename}"

        cap.release()

    def _find_highest_speed_player(self, output_folder: str) -> Dict:
        """
        Find the player with the highest hit speed
        """
        max_speed = 0
        best_player_id = None
        best_hit = None

        for player_id, hits in self.player_hits.items():
            for hit in hits:
                if hit["speed"] > max_speed:
                    max_speed = hit["speed"]
                    best_player_id = player_id
                    best_hit = hit

        if best_player_id is None:
            return {
                "player_image_url": None,
                "speed": 0
            }

        return {
            "player_image_url": self.player_images.get(best_player_id),
            "speed": round(max_speed, 2)
        }

    def _calculate_average_stats(self) -> Dict:
        """
        Calculate global average stats across all players
        """
        all_shoulder_angles = []
        all_knee_angles = []

        for player_id, poses in self.player_poses.items():
            for pose in poses:
                if pose["shoulder_angle"] > 0:
                    all_shoulder_angles.append(pose["shoulder_angle"])
                if pose["knee_bend_angle"] > 0:
                    all_knee_angles.append(pose["knee_bend_angle"])

        return {
            "avg_shoulder_angle": round(np.mean(all_shoulder_angles), 2) if all_shoulder_angles else 0,
            "avg_knee_bend_angle": round(np.mean(all_knee_angles), 2) if all_knee_angles else 0
        }

    def _calculate_player_rankings(
        self,
        court_geometry: CourtGeometry,
        output_folder: str
    ) -> List[Dict]:
        """
        Calculate player rankings based on multiple factors
        """
        player_stats = []

        for player_id in self.player_hits.keys():
            hits = self.player_hits[player_id]
            poses = self.player_poses.get(player_id, [])

            if not hits:
                continue

            # Calculate in-court ratio
            total_hits = len(hits)
            in_court_count = sum(1 for h in hits if h.get("in_court", False))
            in_court_ratio = in_court_count / total_hits if total_hits > 0 else 0

            # Calculate average hit speed
            speeds = [h["speed"] for h in hits if h["speed"] > 0]
            avg_speed = np.mean(speeds) if speeds else 0

            # Calculate average angles
            shoulder_angles = [p["shoulder_angle"] for p in poses if p["shoulder_angle"] > 0]
            knee_angles = [p["knee_bend_angle"] for p in poses if p["knee_bend_angle"] > 0]

            avg_shoulder = np.mean(shoulder_angles) if shoulder_angles else 0
            avg_knee = np.mean(knee_angles) if knee_angles else 0

            # Calculate composite score
            score = self._calculate_score(in_court_ratio, avg_speed, avg_shoulder, avg_knee)

            player_stats.append({
                "player_id": player_id,
                "score": round(score, 2),
                "player_image_url": self.player_images.get(player_id),
                "in_court_ratio": round(in_court_ratio, 4),
                "avg_hit_speed": round(avg_speed, 2),
                "avg_shoulder_angle": round(avg_shoulder, 2),
                "avg_knee_bend_angle": round(avg_knee, 2)
            })

        # Sort by score descending
        player_stats.sort(key=lambda x: x["score"], reverse=True)

        # Add rank
        for i, stats in enumerate(player_stats):
            stats["rank"] = i + 1

        return player_stats

    def _calculate_score(
        self,
        in_court_ratio: float,
        avg_speed: float,
        avg_shoulder: float,
        avg_knee: float
    ) -> float:
        """
        Calculate composite player score

        Weights:
        - in_court_ratio: 40%
        - avg_speed: 30%
        - shoulder angle quality: 15%
        - knee bend quality: 15%
        """
        # In-court score (40%)
        in_court_score = in_court_ratio * 40

        # Speed score (30%) - normalize assuming max ~500 pixels/second
        speed_score = min(30, (avg_speed / 500) * 30)

        # Shoulder angle score (15%) - ideal is 60-120 degrees
        if 60 <= avg_shoulder <= 120:
            shoulder_score = 15
        elif 40 <= avg_shoulder < 60 or 120 < avg_shoulder <= 140:
            shoulder_score = 10
        elif avg_shoulder > 0:
            shoulder_score = 5
        else:
            shoulder_score = 0

        # Knee bend score (15%) - ideal is 120-160 degrees
        if 120 <= avg_knee <= 160:
            knee_score = 15
        elif 100 <= avg_knee < 120 or 160 < avg_knee <= 180:
            knee_score = 10
        elif avg_knee > 0:
            knee_score = 5
        else:
            knee_score = 0

        return in_court_score + speed_score + shoulder_score + knee_score

    def _calculate_match_statistics(
        self,
        ball_positions: List,
        direction_flags: List,
        court_geometry: CourtGeometry
    ) -> Dict:
        """
        Calculate match-level statistics
        """
        total_hits = sum(1 for f in direction_flags if f == 2)
        total_bounces = sum(1 for f in direction_flags if f == 1)

        # Count in-court vs out-court for all hits
        in_court = 0
        out_court = 0

        for player_id, hits in self.player_hits.items():
            for hit in hits:
                if hit.get("in_court", False):
                    in_court += 1
                else:
                    out_court += 1

        total_with_landing = in_court + out_court
        in_court_ratio = in_court / total_with_landing if total_with_landing > 0 else 0
        out_court_ratio = out_court / total_with_landing if total_with_landing > 0 else 0

        # Rally ratio: percentage of frames with active ball exchange
        rally_ratio = self._calculate_rally_ratio(direction_flags)

        return {
            "rally_ratio": round(rally_ratio, 4),
            "out_court_ratio": round(out_court_ratio, 4),
            "in_court_ratio": round(in_court_ratio, 4)
        }

    def _calculate_rally_ratio(self, direction_flags: List) -> float:
        """
        Calculate rally ratio (percentage of time in active rallies)
        """
        if not direction_flags:
            return 0.0

        rally_frames = 0
        total_frames = len(direction_flags)

        in_rally = False
        rally_start = -1
        last_hit_frame = -1

        for i, flag in enumerate(direction_flags):
            if flag == 2:  # Hit by player
                if not in_rally:
                    in_rally = True
                    rally_start = i
                last_hit_frame = i
            elif in_rally:
                # End rally if no hit for 60 frames (about 2 seconds at 30fps)
                if i - last_hit_frame > 60:
                    rally_duration = last_hit_frame - rally_start
                    if rally_duration > 0:
                        rally_frames += rally_duration
                    in_rally = False

        # Handle rally extending to end
        if in_rally and last_hit_frame > rally_start:
            rally_frames += last_hit_frame - rally_start

        return rally_frames / total_frames if total_frames > 0 else 0.0
