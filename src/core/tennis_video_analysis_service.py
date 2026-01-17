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
from .meme_analyzer import MemeAnalyzer
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
        self.player_positions = defaultdict(list)  # player_id -> [(frame, x, y), ...]
        self.next_player_id = 1

    def analyze(
        self,
        video_path: str,
        court_bounds: List[Tuple[int, int]],
        output_folder: str,
        pixels_per_meter: Optional[float] = None
    ) -> Dict:
        """
        Main analysis entry point

        Args:
            video_path: Path to video file
            court_bounds: List of 4 corner points defining the court
            output_folder: Folder to save output images
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
            video_path, fps
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

        # 5. Create highlight videos and analyze memes
        print("\n[5/5] Creating highlight videos and analyzing memes...")
        highlights, meme_analysis = self._create_highlights_and_memes(
            video_path, ball_positions, output_folder, fps
        )

        print("\nAnalysis complete!")

        return {
            "highest_speed_player": highest_speed_player,
            "average_stats": average_stats,
            "player_rankings": player_rankings,
            "match_statistics": match_statistics,
            "highlights": highlights,
            "meme_analysis": meme_analysis
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

            # Process batch - Ball detection (uses default conf=0.3)
            batch_ball_positions = self.ball_detector.detect_positions(batch_frames)
            all_ball_positions.extend(batch_ball_positions)

            # Process batch - Person + Pose detection (uses default conf=0.3)
            batch_person_detections = self.pose_estimator.detect_persons_with_pose(batch_frames)
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

            # Check if ball crossed net
            crossed_net = self._check_ball_crossed_net(ball_positions, frame_idx, court_geometry)

            # Determine hit type: serve or return
            hit_type = self._classify_hit_type(player_id, frame_idx, direction_flags, fps)

            # Store hit data
            hit_data = {
                "frame": frame_idx,
                "time_seconds": frame_idx / fps,
                "speed": speed,
                "ball_pos": ball_pos,
                "landing_pos": landing_pos,
                "in_court": in_court,
                "crossed_net": crossed_net,
                "hit_type": hit_type,  # "serve" or "return"
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

            # Store player position for movement analysis
            bbox = best_person["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            self.player_positions[player_id].append((frame_idx, center_x, center_y))

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

    def _check_ball_crossed_net(
        self,
        ball_positions: List,
        hit_frame: int,
        court_geometry: CourtGeometry,
        max_frames: int = 60
    ) -> bool:
        """
        Check if ball crossed the net after a hit
        Uses the net_y position from court geometry
        """
        if hit_frame >= len(ball_positions):
            return False

        hit_pos = ball_positions[hit_frame]
        if hit_pos == (-1, -1):
            return False

        # Get net position (middle of court vertically)
        net_y = court_geometry.get_net_y() if hasattr(court_geometry, 'get_net_y') else None
        if net_y is None:
            # Estimate net as middle of court bounds
            court_bounds = court_geometry.get_bounds() if hasattr(court_geometry, 'get_bounds') else None
            if court_bounds:
                net_y = (court_bounds[1] + court_bounds[3]) / 2
            else:
                return True  # Assume crossed if we can't determine

        # Check if ball trajectory crosses net
        for i in range(hit_frame + 1, min(hit_frame + max_frames, len(ball_positions))):
            curr_pos = ball_positions[i]
            if curr_pos == (-1, -1):
                continue

            # Check if ball moved from one side of net to the other
            if (hit_pos[1] < net_y and curr_pos[1] > net_y) or \
               (hit_pos[1] > net_y and curr_pos[1] < net_y):
                return True

        return False

    def _classify_hit_type(
        self,
        player_id: int,
        frame_idx: int,
        direction_flags: List,
        fps: float
    ) -> str:
        """
        Classify hit as 'serve' or 'return'
        Serve: first hit after a pause (no hits for > 3 seconds)
        Return: all other hits
        """
        # Get previous hits for this player
        player_hits = self.player_hits.get(player_id, [])

        if len(player_hits) == 0:
            # First hit - check if it's start of a rally (likely serve)
            # Look back to see if there were any hits in last 3 seconds
            pause_frames = int(3 * fps)
            has_recent_hit = False

            for i in range(max(0, frame_idx - pause_frames), frame_idx):
                if i < len(direction_flags) and direction_flags[i] == 2:
                    has_recent_hit = True
                    break

            return "serve" if not has_recent_hit else "return"

        # Check time since last hit
        last_hit = player_hits[-1]
        time_since_last = (frame_idx - last_hit["frame"]) / fps

        # If > 3 seconds since last hit, likely a new serve
        if time_since_last > 3.0:
            return "serve"

        return "return"

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
        Calculate player rankings based on multiple factors with detailed statistics
        """
        player_stats = []
        base_url = f"outputs/{os.path.basename(output_folder)}"

        for player_id in self.player_hits.keys():
            hits = self.player_hits[player_id]
            poses = self.player_poses.get(player_id, [])
            positions = self.player_positions.get(player_id, [])

            if not hits:
                continue

            # === DETAILED HIT STATISTICS ===
            # Separate by hit type
            serves = [h for h in hits if h.get("hit_type") == "serve"]
            returns = [h for h in hits if h.get("hit_type") == "return"]

            # Total accuracy stats
            total_hits = len(hits)
            in_court_count = sum(1 for h in hits if h.get("in_court", False))
            out_court_count = sum(1 for h in hits if not h.get("in_court", False) and h.get("crossed_net", True))
            not_crossed_net = sum(1 for h in hits if not h.get("crossed_net", True))

            # Serve stats
            serve_total = len(serves)
            serve_in_court = sum(1 for h in serves if h.get("in_court", False))
            serve_out_court = sum(1 for h in serves if not h.get("in_court", False) and h.get("crossed_net", True))
            serve_not_crossed = sum(1 for h in serves if not h.get("crossed_net", True))
            serve_speeds = [h["speed"] for h in serves if h["speed"] > 0]
            serve_avg_speed = round(np.mean(serve_speeds), 2) if serve_speeds else 0
            serve_max_speed = round(max(serve_speeds), 2) if serve_speeds else 0

            # Return/Drive stats
            return_total = len(returns)
            return_in_court = sum(1 for h in returns if h.get("in_court", False))
            return_out_court = sum(1 for h in returns if not h.get("in_court", False) and h.get("crossed_net", True))
            return_not_crossed = sum(1 for h in returns if not h.get("crossed_net", True))
            return_speeds = [h["speed"] for h in returns if h["speed"] > 0]
            return_avg_speed = round(np.mean(return_speeds), 2) if return_speeds else 0
            return_max_speed = round(max(return_speeds), 2) if return_speeds else 0

            # Overall ratios
            in_court_ratio = in_court_count / total_hits if total_hits > 0 else 0

            # Calculate average hit speed
            all_speeds = [h["speed"] for h in hits if h["speed"] > 0]
            avg_speed = np.mean(all_speeds) if all_speeds else 0
            max_speed = max(all_speeds) if all_speeds else 0

            # Calculate average angles
            shoulder_angles = [p["shoulder_angle"] for p in poses if p["shoulder_angle"] > 0]
            knee_angles = [p["knee_bend_angle"] for p in poses if p["knee_bend_angle"] > 0]

            avg_shoulder = np.mean(shoulder_angles) if shoulder_angles else 0
            avg_knee = np.mean(knee_angles) if knee_angles else 0

            # === GENERATE HEATMAP ===
            heatmap_filename = f"player_{player_id}_heatmap.jpg"
            heatmap_path = os.path.join(output_folder, heatmap_filename)
            heatmap_url = self._generate_court_heatmap(
                positions, court_geometry, heatmap_path
            )

            # Calculate composite score
            score = self._calculate_score(in_court_ratio, avg_speed, avg_shoulder, avg_knee)

            player_stats.append({
                "player_id": player_id,
                "score": round(score, 2),
                "player_image_url": self.player_images.get(player_id),
                "heatmap_url": f"{base_url}/{heatmap_filename}" if heatmap_url else None,

                # Overall accuracy
                "accuracy": {
                    "total_hits": total_hits,
                    "in_court": in_court_count,
                    "out_court": out_court_count,
                    "not_crossed_net": not_crossed_net,
                    "in_court_ratio": round(in_court_ratio, 4)
                },

                # Serve statistics
                "serve_stats": {
                    "total": serve_total,
                    "in_court": serve_in_court,
                    "out_court": serve_out_court,
                    "not_crossed_net": serve_not_crossed,
                    "avg_speed": serve_avg_speed,
                    "max_speed": serve_max_speed
                },

                # Return/Drive statistics
                "return_stats": {
                    "total": return_total,
                    "in_court": return_in_court,
                    "out_court": return_out_court,
                    "not_crossed_net": return_not_crossed,
                    "avg_speed": return_avg_speed,
                    "max_speed": return_max_speed
                },

                # Speed stats
                "avg_hit_speed": round(avg_speed, 2),
                "max_hit_speed": round(max_speed, 2),

                # Pose stats
                "avg_shoulder_angle": round(avg_shoulder, 2),
                "avg_knee_bend_angle": round(avg_knee, 2)
            })

        # Sort by score descending
        player_stats.sort(key=lambda x: x["score"], reverse=True)

        # Add rank
        for i, stats in enumerate(player_stats):
            stats["rank"] = i + 1

        return player_stats

    def _generate_court_heatmap(
        self,
        positions: List[Tuple],
        court_geometry: CourtGeometry,
        output_path: str,
        size: Tuple[int, int] = (400, 600)
    ) -> Optional[str]:
        """
        Generate a heatmap showing player court coverage

        Args:
            positions: List of (frame, x, y) tuples
            court_geometry: Court geometry for bounds
            output_path: Path to save heatmap image
            size: Output image size (width, height)

        Returns:
            Output path if successful, None otherwise
        """
        if not positions:
            return None

        try:
            width, height = size

            # Create blank heatmap
            heatmap = np.zeros((height, width), dtype=np.float32)

            # Get court bounds for normalization
            court_bounds = court_geometry.get_bounds() if hasattr(court_geometry, 'get_bounds') else None
            if court_bounds:
                min_x, min_y, max_x, max_y = court_bounds
            else:
                # Use position bounds
                x_coords = [p[1] for p in positions]
                y_coords = [p[2] for p in positions]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

            # Add margin
            x_range = max_x - min_x if max_x > min_x else 1
            y_range = max_y - min_y if max_y > min_y else 1

            # Normalize positions and accumulate heatmap
            for _, x, y in positions:
                # Normalize to image coordinates
                norm_x = int((x - min_x) / x_range * (width - 1))
                norm_y = int((y - min_y) / y_range * (height - 1))

                # Clamp values
                norm_x = max(0, min(width - 1, norm_x))
                norm_y = max(0, min(height - 1, norm_y))

                # Add gaussian-like point
                for dy in range(-15, 16):
                    for dx in range(-15, 16):
                        py, px = norm_y + dy, norm_x + dx
                        if 0 <= py < height and 0 <= px < width:
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist < 15:
                                heatmap[py, px] += math.exp(-dist * dist / 50)

            # Normalize to 0-255
            if heatmap.max() > 0:
                heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

            # Apply colormap
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Draw court outline
            court_color = (255, 255, 255)
            cv2.rectangle(heatmap_colored, (10, 10), (width-10, height-10), court_color, 2)

            # Draw net line (horizontal middle)
            net_y = height // 2
            cv2.line(heatmap_colored, (10, net_y), (width-10, net_y), court_color, 2)

            # Draw service boxes
            cv2.line(heatmap_colored, (width//2, 10), (width//2, height-10), court_color, 1)

            # Add title
            cv2.putText(heatmap_colored, "Court Coverage", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Save
            cv2.imwrite(output_path, heatmap_colored)
            return output_path

        except Exception as e:
            print(f"  [WARN] Failed to generate heatmap: {e}")
            return None

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

    def _create_highlights_and_memes(
        self,
        video_path: str,
        ball_positions: List,
        output_folder: str,
        fps: float
    ) -> Tuple[Dict, Dict]:
        """
        Create highlight videos for each player and analyze memes

        Args:
            video_path: Path to original video
            ball_positions: List of ball positions
            output_folder: Output folder for highlight videos
            fps: Video FPS

        Returns:
            Tuple of (highlights_dict, meme_analysis_dict)
        """
        highlights = {}
        meme_analysis = {}

        # Skip if no player hits
        if not self.player_hits:
            print("  No player hits found, skipping highlights")
            return highlights, meme_analysis

        # Initialize MemeAnalyzer
        try:
            meme_analyzer = MemeAnalyzer()
        except Exception as e:
            print(f"  [WARN] Could not load MemeAnalyzer: {e}")
            meme_analyzer = None

        # Prepare player stats for meme analysis
        player_stats = {}
        for player_id in self.player_hits.keys():
            hits = self.player_hits[player_id]
            in_court_count = sum(1 for h in hits if h.get("in_court", False))
            player_stats[player_id] = {
                "accuracy": {
                    "total_hits": len(hits),
                    "in_court": in_court_count
                }
            }

        # Analyze memes
        if meme_analyzer:
            try:
                meme_analysis = meme_analyzer.analyze_shots(
                    ball_positions=ball_positions,
                    ball_hits_by_person=dict(self.player_hits),
                    player_stats=player_stats,
                    player_positions=dict(self.player_positions),
                    fps=fps
                )
                print(f"  Meme analysis complete: {len(meme_analysis)} players analyzed")
            except Exception as e:
                print(f"  [WARN] Meme analysis failed: {e}")

        # Read video frames for highlight creation
        print("  Reading video frames for highlights...")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            print("  No frames read, skipping highlights")
            return highlights, meme_analysis

        print(f"  Read {len(frames)} frames")

        # Create highlight videos for each player
        base_url = f"outputs/{os.path.basename(output_folder)}"

        for player_id, hits in self.player_hits.items():
            if len(hits) == 0:
                continue

            print(f"  Creating highlights for player {player_id} ({len(hits)} hits)...")

            player_highlights = self._create_player_highlight_videos(
                frames=frames,
                player_id=player_id,
                hits=hits,
                output_folder=output_folder,
                fps=fps,
                base_url=base_url,
                meme_analyzer=meme_analyzer,
                meme_analysis=meme_analysis.get(player_id, {})
            )

            if player_highlights:
                highlights[player_id] = player_highlights

        # Clean up frames to free memory
        del frames
        gc.collect()

        return highlights, meme_analysis

    def _create_player_highlight_videos(
        self,
        frames: List,
        player_id: int,
        hits: List[Dict],
        output_folder: str,
        fps: float,
        base_url: str,
        meme_analyzer=None,
        meme_analysis: Dict = None
    ) -> Dict:
        """
        Create highlight videos for a single player

        Args:
            frames: All video frames
            player_id: Player ID
            hits: List of hit events for this player
            output_folder: Output folder
            fps: Video FPS
            base_url: Base URL for video paths
            meme_analyzer: MemeAnalyzer instance
            meme_analysis: Meme analysis for this player

        Returns:
            Dict with highlight info
        """
        MIN_DURATION_SECONDS = 10
        min_duration_frames = int(MIN_DURATION_SECONDS * fps)
        padding_frames = 15
        min_frames = 30

        # Sort hits by frame
        sorted_hits = sorted(hits, key=lambda h: h["frame"])

        # Group hits into sequences (continuous hits within 3 seconds)
        max_gap_frames = int(3 * fps)
        sequences = []
        current_sequence = [sorted_hits[0]]

        for i in range(1, len(sorted_hits)):
            if sorted_hits[i]["frame"] - sorted_hits[i-1]["frame"] <= max_gap_frames:
                current_sequence.append(sorted_hits[i])
            else:
                sequences.append(current_sequence)
                current_sequence = [sorted_hits[i]]
        sequences.append(current_sequence)

        # Create highlights for each sequence
        highlight_clips = []
        memes_for_player = meme_analysis.get("memes", []) if meme_analysis else []

        for seq_idx, sequence in enumerate(sequences):
            if len(sequence) == 0:
                continue

            # Collect frames for this sequence
            highlight_frames_set = set()
            for hit in sequence:
                hit_frame = hit["frame"]
                start_frame = max(0, hit_frame - padding_frames)
                end_frame = min(len(frames), hit_frame + padding_frames + 1)
                for f in range(start_frame, end_frame):
                    highlight_frames_set.add(f)

            highlight_frame_indices = sorted(list(highlight_frames_set))

            if len(highlight_frame_indices) < min_frames:
                continue

            # Extend if less than minimum duration
            if len(highlight_frame_indices) < min_duration_frames:
                needed_frames = min_duration_frames - len(highlight_frame_indices)
                extend_before = needed_frames // 2
                extend_after = needed_frames - extend_before

                first_frame = highlight_frame_indices[0]
                last_frame = highlight_frame_indices[-1]

                new_start = max(0, first_frame - extend_before)
                for f in range(new_start, first_frame):
                    highlight_frames_set.add(f)

                new_end = min(len(frames), last_frame + extend_after + 1)
                for f in range(last_frame + 1, new_end):
                    highlight_frames_set.add(f)

                highlight_frame_indices = sorted(list(highlight_frames_set))

            # Get frame dimensions
            first_frame = frames[highlight_frame_indices[0]]
            height, width = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Create highlight video
            highlight_filename = f"player_{player_id}_highlight_{seq_idx + 1}.mp4"
            highlight_path = os.path.join(output_folder, highlight_filename)
            out_highlight = cv2.VideoWriter(highlight_path, fourcc, fps, (width, height))

            # Save cropped player image from the best hit in this sequence (for clip thumbnail)
            player_crop_filename = f"player_{player_id}_highlight_{seq_idx + 1}_crop.jpg"
            player_crop_path = os.path.join(output_folder, player_crop_filename)
            best_crop_saved = False

            # Track hit crops for each individual hit
            hit_crops = []
            hit_crop_counter = 0

            try:
                for frame_idx in highlight_frame_indices:
                    frame = frames[frame_idx].copy()

                    # Mark hit frames
                    for hit_idx, hit in enumerate(sequence):
                        if hit["frame"] == frame_idx:
                            ball_pos = hit["ball_pos"]
                            cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 15, (0, 0, 255), 3)
                            cv2.putText(frame, "HIT!", (int(ball_pos[0]) + 20, int(ball_pos[1])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                            # Draw player bbox
                            bbox = hit["bbox"]
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"Player {player_id}", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            # Add padding to crop
                            pad = 30
                            crop_x1 = max(0, int(x1) - pad)
                            crop_y1 = max(0, int(y1) - pad)
                            crop_x2 = min(width, int(x2) + pad)
                            crop_y2 = min(height, int(y2) + pad)

                            player_crop = frames[frame_idx][crop_y1:crop_y2, crop_x1:crop_x2]

                            if player_crop.size > 0:
                                # Save crop for each individual hit
                                hit_crop_counter += 1
                                hit_crop_filename = f"player_{player_id}_highlight_{seq_idx + 1}_hit_{hit_crop_counter}.jpg"
                                hit_crop_path = os.path.join(output_folder, hit_crop_filename)
                                cv2.imwrite(hit_crop_path, player_crop)
                                hit_crops.append({
                                    "hit_index": hit_crop_counter,
                                    "frame": frame_idx,
                                    "image_url": f"{base_url}/{hit_crop_filename}",
                                    "speed": hit.get("speed", 0),
                                    "hit_type": hit.get("hit_type", "unknown"),
                                    "in_court": hit.get("in_court", False),
                                    "crossed_net": hit.get("crossed_net", True)
                                })

                                # Also save first crop as the main clip thumbnail
                                if not best_crop_saved:
                                    cv2.imwrite(player_crop_path, player_crop)
                                    best_crop_saved = True
                            break

                    out_highlight.write(frame)
            finally:
                out_highlight.release()

            # Clip info with individual hit images
            clip_info = {
                "highlight_video": f"{base_url}/{highlight_filename}",
                "player_image": f"{base_url}/{player_crop_filename}" if best_crop_saved else None,
                "hit_images": hit_crops,  # List of crop images for each hit
                "clip_index": seq_idx + 1,
                "hit_count": len(sequence),
                "total_frames": len(highlight_frame_indices),
                "duration_seconds": round(len(highlight_frame_indices) / fps, 2),
                "start_frame": highlight_frame_indices[0],
                "end_frame": highlight_frame_indices[-1],
                "hits": [{"frame": h["frame"], "ball_pos": h["ball_pos"], "speed": h.get("speed", 0), "hit_type": h.get("hit_type", "unknown")} for h in sequence]
            }
            highlight_clips.append(clip_info)

        if not highlight_clips:
            return None

        return {
            "player_id": player_id,
            "highlights": highlight_clips,
            "total_clips": len(highlight_clips),
            "total_hits": len(hits),
            "total_duration_seconds": round(sum(c["duration_seconds"] for c in highlight_clips), 2),
            "memes": memes_for_player
        }
