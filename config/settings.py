"""
Configuration settings for Tennis Analysis API
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model paths
MODEL_DIR = BASE_DIR / "src/models"
BALL_MODEL_PATH = str(MODEL_DIR / "ball_best.pt")
PERSON_MODEL_PATH = "yolo11m.pt" # Nano for ball-person intersection
POSE_MODEL_PATH = "yolo11m-pose.pt"  # Medium for better pose detection accuracy

# Upload and output folders
UPLOAD_FOLDER = str(BASE_DIR / "uploads")
OUTPUT_FOLDER = str(BASE_DIR / "outputs")

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "2803"))
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Server base URL for file access (used in API responses)
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "https://durham-cooking-shoe-behaviour.trycloudflare.com")

# File upload settings
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# Analysis default parameters
DEFAULT_BALL_CONF = 0.7
DEFAULT_PERSON_CONF = 0.6
DEFAULT_ANGLE_THRESHOLD = 50
DEFAULT_INTERSECTION_THRESHOLD = 100
DEFAULT_COURT_BOUNDS = (100, 100, 400, 500)

# Cleanup settings
CLEANUP_HOURS = 3  # Delete files older than 3 hours

# Memory optimization settings
MAX_FRAME_HEIGHT = 480  # Maximum frame height for inference - reduced for speed
ENABLE_FRAME_RESIZE = True  # Enable frame resizing before inference


class Settings:
    """Settings class for easy access"""

    def __init__(self):
        self.base_dir = BASE_DIR
        self.model_dir = MODEL_DIR
        self.ball_model_path = BALL_MODEL_PATH
        self.person_model_path = PERSON_MODEL_PATH
        self.pose_model_path = POSE_MODEL_PATH
        self.upload_folder = UPLOAD_FOLDER
        self.output_folder = OUTPUT_FOLDER
        self.api_host = API_HOST
        self.api_port = API_PORT
        self.debug = DEBUG
        self.server_base_url = SERVER_BASE_URL
        self.max_content_length = MAX_CONTENT_LENGTH
        self.allowed_extensions = ALLOWED_EXTENSIONS
        self.default_ball_conf = DEFAULT_BALL_CONF
        self.default_person_conf = DEFAULT_PERSON_CONF
        self.default_angle_threshold = DEFAULT_ANGLE_THRESHOLD
        self.default_intersection_threshold = DEFAULT_INTERSECTION_THRESHOLD
        self.default_court_bounds = DEFAULT_COURT_BOUNDS
        self.cleanup_hours = CLEANUP_HOURS
        self.max_frame_height = MAX_FRAME_HEIGHT
        self.enable_frame_resize = ENABLE_FRAME_RESIZE


# Global settings instance
settings = Settings()
