# =============================================================================
# API ROUTES FOR TENNIS ANALYSIS
# =============================================================================

from flask import Blueprint, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from src.core.tennis_analysis_module import TennisAnalysisModule
from config.settings import settings
import cv2
import os
import uuid
from datetime import datetime, timedelta
import traceback
import shutil

# Create Blueprint
api_bp = Blueprint("api", __name__)

# Initialize Tennis Analysis Module
analyzer = TennisAnalysisModule(
    ball_model_path=settings.ball_model_path,
    person_model_path=settings.person_model_path,
    pose_model_path=settings.pose_model_path,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def allowed_file(filename):
    """Kiểm tra file extension có hợp lệ không"""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in settings.allowed_extensions
    )


def save_cropped_image(image, output_folder, prefix, identifier):
    """Lưu ảnh crop và trả về đường dẫn"""
    if image is None:
        return None

    filename = f"{prefix}_{identifier}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, image)
    return filename


def generate_file_url(filename, folder):
    """Tạo URL để truy cập file"""
    return url_for("serve_file", folder=folder, filename=filename, _external=True)


# =============================================================================
# API ENDPOINTS
# =============================================================================


@api_bp.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "service": "Tennis Analysis API",
            "timestamp": datetime.now().isoformat(),
        }
    )


@api_bp.route("/api/analyze", methods=["POST"])
def analyze_video():
    """
    Endpoint chính để phân tích video tennis

    Parameters (form-data):
        - video: Video file (required)
        - ball_conf: Ball detection confidence (default: 0.7)
        - person_conf: Person detection confidence (default: 0.6)
        - angle_threshold: Angle threshold (default: 50)
        - intersection_threshold: Intersection threshold (default: 100)
        - court_bounds: Court bounds as "x1,y1,x2,y2" (default: "100,100,400,500")

    Returns:
        JSON trực tiếp với kết quả phân tích và links đến hình ảnh/video
    """
    try:
        # Kiểm tra file có được upload không
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        file = request.files["video"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "error": f"Invalid file type. Allowed: {settings.allowed_extensions}"
                    }
                ),
                400,
            )

        # Lưu video upload
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        video_path = os.path.join(settings.upload_folder, unique_filename)
        file.save(video_path)

        # Tạo thư mục output riêng cho request này
        request_id = uuid.uuid4().hex
        request_output_folder = os.path.join(settings.output_folder, request_id)
        os.makedirs(request_output_folder, exist_ok=True)

        # Lấy parameters từ request
        ball_conf = float(request.form.get("ball_conf", settings.default_ball_conf))
        person_conf = float(
            request.form.get("person_conf", settings.default_person_conf)
        )
        angle_threshold = float(
            request.form.get("angle_threshold", settings.default_angle_threshold)
        )
        intersection_threshold = float(
            request.form.get(
                "intersection_threshold", settings.default_intersection_threshold
            )
        )

        # Parse court bounds
        court_bounds_str = request.form.get(
            "court_bounds", ",".join(map(str, settings.default_court_bounds))
        )
        court_bounds = tuple(map(int, court_bounds_str.split(",")))

        # Phân tích video
        results = analyzer.analyze_video(
            video_path=video_path,
            ball_conf=ball_conf,
            person_conf=person_conf,
            angle_threshold=angle_threshold,
            intersection_threshold=intersection_threshold,
            court_bounds=court_bounds,
        )

        # Xử lý kết quả và tạo URLs - Trả về trực tiếp
        result = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "expires_at": (
                datetime.now() + timedelta(hours=settings.cleanup_hours)
            ).isoformat(),
            "highest_speed_info": {},
            "best_players": [],
            "match_statistics": {},
            "visualization_video_url": None,
        }

        # 1. Xử lý highest speed info
        highest_speed = results["highest_speed_info"]
        cropped_filename = save_cropped_image(
            highest_speed["cropped_image"],
            request_output_folder,
            "highest_speed",
            "player",
        )

        result["highest_speed_info"] = {
            "frame": highest_speed["frame"],
            "time_seconds": round(highest_speed["time_seconds"], 2),
            "velocity": round(highest_speed["velocity"], 2),
            "person_id": highest_speed["person_id"],
            "shoulder_angle": round(highest_speed["shoulder_angle"], 2),
            "knee_bend_angle": round(highest_speed["knee_bend_angle"], 2),
            "cropped_image_url": (
                generate_file_url(cropped_filename, request_id)
                if cropped_filename
                else None
            ),
        }

        # 2. Xử lý best players
        for rank, player in enumerate(results["best_players"], 1):
            cropped_filename = save_cropped_image(
                player["cropped_image"],
                request_output_folder,
                f'player_{player["player_id"]}_rank_{rank}',
                "crop",
            )

            player_data = {
                "rank": rank,
                "player_id": player["player_id"],
                "score": round(player["score"], 2),
                "in_court_ratio": round(player["in_court_ratio"], 4),
                "avg_ball_speed": round(player["avg_ball_speed"], 2),
                "avg_shoulder_angle": round(player["avg_shoulder_angle"], 2),
                "avg_knee_bend_angle": round(player["avg_knee_bend_angle"], 2),
                "total_hits": player["total_hits"],
                "cropped_image_url": (
                    generate_file_url(cropped_filename, request_id)
                    if cropped_filename
                    else None
                ),
            }
            result["best_players"].append(player_data)

        # 3. Xử lý match statistics
        stats = results["match_statistics"]
        result["match_statistics"] = {
            "rally_ratio": round(stats["rally_ratio"], 4),
            "in_court_ratio": round(stats["in_court_ratio"], 4),
            "out_court_ratio": round(stats["out_court_ratio"], 4),
            "total_hits": stats["total_hits"],
            "total_in_court": stats["total_in_court"],
            "total_out_court": stats["total_out_court"],
        }

        # 4. Xử lý visualization video
        if results["visualization_video_path"] and os.path.exists(
            results["visualization_video_path"]
        ):
            # Copy video vào output folder
            video_filename = f"visualization_{request_id}.mp4"
            new_video_path = os.path.join(request_output_folder, video_filename)
            shutil.copy2(results["visualization_video_path"], new_video_path)
            result["visualization_video_url"] = generate_file_url(
                video_filename, request_id
            )

        # Trả về trực tiếp result JSON
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@api_bp.route("/api/results/<request_id>", methods=["GET"])
def get_results(request_id):
    """
    Lấy danh sách tất cả files của một request
    """
    try:
        request_folder = os.path.join(settings.output_folder, request_id)

        if not os.path.exists(request_folder):
            return jsonify({"error": "Request ID not found"}), 404

        files = os.listdir(request_folder)
        file_urls = {
            filename: generate_file_url(filename, request_id) for filename in files
        }

        return jsonify({"request_id": request_id, "files": file_urls}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def create_api_blueprint():
    """Factory function to create and return the API blueprint"""
    return api_bp
