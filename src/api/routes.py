# =============================================================================
# API ROUTES FOR TENNIS ANALYSIS
# =============================================================================

from flask import Blueprint, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from src.core.tennis_analysis_module import TennisAnalysisModule
from src.core.var_detector import VarDetector
from src.core.player_analysis_service import PlayerAnalysisService
from config.settings import settings
import cv2
import os
import uuid
from datetime import datetime, timedelta
import traceback
import shutil
import threading
import time

# Create Blueprint
api_bp = Blueprint("api", __name__)

# Initialize Tennis Analysis Module
analyzer = TennisAnalysisModule(
    ball_model_path=settings.ball_model_path,
    person_model_path=settings.person_model_path,
    pose_model_path=settings.pose_model_path,
)

# Initialize VAR Detector
var_detector = VarDetector(model_path=settings.ball_model_path, conf=0.8, batch_size=32)

# Initialize Player Analysis Service
player_analysis_service = PlayerAnalysisService(
    ball_model_path=settings.ball_model_path,
    person_model_path=settings.person_model_path,
    pose_model_path=settings.pose_model_path
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def schedule_video_deletion(video_path, delay_hours=3):
    """
    Lên lịch xóa video sau một khoảng thời gian nhất định

    Args:
        video_path: Đường dẫn đến video cần xóa
        delay_hours: Số giờ chờ trước khi xóa (mặc định: 3)
    """

    def delete_video():
        try:
            time.sleep(delay_hours * 3600)  # Chuyển giờ sang giây
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Deleted video after {delay_hours}h: {video_path}")
            else:
                print(f"[CLEANUP] Video already deleted: {video_path}")
        except Exception as e:
            print(f"[CLEANUP ERROR] Failed to delete {video_path}: {e}")

    # Tạo thread daemon để xóa video
    deletion_thread = threading.Thread(target=delete_video, daemon=True)
    deletion_thread.start()
    print(f"[CLEANUP] Scheduled deletion for {video_path} in {delay_hours} hours")


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

        # Xóa video upload ngay sau khi xử lý xong
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Deleted uploaded video immediately: {video_path}")
        except Exception as cleanup_error:
            print(f"[CLEANUP ERROR] Failed to delete {video_path}: {cleanup_error}")

        # Trả về trực tiếp result JSON
        return jsonify(result), 200

    except Exception as e:
        # Nếu có lỗi, vẫn cố gắng xóa video đã upload
        try:
            if "video_path" in locals() and os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Deleted uploaded video after error: {video_path}")
        except:
            pass
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


@api_bp.route("/api/check_var", methods=["POST"])
def check_var():
    """
    Endpoint để kiểm tra VAR (Video Assistant Referee) cho video bóng đá

    Parameters (form-data):
        - video: Video file (required)

    Returns:
        JSON với URLs đến các video đã xử lý (crop, mask) và video gốc
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

        # Phân tích video với VAR Detector
        results = var_detector.detect_video(
            video_path, output_folder=request_output_folder
        )

        # Copy các video kết quả vào output folder và tạo URLs
        crop_filename = f"var_crop_{request_id}.mp4"
        mask_filename = f"var_mask_{request_id}.mp4"

        crop_output_path = os.path.join(request_output_folder, crop_filename)
        mask_output_path = os.path.join(request_output_folder, mask_filename)

        # Copy files từ thư mục tạm sang output folder
        if os.path.exists(results["crop"]):
            shutil.copy2(results["crop"], crop_output_path)
            os.remove(results["crop"])  # Xóa file tạm

        if os.path.exists(results["mask"]):
            shutil.copy2(results["mask"], mask_output_path)
            os.remove(results["mask"])  # Xóa file tạm

        # Tạo URLs cho view và download
        crop_view_url = generate_file_url(crop_filename, request_id)
        mask_view_url = generate_file_url(mask_filename, request_id)

        # Tạo download URLs (thêm parameter download=true)
        crop_download_url = f"{crop_view_url}?download=true"
        mask_download_url = f"{mask_view_url}?download=true"

        # Tạo response
        result = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "expires_at": (
                datetime.now() + timedelta(hours=settings.cleanup_hours)
            ).isoformat(),
            "videos": {
                "crop": {
                    "view_url": crop_view_url,
                    "download_url": crop_download_url,
                    "filename": crop_filename,
                },
                "mask": {
                    "view_url": mask_view_url,
                    "download_url": mask_download_url,
                    "filename": mask_filename,
                },
            },
            "original_video": results["origin"],
        }

        # Lên lịch xóa video sau 3 giờ
        schedule_video_deletion(video_path, delay_hours=3)

        return jsonify(result), 200

    except Exception as e:
        # Nếu có lỗi, vẫn cố gắng xóa video đã upload ngay lập tức
        try:
            if "video_path" in locals() and os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Deleted uploaded video after error: {video_path}")
        except:
            pass
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@api_bp.route("/api/player-analysis", methods=["POST"])
def player_analysis():
    """
    Endpoint phân tích chi tiết người chơi tennis với 8 chỉ số

    Parameters (form-data):
        - video: Video file (required)
        - court_points: JSON string của 12 điểm tọa độ sân (required)
          Ví dụ: "[[361,139],[481,132],[560,130],[664,131],[981,153],[887,338],[714,641],[372,457],[288,408],[244,372],[169,324],[270,224]]"
        - net_start_idx: Index điểm bắt đầu lưới (default: 2)
        - net_end_idx: Index điểm kết thúc lưới (default: 8)
        - ball_conf: Ball detection confidence (default: 0.7)
        - person_conf: Person detection confidence (default: 0.6)
        - angle_threshold: Angle threshold (default: 50)
        - intersection_threshold: Intersection threshold (default: 100)

    Returns:
        JSON với 8 chỉ số phân tích và links đến hình ảnh/video
    """
    import json

    try:
        # Kiểm tra file có được upload không
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        file = request.files["video"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": f"Invalid file type. Allowed: {settings.allowed_extensions}"
            }), 400

        # Kiểm tra court_points
        court_points_str = request.form.get("court_points")
        if not court_points_str:
            return jsonify({"error": "court_points is required"}), 400

        try:
            court_points = json.loads(court_points_str)
            court_points = [tuple(p) for p in court_points]
            if len(court_points) != 12:
                return jsonify({"error": "court_points must have exactly 12 points"}), 400
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid court_points JSON format"}), 400

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
        net_start_idx = int(request.form.get("net_start_idx", 2))
        net_end_idx = int(request.form.get("net_end_idx", 8))
        ball_conf = float(request.form.get("ball_conf", settings.default_ball_conf))
        person_conf = float(request.form.get("person_conf", settings.default_person_conf))
        angle_threshold = float(request.form.get("angle_threshold", settings.default_angle_threshold))
        intersection_threshold = float(request.form.get("intersection_threshold", settings.default_intersection_threshold))

        # Base URL cho files (dạng: outputs/request_id)
        base_url = f"outputs/{request_id}"

        # Phân tích video
        result = player_analysis_service.analyze(
            video_path=video_path,
            court_points=court_points,
            output_folder=request_output_folder,
            net_start_idx=net_start_idx,
            net_end_idx=net_end_idx,
            ball_conf=ball_conf,
            person_conf=person_conf,
            angle_threshold=angle_threshold,
            intersection_threshold=intersection_threshold,
            base_url=base_url
        )

        # Thêm request_id và expires_at
        result["request_id"] = request_id
        result["expires_at"] = (
            datetime.now() + timedelta(hours=settings.cleanup_hours)
        ).isoformat()

        # Xóa video upload ngay sau khi xử lý xong
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Deleted uploaded video immediately: {video_path}")
        except Exception as cleanup_error:
            print(f"[CLEANUP ERROR] Failed to delete {video_path}: {cleanup_error}")

        return jsonify(result), 200

    except Exception as e:
        # Nếu có lỗi, vẫn cố gắng xóa video đã upload
        try:
            if "video_path" in locals() and os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Deleted uploaded video after error: {video_path}")
        except:
            pass
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def create_api_blueprint():
    """Factory function to create and return the API blueprint"""
    return api_bp
