# =============================================================================
# API ROUTES FOR TENNIS ANALYSIS
# =============================================================================

from flask import Blueprint, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from src.core.tennis_analysis_module import TennisAnalysisModule
from src.core.var_detector import VarDetector
from config.settings import settings
import cv2
import os
import uuid
from datetime import datetime, timedelta
import traceback
import shutil
import threading
import time
import requests

# Create Blueprint
api_bp = Blueprint("api", __name__)


def create_analyzer():
    """Tạo instance mới của TennisAnalysisModule cho mỗi request để tránh race condition"""
    return TennisAnalysisModule(
        ball_model_path=settings.ball_model_path,
        person_model_path=settings.person_model_path,
        pose_model_path=settings.pose_model_path,
    )


def create_var_detector():
    """Tạo instance mới của VarDetector cho mỗi request để tránh race condition"""
    return VarDetector(model_path=settings.ball_model_path, conf=0.8, batch_size=32)


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


def validate_float_param(value, default, param_name, min_val=0.0, max_val=1.0):
    """Validate và convert float parameter với error handling"""
    try:
        result = float(value) if value else default
        if not (min_val <= result <= max_val):
            raise ValueError(f"{param_name} must be between {min_val} and {max_val}")
        return result
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid {param_name}: {value}. Error: {str(e)}")


def validate_court_bounds(court_bounds_str, default_bounds):
    """Validate và parse court_bounds parameter"""
    if not court_bounds_str:
        return default_bounds

    try:
        parts = court_bounds_str.split(",")
        if len(parts) != 4:
            raise ValueError(f"court_bounds must have exactly 4 values (x1,y1,x2,y2), got {len(parts)}")

        bounds = tuple(int(p.strip()) for p in parts)

        # Validate logic: x2 > x1, y2 > y1
        x1, y1, x2, y2 = bounds
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid court_bounds: x2 must > x1 and y2 must > y1")

        return bounds
    except ValueError as e:
        raise ValueError(f"Invalid court_bounds format: {court_bounds_str}. Error: {str(e)}")


def process_video_async(
    video_path,
    request_id,
    request_output_folder,
    callback_url,
    original_video_path,
    ball_conf,
    person_conf,
    angle_threshold,
    intersection_threshold,
    court_bounds,
):
    """
    Xử lý video trong background thread và gọi callback khi hoàn thành

    Args:
        video_path: Đường dẫn video đã upload (local)
        request_id: ID của request
        request_output_folder: Thư mục output
        callback_url: URL để gọi callback khi xử lý xong
        original_video_path: Đường dẫn video gốc trên server (để trả về trong callback)
        ball_conf, person_conf, angle_threshold, intersection_threshold, court_bounds: Các parameters phân tích
    """
    try:
        print(f"[ASYNC] Starting video analysis for request {request_id}")

        # Tạo analyzer instance mới cho request này (tránh race condition)
        analyzer = create_analyzer()

        # Phân tích video
        results = analyzer.analyze_video(
            video_path=video_path,
            ball_conf=ball_conf,
            person_conf=person_conf,
            angle_threshold=angle_threshold,
            intersection_threshold=intersection_threshold,
            court_bounds=court_bounds,
        )

        # Xử lý kết quả và tạo analysis_result
        analysis_result = {
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

        analysis_result["highest_speed_info"] = {
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
            analysis_result["best_players"].append(player_data)

        # 3. Xử lý match statistics
        stats = results["match_statistics"]
        analysis_result["match_statistics"] = {
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
            video_filename = f"visualization_{request_id}.mp4"
            new_video_path = os.path.join(request_output_folder, video_filename)
            shutil.copy2(results["visualization_video_path"], new_video_path)
            analysis_result["visualization_video_url"] = generate_file_url(
                video_filename, request_id
            )
            # Xóa temp visualization video sau khi copy
            try:
                os.remove(results["visualization_video_path"])
                print(f"[CLEANUP] Deleted temp visualization video")
            except Exception as e:
                print(f"[CLEANUP WARNING] Failed to delete temp video: {e}")

        # Xóa video upload sau khi xử lý xong
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Deleted uploaded video: {video_path}")
        except Exception as cleanup_error:
            print(f"[CLEANUP ERROR] Failed to delete {video_path}: {cleanup_error}")

        # Tạo callback payload theo format server yêu cầu
        callback_payload = {
            "video_path": original_video_path,
            "request_id": request_id,
            "status": "success",
            "analysis_result": analysis_result,
        }

        # Gọi callback với retry logic (tối đa 3 lần)
        print(f"[ASYNC] Sending callback to {callback_url}")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    callback_url,
                    json=callback_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                response_text = response.text[:200] if response.text else ""
                print(f"[ASYNC] Callback response: {response.status_code} - {response_text}")
                if response.status_code < 500:  # Success hoặc client error, không retry
                    break
            except Exception as callback_error:
                print(f"[ASYNC ERROR] Callback attempt {attempt + 1}/{max_retries} failed: {callback_error}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

        print(f"[ASYNC] Completed video analysis for request {request_id}")

    except Exception as e:
        print(f"[ASYNC ERROR] Failed to process request {request_id}: {e}")

        # Xóa video upload nếu có lỗi
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Deleted uploaded video after error: {video_path}")
        except:
            pass

        # Xóa output folder nếu có lỗi (tránh tích tụ thư mục rác)
        try:
            if os.path.exists(request_output_folder):
                shutil.rmtree(request_output_folder)
                print(f"[CLEANUP] Deleted output folder after error: {request_output_folder}")
        except Exception as cleanup_error:
            print(f"[CLEANUP ERROR] Failed to delete output folder: {cleanup_error}")

        # Tạo error callback payload theo format server yêu cầu
        error_payload = {
            "video_path": original_video_path,
            "request_id": request_id,
            "status": "failed",
            "error": str(e),
        }

        # Gọi error callback với retry logic (tối đa 3 lần)
        print(f"[ASYNC] Sending error callback to {callback_url}")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                requests.post(
                    callback_url,
                    json=error_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                print(f"[ASYNC] Error callback sent successfully")
                break
            except Exception as callback_error:
                print(f"[ASYNC ERROR] Error callback attempt {attempt + 1}/{max_retries} failed: {callback_error}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)


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
    Endpoint phân tích video tennis (async - luôn trả về ngay, callback khi xong)

    Parameters (form-data):
        - video: Video file (required)
        - callback_url: URL để gọi callback khi xử lý xong (required)
        - original_video_path: Đường dẫn video gốc trên server (required)
        - ball_conf: Ball detection confidence (default: 0.7)
        - person_conf: Person detection confidence (default: 0.6)
        - angle_threshold: Angle threshold (default: 50)
        - intersection_threshold: Intersection threshold (default: 100)
        - court_bounds: Court bounds as "x1,y1,x2,y2" (default: "100,100,400,500")

    Returns:
        Trả về ngay: {"status": "processing", "request_id": "..."}
        Khi xử lý xong sẽ POST callback đến callback_url với kết quả
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

        # Kiểm tra callback_url (required)
        callback_url = request.form.get("callback_url")
        if not callback_url:
            return jsonify({"error": "callback_url is required"}), 400

        original_video_path = request.form.get("original_video_path", "")

        # Lưu video upload
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        video_path = os.path.join(settings.upload_folder, unique_filename)
        file.save(video_path)

        # Tạo thư mục output riêng cho request này
        request_id = uuid.uuid4().hex
        request_output_folder = os.path.join(settings.output_folder, request_id)
        os.makedirs(request_output_folder, exist_ok=True)

        # Lấy và validate parameters từ request
        try:
            ball_conf = validate_float_param(
                request.form.get("ball_conf"),
                settings.default_ball_conf,
                "ball_conf", 0.0, 1.0
            )
            person_conf = validate_float_param(
                request.form.get("person_conf"),
                settings.default_person_conf,
                "person_conf", 0.0, 1.0
            )
            angle_threshold = validate_float_param(
                request.form.get("angle_threshold"),
                settings.default_angle_threshold,
                "angle_threshold", 0.0, 180.0
            )
            intersection_threshold = validate_float_param(
                request.form.get("intersection_threshold"),
                settings.default_intersection_threshold,
                "intersection_threshold", 0.0, 1000.0
            )
            court_bounds = validate_court_bounds(
                request.form.get("court_bounds"),
                settings.default_court_bounds
            )
        except ValueError as e:
            # Xóa video đã upload nếu validation fail
            if os.path.exists(video_path):
                os.remove(video_path)
            return jsonify({"error": str(e)}), 400

        print(f"[ASYNC] Received request {request_id}, callback: {callback_url}")

        # Tạo background thread để xử lý video
        processing_thread = threading.Thread(
            target=process_video_async,
            args=(
                video_path,
                request_id,
                request_output_folder,
                callback_url,
                original_video_path,
                ball_conf,
                person_conf,
                angle_threshold,
                intersection_threshold,
                court_bounds,
            ),
            daemon=True,
        )
        processing_thread.start()

        # Trả về ngay lập tức
        return jsonify({
            "status": "processing",
            "request_id": request_id,
            "message": "Video is being processed. Results will be sent to callback URL.",
        }), 202

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
    video_path = None
    request_output_folder = None

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

        # Tạo instance mới của VarDetector cho request này (thread safety)
        var_detector = create_var_detector()

        # Phân tích video với VAR Detector
        results = var_detector.detect_video(
            video_path, output_folder=request_output_folder
        )

        # Validate results dictionary
        if not results or not isinstance(results, dict):
            raise ValueError("VAR detection returned invalid results")

        # Copy các video kết quả vào output folder và tạo URLs
        crop_filename = f"var_crop_{request_id}.mp4"
        mask_filename = f"var_mask_{request_id}.mp4"

        crop_output_path = os.path.join(request_output_folder, crop_filename)
        mask_output_path = os.path.join(request_output_folder, mask_filename)

        # Prepare response videos dict
        videos_result = {}

        # Copy crop file nếu tồn tại
        crop_path = results.get("crop")
        if crop_path and os.path.exists(crop_path):
            shutil.copy2(crop_path, crop_output_path)
            os.remove(crop_path)  # Xóa file tạm
            crop_view_url = generate_file_url(crop_filename, request_id)
            videos_result["crop"] = {
                "view_url": crop_view_url,
                "download_url": f"{crop_view_url}?download=true",
                "filename": crop_filename,
            }
        else:
            print(f"[WARNING] Crop video not found or not generated: {crop_path}")
            videos_result["crop"] = None

        # Copy mask file nếu tồn tại
        mask_path = results.get("mask")
        if mask_path and os.path.exists(mask_path):
            shutil.copy2(mask_path, mask_output_path)
            os.remove(mask_path)  # Xóa file tạm
            mask_view_url = generate_file_url(mask_filename, request_id)
            videos_result["mask"] = {
                "view_url": mask_view_url,
                "download_url": f"{mask_view_url}?download=true",
                "filename": mask_filename,
            }
        else:
            print(f"[WARNING] Mask video not found or not generated: {mask_path}")
            videos_result["mask"] = None

        # Kiểm tra xem có ít nhất 1 video được tạo không
        if videos_result["crop"] is None and videos_result["mask"] is None:
            raise ValueError("No output videos were generated. Check if the video contains detectable content.")

        # Tạo response
        result = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "expires_at": (
                datetime.now() + timedelta(hours=settings.cleanup_hours)
            ).isoformat(),
            "videos": videos_result,
            "original_video": results.get("origin", video_path),
        }

        # Lên lịch xóa video sau 3 giờ
        schedule_video_deletion(video_path, delay_hours=3)

        return jsonify(result), 200

    except Exception as e:
        # Cleanup on error
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Deleted uploaded video after error: {video_path}")
        except Exception as cleanup_error:
            print(f"[CLEANUP ERROR] Failed to delete video: {cleanup_error}")

        try:
            if request_output_folder and os.path.exists(request_output_folder):
                shutil.rmtree(request_output_folder)
                print(f"[CLEANUP] Deleted output folder after error: {request_output_folder}")
        except Exception as cleanup_error:
            print(f"[CLEANUP ERROR] Failed to delete output folder: {cleanup_error}")

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def create_api_blueprint():
    """Factory function to create and return the API blueprint"""
    return api_bp
