# =============================================================================
# API ROUTES FOR TENNIS ANALYSIS
# =============================================================================

from flask import Blueprint, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from src.core.tennis_analysis_module import TennisAnalysisModule
from src.core.var_detector import VarDetector
from src.core.player_analysis_service import PlayerAnalysisService
from src.core.tennis_video_analysis_service import TennisVideoAnalysisService
from src.core.gpu_queue_manager import gpu_queue, TaskPriority
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
import json

# Create Blueprint
api_bp = Blueprint("api", __name__)

# Initialize Tennis Analysis Module
analyzer = TennisAnalysisModule(
    ball_model_path=settings.ball_model_path,
    person_model_path=settings.person_model_path,
    pose_model_path=settings.pose_model_path,
)

# Initialize VAR Detector (uses default conf=0.3)
var_detector = VarDetector(model_path=settings.ball_model_path, batch_size=32)


def create_player_analysis_service():
    """Tạo instance mới của PlayerAnalysisService cho mỗi request để tránh race condition"""
    return PlayerAnalysisService(
        ball_model_path=settings.ball_model_path,
        person_model_path=settings.person_model_path,
        batch_size=8,  # Reduced from 16 to 8 for memory safety
    )


def create_tennis_video_analysis_service():
    """Tạo instance mới của TennisVideoAnalysisService cho mỗi request"""
    return TennisVideoAnalysisService(
        ball_model_path=settings.ball_model_path,
        person_model_path=settings.person_model_path,
        pose_model_path=settings.pose_model_path,
        batch_size=settings.tennis_analysis_batch_size,
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


def convert_paths_to_urls(data, request_id, file_server_url):
    """
    Đệ quy convert tất cả các đường dẫn file trong dict thành full URL

    Args:
        data: Dict hoặc value cần convert
        request_id: ID của request để tạo URL
        file_server_url: File server URL for serving files (e.g., https://download-linevision.ngrok.app)

    Returns:
        Dict với các path đã được convert thành URL
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = convert_paths_to_urls(value, request_id, file_server_url)
        return result
    elif isinstance(data, list):
        return [convert_paths_to_urls(item, request_id, file_server_url) for item in data]
    elif isinstance(data, str):
        # Kiểm tra nếu là đường dẫn file (chứa outputs/ hoặc kết thúc bằng extension)
        file_extensions = (".png", ".jpg", ".jpeg", ".mp4", ".avi", ".mov", ".json")
        if data.startswith(f"outputs/{request_id}/") or (
            f"outputs/{request_id}" in data and data.lower().endswith(file_extensions)
        ):
            # Tạo full URL với đường dẫn /outputs/
            full_url = f"{file_server_url.rstrip('/')}/{data}"
            return full_url
        return data
    else:
        return data


def convert_numpy_types(data):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    import numpy as np

    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


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


def process_var_async(video_path, request_output_folder, request_id, callback_url=None):
    """
    Xử lý VAR trong background với priority cao nhất.
    Các task khác sẽ tạm dừng cho đến khi VAR hoàn thành.
    """
    if callback_url is None:
        callback_url = "http://linevision.asia/save_var"

    try:
        print(f"[VAR ASYNC] Bắt đầu xử lý VAR cho request {request_id}")

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

        # Tạo URLs với file server URL
        base_url = settings.file_server_url.rstrip("/")
        crop_view_url = f"{base_url}/outputs/{request_id}/{crop_filename}"
        mask_view_url = f"{base_url}/outputs/{request_id}/{mask_filename}"

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
            "status": "completed",
        }

        print(f"[VAR ASYNC] Phân tích VAR hoàn thành cho request {request_id}")

        # Xóa video upload sau khi xử lý xong
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Đã xóa video upload: {video_path}")
        except Exception as cleanup_error:
            print(f"[CLEANUP ERROR] Không thể xóa video: {cleanup_error}")

        # Gọi callback với retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(
                    f"[VAR CALLBACK] Gửi kết quả đến {callback_url} (lần {attempt + 1}/{max_retries})"
                )
                response = requests.post(
                    callback_url,
                    json=result,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )

                if response.status_code == 200:
                    print(f"[VAR CALLBACK] Thành công cho request {request_id}")
                    break
                else:
                    print(
                        f"[VAR CALLBACK] Lỗi HTTP {response.status_code}: {response.text}"
                    )

            except requests.exceptions.RequestException as e:
                print(f"[VAR CALLBACK] Lỗi request (lần {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(f"[VAR CALLBACK] Đợi {wait_time}s trước khi thử lại...")
                time.sleep(wait_time)
        else:
            print(
                f"[VAR CALLBACK] Thất bại sau {max_retries} lần thử cho request {request_id}"
            )

        return result

    except Exception as e:
        print(f"[VAR ASYNC ERROR] Lỗi xử lý VAR request {request_id}: {e}")
        print(traceback.format_exc())

        # Cleanup on error
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except:
            pass
        try:
            if os.path.exists(request_output_folder):
                shutil.rmtree(request_output_folder)
        except:
            pass

        # Gửi thông báo lỗi đến callback
        error_payload = {"request_id": request_id, "status": "failed", "error": str(e)}

        try:
            requests.post(
                callback_url,
                json=error_payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
        except:
            print(f"[VAR CALLBACK ERROR] Không thể gửi thông báo lỗi")

        raise e


@api_bp.route("/api/check_var-async", methods=["POST"])
def check_var_async():
    """
    Endpoint để kiểm tra VAR (Video Assistant Referee) với PRIORITY CAO NHẤT.

    Khi gọi API này, TẤT CẢ các task khác (player-analysis-async, etc.)
    sẽ TẠM DỪNG cho đến khi VAR hoàn thành.

    Parameters (form-data):
        - video: Video file (required)
        - callback_url: URL để gọi callback khi hoàn thành (optional, default: http://linevision.asia/save_var)

    Returns:
        JSON xác nhận đã nhận request và bắt đầu xử lý với priority cao nhất
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

        # Lấy callback URL từ request (optional)
        callback_url = request.form.get(
            "callback_url", "http://linevision.asia/save_var"
        )

        # Submit vào GPU queue với priority VAR (cao nhất)
        # Sử dụng submit_var để đảm bảo các task khác tạm dừng
        gpu_queue.submit_var(
            task_id=request_id,
            func=process_var_async,
            args=(video_path, request_output_folder, request_id, callback_url),
        )

        # Lấy queue status
        queue_status = gpu_queue.get_queue_status()

        return (
            jsonify(
                {
                    "status": "queued",
                    "priority": "VAR (highest)",
                    "message": "VAR request đã được ưu tiên cao nhất. Các task khác sẽ tạm dừng.",
                    "request_id": request_id,
                    "queue_position": 1,  # VAR luôn ở vị trí đầu tiên
                    "queue_status": queue_status,
                    "callback_url": callback_url,
                    "var_status": queue_status.get("var_status", {}),
                }
            ),
            202,
        )

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def process_player_analysis_async(
    video_path,
    court_bounds,
    request_output_folder,
    request_id,
    court_id="",
    original_filename="",
):
    """
    Xử lý phân tích người chơi trong background thread và gọi callback khi hoàn thành
    Sử dụng TennisVideoAnalysisService với 4 điểm court_bounds
    """
    callback_url = "http://linevision.asia/save_json"

    try:
        print(f"[ASYNC] Bắt đầu phân tích player cho request {request_id}")

        # Tạo instance mới của TennisVideoAnalysisService (thread safety)
        service = create_tennis_video_analysis_service()

        # Phân tích video (uses default conf=0.3)
        result = service.analyze(
            video_path=video_path,
            court_bounds=court_bounds,
            output_folder=request_output_folder,
        )

        # Thêm metadata
        result["request_id"] = request_id
        result["file_name"] = original_filename
        result["court_id"] = court_id
        result["timestamp"] = datetime.now().isoformat()
        result["expires_at"] = (
            datetime.now() + timedelta(hours=settings.cleanup_hours)
        ).isoformat()

        # Convert tất cả paths thành full URLs (sử dụng file server URL)
        result = convert_paths_to_urls(result, request_id, settings.file_server_url)

        # Convert numpy types to native Python types for JSON serialization
        result = convert_numpy_types(result)

        print(
            f"[ASYNC] Phân tích hoàn thành cho request {request_id}, file_name: {original_filename}"
        )

        # Xóa video upload sau khi xử lý xong
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Đã xóa video upload: {video_path}")
        except Exception as cleanup_error:
            print(f"[CLEANUP ERROR] Không thể xóa video: {cleanup_error}")

        # Gọi callback với retry logic
        callback_data = {
            "file_name": original_filename,
            "court_id": court_id,
            "data": result,
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(
                    f"[CALLBACK] Gửi kết quả đến {callback_url} (lần {attempt + 1}/{max_retries})"
                )
                response = requests.post(
                    callback_url,
                    json=callback_data,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )

                if response.status_code == 200:
                    print(f"[CALLBACK] Thành công cho request {request_id}")
                    break
                else:
                    print(
                        f"[CALLBACK] Lỗi HTTP {response.status_code}: {response.text}"
                    )

            except requests.exceptions.RequestException as e:
                print(f"[CALLBACK] Lỗi request (lần {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(f"[CALLBACK] Đợi {wait_time}s trước khi thử lại...")
                time.sleep(wait_time)
        else:
            print(
                f"[CALLBACK] Thất bại sau {max_retries} lần thử cho request {request_id}"
            )

    except Exception as e:
        print(f"[ASYNC ERROR] Lỗi xử lý request {request_id}: {e}")
        print(traceback.format_exc())

        # Cleanup on error
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except:
            pass
        try:
            if os.path.exists(request_output_folder):
                shutil.rmtree(request_output_folder)
        except:
            pass

        # Gửi thông báo lỗi đến callback
        error_payload = {
            "file_name": original_filename,
            "court_id": court_id,
            "request_id": request_id,
            "status": "failed",
            "error": str(e),
        }

        try:
            requests.post(
                callback_url,
                json=error_payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
        except:
            print(f"[CALLBACK ERROR] Không thể gửi thông báo lỗi")


@api_bp.route("/api/player-analysis-async", methods=["POST"])
def player_analysis_async():
    """
    Endpoint phân tích người chơi tennis (async với callback)

    Phân tích xong sẽ tự động gọi POST http://linevision.asia/save_json
    với kết quả phân tích

    Parameters (form-data):
        - video: Video file (required)
        - court_bounds: JSON string của 4 điểm góc sân (required)
          Example: "[[100,100],[500,100],[500,600],[100,600]]"
        - court_id: ID của sân tennis (required)

    Returns:
        JSON xác nhận đã nhận request và bắt đầu xử lý
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

        # Validate court_bounds (4 điểm)
        court_bounds_str = request.form.get("court_bounds")
        if not court_bounds_str:
            return (
                jsonify(
                    {
                        "error": "court_bounds is required. Example: [[100,100],[500,100],[500,600],[100,600]]"
                    }
                ),
                400,
            )

        try:
            court_bounds = json.loads(court_bounds_str)
            if not isinstance(court_bounds, list) or len(court_bounds) != 4:
                return (
                    jsonify(
                        {
                            "error": "court_bounds must have exactly 4 points. Example: [[100,100],[500,100],[500,600],[100,600]]"
                        }
                    ),
                    400,
                )
            # Convert to list of tuples
            court_bounds = [tuple(p) for p in court_bounds]
        except json.JSONDecodeError:
            return (
                jsonify(
                    {
                        "error": "Invalid court_bounds JSON format. Example: [[100,100],[500,100],[500,600],[100,600]]"
                    }
                ),
                400,
            )

        # Validate court_id
        court_id = request.form.get("court_id")
        if not court_id:
            return jsonify({"error": "court_id is required"}), 400

        # Save original filename
        original_filename = file.filename

        # Lưu video upload
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        video_path = os.path.join(settings.upload_folder, unique_filename)
        file.save(video_path)

        # Tạo thư mục output riêng cho request này
        request_id = uuid.uuid4().hex
        request_output_folder = os.path.join(settings.output_folder, request_id)
        os.makedirs(request_output_folder, exist_ok=True)

        # Submit vào GPU queue
        gpu_queue.submit(
            task_id=request_id,
            func=process_player_analysis_async,
            args=(
                video_path,
                court_bounds,
                request_output_folder,
                request_id,
                court_id,
                original_filename,
            ),
        )

        # Lấy queue status
        queue_status = gpu_queue.get_queue_status()

        return (
            jsonify(
                {
                    "status": "queued",
                    "message": f"Video added to queue. Position: {queue_status['queue_size']}",
                    "request_id": request_id,
                    "file_name": original_filename,
                    "court_id": court_id,
                    "queue_position": queue_status["queue_size"],
                    "queue_status": queue_status,
                    "callback_url": "http://linevision.asia/save_json",
                }
            ),
            202,
        )

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@api_bp.route("/api/gpu-queue-status", methods=["GET"])
def get_gpu_queue_status():
    """
    Lấy trạng thái của GPU queue

    Returns:
        JSON với thông tin queue: max_concurrent, active_tasks, queue_size, etc.
    """
    status = gpu_queue.get_queue_status()
    return jsonify(status), 200


# =============================================================================
# TENNIS VIDEO ANALYSIS ENDPOINT
# =============================================================================


@api_bp.route("/api/analysis-realtime", methods=["POST"])
def tennis_video_analysis():
    """
    Tennis video analysis endpoint (synchronous)

    Phân tích video tennis với các thông tin:
    1. Người có tốc độ đánh cao nhất (hình ảnh + tốc độ)
    2. Góc mở vai và góc khụy gối trung bình
    3. Bảng xếp hạng người chơi
    4. Thống kê trận đấu (tỉ lệ đối kháng, bóng trong/ngoài sân)

    Parameters (form-data):
        - video: Video file (required, max 5 phút)
        - court_bounds: JSON string của 4 điểm góc sân (required)
          Example: "[[100,100],[500,100],[500,600],[100,600]]"
        - court_id: ID của sân tennis (required)

    Returns:
        JSON với kết quả phân tích:
        {
            "request_id": "uuid",
            "file_name": "string (tên file gốc upload)",
            "court_id": "string",
            "timestamp": "ISO-8601",
            "expires_at": "ISO-8601",
            "highest_speed_player": {
                "player_image_url": "string",
                "speed": float
            },
            "average_stats": {
                "avg_shoulder_angle": float,
                "avg_knee_bend_angle": float
            },
            "player_rankings": [
                {
                    "rank": int,
                    "player_id": int,
                    "score": float,
                    "player_image_url": "string",
                    "in_court_ratio": float,
                    "avg_hit_speed": float,
                    "avg_shoulder_angle": float,
                    "avg_knee_bend_angle": float
                }
            ],
            "match_statistics": {
                "rally_ratio": float,
                "out_court_ratio": float,
                "in_court_ratio": float
            }
        }
    """
    video_path = None
    request_output_folder = None

    try:
        # Validate video file
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

        # Validate court_bounds
        court_bounds_str = request.form.get("court_bounds")
        if not court_bounds_str:
            return (
                jsonify(
                    {
                        "error": "court_bounds is required. Example: [[100,100],[500,100],[500,600],[100,600]]"
                    }
                ),
                400,
            )

        try:
            court_bounds = json.loads(court_bounds_str)
            if not isinstance(court_bounds, list) or len(court_bounds) != 4:
                return (
                    jsonify(
                        {
                            "error": "court_bounds must have exactly 4 points. Example: [[100,100],[500,100],[500,600],[100,600]]"
                        }
                    ),
                    400,
                )
            # Convert to list of tuples
            court_bounds = [tuple(p) for p in court_bounds]
        except json.JSONDecodeError:
            return (
                jsonify(
                    {
                        "error": "Invalid court_bounds JSON format. Example: [[100,100],[500,100],[500,600],[100,600]]"
                    }
                ),
                400,
            )

        # Validate court_id
        court_id = request.form.get("court_id")
        if not court_id:
            return jsonify({"error": "court_id is required"}), 400

        # Save original filename
        original_filename = file.filename

        # Save video temporarily
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        video_path = os.path.join(settings.upload_folder, unique_filename)
        file.save(video_path)

        # Validate video duration (max 5 minutes)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_seconds = total_frames / fps if fps > 0 else 0
        cap.release()

        if duration_seconds > settings.max_video_duration_seconds:
            os.remove(video_path)
            return (
                jsonify(
                    {
                        "error": f"Video exceeds {settings.max_video_duration_seconds // 60} minute limit. Duration: {duration_seconds:.1f}s"
                    }
                ),
                400,
            )

        # Setup output folder
        request_id = uuid.uuid4().hex
        request_output_folder = os.path.join(settings.output_folder, request_id)
        os.makedirs(request_output_folder, exist_ok=True)

        # Create service and analyze (uses default conf=0.3)
        print(f"\n[TENNIS-ANALYSIS] Starting analysis for request: {request_id}")
        print(
            f"[TENNIS-ANALYSIS] Video duration: {duration_seconds:.1f}s, FPS: {fps:.1f}"
        )

        service = create_tennis_video_analysis_service()
        result = service.analyze(
            video_path=video_path,
            court_bounds=court_bounds,
            output_folder=request_output_folder,
        )

        # Add metadata
        result["request_id"] = request_id
        result["file_name"] = original_filename
        result["court_id"] = court_id
        result["timestamp"] = datetime.now().isoformat()
        result["expires_at"] = (
            datetime.now() + timedelta(hours=settings.cleanup_hours)
        ).isoformat()

        # Convert paths to URLs (use file server URL)
        result = convert_paths_to_urls(result, request_id, settings.file_server_url)

        # Convert numpy types to native Python types for JSON serialization
        result = convert_numpy_types(result)

        # Callback to server with analysis results
        callback_data = {
            "file_name": original_filename,
            "court_id": court_id,
            "data": result,
        }
        try:
            callback_response = requests.post(
                "http://linevision.asia/save_json_realtime",
                headers={"Content-Type": "application/json"},
                json=callback_data,
                timeout=30,
            )
            print(
                f"[TENNIS-ANALYSIS] Callback response: {callback_response.status_code}"
            )
        except Exception as callback_error:
            print(f"[TENNIS-ANALYSIS] Callback failed: {callback_error}")

        # Cleanup uploaded video immediately
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            print(f"[TENNIS-ANALYSIS] Cleaned up uploaded video: {video_path}")

        # Schedule output folder cleanup
        schedule_folder_deletion(request_output_folder, settings.cleanup_hours)

        print(f"[TENNIS-ANALYSIS] Analysis completed for request: {request_id}")

        return jsonify(result), 200

    except Exception as e:
        print(f"[TENNIS-ANALYSIS ERROR] {str(e)}")
        print(traceback.format_exc())

        # Cleanup on error
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def schedule_folder_deletion(folder_path, delay_hours=3):
    """
    Lên lịch xóa folder sau một khoảng thời gian nhất định

    Args:
        folder_path: Đường dẫn đến folder cần xóa
        delay_hours: Số giờ chờ trước khi xóa (mặc định: 3)
    """

    def delete_folder():
        try:
            time.sleep(delay_hours * 3600)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"[CLEANUP] Deleted folder after {delay_hours}h: {folder_path}")
            else:
                print(f"[CLEANUP] Folder already deleted: {folder_path}")
        except Exception as e:
            print(f"[CLEANUP ERROR] Failed to delete {folder_path}: {e}")

    deletion_thread = threading.Thread(target=delete_folder, daemon=True)
    deletion_thread.start()
    print(
        f"[CLEANUP] Scheduled folder deletion for {folder_path} in {delay_hours} hours"
    )


@api_bp.route("/api/task-status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """
    Lấy trạng thái của một task cụ thể

    Args:
        task_id: ID của task (request_id)

    Returns:
        JSON với thông tin task
    """
    task = gpu_queue.get_task(task_id)
    if task is None:
        return jsonify({"error": "Task not found"}), 404

    return (
        jsonify(
            {
                "task_id": task.task_id,
                "status": task.status.value,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "error": task.error,
            }
        ),
        200,
    )


def create_api_blueprint():
    """Factory function to create and return the API blueprint"""
    return api_bp
