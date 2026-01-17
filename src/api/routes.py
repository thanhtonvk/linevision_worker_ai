# =============================================================================
# API ROUTES FOR VAR (Video Assistant Referee)
# =============================================================================

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from src.core.var_detector import VarDetector
from src.core.gpu_queue_manager import gpu_queue
from config.settings import settings
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

# Initialize VAR Detector (uses default conf=0.3)
var_detector = VarDetector(model_path=settings.ball_model_path, batch_size=32)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def allowed_file(filename):
    """Kiểm tra file extension có hợp lệ không"""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in settings.allowed_extensions
    )


def schedule_folder_deletion(folder_path, delay_hours=3):
    """
    Lên lịch xóa folder sau một khoảng thời gian nhất định
    """
    def delete_folder():
        try:
            time.sleep(delay_hours * 3600)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"[CLEANUP] Deleted folder after {delay_hours}h: {folder_path}")
        except Exception as e:
            print(f"[CLEANUP ERROR] Failed to delete {folder_path}: {e}")

    deletion_thread = threading.Thread(target=delete_folder, daemon=True)
    deletion_thread.start()


# =============================================================================
# API ENDPOINTS
# =============================================================================


@api_bp.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "VAR Analysis API",
        "timestamp": datetime.now().isoformat(),
    })


@api_bp.route("/api/gpu-queue-status", methods=["GET"])
def get_gpu_queue_status():
    """Lấy trạng thái của GPU queue"""
    status = gpu_queue.get_queue_status()
    return jsonify(status), 200


@api_bp.route("/api/task-status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """Lấy trạng thái của một task cụ thể"""
    task = gpu_queue.get_task(task_id)
    if task is None:
        return jsonify({"error": "Task not found"}), 404

    return jsonify({
        "task_id": task.task_id,
        "status": task.status.value,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "error": task.error,
    }), 200


@api_bp.route("/api/results/<request_id>", methods=["GET"])
def get_results(request_id):
    """Lấy danh sách tất cả files của một request"""
    try:
        request_folder = os.path.join(settings.output_folder, request_id)

        if not os.path.exists(request_folder):
            return jsonify({"error": "Request ID not found"}), 404

        files = os.listdir(request_folder)
        base_url = settings.file_server_url.rstrip("/")
        file_urls = {
            filename: f"{base_url}/outputs/{request_id}/{filename}"
            for filename in files
        }

        return jsonify({"request_id": request_id, "files": file_urls}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# VAR ASYNC PROCESSING
# =============================================================================


def process_var_async(video_path, request_output_folder, request_id, callback_url=None):
    """
    Xử lý VAR trong background với priority cao nhất.
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
            os.remove(results["crop"])

        if os.path.exists(results["mask"]):
            shutil.copy2(results["mask"], mask_output_path)
            os.remove(results["mask"])

        # Tạo URLs với file server URL
        base_url = settings.file_server_url.rstrip("/")
        crop_view_url = f"{base_url}/outputs/{request_id}/{crop_filename}"
        mask_view_url = f"{base_url}/outputs/{request_id}/{mask_filename}"

        # Tạo download URLs
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

        # Schedule output folder cleanup
        schedule_folder_deletion(request_output_folder, settings.cleanup_hours)

        # Gọi callback với retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[VAR CALLBACK] Gửi kết quả đến {callback_url} (lần {attempt + 1}/{max_retries})")
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
                    print(f"[VAR CALLBACK] Lỗi HTTP {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                print(f"[VAR CALLBACK] Lỗi request (lần {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"[VAR CALLBACK] Đợi {wait_time}s trước khi thử lại...")
                time.sleep(wait_time)
        else:
            print(f"[VAR CALLBACK] Thất bại sau {max_retries} lần thử cho request {request_id}")

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
            return jsonify({
                "error": f"Invalid file type. Allowed: {settings.allowed_extensions}"
            }), 400

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
        callback_url = request.form.get("callback_url", "http://linevision.asia/save_var")

        # Submit vào GPU queue với priority VAR (cao nhất)
        gpu_queue.submit_var(
            task_id=request_id,
            func=process_var_async,
            args=(video_path, request_output_folder, request_id, callback_url),
        )

        # Lấy queue status
        queue_status = gpu_queue.get_queue_status()

        return jsonify({
            "status": "queued",
            "priority": "VAR (highest)",
            "message": "VAR request đã được ưu tiên cao nhất.",
            "request_id": request_id,
            "queue_position": 1,
            "queue_status": queue_status,
            "callback_url": callback_url,
            "var_status": queue_status.get("var_status", {}),
        }), 202

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def create_api_blueprint():
    """Factory function to create and return the API blueprint"""
    return api_bp
