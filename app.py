# =============================================================================
# FLASK API FOR TENNIS ANALYSIS MODULE
# =============================================================================

from flask import Flask, send_from_directory, jsonify
from src.api.routes import create_api_blueprint
from config.settings import settings
import os
from datetime import datetime, timedelta
import shutil
import threading
import time
import subprocess

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

app.config["UPLOAD_FOLDER"] = settings.upload_folder
app.config["OUTPUT_FOLDER"] = settings.output_folder
app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length

# Register API Blueprint
api_bp = create_api_blueprint()
app.register_blueprint(api_bp)


# =============================================================================
# CLEANUP FUNCTION
# =============================================================================


def cleanup_old_files():
    """
    Xóa các file và folder cũ hơn 10 phút trong OUTPUT_FOLDER và UPLOAD_FOLDER
    Chạy trong background thread mỗi 1 phút
    """
    while True:
        try:
            now = datetime.now()
            cutoff_time = now - timedelta(hours=settings.cleanup_hours)

            # Kiểm tra từng folder trong OUTPUT_FOLDER
            if os.path.exists(settings.output_folder):
                for folder_name in os.listdir(settings.output_folder):
                    folder_path = os.path.join(settings.output_folder, folder_name)

                    if os.path.isdir(folder_path):
                        # Lấy thời gian tạo folder
                        folder_mtime = datetime.fromtimestamp(
                            os.path.getmtime(folder_path)
                        )

                        # Nếu folder cũ hơn cleanup_hours, xóa nó
                        if folder_mtime < cutoff_time:
                            shutil.rmtree(folder_path)
                            print(
                                f"[CLEANUP] Deleted old folder: {folder_name} (created at {folder_mtime})"
                            )

            # Xóa video upload cũ trong UPLOAD_FOLDER
            if os.path.exists(settings.upload_folder):
                for file_name in os.listdir(settings.upload_folder):
                    file_path = os.path.join(settings.upload_folder, file_name)

                    if os.path.isfile(file_path):
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

                        if file_mtime < cutoff_time:
                            os.remove(file_path)
                            print(
                                f"[CLEANUP] Deleted old upload: {file_name} (created at {file_mtime})"
                            )

        except Exception as e:
            print(f"[CLEANUP ERROR] {e}")

        # Chạy cleanup mỗi 1 phút để kịp thời xóa file cũ
        time.sleep(60)


def clear_system_trash():
    """
    Xóa thùng rác hệ thống (macOS) mỗi 10 phút
    Chạy trong background thread riêng biệt
    """
    # Đường dẫn thùng rác macOS
    trash_path = os.path.expanduser("~/.Trash/*")

    while True:
        try:
            # Xóa tất cả file trong thùng rác macOS
            result = subprocess.run(
                f"rm -rf {trash_path}", shell=True, capture_output=True, text=True
            )

            if result.returncode == 0:
                print(f"[TRASH CLEANUP] Đã dọn sạch thùng rác hệ thống macOS")
            else:
                if result.stderr:
                    print(f"[TRASH CLEANUP ERROR] {result.stderr}")

        except Exception as e:
            print(f"[TRASH CLEANUP ERROR] {e}")

        # Chạy cleanup thùng rác mỗi 10 phút
        print(f"[TRASH CLEANUP] Chờ 10 phút để dọn lại...")
        time.sleep(600)  # 600 giây = 10 phút


# =============================================================================
# FILE SERVING ROUTE
# =============================================================================


@app.route("/files/<folder>/<filename>")
def serve_file(folder, filename):
    """
    Serve static files (images and videos)
    Query parameter: download=true to force download
    """
    try:
        from flask import request

        file_path = os.path.join(app.config["OUTPUT_FOLDER"], folder)
        # Check if download parameter is set
        download = request.args.get("download", "false").lower() == "true"
        return send_from_directory(file_path, filename, as_attachment=download)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


# =============================================================================
# HOME PAGE
# =============================================================================


@app.route("/")
def index():
    """
    API documentation page
    """
    docs = """
    <h1>Tennis Analysis API</h1>
    <h2>Endpoints:</h2>
    <ul>
        <li><b>GET /api/health</b> - Health check</li>
        <li><b>POST /api/analyze</b> - Analyze tennis video
            <ul>
                <li>Parameters (form-data):
                    <ul>
                        <li>video (file, required): Video file</li>
                        <li>ball_conf (float, optional): Ball detection confidence (default: 0.7)</li>
                        <li>person_conf (float, optional): Person detection confidence (default: 0.6)</li>
                        <li>angle_threshold (float, optional): Angle threshold (default: 50)</li>
                        <li>intersection_threshold (float, optional): Intersection threshold (default: 100)</li>
                        <li>court_bounds (string, optional): Court bounds as "x1,y1,x2,y2" (default: "100,100,400,500")</li>
                    </ul>
                </li>
            </ul>
        </li>
        <li><b>POST /api/check_var</b> - VAR (Video Assistant Referee) analysis for football videos
            <ul>
                <li>Parameters (form-data):
                    <ul>
                        <li>video (file, required): Video file</li>
                    </ul>
                </li>
                <li>Returns: JSON with URLs to processed videos (crop, mask) and original video path</li>
            </ul>
        </li>
        <li><b>GET /files/&lt;folder&gt;/&lt;filename&gt;</b> - Serve output files</li>
        <li><b>GET /api/results/&lt;request_id&gt;</b> - Get all files for a request</li>
    </ul>
    <h3>Response Format:</h3>
    <p>API trả về trực tiếp JSON result, không có wrapper {success: true, data: ...}</p>
    <h3>⚠️ Important Notes:</h3>
    <ul>
        <li><b>Auto Cleanup:</b> Files (images and videos) are automatically deleted after 10 minutes to save disk space</li>
        <li><b>Download Files:</b> Make sure to download important results within 10 minutes</li>
        <li><b>Cleanup Schedule:</b> Cleanup runs every 1 minute in the background</li>
        <li><b>Expiration Time:</b> Each response includes an 'expires_at' field showing when files will be deleted</li>
    </ul>
    """
    return docs


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    # Khởi động cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()
    print(f"[CLEANUP] Background cleanup thread started (runs every 1 minute)")
    print(f"[CLEANUP] Files older than 10 minutes will be automatically deleted")

    # Khởi động trash cleanup thread
    trash_cleanup_thread = threading.Thread(target=clear_system_trash, daemon=True)
    trash_cleanup_thread.start()
    print(
        f"[TRASH CLEANUP] System trash cleanup thread started (runs every 10 minutes)"
    )
    print(f"[TRASH CLEANUP] macOS Trash will be cleared every 10 minutes")

    # Tăng timeout để xử lý video lớn
    from werkzeug.serving import WSGIRequestHandler

    WSGIRequestHandler.protocol_version = "HTTP/1.1"

    print(f"🚀 Starting Tennis Analysis API on {settings.api_host}:{settings.api_port}")
    print(f"📁 Upload folder: {settings.upload_folder}")
    print(f"📁 Output folder: {settings.output_folder}")
    print(f"🤖 Models loaded from: {settings.model_dir}")

    app.run(
        host=settings.api_host,
        port=settings.api_port,
        debug=settings.debug,
        threaded=True,  # Enable threading để xử lý nhiều request
    )
