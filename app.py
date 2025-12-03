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
    Xóa các file và folder cũ hơn 24 giờ trong OUTPUT_FOLDER và UPLOAD_FOLDER
    Chạy trong background thread
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

        # Chạy cleanup mỗi 1 giờ
        time.sleep(3600)


# =============================================================================
# FILE SERVING ROUTE
# =============================================================================


@app.route("/files/<folder>/<filename>")
def serve_file(folder, filename):
    """
    Serve static files (images and videos)
    """
    try:
        file_path = os.path.join(app.config["OUTPUT_FOLDER"], folder)
        return send_from_directory(file_path, filename)
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
        <li><b>GET /files/&lt;folder&gt;/&lt;filename&gt;</b> - Serve output files</li>
        <li><b>GET /api/results/&lt;request_id&gt;</b> - Get all files for a request</li>
    </ul>
    <h3>Response Format:</h3>
    <p>API trả về trực tiếp JSON result, không có wrapper {success: true, data: ...}</p>
    <h3>⚠️ Important Notes:</h3>
    <ul>
        <li><b>Auto Cleanup:</b> Files (images and videos) are automatically deleted after 24 hours to save disk space</li>
        <li><b>Download Files:</b> Make sure to download important results within 24 hours</li>
        <li><b>Cleanup Schedule:</b> Cleanup runs every 1 hour in the background</li>
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
    print(f"[CLEANUP] Background cleanup thread started (runs every 1 hour)")
    print(
        f"[CLEANUP] Files older than {settings.cleanup_hours} hours will be automatically deleted"
    )

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
