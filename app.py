# =============================================================================
# FLASK API FOR TENNIS ANALYSIS MODULE
# =============================================================================

from flask import Flask, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from tennis_analysis_module import TennisAnalysisModule
import cv2
import os
import uuid
from datetime import datetime, timedelta
import traceback
import shutil
import threading
import time

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Cấu hình upload
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max file size

# Tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Khởi tạo Tennis Analysis Module
analyzer = TennisAnalysisModule(
    ball_model_path="ball_best.pt",
    person_model_path="yolov8m.pt",
    pose_model_path="yolov8n-pose.pt",
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def allowed_file(filename):
    """Kiểm tra file extension có hợp lệ không"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_cropped_image(image, output_folder, prefix, identifier):
    """Lưu ảnh crop và trả về đường dẫn"""
    if image is None:
        return None

    filename = f"{prefix}_{identifier}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(output_folder, filename)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "service": "Tennis Analysis API",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/analyze", methods=["POST"])
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
                jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}),
                400,
            )

        # Lưu video upload
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(video_path)

        # Tạo thư mục output riêng cho request này
        request_id = uuid.uuid4().hex
        request_output_folder = os.path.join(app.config["OUTPUT_FOLDER"], request_id)
        os.makedirs(request_output_folder, exist_ok=True)

        # Lấy parameters từ request
        ball_conf = float(request.form.get("ball_conf", 0.7))
        person_conf = float(request.form.get("person_conf", 0.6))
        angle_threshold = float(request.form.get("angle_threshold", 50))
        intersection_threshold = float(request.form.get("intersection_threshold", 100))

        # Parse court bounds
        court_bounds_str = request.form.get("court_bounds", "100,100,400,500")
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

        # Xóa video upload để tiết kiệm dung lượng (tùy chọn)
        # os.remove(video_path)

        # Trả về trực tiếp result JSON
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


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


@app.route("/api/results/<request_id>", methods=["GET"])
def get_results(request_id):
    """
    Lấy danh sách tất cả files của một request
    """
    try:
        request_folder = os.path.join(app.config["OUTPUT_FOLDER"], request_id)

        if not os.path.exists(request_folder):
            return jsonify({"error": "Request ID not found"}), 404

        files = os.listdir(request_folder)
        file_urls = {
            filename: generate_file_url(filename, request_id) for filename in files
        }

        return jsonify({"request_id": request_id, "files": file_urls}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    """
    return docs

```
        if not allowed_file(file.filename):
            return (
                jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}),
                400,
            )

        # Lưu video upload
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(video_path)

        # Tạo thư mục output riêng cho request này
        request_id = uuid.uuid4().hex
        request_output_folder = os.path.join(app.config["OUTPUT_FOLDER"], request_id)
        os.makedirs(request_output_folder, exist_ok=True)

        # Lấy parameters từ request
        ball_conf = float(request.form.get("ball_conf", 0.7))
        person_conf = float(request.form.get("person_conf", 0.6))
        angle_threshold = float(request.form.get("angle_threshold", 50))
        intersection_threshold = float(request.form.get("intersection_threshold", 100))

        # Parse court bounds
        court_bounds_str = request.form.get("court_bounds", "100,100,400,500")
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
                "avg_knee_bend_angle": round(player["knee_bend_angle"], 2),
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

        # Xóa video upload để tiết kiệm dung lượng (tùy chọn)
        # os.remove(video_path)

        # Trả về trực tiếp result JSON
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


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


@app.route("/api/results/<request_id>", methods=["GET"])
def get_results(request_id):
    """
    Lấy danh sách tất cả files của một request
    """
    try:
        request_folder = os.path.join(app.config["OUTPUT_FOLDER"], request_id)

        if not os.path.exists(request_folder):
            return jsonify({"error": "Request ID not found"}), 404

        files = os.listdir(request_folder)
        file_urls = {
            filename: generate_file_url(filename, request_id) for filename in files
        }

        return jsonify({"request_id": request_id, "files": file_urls}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    """
    return docs


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == '__main__':
    # Khởi động cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()
    print("[CLEANUP] Background cleanup thread started (runs every 1 hour)")
    
    # Tăng timeout để xử lý video lớn
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True  # Enable threading để xử lý nhiều request
    )
```
