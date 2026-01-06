# =============================================================================
# FILE SERVER API - DEDICATED SERVER FOR SERVING FILES
# =============================================================================
# Separate server for serving analysis output files (videos, images)
# This reduces load on the main analysis API
# =============================================================================

from flask import Flask, send_from_directory, jsonify, request
from config.settings import settings
import os

app = Flask(__name__)


# Simple CORS support without flask_cors dependency
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Configuration
app.config["OUTPUT_FOLDER"] = settings.output_folder
app.config["UPLOAD_FOLDER"] = settings.upload_folder

# File server settings
FILE_SERVER_HOST = os.getenv("FILE_SERVER_HOST", "0.0.0.0")
FILE_SERVER_PORT = int(os.getenv("FILE_SERVER_PORT", "2804"))


# =============================================================================
# FILE SERVING ROUTES
# =============================================================================


@app.route("/")
def index():
    """API documentation page"""
    docs = """
    <h1>LineVision File Server</h1>
    <h2>Dedicated server for serving analysis output files</h2>
    <h3>Endpoints:</h3>
    <ul>
        <li><b>GET /health</b> - Health check</li>
        <li><b>GET /outputs/&lt;folder&gt;/&lt;filename&gt;</b> - Serve output files (videos, images, heatmaps)</li>
        <li><b>GET /files/&lt;folder&gt;/&lt;filename&gt;</b> - Alternative route for output files</li>
        <li><b>GET /list/&lt;folder&gt;</b> - List all files in a folder</li>
    </ul>
    <h3>Query Parameters:</h3>
    <ul>
        <li><b>download=true</b> - Force download instead of inline viewing</li>
    </ul>
    <h3>Examples:</h3>
    <ul>
        <li>View image: <code>/outputs/abc123/player_1_heatmap.jpg</code></li>
        <li>Download video: <code>/outputs/abc123/player_1_highlight_1.mp4?download=true</code></li>
        <li>List files: <code>/list/abc123</code></li>
    </ul>
    """
    return docs


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "LineVision File Server",
        "output_folder": settings.output_folder,
        "output_folder_exists": os.path.exists(settings.output_folder)
    })


@app.route("/outputs/<folder>/<filename>")
def serve_output_file(folder, filename):
    """
    Serve files from outputs folder
    URL: /outputs/{request_id}/{filename}
    Query parameter: download=true to force download
    """
    try:
        file_path = os.path.join(app.config["OUTPUT_FOLDER"], folder)
        full_file_path = os.path.join(file_path, filename)

        if not os.path.exists(full_file_path):
            return jsonify({
                "error": "File not found",
                "requested_file": filename,
                "directory": folder,
                "directory_exists": os.path.exists(file_path)
            }), 404

        # Check if download parameter is set
        download = request.args.get("download", "false").lower() == "true"

        # Set appropriate content type for common file types
        return send_from_directory(file_path, filename, as_attachment=download)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/files/<folder>/<filename>")
def serve_file(folder, filename):
    """
    Alternative route for serving files
    URL: /files/{request_id}/{filename}
    """
    return serve_output_file(folder, filename)


@app.route("/list/<folder>")
def list_files(folder):
    """
    List all files in a folder
    URL: /list/{request_id}
    """
    try:
        folder_path = os.path.join(app.config["OUTPUT_FOLDER"], folder)

        if not os.path.exists(folder_path):
            return jsonify({
                "error": "Folder not found",
                "folder": folder
            }), 404

        files = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                files.append({
                    "filename": filename,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "url": f"/outputs/{folder}/{filename}"
                })

        return jsonify({
            "folder": folder,
            "total_files": len(files),
            "files": files
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/<folder>/<filename>")
def download_file(folder, filename):
    """
    Force download a file
    URL: /download/{request_id}/{filename}
    """
    try:
        file_path = os.path.join(app.config["OUTPUT_FOLDER"], folder)
        full_file_path = os.path.join(file_path, filename)

        if not os.path.exists(full_file_path):
            return jsonify({
                "error": "File not found",
                "requested_file": filename
            }), 404

        return send_from_directory(file_path, filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("       LINEVISION FILE SERVER - STARTING")
    print("=" * 60)
    print(f"📁 Output folder: {settings.output_folder}")
    print(f"🌐 Server URL: http://{FILE_SERVER_HOST}:{FILE_SERVER_PORT}")
    print(f"🔗 Public URL: https://download-linevision.ngrok.app")
    print("=" * 60)

    app.run(
        host=FILE_SERVER_HOST,
        port=FILE_SERVER_PORT,
        debug=False,
        threaded=True
    )
