"""
Flask server for serving video files with Range request support.
"""
import os
import threading
from flask import Flask, send_file
from pathlib import Path
from utils.logger import get_logger
from utils.session_state import SessionStateManager

logger = get_logger('server')


def make_video_app(root_dir: str) -> Flask:
    """
    Create a Flask app to serve video files with Range requests support.

    Args:
        root_dir: Root directory containing video files

    Returns:
        Configured Flask application
    """
    app = Flask(__name__)

    @app.after_request
    def add_cors(resp):
        """Add CORS headers to all responses."""
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Range, Origin, Content-Type, Accept"
        resp.headers["Access-Control-Expose-Headers"] = "Content-Range, Accept-Ranges, Content-Length"
        resp.headers["Accept-Ranges"] = "bytes"
        return resp

    @app.route("/video/<path:filename>")
    def serve_video(filename: str):
        """
        Serve video file with Range request support.

        Args:
            filename: Video filename

        Returns:
            Video file response
        """
        filepath = os.path.join(root_dir, filename)
        if not os.path.isfile(filepath):
            logger.warning(f"Video file not found: {filepath}")
            return "Not found", 404

        logger.debug(f"Serving video: {filename}")
        return send_file(filepath, mimetype="video/mp4", as_attachment=False, conditional=True)

    logger.info(f"Created Flask video app with root: {root_dir}")
    return app


def ensure_server_once(app: Flask, host: str = "127.0.0.1", port: int = 8801) -> None:
    """
    Ensure the Flask server is started only once in a background thread.

    Args:
        app: Flask application to run
        host: Host address to bind to
        port: Port number to bind to
    """
    if not SessionStateManager.is_server_started():
        def run_server():
            app.run(host=host, port=port, debug=False, use_reloader=False)

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        SessionStateManager.set_server_started(True)
        logger.info(f"Local video server started on http://{host}:{port}")
    else:
        logger.debug("Server already started, skipping initialization")
