"""
backend/app.py
==============
Entrypoint and application factory for the Industrial AI Inspector Flask server.
Serves the Vanilla HTML/CSS/JS frontend and REST API endpoints from the same origin.
"""

import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct execution (python backend/app.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, send_from_directory, jsonify
from backend.routes.health import health_bp
from backend.routes.inspect import inspect_bp


def create_app(config: dict | None = None) -> Flask:
    """
    Application factory for the Flask backend.
    
    Configures static file resolution pointing directly to the frontend/
    directory, registers API blueprints, and ensures clean error responses.
    """
    project_root = Path(__file__).resolve().parent.parent
    frontend_dir = project_root / "frontend"

    app = Flask(
        __name__,
        static_folder=str(frontend_dir),
        static_url_path="",
    )

    # 10 MB upload ceiling
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    if config:
        app.config.update(config)

    # Register API blueprints
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(inspect_bp, url_prefix="/api")

    # Serve the single-page industrial interface
    @app.route("/")
    def serve_index():
        return send_from_directory(str(frontend_dir), "index.html")

    # Serve static assets or fallback with explicit 404
    @app.route("/<path:filename>")
    def serve_frontend_assets(filename: str):
        target = frontend_dir / filename
        if target.is_file():
            return send_from_directory(str(frontend_dir), filename)
        return jsonify({"error": f"File '{filename}' not found"}), 404

    @app.errorhandler(404)
    def handle_not_found(error):
        return jsonify({"error": "Endpoint or resource not found", "status_code": 404}), 404

    @app.errorhandler(413)
    def handle_large_file(error):
        return jsonify({"error": "File size exceeds the 10 MB maximum limit.", "status_code": 413}), 413

    @app.errorhandler(500)
    def handle_server_error(error):
        return jsonify({"error": "Internal server error", "status_code": 500}), 500

    return app


# Expose application instance for WSGI production servers (e.g., Gunicorn on Render)
app = create_app()


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "yes")

    print(f"[*] Starting Industrial AI Inspector server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
