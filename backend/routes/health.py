"""
backend/routes/health.py
========================
Health check and diagnostic route endpoints.
"""

from datetime import datetime, timezone
import platform
import sys
from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint returning service status and environment metadata.
    Responds to GET /api/health
    """
    return (
        jsonify(
            {
                "status": "healthy",
                "service": "Industrial AI Inspector API",
                "version": "0.1.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "environment": {
                    "python": sys.version.split()[0],
                    "platform": platform.system(),
                    "phase": "Phase 1: Project Setup & Environment Verification",
                },
            }
        ),
        200,
    )
