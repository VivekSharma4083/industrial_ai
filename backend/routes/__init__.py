"""
backend/routes/__init__.py
==========================
Route blueprints package.
"""

from backend.routes.health import health_bp

__all__ = ["health_bp"]
