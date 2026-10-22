"""
tests/test_health.py
====================
Unit and integration tests for Flask health endpoints, static asset resolution,
and core environment imports.
"""

import pytest
from backend.app import create_app


@pytest.fixture
def client():
    """Create a test client configured for testing."""
    app = create_app({"TESTING": True})
    with app.test_client() as test_client:
        yield test_client


def test_core_imports():
    """Verify that core Phase 1 dependencies are importable and functional."""
    import flask
    import PIL
    import pytest as pt

    assert flask is not None
    assert PIL is not None
    assert pt is not None


def test_health_endpoint(client):
    """
    Test that GET /api/health returns 200 OK with expected JSON structure.
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.is_json

    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "Industrial AI Inspector API"
    assert data["version"] == "0.1.0"
    assert "timestamp" in data
    assert "environment" in data
    assert "python" in data["environment"]
    assert "platform" in data["environment"]


def test_index_route(client):
    """
    Test that GET / serves index.html with appropriate title and status 200.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert b"INDUSTRIAL AI INSPECTOR" in response.data
    assert b"text/html" in response.content_type.encode()


def test_static_asset_serving(client):
    """
    Test that CSS and JS static assets are served from the frontend directory.
    """
    css_res = client.get("/css/styles.css")
    assert css_res.status_code == 200
    assert b"--bg-primary" in css_res.data

    js_res = client.get("/js/app.js")
    assert js_res.status_code == 200
    assert b"checkHealth" in js_res.data


def test_not_found_handling(client):
    """
    Test that requesting non-existent files or endpoints returns 404 JSON.
    """
    response = client.get("/api/non_existent_endpoint")
    assert response.status_code == 404
    assert response.is_json
