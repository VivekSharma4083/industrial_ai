"""
tests/test_api_inspection.py
=============================
Integration and unit tests for the inspection REST API, model service,
multi-model inference, threshold calibration modes, and history caching.
"""

import io
import json
import numpy as np
from PIL import Image
import pytest

from backend.app import create_app
from backend.services.model_service import ModelService


@pytest.fixture(scope="module")
def app():
    """Create Flask application fixture configured for testing."""
    test_app = create_app({"TESTING": True})
    return test_app


@pytest.fixture(scope="module")
def client(app):
    """Test client fixture."""
    return app.test_client()


def _create_dummy_image_bytes(size=(224, 224), color=(128, 128, 128)) -> bytes:
    """Generate in-memory PNG image bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_models_endpoint(client):
    """Verify GET /api/models returns model readiness and threshold metadata."""
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.get_json()

    assert "status" in data
    assert data["target_category"] == "bottle"
    assert "models" in data

    # Verify Feature Reconstruction metadata
    assert "reconstruction" in data["models"]
    recon = data["models"]["reconstruction"]
    assert recon["name"] == "Feature Reconstruction Autoencoder"
    assert recon["loaded"] is True
    assert "thresholds" in recon
    assert recon["thresholds"]["unbiased"] == pytest.approx(23.7207, abs=1e-3)
    assert recon["thresholds"]["optimal"] == pytest.approx(23.6698, abs=1e-3)

    # Verify PatchCore metadata
    assert "patchcore" in data["models"]
    patch = data["models"]["patchcore"]
    assert patch["name"] == "PatchCore Baseline"
    assert patch["loaded"] is True
    assert "thresholds" in patch
    assert patch["thresholds"]["unbiased"] == pytest.approx(10.0016, abs=1e-3)
    assert patch["thresholds"]["optimal"] == pytest.approx(13.1616, abs=1e-3)


def test_eval_summary_endpoint(client):
    """Verify GET /api/eval-summary returns saved empirical benchmark results."""
    res = client.get("/api/eval-summary")
    assert res.status_code == 200
    data = res.get_json()

    assert data["category"] == "bottle"
    assert data["test_samples"] == 83
    assert "comparison" in data

    comp = data["comparison"]
    assert comp["image_roc_auc"]["reconstruction"] == pytest.approx(0.9968, abs=1e-3)
    assert comp["image_roc_auc"]["patchcore"] == pytest.approx(1.0000, abs=1e-3)
    assert comp["pixel_roc_auc"]["reconstruction"] == pytest.approx(0.9641, abs=1e-3)
    assert comp["pixel_roc_auc"]["patchcore"] == pytest.approx(0.9797, abs=1e-3)
    assert comp["false_negatives"]["reconstruction"] == 3
    assert comp["false_negatives"]["patchcore"] == 0


def test_samples_endpoint(client):
    """Verify GET /api/samples returns sample presets."""
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.get_json()

    assert "samples" in data
    assert len(data["samples"]) == 4

    categories = [s["category"] for s in data["samples"]]
    assert "good" in categories
    assert "broken_large" in categories
    assert "broken_small" in categories
    assert "contamination" in categories


def test_inspect_missing_image_payload(client):
    """Verify POST /api/inspect returns 400 when no image is uploaded."""
    res = client.post("/api/inspect", data={})
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_inspect_corrupt_file_payload(client):
    """Verify POST /api/inspect rejects corrupted non-image files."""
    corrupt_data = {
        "image": (io.BytesIO(b"This is not a valid image content"), "corrupt.png"),
        "model": "both",
    }
    res = client.post("/api/inspect", data=corrupt_data, content_type="multipart/form-data")
    assert res.status_code == 400
    data = res.get_json()
    assert "Invalid or corrupted image format" in data["error"]


def test_inspect_with_sample_preset(client):
    """Verify POST /api/inspect executes inference on a preset sample."""
    payload = {
        "sample_id": "sample_good_000",
        "model": "both",
        "threshold_mode": "unbiased",
    }
    res = client.post("/api/inspect", data=payload)
    assert res.status_code == 200
    data = res.get_json()

    assert "inspection_id" in data
    assert "preview_image" in data
    assert data["preview_image"].startswith("data:image/png;base64,")

    # Both models should be evaluated
    assert "reconstruction" in data
    assert "patchcore" in data
    assert "summary" in data

    recon = data["reconstruction"]
    assert "score" in recon
    assert recon["threshold"] == pytest.approx(23.7207, abs=1e-3)
    assert "is_anomaly" in recon
    assert recon["heatmap_image"].startswith("data:image/png;base64,")
    assert recon["overlay_image"].startswith("data:image/png;base64,")

    patch = data["patchcore"]
    assert "score" in patch
    assert patch["threshold"] == pytest.approx(10.0016, abs=1e-3)
    assert "is_anomaly" in patch
    assert patch["heatmap_image"].startswith("data:image/png;base64,")
    assert patch["overlay_image"].startswith("data:image/png;base64,")

    summary = data["summary"]
    assert summary["agreement"] in ("CONCORDANT_NORMAL", "CONCORDANT_DEFECT", "DISCORDANT")
    assert summary["total_latency_ms"] > 0


def test_inspect_single_model_reconstruction(client):
    """Verify POST /api/inspect runs only reconstruction when selected."""
    img_bytes = _create_dummy_image_bytes()
    data_payload = {
        "image": (io.BytesIO(img_bytes), "test_dummy.png"),
        "model": "reconstruction",
        "threshold_mode": "unbiased",
    }
    res = client.post("/api/inspect", data=data_payload, content_type="multipart/form-data")
    assert res.status_code == 200
    data = res.get_json()

    assert "reconstruction" in data
    assert "patchcore" not in data
    assert data["model_choice"] == "reconstruction"


def test_inspect_single_model_patchcore(client):
    """Verify POST /api/inspect runs only PatchCore when selected."""
    img_bytes = _create_dummy_image_bytes()
    data_payload = {
        "image": (io.BytesIO(img_bytes), "test_dummy.png"),
        "model": "patchcore",
        "threshold_mode": "unbiased",
    }
    res = client.post("/api/inspect", data=data_payload, content_type="multipart/form-data")
    assert res.status_code == 200
    data = res.get_json()

    assert "patchcore" in data
    assert "reconstruction" not in data
    assert data["model_choice"] == "patchcore"


def test_inspect_threshold_modes(client):
    """Verify optimal and custom threshold mode overrides."""
    img_bytes = _create_dummy_image_bytes()

    # 1. Optimal threshold mode
    res_opt = client.post(
        "/api/inspect",
        data={
            "image": (io.BytesIO(img_bytes), "test_opt.png"),
            "model": "both",
            "threshold_mode": "optimal",
        },
        content_type="multipart/form-data",
    )
    assert res_opt.status_code == 200
    data_opt = res_opt.get_json()
    assert data_opt["reconstruction"]["threshold"] == pytest.approx(23.6698, abs=1e-3)
    assert data_opt["patchcore"]["threshold"] == pytest.approx(13.1616, abs=1e-3)

    # 2. Custom threshold mode
    res_cust = client.post(
        "/api/inspect",
        data={
            "image": (io.BytesIO(img_bytes), "test_cust.png"),
            "model": "both",
            "threshold_mode": "custom",
            "custom_threshold_recon": "25.0",
            "custom_threshold_patch": "15.0",
        },
        content_type="multipart/form-data",
    )
    assert res_cust.status_code == 200
    data_cust = res_cust.get_json()
    assert data_cust["reconstruction"]["threshold"] == 25.0
    assert data_cust["patchcore"]["threshold"] == 15.0


def test_history_workflow(client):
    """Verify inspection history storage, retrieval, item detail, and clearing."""
    # 1. Inspect an image to guarantee at least one history entry
    img_bytes = _create_dummy_image_bytes()
    inspect_res = client.post(
        "/api/inspect",
        data={
            "image": (io.BytesIO(img_bytes), "history_check.png"),
            "model": "both",
            "threshold_mode": "unbiased",
        },
        content_type="multipart/form-data",
    )
    assert inspect_res.status_code == 200
    item_id = inspect_res.get_json()["inspection_id"]

    # 2. Retrieve history list
    res_hist = client.get("/api/history")
    assert res_hist.status_code == 200
    hist_data = res_hist.get_json()
    assert hist_data["count"] >= 1
    found = any(h["id"] == item_id for h in hist_data["history"])
    assert found is True

    # 3. Retrieve history detail
    res_detail = client.get(f"/api/history/{item_id}")
    assert res_detail.status_code == 200
    detail_data = res_detail.get_json()
    assert detail_data["inspection_id"] == item_id
    assert "preview_image" in detail_data

    # 4. Clear history
    res_clear = client.delete("/api/history")
    assert res_clear.status_code == 200

    # 5. Verify history is empty
    res_empty = client.get("/api/history")
    assert res_empty.status_code == 200
    assert res_empty.get_json()["count"] == 0
