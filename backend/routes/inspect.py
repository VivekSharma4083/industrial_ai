"""
backend/routes/inspect.py
=========================
Flask Blueprint for anomaly inspection, model diagnostics, evaluation benchmarks,
and inspection history.
"""

import io
import logging
from flask import Blueprint, Response, jsonify, request

from backend.services.model_service import ModelService

logger = logging.getLogger(__name__)
inspect_bp = Blueprint("inspect", __name__)


@inspect_bp.route("/models", methods=["GET"])
def get_models():
    """
    Return model readiness, loaded backbones, parameter counts, and default thresholds.
    Responds to GET /api/models
    """
    service = ModelService.get_instance()
    status = service.get_model_status()
    return jsonify(status), 200


@inspect_bp.route("/inspect", methods=["POST"])
def inspect_image():
    """
    Execute visual anomaly inspection on an uploaded image or sample preset.
    Responds to POST /api/inspect

    Accepts multipart/form-data:
      - image: File upload (PNG, JPEG, BMP)
      - sample_id: (Optional) preset sample identifier if not uploading a file
      - model: 'reconstruction' | 'patchcore' | 'both' (default: 'both')
      - threshold_mode: 'unbiased' | 'optimal' | 'custom' (default: 'unbiased')
      - custom_threshold_recon: (Optional) float threshold for reconstruction
      - custom_threshold_patch: (Optional) float threshold for PatchCore
    """
    service = ModelService.get_instance()

    # Determine model choice & threshold mode from form or JSON
    if request.is_json:
        payload = request.get_json() or {}
        model_choice = payload.get("model", "both")
        threshold_mode = payload.get("threshold_mode", "unbiased")
        custom_recon = payload.get("custom_threshold_recon")
        custom_patch = payload.get("custom_threshold_patch")
        sample_id = payload.get("sample_id")
        uploaded_file = None
    else:
        model_choice = request.form.get("model", "both")
        threshold_mode = request.form.get("threshold_mode", "unbiased")
        custom_recon = request.form.get("custom_threshold_recon")
        custom_patch = request.form.get("custom_threshold_patch")
        sample_id = request.form.get("sample_id")
        uploaded_file = request.files.get("image")

    # Retrieve image payload
    image_bytes = None
    source_name = "uploaded_image.png"

    if uploaded_file and uploaded_file.filename != "":
        source_name = uploaded_file.filename
        image_bytes = uploaded_file.read()
    elif sample_id:
        sample_data = service.get_sample_image_bytes(sample_id)
        if sample_data is None:
            return jsonify({"error": f"Sample preset '{sample_id}' not found on server."}), 404
        image_bytes, source_name = sample_data
    else:
        return jsonify({
            "error": "No image provided. Please upload an image file under 'image' or specify 'sample_id'."
        }), 400

    if not image_bytes:
        return jsonify({"error": "Empty image payload received."}), 400

    # Parse custom threshold overrides safely
    c_recon = float(custom_recon) if custom_recon not in (None, "") else None
    c_patch = float(custom_patch) if custom_patch not in (None, "") else None

    try:
        results = service.inspect(
            image_bytes=image_bytes,
            model_choice=model_choice,
            threshold_mode=threshold_mode,
            custom_threshold_recon=c_recon,
            custom_threshold_patch=c_patch,
            source_name=source_name,
        )
        return jsonify(results), 200
    except ValueError as val_err:
        return jsonify({"error": str(val_err)}), 400
    except RuntimeError as run_err:
        return jsonify({"error": str(run_err)}), 503
    except Exception as exc:
        logger.exception("Inspection exception: %s", exc)
        return jsonify({"error": f"Internal inspection error: {exc}"}), 500


@inspect_bp.route("/history", methods=["GET"])
def get_history():
    """
    Return recent inspection history log.
    Responds to GET /api/history?limit=50
    """
    limit = request.args.get("limit", default=50, type=int)
    service = ModelService.get_instance()
    items = service.get_history(limit=limit)
    return jsonify({"count": len(items), "history": items}), 200


@inspect_bp.route("/history/<inspection_id>", methods=["GET"])
def get_history_detail(inspection_id: str):
    """
    Retrieve complete inspection outcome with full heatmaps for a past inspection.
    Responds to GET /api/history/<inspection_id>
    """
    service = ModelService.get_instance()
    item = service.get_history_item(inspection_id)
    if item is None:
        return jsonify({"error": f"Inspection record '{inspection_id}' not found."}), 404
    return jsonify(item), 200


@inspect_bp.route("/history", methods=["DELETE"])
def clear_history():
    """
    Clear all in-memory inspection records.
    Responds to DELETE /api/history
    """
    service = ModelService.get_instance()
    service.clear_history()
    return jsonify({"status": "cleared", "message": "Inspection history cleared."}), 200


@inspect_bp.route("/eval-summary", methods=["GET"])
def get_evaluation_summary():
    """
    Return genuine empirical benchmark evaluation summary for both models.
    Responds to GET /api/eval-summary
    """
    service = ModelService.get_instance()
    summary = service.get_eval_summary()
    return jsonify(summary), 200


@inspect_bp.route("/samples", methods=["GET"])
def get_sample_presets():
    """
    Return list of pre-configured sample product images with metadata and thumbnails.
    Responds to GET /api/samples
    """
    service = ModelService.get_instance()
    presets = service.get_sample_presets()
    return jsonify({"count": len(presets), "samples": presets}), 200


@inspect_bp.route("/samples/<sample_id>", methods=["GET"])
def get_sample_image(sample_id: str):
    """
    Retrieve the raw sample image as image/png.
    Responds to GET /api/samples/<sample_id>
    """
    service = ModelService.get_instance()
    sample_data = service.get_sample_image_bytes(sample_id)
    if sample_data is None:
        return jsonify({"error": f"Sample image '{sample_id}' not found."}), 404

    image_bytes, filename = sample_data
    return Response(
        image_bytes,
        mimetype="image/png",
        headers={"Content-Disposition": f"inline; filename={filename}"},
    )
