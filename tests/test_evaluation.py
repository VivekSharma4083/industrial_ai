"""
tests/test_evaluation.py
========================
Unit tests for the evaluation metrics module:
- Image-level AUROC & threshold computation
- Pixel-level AUROC localization
- Defect category distribution analysis
- Latency benchmarking
- Multi-panel visual overlay generator
"""

import numpy as np
from PIL import Image
import pytest
import torch

from src.evaluation.metrics import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_per_defect_stats,
    benchmark_inference_latency,
    generate_evaluation_overlay,
    ImageEvaluationResult,
    PixelEvaluationResult,
)
from src.models.feature_extractor import ResNetFeatureExtractor
from src.models.reconstruction import FeatureReconstructionAutoencoder
from src.models.anomaly_detector import FeatureAnomalyDetector


def test_compute_image_auroc_perfect_separation():
    """Test that perfectly separated scores yield AUROC=1.0 and correct metrics."""
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_scores = [0.1, 0.2, 0.15, 0.25, 0.8, 0.85, 0.9, 0.95]

    res = compute_image_auroc(y_true, y_scores)

    assert isinstance(res, ImageEvaluationResult)
    assert res.roc_auc == 1.0
    assert res.true_positives == 4
    assert res.true_negatives == 4
    assert res.false_positives == 0
    assert res.false_negatives == 0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1_score == 1.0
    assert res.accuracy == 1.0
    assert res.total_samples == 8


def test_compute_image_auroc_inverted_separation():
    """Test that completely inverted predictions yield AUROC=0.0."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.9, 0.8, 0.2, 0.1]

    res = compute_image_auroc(y_true, y_scores)
    assert res.roc_auc == 0.0


def test_compute_image_auroc_error_handling():
    """Test invalid inputs (empty lists, single-class inputs)."""
    with pytest.raises(ValueError, match="empty label array"):
        compute_image_auroc([], [])

    with pytest.raises(ValueError, match="both positive and negative"):
        compute_image_auroc([0, 0, 0], [0.1, 0.2, 0.3])


def test_compute_pixel_auroc_perfect_localization():
    """Test pixel AUROC when error maps cleanly highlight defective pixels."""
    # Create two synthetic 16x16 samples
    mask1 = np.zeros((16, 16), dtype=np.uint8)
    mask1[4:8, 4:8] = 1  # 16 defect pixels

    map1 = np.zeros((16, 16), dtype=np.float32)
    map1[4:8, 4:8] = 0.95  # High reconstruction error on defect

    mask2 = np.zeros((16, 16), dtype=np.uint8)  # Normal sample (all 0)
    map2 = np.full((16, 16), 0.05, dtype=np.float32)  # Low error

    res = compute_pixel_auroc([mask1, mask2], [map1, map2])

    assert isinstance(res, PixelEvaluationResult)
    assert res.roc_auc == 1.0
    assert res.total_pixels == 512
    assert res.defect_pixels == 16
    assert res.normal_pixels == 496


def test_compute_pixel_auroc_mismatches_and_edge_cases():
    """Test validation guards on count mismatch and dimension mismatch."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    anomaly_map = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="Mismatch in counts"):
        compute_pixel_auroc([mask], [anomaly_map, anomaly_map])

    with pytest.raises(ValueError, match="shape mismatch"):
        diff_shape_map = np.zeros((12, 12), dtype=np.float32)
        compute_pixel_auroc([mask], [diff_shape_map])

    with pytest.raises(ValueError, match="both defective and normal pixels"):
        # All zeros has only 1 class
        compute_pixel_auroc([mask], [anomaly_map])


def test_compute_per_defect_stats():
    """Test summary statistics grouped by defect names."""
    labels = ["good", "good", "broken_large", "broken_large", "contamination"]
    scores = [0.1, 0.2, 0.9, 0.7, 0.5]

    stats = compute_per_defect_stats(labels, scores)

    assert "good" in stats
    assert "broken_large" in stats
    assert "contamination" in stats

    assert stats["good"].count == 2
    assert np.isclose(stats["good"].mean_score, 0.15)
    assert stats["broken_large"].count == 2
    assert np.isclose(stats["broken_large"].mean_score, 0.8)
    assert stats["contamination"].count == 1
    assert np.isclose(stats["contamination"].mean_score, 0.5)


def test_benchmark_inference_latency():
    """Test monotonic CPU latency benchmark execution with synthetic detector."""
    feature_extractor = ResNetFeatureExtractor(layer_name="layer2", pretrained=False, device="cpu")
    recon_model = FeatureReconstructionAutoencoder(in_channels=feature_extractor.out_channels)
    detector = FeatureAnomalyDetector(feature_extractor, recon_model, device="cpu")

    dummy_img = Image.new("RGB", (224, 224), (128, 128, 128))
    result = benchmark_inference_latency(detector, dummy_img, warmup_runs=1, timed_runs=3)

    assert result.timed_runs == 3
    assert result.mean_ms > 0.0
    assert result.throughput_fps > 0.0
    assert result.p50_ms > 0.0


def test_generate_evaluation_overlay():
    """Test multi-panel visualization creation and stitched resolution."""
    img = Image.new("RGB", (64, 64), (100, 150, 200))
    heat = np.random.rand(64, 64).astype(np.float32)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 1

    # 4 panels: original, heatmap, overlay, mask
    overlay = generate_evaluation_overlay(img, heat, mask)

    assert isinstance(overlay, Image.Image)
    assert overlay.size == (64 * 4, 64)
    assert overlay.mode == "RGB"
