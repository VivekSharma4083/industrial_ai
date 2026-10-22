"""
tests/test_patchcore.py
=======================
Focused unit and integration test suite for the PatchCore anomaly detection baseline:
- Multi-scale patch embedding extraction and spatial alignment
- Deterministic Johnson-Lindenstrauss greedy k-Center coreset subsampling
- Memory bank construction from normal training data
- Chunked nearest-neighbor distance calculation
- Spatial anomaly map dimensions, upsampling, and Gaussian smoothing
- Checkpoint serialization and fresh reload
- Evaluation metric compatibility
"""

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.patchcore import (
    PatchCoreConfig,
    PatchEmbeddingExtractor,
    GreedyCoresetSampler,
    PatchCoreDetector,
)
from src.evaluation.metrics import compute_image_auroc, compute_pixel_auroc


def test_patch_embedding_shape_and_finite():
    """Verify that multi-scale patch extraction produces correct shape and finite values."""
    # Untrained ResNet for fast offline unit testing
    extractor = PatchEmbeddingExtractor(
        layers=("layer2", "layer3"),
        pretrained=False,
        device="cpu",
        image_size=(224, 224),
        target_spatial_size=(28, 28),
    )

    dummy_input = torch.randn(2, 3, 224, 224)
    patches, spatial_shape = extractor(dummy_input)

    # layer2 (512) + layer3 (1024) = 1536 channels
    # 2 images * 28 * 28 = 1568 total patches
    assert spatial_shape == (28, 28)
    assert patches.shape == (2 * 28 * 28, 1536)
    assert torch.all(torch.isfinite(patches))


def test_greedy_coreset_sampler_determinism_and_ratio():
    """Verify that GreedyCoresetSampler is deterministic and respects coreset ratio."""
    features = torch.randn(200, 64)

    coreset1, idx1 = GreedyCoresetSampler.subsample(features, coreset_ratio=0.1, seed=42)
    coreset2, idx2 = GreedyCoresetSampler.subsample(features, coreset_ratio=0.1, seed=42)

    # 10% of 200 = 20 points
    assert len(idx1) == 20
    assert coreset1.shape == (20, 64)
    assert idx1 == idx2
    assert torch.allclose(coreset1, coreset2)

    # Verify that all selected indices are unique
    assert len(set(idx1)) == 20


def test_greedy_coreset_sampler_boundary_cases():
    """Verify coreset sampler handles edge cases (ratio=1.0, very small N)."""
    features = torch.randn(5, 16)
    coreset, idx = GreedyCoresetSampler.subsample(features, coreset_ratio=1.0)
    assert len(idx) == 5
    assert torch.allclose(coreset, features)

    # Very small ratio still retains at least 1 point
    coreset_min, idx_min = GreedyCoresetSampler.subsample(features, coreset_ratio=0.001)
    assert len(idx_min) == 1


def test_patchcore_fit_and_chunked_scoring():
    """Verify memory bank construction and chunked nearest-neighbor scoring."""
    extractor = PatchEmbeddingExtractor(
        layers=("layer2", "layer3"),
        pretrained=False,
        device="cpu",
    )
    cfg = PatchCoreConfig(coreset_ratio=0.1, chunk_size=128)
    detector = PatchCoreDetector(extractor, config=cfg)

    # Synthetic normal training data: 3 samples
    dummy_data = torch.rand(3, 3, 224, 224)
    train_loader = DataLoader(TensorDataset(dummy_data), batch_size=2)

    stats = detector.fit(train_loader)

    assert stats["train_samples"] == 3
    assert detector.memory_bank is not None
    # 3 samples * 784 patches = 2352 raw patches. 10% coreset = 235 patches
    assert detector.memory_bank.shape[0] == 235
    assert detector.memory_bank.shape[1] == 1536

    # Test inference on a single image
    test_img = Image.new("RGB", (224, 224), (100, 150, 200))
    result = detector.inspect_image(test_img)

    assert result.image_score > 0.0
    assert np.isfinite(result.image_score)
    assert result.anomaly_map.shape == (224, 224)
    assert result.normalized_map.shape == (224, 224)
    assert 0.0 <= np.min(result.normalized_map) <= np.max(result.normalized_map) <= 1.0


def test_patchcore_checkpoint_save_and_load(tmp_path):
    """Verify PatchCore memory bank serialization and restoration."""
    extractor = PatchEmbeddingExtractor(layers=("layer2", "layer3"), pretrained=False, device="cpu")
    detector = PatchCoreDetector(extractor)

    # Manually populate memory bank with known synthetic vectors
    dummy_bank = torch.randn(50, 1536)
    detector.memory_bank = dummy_bank

    ckpt_file = tmp_path / "patchcore_test.pth"
    detector.save(ckpt_file)

    # Create fresh detector and load checkpoint
    fresh_detector = PatchCoreDetector(extractor)
    assert fresh_detector.memory_bank is None

    fresh_detector.load(ckpt_file)
    assert fresh_detector.memory_bank is not None
    assert fresh_detector.memory_bank.shape == (50, 1536)
    assert torch.allclose(fresh_detector.memory_bank, dummy_bank)


def test_patchcore_evaluation_metric_compatibility():
    """Verify PatchCore outputs are fully compatible with evaluation metrics."""
    extractor = PatchEmbeddingExtractor(layers=("layer2", "layer3"), pretrained=False, device="cpu")
    detector = PatchCoreDetector(extractor)
    detector.memory_bank = torch.randn(20, 1536)

    # Run inspection on two synthetic samples (1 normal, 1 anomalous)
    normal_img = Image.new("RGB", (64, 64), (50, 50, 50))
    defect_img = Image.new("RGB", (64, 64), (200, 200, 200))

    res_norm = detector.inspect_image(normal_img)
    res_def = detector.inspect_image(defect_img)

    # Evaluate image AUROC
    y_true = [0, 1]
    y_scores = [res_norm.image_score, res_def.image_score]
    # Set artificial separation to ensure dual-class validity
    if y_scores[0] == y_scores[1]:
        y_scores[1] += 1.0
    img_eval = compute_image_auroc(y_true, y_scores)
    assert 0.0 <= img_eval.roc_auc <= 1.0

    # Evaluate pixel AUROC
    mask_norm = np.zeros((64, 64), dtype=np.uint8)
    mask_def = np.zeros((64, 64), dtype=np.uint8)
    mask_def[10:30, 10:30] = 1

    pix_eval = compute_pixel_auroc([mask_norm, mask_def], [res_norm.anomaly_map, res_def.anomaly_map])
    assert 0.0 <= pix_eval.roc_auc <= 1.0
