"""
tests/test_models.py
====================
Unit tests for ResNet feature extraction, convolutional feature reconstruction,
anomaly scoring, safe map normalization, checkpoint persistence, and CPU execution.
All tests utilize explicitly untrained (pretrained=False) models or synthetic tensors
so they execute deterministically offline without downloading external weights.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.feature_extractor import ResNetFeatureExtractor
from src.models.reconstruction import FeatureReconstructionAutoencoder
from src.models.anomaly_detector import FeatureAnomalyDetector, AnomalyResult
from src.models.trainer import ReconstructionTrainer, TrainingConfig


@pytest.fixture
def untrained_extractor() -> ResNetFeatureExtractor:
    """Fixture providing an untrained, lightweight ResNet50 layer2 extractor on CPU."""
    return ResNetFeatureExtractor(
        layer_name="layer2",
        pretrained=False,  # Explicitly avoids external network weight download
        device="cpu",
        image_size=(224, 224),
    )


@pytest.fixture
def autoencoder() -> FeatureReconstructionAutoencoder:
    """Fixture providing a feature reconstruction autoencoder matching layer2 (512 channels)."""
    return FeatureReconstructionAutoencoder(in_channels=512)


def test_feature_extractor_output_shape(untrained_extractor: ResNetFeatureExtractor):
    """Verify feature extractor produces expected spatial tensor shape for 224x224 input."""
    dummy_input = torch.rand(2, 3, 224, 224)
    features = untrained_extractor(dummy_input)

    # For layer2: 512 channels, downsampled by factor of 8 (224 / 8 = 28)
    assert features.shape == (2, 512, 28, 28)
    assert features.device.type == "cpu"
    # Ensure all backbone parameters are strictly frozen
    for p in untrained_extractor.parameters():
        assert not p.requires_grad


def test_feature_extractor_configurable_layer():
    """Verify layer3 produces (1024, 14, 14) and invalid layer names raise ValueError."""
    extractor_l3 = ResNetFeatureExtractor(layer_name="layer3", pretrained=False, device="cpu")
    dummy_input = torch.rand(1, 3, 224, 224)
    features = extractor_l3(dummy_input)
    assert features.shape == (1, 1024, 14, 14)

    with pytest.raises(ValueError, match="Invalid layer_name"):
        ResNetFeatureExtractor(layer_name="invalid_layer", pretrained=False)


def test_reconstruction_shape_and_parameters(autoencoder: FeatureReconstructionAutoencoder):
    """Verify reconstruction network preserves spatial feature grid dimensions."""
    dummy_features = torch.rand(2, 512, 28, 28)
    reconstructed = autoencoder(dummy_features)

    assert reconstructed.shape == dummy_features.shape
    assert autoencoder.count_parameters() > 0


def test_reconstruction_loss_finite(autoencoder: FeatureReconstructionAutoencoder):
    """Verify MSE, Cosine, and Combined losses compute finite positive values."""
    f1 = torch.rand(2, 512, 14, 14)
    f2 = torch.rand(2, 512, 14, 14)

    loss_mse = autoencoder.compute_loss(f1, f2, loss_type="mse")
    loss_cos = autoencoder.compute_loss(f1, f2, loss_type="cosine")
    loss_comb = autoencoder.compute_loss(f1, f2, loss_type="combined")

    assert torch.isfinite(loss_mse) and loss_mse.item() >= 0.0
    assert torch.isfinite(loss_cos) and loss_cos.item() >= 0.0
    assert torch.isfinite(loss_comb) and loss_comb.item() >= 0.0

    with pytest.raises(ValueError, match="Unknown loss_type"):
        autoencoder.compute_loss(f1, f2, loss_type="invalid_loss")


def test_anomaly_detector_dimensions_and_scoring(
    untrained_extractor: ResNetFeatureExtractor,
    autoencoder: FeatureReconstructionAutoencoder,
):
    """
    Verify the anomaly detector outputs spatial maps matching original image dimensions
    and yields finite scores.
    """
    detector = FeatureAnomalyDetector(
        feature_extractor=untrained_extractor,
        reconstruction_model=autoencoder,
        device="cpu",
        score_aggregation="max",
    )

    # Test with arbitrary non-square PIL image
    img = Image.new("RGB", (320, 180), color=(100, 150, 200))
    result = detector.inspect_image(img)

    assert isinstance(result, AnomalyResult)
    assert result.orig_size == (320, 180)
    assert result.anomaly_map.shape == (180, 320)
    assert result.normalized_map.shape == (180, 320)
    assert np.isfinite(result.image_score)
    assert 0.0 <= np.min(result.normalized_map) <= np.max(result.normalized_map) <= 1.0


def test_constant_map_normalization_edge_cases(
    untrained_extractor: ResNetFeatureExtractor,
    autoencoder: FeatureReconstructionAutoencoder,
):
    """
    Verify safe normalization prevents division by zero when the error map is flat.
    """
    detector = FeatureAnomalyDetector(untrained_extractor, autoencoder)

    # 1. Perfectly flat zero map
    zeros_map = np.zeros((50, 50), dtype=np.float32)
    norm_zeros = detector._safe_normalize(zeros_map)
    assert np.all(norm_zeros == 0.0)

    # 2. Flat non-zero constant map
    const_map = np.full((50, 50), 42.0, dtype=np.float32)
    norm_const = detector._safe_normalize(const_map)
    assert np.all(norm_const == 0.0)

    # 3. Map containing NaN or Infinity
    dirty_map = np.array([[np.nan, 10.0], [np.inf, -np.inf]], dtype=np.float32)
    norm_dirty = detector._safe_normalize(dirty_map)
    assert np.all(np.isfinite(norm_dirty))
    assert 0.0 <= np.min(norm_dirty) <= np.max(norm_dirty) <= 1.0


def test_deterministic_inference_eval_mode(
    untrained_extractor: ResNetFeatureExtractor,
    autoencoder: FeatureReconstructionAutoencoder,
):
    """Verify repeated inference on identical input yields deterministic identical outputs."""
    detector = FeatureAnomalyDetector(untrained_extractor, autoencoder)
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))

    res1 = detector.inspect_image(img)
    res2 = detector.inspect_image(img)

    np.testing.assert_allclose(res1.anomaly_map, res2.anomaly_map, rtol=1e-5, atol=1e-5)
    assert res1.image_score == pytest.approx(res2.image_score, rel=1e-5)


def test_checkpoint_save_and_load(
    untrained_extractor: ResNetFeatureExtractor,
    autoencoder: FeatureReconstructionAutoencoder,
    tmp_path: Path,
):
    """Verify trainer checkpoint serialization and weight restoration."""
    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        output_dir=tmp_path / "checkpoints",
        checkpoint_name="test_model.pth",
    )
    trainer = ReconstructionTrainer(untrained_extractor, autoencoder, config=config)

    # Save initial state
    save_file = tmp_path / "checkpoints" / "test_model.pth"
    saved_path = trainer.save_checkpoint(save_file, extra_metadata={"run": "unit_test"})
    assert saved_path.is_file()

    # Create new fresh autoencoder with altered parameters
    new_autoencoder = FeatureReconstructionAutoencoder(in_channels=512)
    with torch.no_grad():
        for p in new_autoencoder.parameters():
            p.add_(1.0)  # modify weights

    new_trainer = ReconstructionTrainer(untrained_extractor, new_autoencoder, config=config)
    checkpoint_data = new_trainer.load_checkpoint(save_file)

    assert "model_state_dict" in checkpoint_data
    assert checkpoint_data["feature_layer"] == "layer2"
    assert checkpoint_data["extra_metadata"]["run"] == "unit_test"

    # Verify restored weights match original
    for (k1, v1), (k2, v2) in zip(
        autoencoder.state_dict().items(), new_autoencoder.state_dict().items()
    ):
        assert k1 == k2
        torch.testing.assert_close(v1, v2)


def test_trainer_synthetic_epoch(
    untrained_extractor: ResNetFeatureExtractor,
    autoencoder: FeatureReconstructionAutoencoder,
):
    """Verify trainer completes a single training step on synthetic image tensors."""
    trainer = ReconstructionTrainer(untrained_extractor, autoencoder)

    # Batch of 2 synthetic images
    synthetic_images = torch.rand(2, 3, 224, 224)
    loader = DataLoader(TensorDataset(synthetic_images), batch_size=2)

    initial_loss = trainer.train_epoch(loader)
    assert np.isfinite(initial_loss)
    assert initial_loss >= 0.0

    # Ensure empty loader raises ValueError
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 3, 224, 224)), batch_size=2)
    with pytest.raises(ValueError, match="0 batches"):
        trainer.train_epoch(empty_loader)
