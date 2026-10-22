#!/usr/bin/env python3
"""
scripts/smoke_test_models.py
============================
Pre-flight smoke test verifying the end-to-end CPU anomaly detection pipeline:
Feature Extraction -> Reconstruction -> Error Map -> Bilinear Upsampling -> Anomaly Scoring.
Uses synthetic input tensors to verify execution without external downloads.
"""

from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.feature_extractor import ResNetFeatureExtractor
from src.models.reconstruction import FeatureReconstructionAutoencoder
from src.models.anomaly_detector import FeatureAnomalyDetector


def main() -> int:
    print("=" * 65)
    print(" Industrial AI Inspector -- CPU Model Pipeline Smoke Test")
    print("=" * 65)

    device = "cpu"
    layer_name = "layer2"
    image_size = (224, 224)

    # 1. Initialize Feature Extractor (Untrained for offline smoke test)
    print(f"[*] Initializing ResNetFeatureExtractor (layer={layer_name}, pretrained=False, device={device})...")
    extractor = ResNetFeatureExtractor(
        layer_name=layer_name,
        pretrained=False,
        device=device,
        image_size=image_size,
    )
    feat_shape = extractor.get_feature_shape()
    print(f"    [+] ResNet50 '{layer_name}' channels: {extractor.out_channels}")
    print(f"    [+] Expected feature map spatial shape: {feat_shape}")

    # 2. Initialize Reconstruction Autoencoder
    print("[*] Initializing FeatureReconstructionAutoencoder...")
    recon = FeatureReconstructionAutoencoder(
        in_channels=extractor.out_channels,
    )
    print(f"    [+] Autoencoder trainable parameters: {recon.count_parameters():,}")

    # 3. Initialize Anomaly Scoring Engine
    print("[*] Initializing FeatureAnomalyDetector (score_aggregation='max')...")
    detector = FeatureAnomalyDetector(
        feature_extractor=extractor,
        reconstruction_model=recon,
        device=device,
        score_aggregation="max",
        gaussian_sigma=4.0,
        gaussian_kernel_size=7,
    )

    # 4. Create synthetic test image with artificial anomaly patch
    print("[*] Generating synthetic test image with simulated defect...")
    orig_w, orig_h = 300, 200
    test_img = Image.new("RGB", (orig_w, orig_h), color=(220, 220, 220))
    draw = ImageDraw.Draw(test_img)
    # Draw simulated defect: dark rectangle
    draw.rectangle([60, 40, 140, 120], fill=(20, 20, 20))

    # 5. Execute Inspection
    print("[*] Executing inspection pass...")
    result = detector.inspect_image(test_img, threshold=None)

    print("\n" + "-" * 65)
    print(" Inspection Telemetry:")
    print("-" * 65)
    print(f"  - Original Dimensions:     {result.orig_size} (width, height)")
    print(f"  - Anomaly Map Shape:       {result.anomaly_map.shape}")
    print(f"  - Raw Anomaly Score:       {result.image_score:.6f}")
    print(f"  - Map Min / Max:           {np.min(result.anomaly_map):.6f} / {np.max(result.anomaly_map):.6f}")
    print(f"  - Normalized Map Bounds:   [{np.min(result.normalized_map):.4f}, {np.max(result.normalized_map):.4f}]")
    print(f"  - Model Metadata:          {result.metadata}")
    print("-" * 65)

    # Validations
    assert result.anomaly_map.shape == (orig_h, orig_w), "Output map shape does not match original image dimensions!"
    assert np.isfinite(result.image_score), "Image score is not finite!"
    assert 0.0 <= np.min(result.normalized_map) <= np.max(result.normalized_map) <= 1.0, "Normalized map out of bounds!"

    # 6. Test Constant Map Edge Case
    print("\n[*] Testing edge case: perfectly flat/constant input map...")
    flat_map = np.full((100, 100), 5.0, dtype=np.float32)
    flat_norm = detector._safe_normalize(flat_map)
    assert np.all(flat_norm == 0.0), "Constant map did not normalize to zeros!"
    assert np.isfinite(flat_norm).all(), "Constant map produced non-finite values!"
    print("    [+] Safe normalization correctly handled zero-variance edge case.")

    print("\n" + "=" * 65)
    print("[SUCCESS] CPU Model Pipeline Smoke Test passed completely!")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
