"""
src/models/anomaly_detector.py
==============================
Inference and scoring engine for feature-reconstruction anomaly detection.
Generates pixel-level explainable anomaly heatmaps and aggregated image-level
anomaly scores.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feature_extractor import ResNetFeatureExtractor
from src.models.reconstruction import FeatureReconstructionAutoencoder


@dataclass
class AnomalyResult:
    """
    Structured outcome of an anomaly inspection pass.
    """
    image_score: float
    anomaly_map: np.ndarray          # Unscaled spatial reconstruction error map (H, W)
    normalized_map: np.ndarray       # Normalized heatmap in [0.0, 1.0] for visualization (H, W)
    orig_size: Tuple[int, int]       # (width, height) of original image
    threshold: Optional[float] = None
    is_anomaly: Optional[bool] = None  # None unless a calibrated threshold is provided
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_score": float(self.image_score),
            "orig_size": self.orig_size,
            "threshold": self.threshold,
            "is_anomaly": self.is_anomaly,
            "metadata": self.metadata or {},
        }


def _get_gaussian_kernel(kernel_size: int = 7, sigma: float = 4.0) -> torch.Tensor:
    """
    Generate a 2D Gaussian kernel in pure PyTorch (no external dependencies).
    """
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    gaussian = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2.0 * sigma ** 2))
    gaussian = gaussian / torch.sum(gaussian)
    return gaussian.view(1, 1, kernel_size, kernel_size)


class FeatureAnomalyDetector:
    """
    End-to-end anomaly detection and localization pipeline combining a frozen
    feature extractor with a trained spatial reconstruction autoencoder.
    """

    def __init__(
        self,
        feature_extractor: ResNetFeatureExtractor,
        reconstruction_model: FeatureReconstructionAutoencoder,
        device: Union[str, torch.device] = "cpu",
        score_aggregation: str = "max",
        top_k_ratio: float = 0.01,
        gaussian_sigma: float = 4.0,
        gaussian_kernel_size: int = 7,
    ):
        self.device = torch.device(device)
        self.feature_extractor = feature_extractor.to(self.device).eval()
        self.reconstruction_model = reconstruction_model.to(self.device).eval()
        self.score_aggregation = score_aggregation.lower().strip()
        self.top_k_ratio = top_k_ratio
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_kernel_size = gaussian_kernel_size

        if self.score_aggregation not in ("max", "top_k", "mean"):
            raise ValueError(
                f"Invalid score_aggregation '{self.score_aggregation}'. "
                f"Supported options: 'max', 'top_k', 'mean'."
            )

    @torch.no_grad()
    def inspect_image(
        self,
        image: Union[Image.Image, torch.Tensor],
        threshold: Optional[float] = None,
    ) -> AnomalyResult:
        """
        Perform deterministic anomaly inspection on a single input image.

        Important Notice on Scoring and Thresholds:
        ------------------------------------------
        The returned image_score represents raw reconstruction error and is NOT
        a calibrated probability. A classification threshold must be empirically
        determined using a balanced validation set (e.g. via ROC analysis).

        Args:
            image: PIL Image or Tensor.
            threshold: Optional decision boundary to classify normal vs anomalous.

        Returns:
            AnomalyResult with spatial heatmaps and image-level scores.
        """
        # Determine original dimensions
        if isinstance(image, Image.Image):
            orig_w, orig_h = image.size
        elif isinstance(image, torch.Tensor):
            orig_h, orig_w = image.shape[-2:]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 1. Preprocess input
        x = self.feature_extractor.preprocess(image).to(self.device)

        # 2. Extract intermediate spatial features
        features = self.feature_extractor(x)  # (1, C, H_f, W_f)

        # 3. Reconstruct feature map
        reconstructed = self.reconstruction_model(features)  # (1, C, H_f, W_f)

        # 4. Compute spatial L2 reconstruction error map across channels
        # Shape: (1, 1, H_f, W_f)
        diff = features - reconstructed
        error_map = torch.norm(diff, p=2, dim=1, keepdim=True)

        # 5. Apply Gaussian smoothing to reduce patch-boundary artifacts
        if self.gaussian_sigma > 0 and self.gaussian_kernel_size > 1:
            kernel = _get_gaussian_kernel(
                kernel_size=self.gaussian_kernel_size,
                sigma=self.gaussian_sigma,
            ).to(self.device)
            padding = self.gaussian_kernel_size // 2
            error_map = F.conv2d(error_map, kernel, padding=padding)

        # 6. Upsample to original image resolution (orig_h, orig_w)
        upsampled_map = F.interpolate(
            error_map,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        anomaly_map_np = upsampled_map.squeeze().cpu().numpy()  # (orig_h, orig_w)

        # 7. Safe normalization for visualization (prevent divide-by-zero on flat maps)
        norm_map_np = self._safe_normalize(anomaly_map_np)

        # 8. Compute image-level anomaly score
        image_score = self._compute_image_score(anomaly_map_np)

        # 9. Evaluate threshold if provided
        is_anomaly = None
        if threshold is not None:
            is_anomaly = bool(image_score >= threshold)

        metadata = {
            "feature_layer": self.feature_extractor.layer_name,
            "feature_shape": list(features.shape),
            "reconstruction_channels": self.reconstruction_model.in_channels,
            "score_aggregation": self.score_aggregation,
            "gaussian_sigma": self.gaussian_sigma,
            "device": str(self.device),
        }

        return AnomalyResult(
            image_score=float(image_score),
            anomaly_map=anomaly_map_np,
            normalized_map=norm_map_np,
            orig_size=(orig_w, orig_h),
            threshold=threshold,
            is_anomaly=is_anomaly,
            metadata=metadata,
        )

    def _compute_image_score(self, error_map: np.ndarray) -> float:
        """
        Aggregate pixel-level error map into a single image-level anomaly metric.
        """
        if self.score_aggregation == "max":
            return float(np.max(error_map))
        elif self.score_aggregation == "mean":
            return float(np.mean(error_map))
        elif self.score_aggregation == "top_k":
            # Average of top k% pixels (robust to single-pixel outliers while sensitive to defects)
            flat = error_map.flatten()
            k = max(1, int(len(flat) * self.top_k_ratio))
            top_k_values = np.partition(flat, -k)[-k:]
            return float(np.mean(top_k_values))
        return float(np.max(error_map))

    @staticmethod
    def _safe_normalize(raw_map: np.ndarray) -> np.ndarray:
        """
        Safely scale values to [0.0, 1.0].
        Handles constant maps, NaNs, and infinite edge cases.
        """
        # Guard against non-finite values
        clean_map = np.nan_to_num(raw_map, nan=0.0, posinf=0.0, neginf=0.0)
        min_val = float(np.min(clean_map))
        max_val = float(np.max(clean_map))
        spread = max_val - min_val

        # Constant map threshold (e.g. perfectly flat reconstruction)
        if spread < 1e-8:
            return np.zeros_like(clean_map, dtype=np.float32)

        normalized = (clean_map - min_val) / spread
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)
