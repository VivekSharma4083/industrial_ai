"""
src/models/patchcore.py
=======================
CPU-compatible PatchCore anomaly detection baseline for industrial visual inspection.
Extracts multi-scale neighborhood-aggregated patch embeddings from intermediate
ResNet-50 layers (layer2 and layer3), applies greedy k-Center coreset subsampling
with Johnson-Lindenstrauss random projection, and scores unseen patches via chunked
nearest-neighbor distance queries.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms.functional as TF

from src.models.anomaly_detector import AnomalyResult, _get_gaussian_kernel

# Standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class PatchCoreConfig:
    """Hyperparameters and configuration settings for PatchCore."""
    layers: Tuple[str, ...] = ("layer2", "layer3")
    image_size: Tuple[int, int] = (224, 224)
    target_spatial_size: Tuple[int, int] = (28, 28)
    patch_pool_kernel: int = 3
    patch_pool_stride: int = 1
    patch_pool_padding: int = 1
    coreset_ratio: float = 0.01          # 1% coreset retention
    proj_dim: int = 128                  # Johnson-Lindenstrauss projection dimension
    seed: int = 42
    chunk_size: int = 256                # Patch query chunk size for memory bounding
    gaussian_kernel_size: int = 7
    gaussian_sigma: float = 4.0
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["layers"] = list(self.layers)
        d["image_size"] = list(self.image_size)
        d["target_spatial_size"] = list(self.target_spatial_size)
        return d


class PatchEmbeddingExtractor(nn.Module):
    """
    Extracts multi-scale, neighborhood-aware patch embeddings from a frozen
    ResNet-50 backbone.
    """

    def __init__(
        self,
        layers: Sequence[str] = ("layer2", "layer3"),
        pretrained: bool = True,
        device: Union[str, torch.device] = "cpu",
        image_size: Tuple[int, int] = (224, 224),
        target_spatial_size: Tuple[int, int] = (28, 28),
        pool_kernel: int = 3,
        pool_stride: int = 1,
        pool_padding: int = 1,
    ):
        super().__init__()
        self.layers = tuple(layers)
        self.device = torch.device(device)
        self.image_size = image_size
        self.target_spatial_size = target_spatial_size

        # Validate requested layers
        known_layers = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        for l in self.layers:
            if l not in known_layers:
                raise ValueError(f"Unknown layer '{l}'. Supported: {list(known_layers.keys())}")

        self.embedding_dim = sum(known_layers[l] for l in self.layers)

        # Load ResNet-50 backbone
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = resnet50(weights=None)

        self.backbone.to(self.device).eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Local spatial neighborhood aggregation
        self.patch_pool = nn.AvgPool2d(
            kernel_size=pool_kernel,
            stride=pool_stride,
            padding=pool_padding,
        )

        # Hook storage
        self._activations: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attach forward hooks to target intermediate modules."""
        for layer_name in self.layers:
            target_mod = getattr(self.backbone, layer_name)

            def get_hook(name: str):
                def hook(module, inp, out):
                    self._activations[name] = out
                return hook

            target_mod.register_forward_hook(get_hook(layer_name))

    def preprocess(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess input image to the format expected by ImageNet-pretrained ResNet.
        """
        if isinstance(image, Image.Image):
            img = image.convert("RGB").resize(self.image_size, Image.Resampling.BILINEAR)
            tensor = TF.to_tensor(img)
            tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            return tensor.unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            tensor = image.clone()
            if tensor.dim() == 3:
                if tensor.shape[-2:] != self.image_size:
                    tensor = TF.resize(tensor, list(self.image_size), antialias=True)
                tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                return tensor.unsqueeze(0).to(self.device)
            elif tensor.dim() == 4:
                if tensor.shape[-2:] != self.image_size:
                    tensor = TF.resize(tensor, list(self.image_size), antialias=True)
                mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
                std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
                return ((tensor - mean) / std).to(self.device)
            else:
                raise ValueError(f"Unsupported tensor dimensions: {tensor.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract aligned and neighborhood-aggregated patch embeddings.

        Args:
            x: Preprocessed input tensor of shape (B, 3, H, W).

        Returns:
            Tuple of:
              - patch_embeddings: Tensor of shape (B * H_target * W_target, embedding_dim)
              - spatial_shape: (H_target, W_target), e.g. (28, 28)
        """
        x = x.to(self.device)
        self._activations.clear()
        _ = self.backbone(x)

        processed_features = []
        b = x.shape[0]
        h_t, w_t = self.target_spatial_size

        for layer_name in self.layers:
            act = self._activations[layer_name]  # (B, C, H_l, W_l)
            # 1. Local spatial pooling for neighborhood context
            pooled = self.patch_pool(act)
            # 2. Spatial alignment via bilinear interpolation
            if pooled.shape[-2:] != (h_t, w_t):
                aligned = F.interpolate(
                    pooled,
                    size=(h_t, w_t),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                aligned = pooled
            processed_features.append(aligned)

        # Concatenate along channel dimension: (B, total_channels, H_t, W_t)
        merged = torch.cat(processed_features, dim=1)
        # Permute to (B, H_t, W_t, total_channels) and flatten spatial patches
        patches = merged.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        return patches, (h_t, w_t)


class GreedyCoresetSampler:
    """
    Deterministic minimax facility-location (k-Center greedy) coreset sampler
    with Johnson-Lindenstrauss random projection for fast CPU convergence.
    """

    @staticmethod
    def subsample(
        features: torch.Tensor,
        coreset_ratio: float = 0.01,
        proj_dim: int = 128,
        seed: int = 42,
        chunk_size: int = 8192,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Subsample representative patch vectors from a feature bank.

        Args:
            features: 2D Tensor of shape (N, D).
            coreset_ratio: Fraction of vectors to retain (0 < ratio <= 1.0).
            proj_dim: Dimension of Johnson-Lindenstrauss projection matrix.
            seed: Random seed for deterministic projection initialization.
            chunk_size: Distance calculation chunk size to avoid memory spikes.

        Returns:
            Tuple of (coreset_tensor of shape (M, D), selected_indices).
        """
        n, d = features.shape
        m = max(1, int(round(n * coreset_ratio)))

        if m >= n:
            return features.clone(), list(range(n))

        device = features.device

        # 1. Construct deterministic Johnson-Lindenstrauss random projection matrix
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        # Random Gaussian projection matrix (D, proj_dim)
        proj_matrix = torch.randn(d, proj_dim, generator=rng, device=device)
        proj_matrix = proj_matrix / np.sqrt(proj_dim)

        # Project features: (N, D) @ (D, proj_dim) -> (N, proj_dim)
        features_proj = torch.matmul(features, proj_matrix)

        # 2. Greedy Minimax k-Center Selection
        # Select initial center: point with maximum norm (or index 0)
        c0 = int(torch.argmax(torch.norm(features_proj, p=2, dim=1)).item())
        selected_indices = [c0]

        # Track minimum distance from each point to the closest selected center
        min_dists = torch.norm(features_proj - features_proj[c0 : c0 + 1], p=2, dim=1)

        for _ in range(1, m):
            # Next center is the point farthest from all currently selected centers
            next_idx = int(torch.argmax(min_dists).item())
            selected_indices.append(next_idx)

            # Update minimum distances efficiently in chunks
            center_vec = features_proj[next_idx : next_idx + 1]
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                dist_chunk = torch.norm(features_proj[start:end] - center_vec, p=2, dim=1)
                min_dists[start:end] = torch.minimum(min_dists[start:end], dist_chunk)

        # Extract selected unprojected features
        selected_tensor = features[selected_indices].clone()
        return selected_tensor, selected_indices


class PatchCoreDetector:
    """
    End-to-end PatchCore anomaly inspection engine:
    - Extracts multi-scale neighborhood patch embeddings.
    - Fits memory bank from normal training samples.
    - Runs chunked nearest-neighbor scoring for fast, memory-bounded CPU inference.
    - Generates spatial anomaly heatmaps and aggregated image scores.
    """

    def __init__(
        self,
        extractor: PatchEmbeddingExtractor,
        config: Optional[PatchCoreConfig] = None,
    ):
        self.config = config or PatchCoreConfig()
        self.device = torch.device(self.config.device)
        self.extractor = extractor.to(self.device).eval()
        self.memory_bank: Optional[torch.Tensor] = None
        self.train_scores: List[float] = []

    @torch.no_grad()
    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        coreset_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Construct memory bank from normal defect-free training images.

        Args:
            dataloader: DataLoader providing normal training images.
            coreset_ratio: Optional override for subsampling ratio.

        Returns:
            Dictionary summarizing memory-bank construction statistics.
        """
        ratio = coreset_ratio if coreset_ratio is not None else self.config.coreset_ratio
        all_patches: List[torch.Tensor] = []

        t0 = time.perf_counter()
        sample_count = 0

        for batch in dataloader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = self.extractor.preprocess(images)
            patches, _ = self.extractor(x)
            all_patches.append(patches.cpu())
            sample_count += images.shape[0] if images.dim() == 4 else 1

        if len(all_patches) == 0:
            raise ValueError("DataLoader provided 0 samples. Cannot construct memory bank.")

        raw_memory = torch.cat(all_patches, dim=0)
        n_raw, dim = raw_memory.shape

        # Run deterministic greedy coreset subsampling
        t_core0 = time.perf_counter()
        coreset, _ = GreedyCoresetSampler.subsample(
            raw_memory,
            coreset_ratio=ratio,
            proj_dim=self.config.proj_dim,
            seed=self.config.seed,
        )
        t_core1 = time.perf_counter()

        self.memory_bank = coreset.to(self.device)
        total_time = time.perf_counter() - t0

        # Compute training score baseline for unsupervised threshold calibration
        self.train_scores.clear()
        for batch in dataloader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            for idx in range(images.shape[0] if images.dim() == 4 else 1):
                img = images[idx] if images.dim() == 4 else images
                res = self.inspect_image(img)
                self.train_scores.append(res.image_score)

        return {
            "train_samples": sample_count,
            "raw_patches": n_raw,
            "coreset_patches": coreset.shape[0],
            "coreset_ratio": ratio,
            "embedding_dim": dim,
            "total_fit_time_s": round(total_time, 2),
            "coreset_time_s": round(t_core1 - t_core0, 2),
            "memory_bank_mb": round((coreset.numel() * 4) / (1024 * 1024), 2),
        }

    @torch.no_grad()
    def inspect_image(
        self,
        image: Union[Image.Image, torch.Tensor],
        threshold: Optional[float] = None,
    ) -> AnomalyResult:
        """
        Inspect a single product image and generate an anomaly heatmap and score.

        Args:
            image: PIL Image or Tensor.
            threshold: Optional calibrated threshold.

        Returns:
            AnomalyResult containing image_score, anomaly_map, normalized_map, and metadata.
        """
        if self.memory_bank is None or self.memory_bank.shape[0] == 0:
            raise RuntimeError("PatchCoreDetector memory bank is empty. Call fit() or load() first.")

        # Determine original dimensions
        if isinstance(image, Image.Image):
            orig_w, orig_h = image.size
        elif isinstance(image, torch.Tensor):
            orig_h, orig_w = image.shape[-2:]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 1. Preprocess and extract patch embeddings
        x = self.extractor.preprocess(image).to(self.device)
        patches, (h_feat, w_feat) = self.extractor(x)  # (H_feat * W_feat, dim)

        # 2. Chunked Nearest-Neighbor Distance Query against Memory Bank
        # Compute min distance to memory bank for each patch in memory-safe chunks
        n_patches = patches.shape[0]
        min_distances = []
        chunk_sz = self.config.chunk_size

        for start in range(0, n_patches, chunk_sz):
            end = min(start + chunk_sz, n_patches)
            chunk = patches[start:end]
            # Pairwise Euclidean distance between chunk and memory bank
            dists = torch.cdist(chunk, self.memory_bank)  # (chunk_len, M)
            chunk_mins, _ = torch.min(dists, dim=1)
            min_distances.append(chunk_mins)

        patch_scores = torch.cat(min_distances, dim=0)  # (H_feat * W_feat,)
        score_map = patch_scores.view(1, 1, h_feat, w_feat)

        # 3. Gaussian Spatial Smoothing
        if self.config.gaussian_sigma > 0 and self.config.gaussian_kernel_size > 1:
            kernel = _get_gaussian_kernel(
                kernel_size=self.config.gaussian_kernel_size,
                sigma=self.config.gaussian_sigma,
            ).to(self.device)
            pad = self.config.gaussian_kernel_size // 2
            score_map = F.conv2d(score_map, kernel, padding=pad)

        # 4. Upsample to original image resolution
        upsampled = F.interpolate(
            score_map,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        anomaly_map_np = upsampled.squeeze().cpu().numpy()

        # 5. Safe normalization for visualization
        norm_map_np = self._safe_normalize(anomaly_map_np)

        # 6. Image-level score: maximum smoothed spatial anomaly
        image_score = float(np.max(anomaly_map_np))

        # 7. Evaluate threshold if provided
        is_anomaly = bool(image_score >= threshold) if threshold is not None else None

        metadata = {
            "model_type": "PatchCore",
            "layers": list(self.config.layers),
            "embedding_dim": self.extractor.embedding_dim,
            "memory_bank_size": self.memory_bank.shape[0],
            "coreset_ratio": self.config.coreset_ratio,
            "spatial_patch_grid": [h_feat, w_feat],
            "device": str(self.device),
        }

        return AnomalyResult(
            image_score=image_score,
            anomaly_map=anomaly_map_np,
            normalized_map=norm_map_np,
            orig_size=(orig_w, orig_h),
            threshold=threshold,
            is_anomaly=is_anomaly,
            metadata=metadata,
        )

    @staticmethod
    def _safe_normalize(raw_map: np.ndarray) -> np.ndarray:
        """Safely scale spatial anomaly map to [0.0, 1.0]."""
        clean = np.nan_to_num(raw_map, nan=0.0, posinf=0.0, neginf=0.0)
        min_v = float(np.min(clean))
        max_v = float(np.max(clean))
        spread = max_v - min_v
        if spread < 1e-8:
            return np.zeros_like(clean, dtype=np.float32)
        normed = (clean - min_v) / spread
        return np.clip(normed, 0.0, 1.0).astype(np.float32)

    def save(self, path: Union[str, Path]) -> Path:
        """Serialize memory bank and configuration to disk."""
        if self.memory_bank is None:
            raise RuntimeError("Cannot save an empty PatchCore model.")
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "memory_bank": self.memory_bank.cpu(),
            "config": self.config.to_dict(),
            "train_scores": self.train_scores,
            "embedding_dim": self.extractor.embedding_dim,
        }
        torch.save(payload, p)
        return p

    def load(self, path: Union[str, Path]) -> None:
        """Load memory bank and configuration from disk."""
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"PatchCore checkpoint not found at: '{p}'")
        payload = torch.load(p, map_location=self.device, weights_only=False)
        self.memory_bank = payload["memory_bank"].to(self.device)
        self.train_scores = payload.get("train_scores", [])
