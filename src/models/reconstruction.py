"""
src/models/reconstruction.py
============================
Lightweight convolutional feature reconstruction network for unsupervised anomaly detection.

Rationale:
----------
Pretrained deep CNN features (e.g., ResNet50 intermediate representations) encode
rich semantics of local textures, contours, and object structure. When a convolutional
autoencoder is trained exclusively on defect-free features from normal industrial products,
it learns the constrained low-dimensional manifold of normal visual patterns.

When an anomalous product image (scratches, cracks, contaminants, missing parts) is
presented at test time, the anomalous regions produce out-of-distribution feature vectors.
Because the reconstruction network has never observed these failure modes, it cannot
faithfully reconstruct them. The spatial discrepancy between original and reconstructed
feature maps directly reveals both the presence and exact location of defects.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureReconstructionAutoencoder(nn.Module):
    """
    Lightweight convolutional autoencoder designed to reconstruct intermediate ResNet feature maps.

    Maintains exact spatial grid dimensions (H, W) while compressing and projecting
    features through a low-dimensional bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 512,
        bottleneck_channels: Optional[int] = None,
        intermediate_channels: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = intermediate_channels or max(64, in_channels // 2)
        self.bottleneck_channels = bottleneck_channels or max(32, in_channels // 4)

        # 1. Encoder: 1x1 projection followed by 3x3 spatial context compression
        self.enc1 = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inter_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.enc2 = nn.Conv2d(self.inter_channels, self.bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.bottleneck_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # 2. Latent bottleneck representation
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 3. Decoder: 3x3 spatial expansion followed by 1x1 output projection
        self.dec1 = nn.Conv2d(self.bottleneck_channels, self.inter_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.inter_channels)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        self.dec2 = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: reconstructs input feature map.

        Args:
            x: Spatial feature tensor of shape (B, C, H, W).

        Returns:
            Reconstructed spatial feature tensor of identical shape (B, C, H, W).
        """
        # Encoder
        h = self.act1(self.bn1(self.enc1(x)))
        h = self.act2(self.bn2(self.enc2(h)))

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        h = self.act3(self.bn3(self.dec1(h)))
        out = self.dec2(h)
        return out

    def compute_loss(
        self,
        features: torch.Tensor,
        reconstructed: torch.Tensor,
        loss_type: str = "mse",
        cosine_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss between original features and reconstructed features.

        Args:
            features: Ground-truth feature map (B, C, H, W).
            reconstructed: Reconstructed feature map (B, C, H, W).
            loss_type: "mse", "cosine", or "combined".
            cosine_weight: Weight for cosine distance when loss_type="combined".

        Returns:
            Scalar loss tensor.
        """
        if loss_type == "mse":
            return F.mse_loss(reconstructed, features)
        elif loss_type == "cosine":
            # 1 - cosine similarity averaged over spatial grid and batch
            cos_sim = F.cosine_similarity(reconstructed, features, dim=1)  # (B, H, W)
            return torch.mean(1.0 - cos_sim)
        elif loss_type == "combined":
            mse = F.mse_loss(reconstructed, features)
            cos_sim = F.cosine_similarity(reconstructed, features, dim=1)
            cos_loss = torch.mean(1.0 - cos_sim)
            return mse + cosine_weight * cos_loss
        else:
            raise ValueError(f"Unknown loss_type '{loss_type}'. Supported: 'mse', 'cosine', 'combined'.")

    def count_parameters(self) -> int:
        """Return total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
