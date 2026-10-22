"""
src/models/feature_extractor.py
===============================
Modular ResNet50 intermediate feature extractor for industrial anomaly detection.
Extracts spatial feature representations from specified intermediate ResNet layers
while keeping the backbone frozen. Supports CPU-only inference and standard
ImageNet normalization.
"""

from typing import Dict, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms.functional as TF

# Known output channels for ResNet50 architectural layers
RESNET50_LAYER_CHANNELS = {
    "layer1": 256,
    "layer2": 512,
    "layer3": 1024,
    "layer4": 2048,
}

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResNetFeatureExtractor(nn.Module):
    """
    Frozen ResNet-50 feature extractor targeting intermediate spatial representations.

    Intermediate layers (e.g. layer2 or layer3) retain spatial localization and
    mid-level texture/geometry patterns essential for anomaly detection, unlike
    the final global average pooling layer which discards spatial information.
    """

    def __init__(
        self,
        layer_name: str = "layer2",
        pretrained: bool = True,
        weights: Optional[Union[str, ResNet50_Weights]] = "DEFAULT",
        device: Union[str, torch.device] = "cpu",
        image_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.layer_name = layer_name.lower().strip()
        self.device = torch.device(device)
        self.image_size = image_size
        self.pretrained = pretrained

        if self.layer_name not in RESNET50_LAYER_CHANNELS:
            raise ValueError(
                f"Invalid layer_name '{self.layer_name}'. "
                f"Supported layers: {list(RESNET50_LAYER_CHANNELS.keys())}"
            )

        self.out_channels = RESNET50_LAYER_CHANNELS[self.layer_name]

        # Load ResNet-50 backbone
        if pretrained:
            resolved_weights = (
                ResNet50_Weights.DEFAULT if weights == "DEFAULT" else weights
            )
            try:
                self.backbone = resnet50(weights=resolved_weights)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load torchvision ResNet50 pretrained weights ({exc}). "
                    f"If executing offline or in unit tests, explicitly set pretrained=False."
                ) from exc
        else:
            # Explicitly untrained model (used for unit tests and offline validation)
            self.backbone = resnet50(weights=None)

        self.backbone.to(self.device)

        # Hook storage for intermediate spatial feature maps
        self._features: Dict[str, torch.Tensor] = {}
        self._register_hook()

        # Freeze all backbone parameters
        self.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _register_hook(self) -> None:
        """Register forward hook on the chosen layer to capture activations."""
        target_module = getattr(self.backbone, self.layer_name)

        def hook(module, input_tensor, output_tensor):
            self._features["out"] = output_tensor

        target_module.register_forward_hook(hook)

    def preprocess(
        self, image: Union[Image.Image, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess input image to the format expected by ImageNet-pretrained ResNet.

        Args:
            image: PIL Image or torch.Tensor with values in [0, 1].

        Returns:
            Normalized tensor of shape (1, 3, H, W) on self.device.
        """
        if isinstance(image, Image.Image):
            # Ensure RGB
            img = image.convert("RGB")
            # Resize
            img = img.resize(self.image_size, Image.Resampling.BILINEAR)
            tensor = TF.to_tensor(img)
        elif isinstance(image, torch.Tensor):
            tensor = image.clone()
            if tensor.dim() == 4:
                # Batch of tensors
                if tensor.shape[-2:] != self.image_size:
                    tensor = TF.resize(tensor, list(self.image_size), antialias=True)
                # Normalize batch
                mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
                std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
                return ((tensor - mean) / std).to(self.device)
            elif tensor.dim() == 3:
                if tensor.shape[-2:] != self.image_size:
                    tensor = TF.resize(tensor, list(self.image_size), antialias=True)
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        # Normalize 3D tensor
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        # Add batch dimension: (1, 3, H, W)
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature map for input batch x.

        Args:
            x: Preprocessed input tensor of shape (B, 3, H, W).

        Returns:
            Spatial feature map of shape (B, C, H_feat, W_feat).
        """
        x = x.to(self.device)
        self._features.clear()
        # Forward pass triggers hook on target layer
        _ = self.backbone(x)
        features = self._features["out"]
        return features

    def get_feature_shape(
        self, input_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int, int]:
        """
        Determine (channels, height, width) of extracted features for given input resolution.
        """
        h_in, w_in = input_size or self.image_size
        downsample_factors = {
            "layer1": 4,   # 224 -> 56
            "layer2": 8,   # 224 -> 28
            "layer3": 16,  # 224 -> 14
            "layer4": 32,  # 224 -> 7
        }
        scale = downsample_factors[self.layer_name]
        return (self.out_channels, h_in // scale, w_in // scale)
