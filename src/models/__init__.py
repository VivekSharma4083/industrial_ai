"""
src/models/__init__.py
======================
Model architectures, feature extractors, and inference scoring engines
for Industrial AI Inspector.
"""

from src.models.feature_extractor import ResNetFeatureExtractor, RESNET50_LAYER_CHANNELS
from src.models.reconstruction import FeatureReconstructionAutoencoder
from src.models.anomaly_detector import FeatureAnomalyDetector, AnomalyResult
from src.models.trainer import ReconstructionTrainer, TrainingConfig, set_seed
from src.models.patchcore import (
    PatchCoreConfig,
    PatchEmbeddingExtractor,
    GreedyCoresetSampler,
    PatchCoreDetector,
)

__all__ = [
    "ResNetFeatureExtractor",
    "RESNET50_LAYER_CHANNELS",
    "FeatureReconstructionAutoencoder",
    "FeatureAnomalyDetector",
    "AnomalyResult",
    "ReconstructionTrainer",
    "TrainingConfig",
    "set_seed",
    "PatchCoreConfig",
    "PatchEmbeddingExtractor",
    "GreedyCoresetSampler",
    "PatchCoreDetector",
]
