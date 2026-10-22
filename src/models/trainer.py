"""
src/models/trainer.py
=====================
Modular training manager for the feature-reconstruction autoencoder.
Trains exclusively on normal (defect-free) image representations.
Includes checkpoint management, loss tracking, and reproducibility guarantees.
"""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.feature_extractor import ResNetFeatureExtractor
from src.models.reconstruction import FeatureReconstructionAutoencoder


@dataclass
class TrainingConfig:
    """
    Hyperparameters and runtime settings for reconstruction model training.
    """
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-5
    loss_type: str = "mse"          # "mse", "cosine", or "combined"
    cosine_weight: float = 0.5
    device: str = "cpu"
    seed: int = 42
    output_dir: Union[str, Path] = Path("artifacts/checkpoints")
    checkpoint_name: str = "reconstruction_bottle.pth"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["output_dir"] = str(d["output_dir"])
        return d


def set_seed(seed: int = 42) -> None:
    """Ensure deterministic execution across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReconstructionTrainer:
    """
    Manages the optimization lifecycle for the feature reconstruction autoencoder.
    """

    def __init__(
        self,
        feature_extractor: ResNetFeatureExtractor,
        reconstruction_model: FeatureReconstructionAutoencoder,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)

        # Set reproducible seed
        set_seed(self.config.seed)

        self.feature_extractor = feature_extractor.to(self.device).eval()
        self.reconstruction_model = reconstruction_model.to(self.device)

        # Configure optimizer strictly for the trainable reconstruction model
        self.optimizer = torch.optim.Adam(
            self.reconstruction_model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Execute one training epoch over normal training samples.

        Important: Feature extractor remains completely frozen; only the
        reconstruction network weights are updated.
        """
        if len(dataloader) == 0:
            raise ValueError("Provided DataLoader contains 0 batches.")

        self.reconstruction_model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Handle tuple from MVTecDataset: (image, label_id, mask, metadata)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)

            # 1. Extract representations with frozen backbone
            with torch.no_grad():
                features = self.feature_extractor(images)

            # 2. Reconstruct features
            reconstructed = self.reconstruction_model(features)

            # 3. Compute spatial reconstruction discrepancy
            loss = self.reconstruction_model.compute_loss(
                features,
                reconstructed,
                loss_type=self.config.loss_type,
                cosine_weight=self.config.cosine_weight,
            )

            # Guard against NaN/Inf
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Encountered non-finite loss during training: {loss.item()}")

            # 4. Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return float(avg_loss)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluate reconstruction loss over a validation set of normal samples.
        """
        if len(dataloader) == 0:
            return 0.0

        self.reconstruction_model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(self.device)

            features = self.feature_extractor(images)
            reconstructed = self.reconstruction_model(features)
            loss = self.reconstruction_model.compute_loss(
                features,
                reconstructed,
                loss_type=self.config.loss_type,
                cosine_weight=self.config.cosine_weight,
            )

            total_loss += loss.item()
            num_batches += 1

        return float(total_loss / max(1, num_batches))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_best: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete training routine for config.epochs.

        Args:
            train_loader: DataLoader for defect-free training split (train/good).
            val_loader: Optional normal-only validation loader.
            save_best: If True, saves checkpoint upon completion.

        Returns:
            Dictionary containing final metrics and loss trajectories.
        """
        if len(train_loader) == 0:
            raise ValueError("Training DataLoader contains 0 samples. Cannot start training.")

        print(f"[*] Starting reconstruction model training on {self.device}...")
        print(f"    - Epochs: {self.config.epochs} | Batch size: {self.config.batch_size} | LR: {self.config.lr}")
        print(f"    - Trainable Parameters: {self.reconstruction_model.count_parameters():,}")

        if val_loader is None:
            print("    [!] Notice: No separate normal validation split provided.")
            print("        Evaluation on test anomalies must remain segregated until final testing.")

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            val_str = ""
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                val_str = f" | Val Loss: {val_loss:.6f}"

            print(f"    [Epoch {epoch:02d}/{self.config.epochs:02d}] Train Loss: {train_loss:.6f}{val_str}")

        saved_path = None
        if save_best:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            saved_path = out_dir / self.config.checkpoint_name
            self.save_checkpoint(saved_path)
            print(f"[+] Checkpoint saved to: {saved_path}")

        return {
            "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            "history": self.history,
            "checkpoint_path": str(saved_path) if saved_path else None,
        }

    def save_checkpoint(
        self, path: Union[str, Path], extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Serialize model weights, optimizer state, and configuration to disk.
        """
        target_path = Path(path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self.reconstruction_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "feature_layer": self.feature_extractor.layer_name,
            "feature_channels": self.feature_extractor.out_channels,
            "history": self.history,
            "extra_metadata": extra_metadata or {},
        }
        torch.save(payload, target_path)
        return target_path

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Restore model weights and optimizer state from checkpoint.
        """
        target_path = Path(path).resolve()
        if not target_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: '{target_path}'")

        checkpoint = torch.load(target_path, map_location=self.device)
        self.reconstruction_model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint
