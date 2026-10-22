"""
src/data/config.py
==================
Configuration manager for MVTec Anomaly Detection dataset paths and category settings.
Supports dynamic configuration via environment variables or programmatic overrides.
"""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class DatasetConfig:
    """
    Configuration for MVTec AD dataset directory layout and active category.
    """
    dataset_root: Path
    category: str = "bottle"

    def __init__(self, dataset_root: Path | str | None = None, category: str = "bottle"):
        # Resolve project root dynamically
        project_root = Path(__file__).resolve().parent.parent.parent

        if dataset_root is None:
            env_root = os.environ.get("MVTEC_DATASET_ROOT")
            if env_root:
                self.dataset_root = Path(env_root).resolve()
            else:
                self.dataset_root = (project_root / "data" / "mvtec_anomaly_detection").resolve()
        else:
            self.dataset_root = Path(dataset_root).resolve()

        self.category = category.strip().lower()

    @property
    def category_dir(self) -> Path:
        """
        Directory for the active category (e.g., data/mvtec_anomaly_detection/bottle).
        If dataset_root already terminates in the category name, return dataset_root.
        """
        if self.dataset_root.name.lower() == self.category:
            return self.dataset_root
        return self.dataset_root / self.category

    @property
    def train_dir(self) -> Path:
        """Path to train/ directory."""
        return self.category_dir / "train"

    @property
    def train_good_dir(self) -> Path:
        """Path to train/good/ directory containing pristine normal samples."""
        return self.train_dir / "good"

    @property
    def test_dir(self) -> Path:
        """Path to test/ directory containing normal and defective samples."""
        return self.category_dir / "test"

    @property
    def test_good_dir(self) -> Path:
        """Path to test/good/ directory containing normal test samples."""
        return self.test_dir / "good"

    @property
    def ground_truth_dir(self) -> Path:
        """Path to ground_truth/ directory containing defect binary segmentation masks."""
        return self.category_dir / "ground_truth"

    def to_dict(self) -> dict:
        """Serialize configuration to dictionary."""
        return {
            "dataset_root": str(self.dataset_root),
            "category": self.category,
            "category_dir": str(self.category_dir),
            "train_dir": str(self.train_dir),
            "test_dir": str(self.test_dir),
            "ground_truth_dir": str(self.ground_truth_dir),
            "exists": self.category_dir.exists(),
        }
