"""
src/data/dataset.py
===================
PyTorch Dataset implementation for MVTec Anomaly Detection.
Supports unsupervised one-class training (normal-only) and comprehensive
evaluation (normal + defective with ground-truth masks).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from src.data.config import DatasetConfig
from src.data.validator import MVTecValidator, VALID_EXTENSIONS


@dataclass
class DatasetSample:
    """
    Representation of an individual sample within the MVTec dataset.
    """
    image_path: Path
    split: str                     # "train" or "test"
    label_name: str                # "good" or defect name (e.g. "broken_large")
    is_anomaly: bool               # False for "good", True for defects
    label_id: int                  # 0 for normal, 1 for anomaly
    mask_path: Optional[Path] = None  # None for normal samples


class MVTecDataset(Dataset):
    """
    PyTorch Dataset for MVTec AD categories.
    
    In 'train' split:
        Strictly yields defect-free normal samples (label_id=0, mask=zeros).
    In 'test' split:
        Yields normal test samples and defective test samples with corresponding
        ground-truth binary masks (1=defect, 0=background).
    """

    def __init__(
        self,
        category_dir: Path | str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.category_dir = Path(category_dir).resolve()
        self.split = split.lower().strip()
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size

        if self.split not in ("train", "test"):
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train' or 'test'.")

        if not self.category_dir.exists():
            raise FileNotFoundError(
                f"Dataset category directory not found: '{self.category_dir}'. "
                f"Verify path or configure MVTEC_DATASET_ROOT."
            )

        self.samples: List[DatasetSample] = []
        self._load_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split '{self.split}' in '{self.category_dir}'. "
                f"Ensure the dataset directory contains the expected MVTec folder structure."
            )

    def _load_samples(self) -> None:
        """Scan directory and index samples according to split."""
        target_split_dir = self.category_dir / self.split
        gt_dir = self.category_dir / "ground_truth"

        if not target_split_dir.is_dir():
            return

        if self.split == "train":
            # Training is strictly unsupervised: normal 'good' samples only
            good_dir = target_split_dir / "good"
            if good_dir.is_dir():
                for img_path in sorted(good_dir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
                        self.samples.append(
                            DatasetSample(
                                image_path=img_path,
                                split="train",
                                label_name="good",
                                is_anomaly=False,
                                label_id=0,
                                mask_path=None,
                            )
                        )
        else:
            # Test split includes both normal ('good') and defect subdirectories
            for sub_dir in sorted(target_split_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue

                defect_name = sub_dir.name
                is_normal = defect_name.lower() == "good"

                for img_path in sorted(sub_dir.iterdir()):
                    if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTENSIONS:
                        continue

                    if is_normal:
                        self.samples.append(
                            DatasetSample(
                                image_path=img_path,
                                split="test",
                                label_name="good",
                                is_anomaly=False,
                                label_id=0,
                                mask_path=None,
                            )
                        )
                    else:
                        # Find corresponding ground-truth mask
                        mask_path = None
                        if gt_dir.is_dir():
                            gt_defect_dir = gt_dir / defect_name
                            mask_path = MVTecValidator._find_expected_mask(gt_defect_dir, img_path)

                        self.samples.append(
                            DatasetSample(
                                image_path=img_path,
                                split="test",
                                label_name=defect_name,
                                is_anomaly=True,
                                label_id=1,
                                mask_path=mask_path,
                            )
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, Dict[str, Any]]:
        sample = self.samples[idx]

        # Load input image (ensure RGB format)
        with Image.open(sample.image_path) as raw_img:
            image = raw_img.convert("RGB")
            orig_size = image.size  # (width, height)

        # Load or synthesize ground-truth mask
        if sample.is_anomaly and sample.mask_path is not None and sample.mask_path.is_file():
            with Image.open(sample.mask_path) as raw_mask:
                mask = raw_mask.convert("L")
        else:
            # Normal sample: zero mask matching original image dimensions
            mask = Image.new("L", orig_size, 0)

        # Apply image transform
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            # Default fallback transform
            image = image.resize(self.image_size, Image.Resampling.BILINEAR)
            image_tensor = TF.to_tensor(image)

        # Apply mask transform
        if self.target_transform is not None:
            mask_tensor = self.target_transform(mask)
        else:
            # Default fallback mask transform: resize with nearest neighbor to preserve binary edges
            mask = mask.resize(self.image_size, Image.Resampling.NEAREST)
            mask_tensor = TF.to_tensor(mask)
            # Threshold to ensure strictly binary {0.0, 1.0}
            mask_tensor = (mask_tensor > 0.5).float()

        metadata = {
            "image_path": str(sample.image_path),
            "split": sample.split,
            "label_name": sample.label_name,
            "is_anomaly": sample.is_anomaly,
            "mask_path": str(sample.mask_path) if sample.mask_path else "",
            "orig_size": orig_size,
        }

        return image_tensor, sample.label_id, mask_tensor, metadata

    def get_defect_types(self) -> List[str]:
        """Return unique defect labels present in this dataset split."""
        return sorted(list({s.label_name for s in self.samples if s.is_anomaly}))

    def get_label_distribution(self) -> Dict[str, int]:
        """Return counts of each label in the current dataset split."""
        dist: Dict[str, int] = {}
        for s in self.samples:
            dist[s.label_name] = dist.get(s.label_name, 0) + 1
        return dist
