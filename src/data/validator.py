"""
src/data/validator.py
=====================
Robust inspection and validation pipeline for MVTec Anomaly Detection dataset directories.
Performs path resolution, folder structure checks, image integrity checks,
and defect ground-truth mask correspondence validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ValidationResult:
    """
    Structured outcome of dataset category validation.
    """
    is_valid: bool
    category: str
    category_dir: Path
    train_good_count: int = 0
    test_good_count: int = 0
    defect_counts: Dict[str, int] = field(default_factory=dict)
    mask_counts: Dict[str, int] = field(default_factory=dict)
    missing_masks: List[Tuple[Path, Path]] = field(default_factory=list)
    corrupted_images: List[Tuple[Path, str]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_test_samples(self) -> int:
        return self.test_good_count + sum(self.defect_counts.values())

    @property
    def total_samples(self) -> int:
        return self.train_good_count + self.total_test_samples

    def summary(self) -> str:
        """Format an executive text summary of dataset validation."""
        lines = [
            f"Dataset Validation Report for category: '{self.category}'",
            f"Directory: {self.category_dir}",
            f"Overall Status: {'VALID' if self.is_valid else 'INVALID / INCOMPLETE'}",
            "-" * 55,
            f"Train (good / normal):       {self.train_good_count} samples",
            f"Test (good / normal):        {self.test_good_count} samples",
        ]
        if self.defect_counts:
            lines.append("Test Defect Types:")
            for defect, count in self.defect_counts.items():
                mask_cnt = self.mask_counts.get(defect, 0)
                lines.append(f"  - {defect:18s}: {count:4d} images | {mask_cnt:4d} masks")
        else:
            lines.append("Test Defect Types:           0 detected")

        lines.append(f"Total Images:                {self.total_samples}")
        lines.append("-" * 55)

        if self.issues:
            lines.append(f"ERRORS / BLOCKERS ({len(self.issues)}):")
            for iss in self.issues:
                lines.append(f"  [!] {iss}")

        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            for wrn in self.warnings:
                lines.append(f"  [*] {wrn}")

        if self.corrupted_images:
            lines.append(f"CORRUPTED IMAGES ({len(self.corrupted_images)}):")
            for p, err in self.corrupted_images:
                lines.append(f"  [-] {p.name}: {err}")

        return "\n".join(lines)


class MVTecValidator:
    """
    Validates the directory layout, image files, and ground-truth masks
    for an MVTec AD category folder.
    """

    def __init__(self, category_dir: Path | str, category_name: Optional[str] = None):
        self.category_dir = Path(category_dir).resolve()
        self.category_name = category_name or self.category_dir.name

    def validate(self, check_image_bytes: bool = True) -> ValidationResult:
        """
        Execute full validation of the target category directory.
        
        Args:
            check_image_bytes: If True, opens each image with Pillow to verify it
                               is readable and non-corrupt.
        """
        result = ValidationResult(
            is_valid=True,
            category=self.category_name,
            category_dir=self.category_dir,
        )

        # 1. Base category existence check
        if not self.category_dir.exists():
            result.is_valid = False
            result.issues.append(
                f"Category directory does not exist: '{self.category_dir}'. "
                f"Please ensure MVTec AD '{self.category_name}' is placed in this location."
            )
            return result

        if not self.category_dir.is_dir():
            result.is_valid = False
            result.issues.append(f"Path is a file, not a directory: '{self.category_dir}'")
            return result

        train_dir = self.category_dir / "train"
        test_dir = self.category_dir / "test"
        gt_dir = self.category_dir / "ground_truth"

        # 2. Train split inspection
        if not train_dir.is_dir():
            result.is_valid = False
            result.issues.append(f"Missing mandatory 'train/' directory under '{self.category_dir}'")
        else:
            train_good = train_dir / "good"
            if not train_good.is_dir():
                result.is_valid = False
                result.issues.append(f"Missing 'train/good/' folder for normal samples in '{train_dir}'")
            else:
                train_files = self._get_image_files(train_good)
                result.train_good_count = len(train_files)
                if result.train_good_count == 0:
                    result.is_valid = False
                    result.issues.append(f"No valid image files found in '{train_good}'")
                elif check_image_bytes:
                    self._verify_images(train_files, result)

        # 3. Test split inspection
        if not test_dir.is_dir():
            result.is_valid = False
            result.issues.append(f"Missing mandatory 'test/' directory under '{self.category_dir}'")
            return result

        test_subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
        if not test_subdirs:
            result.is_valid = False
            result.issues.append(f"Test directory is empty: '{test_dir}'")
            return result

        has_test_good = False
        defect_types = []

        for sub in sorted(test_subdirs):
            sub_name = sub.name.lower()
            files = self._get_image_files(sub)

            if sub_name == "good":
                has_test_good = True
                result.test_good_count = len(files)
                if result.test_good_count == 0:
                    result.warnings.append(f"'test/good/' exists but contains 0 images.")
                elif check_image_bytes:
                    self._verify_images(files, result)
            else:
                defect_types.append(sub.name)
                result.defect_counts[sub.name] = len(files)
                if len(files) == 0:
                    result.warnings.append(f"Defect subfolder 'test/{sub.name}' contains 0 images.")
                elif check_image_bytes:
                    self._verify_images(files, result)

        if not has_test_good:
            result.warnings.append(
                f"No 'test/good/' subfolder found in '{test_dir}'. Normal test images are recommended for ROC-AUC evaluation."
            )

        if not defect_types:
            result.is_valid = False
            result.issues.append(f"No defect folders found in '{test_dir}'. Anomaly testing requires at least one defect category.")

        # 4. Ground-truth mask inspection & correspondence
        if gt_dir.is_dir():
            for defect in defect_types:
                gt_defect_dir = gt_dir / defect
                test_defect_dir = test_dir / defect

                if not gt_defect_dir.is_dir():
                    result.is_valid = False
                    result.issues.append(
                        f"Missing ground_truth subfolder for defect '{defect}': '{gt_defect_dir}'"
                    )
                    continue

                mask_files = self._get_image_files(gt_defect_dir)
                result.mask_counts[defect] = len(mask_files)

                # Check 1-to-1 correspondence between test defect images and masks
                test_images = self._get_image_files(test_defect_dir)
                for test_img in test_images:
                    expected_mask = self._find_expected_mask(gt_defect_dir, test_img)
                    if expected_mask is None or not expected_mask.is_file():
                        candidate = gt_defect_dir / f"{test_img.stem}_mask{test_img.suffix}"
                        result.missing_masks.append((test_img, candidate))
                        result.is_valid = False
                        result.issues.append(
                            f"Missing defect mask for '{test_img.relative_to(self.category_dir)}'. Expected: '{candidate.name}'"
                        )
                    elif check_image_bytes:
                        # Verify mask dimensions match image dimensions
                        self._verify_mask_dimensions(test_img, expected_mask, result)
        else:
            if defect_types:
                result.warnings.append(
                    f"'ground_truth/' directory not found under '{self.category_dir}'. "
                    f"Pixel-level localization evaluation (PRO-AUC) will be unavailable."
                )

        if result.corrupted_images:
            result.is_valid = False

        return result

    @staticmethod
    def _get_image_files(directory: Path) -> List[Path]:
        """Collect all files matching image extensions in directory."""
        if not directory.is_dir():
            return []
        return sorted([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])

    @staticmethod
    def _find_expected_mask(gt_defect_dir: Path, test_img: Path) -> Optional[Path]:
        """
        Locate matching mask. Official MVTec format uses `<stem>_mask.png`,
        but also check `<stem>.png` fallback.
        """
        # Primary: 000_mask.png
        cand1 = gt_defect_dir / f"{test_img.stem}_mask{test_img.suffix}"
        if cand1.is_file():
            return cand1
        # Secondary: 000.png
        cand2 = gt_defect_dir / test_img.name
        if cand2.is_file():
            return cand2
        return None

    @staticmethod
    def _verify_images(files: List[Path], result: ValidationResult) -> None:
        """Verify image headers and readability with Pillow."""
        for img_path in files:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                # Re-open to confirm full data decodability (verify() only checks headers)
                with Image.open(img_path) as img:
                    img.load()
            except Exception as exc:
                result.corrupted_images.append((img_path, str(exc)))
                result.issues.append(f"Corrupt image at '{img_path.name}': {exc}")

    @staticmethod
    def _verify_mask_dimensions(img_path: Path, mask_path: Path, result: ValidationResult) -> None:
        """Verify that mask dimensions match the test image."""
        try:
            with Image.open(img_path) as img, Image.open(mask_path) as mask:
                if img.size != mask.size:
                    result.issues.append(
                        f"Dimension mismatch: Image '{img_path.name}' size {img.size} != Mask '{mask_path.name}' size {mask.size}"
                    )
                    result.is_valid = False
        except Exception as exc:
            result.corrupted_images.append((mask_path, f"Failed reading mask: {exc}"))


def validate_mvtec_category(category_dir: Path | str, category_name: Optional[str] = None) -> ValidationResult:
    """Convenience function to validate an MVTec category."""
    validator = MVTecValidator(category_dir, category_name=category_name)
    return validator.validate(check_image_bytes=True)
