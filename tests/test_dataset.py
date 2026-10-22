"""
tests/test_dataset.py
=====================
Automated tests for MVTec dataset configuration, directory structure validation,
PyTorch dataset loading, image integrity checks, and ground-truth mask correspondence.
Uses synthetic temporary image fixtures so tests execute deterministically without
requiring external dataset downloads.
"""

from pathlib import Path
import pytest
from PIL import Image, ImageDraw
import torch

from src.data.config import DatasetConfig
from src.data.validator import MVTecValidator, validate_mvtec_category
from src.data.dataset import MVTecDataset
from scripts.inspect_dataset import generate_contact_sheet


@pytest.fixture
def synthetic_mvtec_category(tmp_path: Path) -> Path:
    """
    Construct a complete synthetic MVTec AD category hierarchy in a temp directory.
    Includes:
      - train/good: 2 normal images
      - test/good: 1 normal test image (no mask expected)
      - test/broken_large: 1 defective image
      - test/contamination: 1 defective image
      - ground_truth/broken_large: 1 binary mask
      - ground_truth/contamination: 1 binary mask
    """
    category_dir = tmp_path / "bottle"

    # Create subdirectories
    train_good = category_dir / "train" / "good"
    test_good = category_dir / "test" / "good"
    test_broken = category_dir / "test" / "broken_large"
    test_contam = category_dir / "test" / "contamination"
    gt_broken = category_dir / "ground_truth" / "broken_large"
    gt_contam = category_dir / "ground_truth" / "contamination"

    for d in [train_good, test_good, test_broken, test_contam, gt_broken, gt_contam]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Normal training images (RGB)
    img1 = Image.new("RGB", (256, 256), color=(200, 200, 200))
    img1.save(train_good / "000.png")
    img2 = Image.new("RGB", (256, 256), color=(210, 210, 210))
    img2.save(train_good / "001.png")

    # 2. Normal test image (RGB, no mask)
    test_norm = Image.new("RGB", (256, 256), color=(205, 205, 205))
    test_norm.save(test_good / "000.png")

    # 3. Defective test images and corresponding masks
    # broken_large
    def_img1 = Image.new("RGB", (256, 256), color=(180, 180, 180))
    draw1 = ImageDraw.Draw(def_img1)
    draw1.rectangle([50, 50, 100, 100], fill=(50, 50, 50))  # dark defect region
    def_img1.save(test_broken / "000.png")

    mask1 = Image.new("L", (256, 256), color=0)
    mask1_draw = ImageDraw.Draw(mask1)
    mask1_draw.rectangle([50, 50, 100, 100], fill=255)  # white anomaly mask
    mask1.save(gt_broken / "000_mask.png")

    # contamination
    def_img2 = Image.new("RGB", (256, 256), color=(190, 190, 190))
    draw2 = ImageDraw.Draw(def_img2)
    draw2.ellipse([120, 120, 160, 160], fill=(30, 30, 30))
    def_img2.save(test_contam / "000.png")

    mask2 = Image.new("L", (256, 256), color=0)
    mask2_draw = ImageDraw.Draw(mask2)
    mask2_draw.ellipse([120, 120, 160, 160], fill=255)
    mask2.save(gt_contam / "000_mask.png")

    return category_dir


def test_dataset_config_paths(tmp_path: Path):
    """Verify DatasetConfig path resolution and property derivations."""
    cfg = DatasetConfig(dataset_root=tmp_path / "data", category="bottle")
    assert cfg.category == "bottle"
    assert cfg.category_dir == (tmp_path / "data" / "bottle").resolve()
    assert cfg.train_dir == cfg.category_dir / "train"
    assert cfg.test_dir == cfg.category_dir / "test"
    assert cfg.ground_truth_dir == cfg.category_dir / "ground_truth"

    info = cfg.to_dict()
    assert info["category"] == "bottle"
    assert "exists" in info


def test_missing_dataset_detection(tmp_path: Path):
    """Verify that absent dataset directories trigger clean, actionable errors."""
    non_existent = tmp_path / "does_not_exist"
    validator = MVTecValidator(non_existent)
    result = validator.validate()

    assert not result.is_valid
    assert len(result.issues) > 0
    assert "does not exist" in result.issues[0]

    # Check PyTorch dataset raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="not found"):
        MVTecDataset(category_dir=non_existent, split="train")


def test_structure_validation_success(synthetic_mvtec_category: Path):
    """Verify full validation passes on a well-formed category structure."""
    result = validate_mvtec_category(synthetic_mvtec_category, category_name="bottle")

    assert result.is_valid
    assert result.train_good_count == 2
    assert result.test_good_count == 1
    assert result.defect_counts.get("broken_large") == 1
    assert result.defect_counts.get("contamination") == 1
    assert result.mask_counts.get("broken_large") == 1
    assert result.mask_counts.get("contamination") == 1
    assert len(result.missing_masks) == 0
    assert len(result.corrupted_images) == 0

    summary_str = result.summary()
    assert "VALID" in summary_str
    assert "broken_large" in summary_str


def test_normal_samples_without_defect_masks(synthetic_mvtec_category: Path):
    """
    Ensure normal training and test images correctly produce zero-masks
    without raising missing-mask errors.
    """
    train_dataset = MVTecDataset(
        synthetic_mvtec_category,
        split="train",
        image_size=(224, 224),
    )
    assert len(train_dataset) == 2

    img_tensor, label_id, mask_tensor, meta = train_dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    assert label_id == 0
    assert not meta["is_anomaly"]
    assert meta["label_name"] == "good"
    assert mask_tensor.shape == (1, 224, 224)
    # Entire mask must be zero for normal sample
    assert torch.sum(mask_tensor) == 0.0


def test_test_split_and_mask_correspondence(synthetic_mvtec_category: Path):
    """
    Verify test split loads both normal and defective samples, accurately
    associating ground-truth binary masks with defect categories.
    """
    test_dataset = MVTecDataset(
        synthetic_mvtec_category,
        split="test",
        image_size=(224, 224),
    )
    # 1 normal + 1 broken_large + 1 contamination = 3 total
    assert len(test_dataset) == 3

    labels = test_dataset.get_label_distribution()
    assert labels["good"] == 1
    assert labels["broken_large"] == 1
    assert labels["contamination"] == 1

    # Find the broken_large sample
    broken_sample = None
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        if item[3]["label_name"] == "broken_large":
            broken_sample = item
            break

    assert broken_sample is not None
    img, lbl_id, mask, meta = broken_sample
    assert lbl_id == 1
    assert meta["is_anomaly"] is True
    assert mask.shape == (1, 224, 224)
    # The mask contains the defect rectangle, so sum must be strictly positive
    assert torch.sum(mask) > 0.0
    # Values must be binary (0.0 or 1.0)
    unique_vals = set(torch.unique(mask).tolist())
    assert unique_vals.issubset({0.0, 1.0})


def test_missing_mask_detection(synthetic_mvtec_category: Path):
    """Verify that removing a defect mask causes validation failure."""
    mask_to_delete = synthetic_mvtec_category / "ground_truth" / "broken_large" / "000_mask.png"
    mask_to_delete.unlink()

    result = validate_mvtec_category(synthetic_mvtec_category)
    assert not result.is_valid
    assert len(result.missing_masks) == 1
    assert "Missing defect mask" in result.summary()


def test_corrupted_image_detection(synthetic_mvtec_category: Path):
    """Verify that non-decodable byte files are caught and flagged as corrupt."""
    bad_file = synthetic_mvtec_category / "train" / "good" / "099_corrupted.png"
    bad_file.write_bytes(b"NOT_A_VALID_PNG_FILE_HEADER")

    result = validate_mvtec_category(synthetic_mvtec_category)
    assert not result.is_valid
    assert len(result.corrupted_images) == 1
    assert result.corrupted_images[0][0].name == "099_corrupted.png"


def test_contact_sheet_generation(synthetic_mvtec_category: Path, tmp_path: Path):
    """Verify contact sheet generation renders properly on synthetic dataset."""
    result = validate_mvtec_category(synthetic_mvtec_category)
    out_file = tmp_path / "artifacts" / "test_sheet.png"

    saved_path = generate_contact_sheet(synthetic_mvtec_category, result, out_file)
    assert saved_path is not None
    assert saved_path.is_file()

    with Image.open(saved_path) as sheet:
        assert sheet.width > 300
        assert sheet.height > 200
