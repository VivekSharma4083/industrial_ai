#!/usr/bin/env python3
"""
scripts/inspect_dataset.py
==========================
Comprehensive diagnostic utility to inspect, validate, and summarize an MVTec AD
category directory, and generate a representative sample contact sheet in artifacts/.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import DatasetConfig
from src.data.validator import MVTecValidator, ValidationResult


def generate_contact_sheet(
    category_dir: Path,
    result: ValidationResult,
    output_path: Path,
    thumb_size: int = 180,
) -> Optional[Path]:
    """
    Generate a visual contact sheet with representative normal and defective samples,
    including their ground-truth defect masks.
    """
    samples_to_render: List[Dict[str, any]] = []

    # 1. Representative Normal Train sample
    train_good_dir = category_dir / "train" / "good"
    if train_good_dir.is_dir():
        train_images = MVTecValidator._get_image_files(train_good_dir)
        if train_images:
            samples_to_render.append({
                "title": "Train: Normal (good)",
                "img_path": train_images[0],
                "mask_path": None,
            })

    # 2. Representative Normal Test sample
    test_good_dir = category_dir / "test" / "good"
    if test_good_dir.is_dir():
        test_good_images = MVTecValidator._get_image_files(test_good_dir)
        if test_good_images:
            samples_to_render.append({
                "title": "Test: Normal (good)",
                "img_path": test_good_images[0],
                "mask_path": None,
            })

    # 3. Representative samples for each defect type
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"

    if test_dir.is_dir():
        for defect in sorted(result.defect_counts.keys()):
            defect_dir = test_dir / defect
            defect_imgs = MVTecValidator._get_image_files(defect_dir)
            if defect_imgs:
                first_img = defect_imgs[0]
                mask_file = None
                if gt_dir.is_dir():
                    mask_file = MVTecValidator._find_expected_mask(gt_dir / defect, first_img)

                samples_to_render.append({
                    "title": f"Test: {defect}",
                    "img_path": first_img,
                    "mask_path": mask_file,
                })

    if not samples_to_render:
        return None

    # Calculate layout
    num_samples = len(samples_to_render)
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols

    cell_w = thumb_size * 2 + 30  # Image + Mask + Padding
    cell_h = thumb_size + 45      # Image + Title
    margin = 20

    sheet_w = margin * 2 + cols * cell_w
    sheet_h = margin * 2 + rows * cell_h + 40  # Header banner

    contact_sheet = Image.new("RGB", (sheet_w, sheet_h), color=(15, 20, 30))
    draw = ImageDraw.Draw(contact_sheet)

    # Header text
    draw.text((margin, 12), f"MVTec AD Dataset Overview: '{result.category}' category", fill=(245, 158, 11))

    for idx, item in enumerate(samples_to_render):
        r = idx // cols
        c = idx % cols
        x_cell = margin + c * cell_w
        y_cell = margin + 40 + r * cell_h

        # Title
        draw.text((x_cell, y_cell), item["title"][:28], fill=(220, 225, 235))

        # Render image
        try:
            with Image.open(item["img_path"]) as img:
                img_thumb = img.convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)
                contact_sheet.paste(img_thumb, (x_cell, y_cell + 20))
        except Exception:
            pass

        # Render mask
        mask_x = x_cell + thumb_size + 10
        if item["mask_path"] and item["mask_path"].is_file():
            try:
                with Image.open(item["mask_path"]) as mask:
                    mask_thumb = mask.convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.NEAREST)
                    contact_sheet.paste(mask_thumb, (mask_x, y_cell + 20))
            except Exception:
                pass
        else:
            # Draw placeholder for normal sample mask (no defect)
            draw.rectangle(
                [mask_x, y_cell + 20, mask_x + thumb_size, y_cell + 20 + thumb_size],
                outline=(50, 60, 80),
                fill=(20, 25, 35),
            )
            draw.text((mask_x + 25, y_cell + 20 + thumb_size // 2 - 8), "[No Defect Mask]", fill=(120, 130, 145))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.save(output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="MVTec AD Dataset Inspection Utility")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to MVTec dataset root directory (default: data/mvtec_anomaly_detection)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="Category name to inspect (default: bottle)",
    )
    parser.add_argument(
        "--generate-visuals",
        action="store_true",
        default=True,
        help="Generate contact sheet visualization in artifacts/ if dataset is present",
    )
    args = parser.parse_args()

    config = DatasetConfig(dataset_root=args.dataset_root, category=args.category)

    print("=" * 65)
    print(" Industrial AI Inspector -- Dataset Inspection & Validation")
    print("=" * 65)
    print(f"[*] Target Category:  {config.category}")
    print(f"[*] Target Directory: {config.category_dir}")
    print("-" * 65)

    if not config.category_dir.exists():
        print("[!] STATUS: Dataset category folder NOT found.")
        print("    The MVTec AD dataset has not been placed in the target directory yet.")
        print()
        print("    Setup Instructions:")
        print(f"    1. Download the '{config.category}' category from MVTec:")
        print("       https://www.mvtec.com/company/research/datasets/mvtec-ad")
        print(f"    2. Extract it so that the following structure exists:")
        print(f"       {config.category_dir}/")
        print("           train/good/")
        print("           test/good/")
        print("           test/<defect_type>/")
        print("           ground_truth/<defect_type>/")
        print()
        print("    Note: MVTec AD is licensed for non-commercial research (CC BY-NC-SA 4.0).")
        print("=" * 65)
        # Graceful diagnostic exit code 0 so automated scripts can distinguish clean missing data reporting
        return 0

    validator = MVTecValidator(config.category_dir, category_name=config.category)
    result = validator.validate(check_image_bytes=True)

    print(result.summary())
    print("-" * 65)

    if args.generate_visuals and result.is_valid:
        artifacts_dir = PROJECT_ROOT / "artifacts" / "contact_sheets"
        out_sheet = artifacts_dir / f"mvtec_{config.category}_overview.png"
        sheet_path = generate_contact_sheet(config.category_dir, result, out_sheet)
        if sheet_path:
            print(f"[+] Saved visual contact sheet to: {sheet_path}")

    print("=" * 65)
    if result.is_valid:
        print(f"[SUCCESS] Dataset '{config.category}' is complete, verified, and ready for model training.")
        return 0
    else:
        print(f"[WARNING] Dataset '{config.category}' has validation errors or missing components. Review report above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
