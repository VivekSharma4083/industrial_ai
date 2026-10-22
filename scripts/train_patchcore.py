#!/usr/bin/env python3
"""
scripts/train_patchcore.py
==========================
Fits the PatchCore memory bank on normal (defect-free) MVTec AD samples.
Applies Johnson-Lindenstrauss random projection and greedy k-Center minimax
coreset subsampling, then serializes the model to disk.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import DatasetConfig
from src.data.dataset import MVTecDataset
from src.models.patchcore import (
    PatchCoreConfig,
    PatchEmbeddingExtractor,
    PatchCoreDetector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit PatchCore Memory Bank on Normal Samples")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec category name (default: bottle)")
    parser.add_argument("--dataset-root", type=str, default=None, help="Root path to MVTec dataset")
    parser.add_argument("--layers", type=str, default="layer2,layer3", help="Comma-separated ResNet50 layers (default: layer2,layer3)")
    parser.add_argument("--coreset-ratio", type=float, default=0.01, help="Coreset subsampling ratio (default: 0.01 = 1%)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for feature extraction (default: 8)")
    parser.add_argument("--device", type=str, default="cpu", help="Target device (default: cpu)")
    parser.add_argument("--output-dir", type=str, default="artifacts/checkpoints", help="Directory to save checkpoint")
    parser.add_argument("--synthetic-run", action="store_true", help="Run 1-second smoke test on synthetic tensors")
    parser.add_argument("--smoke-run", action="store_true", help="Run smoke test on 5 real images before full fitting")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layers = tuple(l.strip() for l in args.layers.split(","))

    print("=" * 68)
    print(" Industrial AI Inspector -- PatchCore Memory Bank Construction")
    print("=" * 68)
    print(f"[*] Target Category:         {args.category}")
    print(f"[*] Backbone Layers:         {layers}")
    print(f"[*] Coreset Ratio:           {args.coreset_ratio:.4f} ({args.coreset_ratio*100:.1f}%)")
    print(f"[*] Target Device:           {args.device}")
    print("-" * 68)

    # 1. Synthetic smoke run
    if args.synthetic_run:
        print("[*] Running synthetic tensor smoke test (no dataset required)...")
        extractor = PatchEmbeddingExtractor(layers=layers, pretrained=False, device=args.device)
        cfg = PatchCoreConfig(layers=layers, coreset_ratio=args.coreset_ratio, device=args.device)
        detector = PatchCoreDetector(extractor, config=cfg)

        dummy_data = torch.rand(4, 3, 224, 224)
        loader = DataLoader(TensorDataset(dummy_data), batch_size=2)
        stats = detector.fit(loader)

        out_path = PROJECT_ROOT / args.output_dir / f"synthetic_patchcore_{args.category}.pth"
        detector.save(out_path)
        print(f"[SUCCESS] Synthetic smoke run completed! Coreset vectors: {stats['coreset_patches']}, saved to {out_path.name}")
        return 0

    # 2. Check dataset directory
    config_dataset = DatasetConfig(dataset_root=args.dataset_root, category=args.category)
    if not config_dataset.category_dir.exists():
        print(f"[!] Dataset folder not found: {config_dataset.category_dir}")
        print("    Please download or extract the dataset first.")
        return 1

    # 3. Load normal training images
    print("[*] Loading defect-free training split (train/good/)...")
    try:
        train_dataset = MVTecDataset(
            category_dir=config_dataset.category_dir,
            split="train",
            image_size=(224, 224),
        )
    except Exception as exc:
        print(f"[!] Error loading training dataset: {exc}")
        return 1

    print(f"[+] Found {len(train_dataset)} normal training images.")

    if args.smoke_run:
        print("[*] Executing real-image smoke test on first 5 samples...")
        train_dataset = Subset(train_dataset, range(min(5, len(train_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 4. Initialize PatchCore Extractor and Detector
    print("[*] Initializing frozen ResNet-50 multi-scale patch extractor...")
    extractor = PatchEmbeddingExtractor(
        layers=layers,
        pretrained=True,
        device=args.device,
        image_size=(224, 224),
        target_spatial_size=(28, 28),
    )
    print(f"    - Extracted embedding dimension: {extractor.embedding_dim} (combined across {layers})")

    cfg = PatchCoreConfig(
        layers=layers,
        coreset_ratio=args.coreset_ratio,
        device=args.device,
    )
    detector = PatchCoreDetector(extractor, config=cfg)

    # 5. Fit memory bank with greedy coreset subsampling
    print(f"[*] Extracting normal patch embeddings and computing coreset (ratio: {args.coreset_ratio})...")
    stats = detector.fit(train_loader)

    print("\n" + "=" * 68)
    print(" PATCHCORE MEMORY BANK CONSTRUCTION SUMMARY")
    print("=" * 68)
    print(f" Processed Training Images:  {stats['train_samples']}")
    print(f" Total Raw Patches:          {stats['raw_patches']:,}")
    print(f" Subsampled Coreset Patches: {stats['coreset_patches']:,} ({stats['coreset_ratio']*100:.1f}%)")
    print(f" Patch Vector Dimension:     {stats['embedding_dim']}")
    print(f" Memory Bank Size on Disk:   {stats['memory_bank_mb']} MB")
    print(f" Total Fitting Time:         {stats['total_fit_time_s']:.2f} s (Coreset k-Center: {stats['coreset_time_s']:.2f} s)")
    print("-" * 68)

    # Training score baseline distribution
    train_scores = np.asarray(detector.train_scores, dtype=np.float64)
    if len(train_scores) > 0:
        mu = float(np.mean(train_scores))
        std = float(np.std(train_scores))
        p99 = float(np.percentile(train_scores, 99))
        print(" Normal Training Score Baseline (Unbiased Calibration):")
        print(f"   Mean (mu):                {mu:.6f}")
        print(f"   Std Dev (sigma):          {std:.6f}")
        print(f"   99th Percentile:          {p99:.6f}")
        print(f"   mu + 3*sigma:             {mu + 3*std:.6f}")
        print(f"   Range:                    {np.min(train_scores):.6f} .. {np.max(train_scores):.6f}")

    # 6. Save checkpoint
    ckpt_name = f"patchcore_{args.category}_smoke.pth" if args.smoke_run else f"patchcore_{args.category}.pth"
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ckpt_name
    detector.save(out_path)

    print("-" * 68)
    print(f"[SUCCESS] PatchCore model checkpoint saved to: {out_path}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
