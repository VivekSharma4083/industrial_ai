#!/usr/bin/env python3
"""
scripts/train.py
================
Command-line utility to train the Feature Reconstruction Autoencoder
on normal samples of an MVTec AD category.
"""

import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import DatasetConfig
from src.data.dataset import MVTecDataset
from src.models.feature_extractor import ResNetFeatureExtractor
from src.models.reconstruction import FeatureReconstructionAutoencoder
from src.models.trainer import ReconstructionTrainer, TrainingConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Feature Reconstruction Autoencoder on MVTec Normal Samples")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec category name (default: bottle)")
    parser.add_argument("--dataset-root", type=str, default=None, help="Root path to MVTec dataset")
    parser.add_argument("--layer", type=str, default="layer2", help="ResNet50 feature layer (default: layer2)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--loss-type", type=str, default="mse", choices=["mse", "cosine", "combined"], help="Loss type")
    parser.add_argument("--no-pretrained", action="store_true", help="Use untrained ResNet50 (for offline tests)")
    parser.add_argument("--output-dir", type=str, default="artifacts/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--synthetic-run", action="store_true", help="Run 1-epoch smoke test on synthetic tensors")

    args = parser.parse_args()

    print("=" * 65)
    print(" Industrial AI Inspector -- Feature Reconstruction Training")
    print("=" * 65)
    print(f"[*] Target Category:       {args.category}")
    print(f"[*] Target ResNet Layer:   {args.layer}")
    print(f"[*] Epochs:                {args.epochs}")
    print(f"[*] Batch Size:            {args.batch_size}")
    print(f"[*] Loss Function:         {args.loss_type}")
    print(f"[*] Pretrained Weights:    {not args.no_pretrained}")
    print("-" * 65)

    device = "cpu"
    print(f"[*] Device Target:         {device}")

    # Check synthetic test flag
    if args.synthetic_run:
        print("[*] Executing synthetic tensor smoke test (no dataset required)...")
        feature_extractor = ResNetFeatureExtractor(
            layer_name=args.layer,
            pretrained=False,
            device=device,
        )
        recon_model = FeatureReconstructionAutoencoder(
            in_channels=feature_extractor.out_channels,
        )
        config = TrainingConfig(
            epochs=1,
            batch_size=2,
            lr=args.lr,
            loss_type=args.loss_type,
            device=device,
            output_dir=PROJECT_ROOT / args.output_dir,
            checkpoint_name=f"synthetic_reconstruction_{args.category}.pth",
        )
        trainer = ReconstructionTrainer(feature_extractor, recon_model, config=config)

        # Create small synthetic batch: 4 samples of shape (3, 224, 224)
        dummy_data = torch.rand(4, 3, 224, 224)
        synthetic_loader = DataLoader(TensorDataset(dummy_data), batch_size=2)

        results = trainer.fit(synthetic_loader, save_best=True)
        print(f"[SUCCESS] Synthetic smoke run completed! Final loss: {results['final_train_loss']:.6f}")
        return 0

    # Real dataset path checking
    config_dataset = DatasetConfig(dataset_root=args.dataset_root, category=args.category)
    if not config_dataset.category_dir.exists():
        print(f"[!] STATUS: Dataset category folder NOT found at:")
        print(f"    '{config_dataset.category_dir}'")
        print()
        print("    Training requires defect-free training images at 'train/good/'.")
        print(f"    Please download or extract the '{args.category}' category following docs/dataset_guide.md.")
        print()
        print("    Tip: To test the end-to-end training pipeline without the dataset, run:")
        print("    python scripts/train.py --synthetic-run")
        print("=" * 65)
        return 0

    # Build dataset & loader
    try:
        train_dataset = MVTecDataset(
            category_dir=config_dataset.category_dir,
            split="train",
            image_size=(224, 224),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Single-process for CPU stability
        )
        print(f"[+] Loaded {len(train_dataset)} normal training samples.")
    except Exception as exc:
        print(f"[!] Error preparing training dataset: {exc}")
        return 1

    # Initialize models
    try:
        feature_extractor = ResNetFeatureExtractor(
            layer_name=args.layer,
            pretrained=not args.no_pretrained,
            device=device,
        )
        recon_model = FeatureReconstructionAutoencoder(
            in_channels=feature_extractor.out_channels,
        )
    except Exception as exc:
        print(f"[!] Model initialization failed: {exc}")
        return 1

    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_type=args.loss_type,
        device=device,
        output_dir=PROJECT_ROOT / args.output_dir,
        checkpoint_name=f"reconstruction_{args.category}.pth",
    )

    trainer = ReconstructionTrainer(feature_extractor, recon_model, config=train_cfg)
    results = trainer.fit(train_loader, save_best=True)

    print("=" * 65)
    print(f"[SUCCESS] Training completed! Model saved to: {results['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
