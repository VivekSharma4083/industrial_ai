#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "=========================================================="
echo " Building Industrial AI Inspector for Production (Render)"
echo "=========================================================="

# 1. Upgrade pip and install pinned dependencies with CPU PyTorch index
echo "[*] Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 2. Pre-cache torchvision ResNet-50 ImageNet weights in build phase
# This prevents downloading 102 MB during cold-start boot at runtime
echo "[*] Pre-caching ResNet-50 ImageNet weights..."
python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.DEFAULT); print('[+] Pretrained ResNet-50 cached successfully.')"

echo "[SUCCESS] Build phase completed successfully."
