# Feature-Reconstruction Visual Anomaly Detection Architecture

## 1. Overview & Conceptual Rationale

Traditional supervised defect detection requires thousands of annotated bounding boxes or pixel segmentation masks across all failure modes. In high-reliability industrial manufacturing, defects are rare, unpredictable, and cost-prohibitive to curate at scale.

**Industrial AI Inspector** utilizes **unsupervised / one-class anomaly detection** based on **deep intermediate feature reconstruction**.

```
Input Image (H, W, 3)
         │
         ▼
[ImageNet Preprocessing & Standardization]
  (Resize to 224x224, Normalized with ImageNet μ & σ)
         │
         ▼
[Frozen ResNet-50 Feature Extractor]
  (Captures mid-level spatial features from layer2: 512 channels, 28x28 grid)
         │
         ▼ Original Feature Map f ∈ ℝ^(B × 512 × 28 × 28)
         ├─────────────────────────────────────────┐
         ▼                                         │
[Feature Reconstruction Autoencoder]               │
  (Trained strictly on defect-free samples)        │
         │                                         │
         ▼ Reconstructed Feature Map f̂             ▼
  ┌────────────────────────────────────────────────┴────────┐
  │ Channel-wise Spatial Euclidean Distance: e = ||f - f̂||₂ │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼ Raw Error Grid (28x28)
              [Gaussian Spatial Smoothing (σ=4.0)]
                             │
                             ▼
       [Bilinear Upsampling to Original Resolution (H, W)]
                             │
                             ▼
         [Safe Variance-Guarded Normalization to [0, 1]]
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
      [Pixel Anomaly Heatmap]  [Image Anomaly Score]
```

---

## 2. Why Intermediate Feature Reconstruction?

| Aspect | Pixel-Space Autoencoders | Feature-Space Reconstruction (Ours) |
| :--- | :--- | :--- |
| **Sensitivity** | Vulnerable to irrelevant pixel noise, illumination shifts, and camera sensor artifacts. | Robust to lighting shifts; focuses on structural semantics and texture continuity. |
| **Output Fidelity** | Prone to blurry pixel reconstructions due to $L_2$ averaging. | Features reside in a semantic manifold where deviations represent true structural defects. |
| **Resolution Trade-off** | Full-resolution autoencoders require massive parameter counts and compute. | Intermediate features (e.g. `layer2` at $28 \times 28$) provide rich spatial localization with only ~1.0M autoencoder parameters (~4 MB). |

---

## 3. Component Specifications

### 3.1 Frozen ResNet50 Feature Extractor (`src/models/feature_extractor.py`)
- **Backbone:** Standard torchvision ResNet-50.
- **Target Layer:** `layer2` (512 channels, $8\times$ downsampled grid) by default; configurable to `layer3` (1024 channels, $16\times$).
- **State:** Completely frozen (`eval()` mode, `requires_grad = False`).
- **Input Normalization:** ImageNet standard ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Execution Target:** Optimized for CPU inference.

### 3.2 Convolutional Reconstruction Autoencoder (`src/models/reconstruction.py`)
- **Input / Output Dimension:** $(B, C, H, W)$ where $C=512, H=28, W=28$ for 224x224 inputs.
- **Encoder:**
  - $1 \times 1$ Conv ($C \to C/2$) + BatchNorm2d + LeakyReLU(0.2)
  - $3 \times 3$ Conv ($C/2 \to C/4$) + BatchNorm2d + LeakyReLU(0.2), padding=1
- **Latent Bottleneck:**
  - $3 \times 3$ Conv ($C/4 \to C/4$) + BatchNorm2d + LeakyReLU(0.2), padding=1
- **Decoder:**
  - $3 \times 3$ Conv ($C/4 \to C/2$) + BatchNorm2d + LeakyReLU(0.2), padding=1
  - $1 \times 1$ Conv ($C/2 \to C$)
- **Parameter Footprint:** ~1,001,472 trainable parameters (~4 MB uncompressed checkpoint).

### 3.3 Anomaly Scoring & Map Generator (`src/models/anomaly_detector.py`)
- **Pixel Discrepancy Metric:** $e(h, w) = \sqrt{\sum_{c=1}^C (f_{c, h, w} - \hat{f}_{c, h, w})^2}$
- **Spatial Smoothing:** Pure PyTorch 2D Gaussian filter ($\sigma=4.0, k=7$) to smooth patch-boundary discretization.
- **Upsampling:** Bilinear interpolation to the original image dimensions $(H_{orig}, W_{orig})$.
- **Safe Normalization:** 
  $$\tilde{e} = \frac{e - \min(e)}{\max(e) - \min(e)}$$
  If $\max(e) - \min(e) < 10^{-8}$ (flat zero-variance map), safe fallback returns an all-zero map to avoid divide-by-zero or NaN artifacts.
- **Image-Level Score Aggregation:**
  - `max`: Highest localized pixel anomaly value (default; sensitive to small scratches and cracks).
  - `top_k`: Average of top 1% anomaly pixels (resilient against single-pixel noise).
  - `mean`: Average spatial error across entire image.

---

## 4. Training Assumptions & Segregation Rules

1. **Defect-Free Training Split Only:**  
   Training strictly ingests normal samples from `train/good/`.
2. **Strict Test Segregation:**  
   The test split (`test/good/` and `test/<defect_type>/`) is never observed during training or model selection.
3. **Loss Formulations:**
   - **MSE:** $\mathcal{L}_{MSE} = \frac{1}{CHW} \sum (f - \hat{f})^2$
   - **Cosine Distance:** $\mathcal{L}_{cos} = 1 - \frac{f \cdot \hat{f}}{\|f\| \|\hat{f}\|}$
   - **Combined (Recommended):** $\mathcal{L}_{comb} = \mathcal{L}_{MSE} + 0.5 \cdot \mathcal{L}_{cos}$

---

## 5. Scoring Disclaimer & Decision Boundaries

> [!WARNING]  
> The raw anomaly score output by `FeatureAnomalyDetector` is a reconstruction distance metric, **not a calibrated class probability**.  
> In industrial deployment, an operational threshold $\tau$ must be selected using a validation dataset via Receiver Operating Characteristic (ROC) or Precision-Recall (PR) analysis based on factory tolerance for False Positive Rate (FPR) vs. False Negative Rate (FNR).

---

## 6. Execution Commands

### CPU Smoke Test
Run end-to-end verification through feature extraction, autoencoding, and scoring on synthetic tensors:
```powershell
python scripts/smoke_test_models.py
```

### Synthetic Training Smoke Run
Run a complete 1-epoch training and checkpoint serialization cycle without real data:
```powershell
python scripts/train.py --synthetic-run
```

### Full Training (When MVTec AD is Present)
```powershell
python scripts/train.py --category bottle --epochs 20 --batch-size 8 --layer layer2
```
Checkpoints will be saved to `artifacts/checkpoints/reconstruction_bottle.pth` (automatically ignored by Git).
