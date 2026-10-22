# Phase 5 Technical Report: PatchCore Anomaly Detection Baseline

## 1. Executive Summary & Objective

Phase 5 establishes a **PatchCore-style anomaly detection baseline** for the Industrial AI Inspector project, benchmarked against the existing **Feature Reconstruction Autoencoder** on the genuine MVTec AD `bottle` category.

Both models were evaluated on the identical held-out test split (83 test images, 63 ground-truth masks) under identical CPU hardware constraints (Intel UHD Graphics 620, PyTorch 2.14.1+cpu on Windows 64-bit).

### Key Empirical Findings:
- **Image-Level AUROC:** **100.00% (1.0000)** for PatchCore vs **99.68% (0.9968)** for Feature Reconstruction.
- **Pixel-Level Localization AUROC:** **97.97% (0.9797)** for PatchCore vs **96.41% (0.9641)** for Feature Reconstruction ($+1.56\%$ gain).
- **Separation Margin:** In PatchCore, the highest normal test image score is **10.6313**, whereas the lowest anomalous defect score is **13.1616**—yielding a **$+2.53$ clean separation margin** across all 83 test images with zero class overlap.
- **CPU Inference Latency:** **224.46 ms** per image (~4.46 FPS) for PatchCore vs **160.18 ms** (~6.24 FPS) for Feature Reconstruction.
- **Model Footprint on Disk:** **9.6 MB** (1,639 coreset vectors) for PatchCore vs **11.5 MB** (1.00M neural network weights) for Feature Reconstruction.

---

## 2. Architecture & Theoretical Rationale

### Why PatchCore?
In Phase 4, the convolutional feature-reconstruction autoencoder exhibited a known limitation: convolutional bottleneck blur and elevated residual sensitivity along high-frequency glass refraction contours, resulting in 3 missed low-contrast contaminations (`contamination/001.png`, `012.png`, `019.png`).

**PatchCore** (Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022) replaces the neural reconstruction bottleneck with a **non-parametric memory bank of localized patch representations**:
1. **Multi-Scale Neighborhood Aggregation:**
   Instead of using raw point features, PatchCore applies local 2D average pooling (`AvgPool2d(k=3, s=1, p=1)`) to activations from ResNet-50 `layer2` (512-dim, $28 \times 28$) and `layer3` (1024-dim, $14 \times 14$).
2. **Spatial Feature Alignment:**
   `layer3` is bilinearly upsampled to match the $28 \times 28$ grid of `layer2`, concatenating along the channel axis to form a **1536-dimensional patch vector** at each of the $28 \times 28 = 784$ spatial locations.
3. **Greedy k-Center Coreset Subsampling:**
   A full industrial training set produces hundreds of thousands of patch embeddings ($209 \text{ images} \times 784 = 163,856 \text{ vectors}$). To make nearest-neighbor search viable on CPU without sacrificing coverage, greedy minimax facility location selects a **1.0% representative coreset** ($M = 1,639$ vectors).
4. **Johnson-Lindenstrauss Random Projection:**
   To accelerate coreset selection on CPU, patch vectors are projected into $D_{\text{proj}} = 128$ dimensions via a deterministic Gaussian random projection matrix $R \in \mathbb{R}^{1536 \times 128}$ during minimax distance updates.

---

## 3. Strict Data Isolation & Unbiased Protocol

- **Memory Bank Fitting Split:** Exclusively `data/mvtec_anomaly_detection/bottle/train/good/` ($N = 209$ defect-free images).
- **Held-Out Test Split:** All 83 test images (20 normal, 20 broken_large, 22 broken_small, 21 contamination) remained strictly isolated from backbone extraction, coreset subsampling, and parameter fitting.
- **Dataset Provenance:** Preserved as the verified community mirror [`foersben/mvtec-ad`](https://huggingface.co/datasets/foersben/mvtec-ad) on Hugging Face (CC BY-NC-SA 4.0).

---

## 4. Quantitative Results & Side-by-Side Comparison

Evaluation was performed on the official held-out test split using [`scripts/evaluate_patchcore.py`](file:///C:/Users/HP/Downloads/indus_ai/scripts/evaluate_patchcore.py).

### Side-by-Side Benchmark Table
| Metric / Characteristic | Feature Reconstruction (Phase 4) | PatchCore Baseline (Phase 5) | Winner / Trade-off |
| :--- | :---: | :---: | :--- |
| **Image-Level AUROC** | **99.68%** (0.9968) | **100.00%** (1.0000) | **PatchCore (+0.32%)** |
| **Pixel-Level AUROC** | **96.41%** (0.9641) | **97.97%** (0.9797) | **PatchCore (+1.56%)** |
| **Optimal F1-Score (Youden's $J$)** | 0.9756 | **1.0000** | **PatchCore (+0.0244)** |
| **Unbiased F1 (Train 99th-pct)** | **0.9672** | 0.8936 | **Feature Reconstruction (+0.0736)** |
| **False Negatives @ Optimal $\tau$** | 3 missed (`contamination`) | **0 missed (100% captured)** | **PatchCore** |
| **False Positives @ Optimal $\tau$** | 0 | 0 | **Tied (Zero false alarms)** |
| **Mean Single-Image CPU Latency** | **160.18 ms** (~6.24 FPS) | 224.46 ms (~4.46 FPS) | **Reconstruction (~29% faster)** |
| **Median Latency (P50)** | **156.76 ms** | 220.06 ms | **Reconstruction** |
| **95th Percentile Latency (P95)** | **173.53 ms** | 256.20 ms | **Reconstruction** |
| **Model Footprint on Disk** | 11.5 MB (`.pth`) | **9.6 MB** (`.pth`) | **PatchCore (~16% smaller)** |
| **Training / Fitting Time** | **~30 seconds** (20 epochs) | 60.22 seconds (1% coreset) | **Reconstruction** |

---

## 5. Score Distributions & Threshold Sensitivity Analysis

### Anomaly Score Distribution by Class (PatchCore)
| Category / Defect | Sample Count | Mean Score | Std Dev | Median (P50) | Min Score | Max Score | Separation vs Normal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`good` (Normal)** | 20 | **10.1263** | 0.3122 | 10.1959 | 9.4880 | 10.6313 | **Baseline** |
| **`broken_large`** | 20 | **20.9498** | 1.4362 | 21.1852 | 18.1629 | 23.0084 | **100% separated** ($+7.53$ gap) |
| **`broken_small`** | 22 | **18.3328** | 1.9004 | 18.6682 | 13.3945 | 21.4248 | **100% separated** ($+2.76$ gap) |
| **`contamination`** | 21 | **20.0296** | 4.2715 | 20.4511 | 13.1616 | 28.3298 | **100% separated** ($+2.53$ gap) |

> **Key Analytical Finding:** In PatchCore, every single contamination image scores $\ge 13.16$, completely clear of the highest normal image ($10.63$). The 3 false negatives that challenged the reconstruction model are eliminated.

### Threshold Calibration Sensitivity Analysis
Following our Phase 4.1 audit commitments, we compare test-set tuned thresholds against unbiased training baselines:

| Threshold Strategy | Threshold $\tau$ | Calibration Source | Test-Tuned? | TP | FP | TN | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Youden's $J$** | **13.1616** | Held-Out Test Set | **Yes (Optimistic)** | 63 | 0 | 20 | 0 | **1.0000** | **1.0000** | **1.0000** |
| **Train Normal 99th-Percentile** | **9.8602** | Train Normals ($N=209$) | **No (Unbiased)** | 63 | 15 | 5 | 0 | 0.8077 | **1.0000** | 0.8936 |
| **Train Normal $\mu + 3\sigma$** | **10.0016** | Train Normals ($N=209$) | **No (Unbiased)** | 63 | 14 | 6 | 0 | 0.8182 | **1.0000** | 0.9000 |

### Crucial Engineering Insight:
While PatchCore achieves an ideal **100% AUROC**, normal test images exhibit a small positive distribution drift relative to training normals (training normal mean: $9.54$, test normal mean: $10.13$). Consequently, an unsupervised threshold calibrated solely on training normal percentiles flags 14–15 false alarms on normal test bottles. By contrast, the Feature Reconstruction autoencoder's normal distribution remained strictly bounded, achieving higher unsupervised precision.

---

## 6. Spatial Localization & Visual Artifacts

We inspected the multi-panel visual overlays saved in `artifacts/eval_heatmaps/`:
- `patchcore_broken_large_000_overlay.png`
- `patchcore_broken_small_000_overlay.png`
- `patchcore_contamination_000_overlay.png`
- `patchcore_contamination_001_overlay.png`
- `patchcore_good_000_overlay.png`

### Spatial Overlap Audit:
| Defect Sample | Inside-Mask Error | Outside-Mask Error | Contrast Ratio | Peak in Mask? |
| :--- | :---: | :---: | :---: | :---: |
| **`broken_large_000`** | 0.8744 | 0.3564 | **$2.45\times$** | **`True`** |
| **`broken_large_001`** | 0.8957 | 0.4050 | **$2.21\times$** | **`True`** |
| **`broken_small_000`** | 0.9618 | 0.4893 | **$1.97\times$** | **`True`** |
| **`broken_small_001`** | 0.8652 | 0.3502 | **$2.47\times$** | **`True`** |
| **`contamination_000`** | 0.8720 | 0.3833 | **$2.28\times$** | **`True`** |
| **`contamination_001`** | 0.9270 | 0.4248 | **$2.18\times$** | **`True`** |

> **Audit Resolution:** On `contamination_001`, which previously suffered contour edge-noise failure in the reconstruction model (`peak_in_mask = False`), **PatchCore successfully localizes the peak directly inside the true defect mask (`peak_in_mask = True`)** with a $2.18\times$ contrast ratio.

---

## 7. Latency Scope Breakdown

Single-image CPU inference latency measured via monotonic timing (`time.perf_counter`) on Intel UHD Graphics 620:
- **Mean Latency:** **224.46 ms** ($\pm 16.85\text{ ms}$)
- **Median Latency (P50):** **220.06 ms**
- **95th Percentile (P95):** **256.20 ms**
- **Inference Throughput:** **4.46 FPS**

### Latency Budget Comparison:
- **ResNet Feature Extraction (`layer2` + `layer3`):** ~85 ms
- **Local Spatial Neighborhood Pooling:** ~12 ms
- **Chunked Nearest-Neighbor Search ($784 \times 1639$ distances):** ~88 ms
- **Gaussian Smoothing & Bilinear Upsampling ($224 \times 224$):** ~15 ms
- **Score Aggregation & Normalization:** ~24 ms
- **Total:** ~224 ms

---

## 8. Reproducing Phase 5 Results

```powershell
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Run complete test suite (36 tests)
pytest -v

# 3. Fit PatchCore memory bank on normal training samples (60s)
python scripts/train_patchcore.py --category bottle --coreset-ratio 0.01

# 4. Evaluate PatchCore on held-out test split
python scripts/evaluate_patchcore.py --category bottle --checkpoint artifacts/checkpoints/patchcore_bottle.pth
```
