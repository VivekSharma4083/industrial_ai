# Phase 4 Experimental Report: Feature Reconstruction on Genuine MVTec Bottle Dataset

## 1. Executive Summary & Objective

Phase 4 establishes the first empirical validation of the **Feature Reconstruction Anomaly Detection** pipeline using **genuine MVTec AD `bottle` images** and **torchvision ImageNet-pretrained ResNet-50 weights**.

All experiments were executed locally under CPU hardware constraints (Intel UHD Graphics 620, PyTorch 2.14.1+cpu on Windows 64-bit). The model achieved:
- **Image-level AUROC: 99.68% (0.9968)**
- **Pixel-level Localization AUROC: 96.41% (0.9641)**
- **Optimal Decision Boundary (Youden's J): 23.6698**
- **Precision at Optimal Threshold: 100.0% (0 False Positives out of 20 normal test samples)**
- **Recall at Optimal Threshold: 95.24% (60 out of 63 defect images detected)**
- **Inference Latency: 157.08 ms per image on CPU (~6.37 FPS throughput)**

---

## 2. Dataset Partition & Provenance Verification

### Dataset Source & Provenance
- **Original Dataset:** MVTec Anomaly Detection (MVTec AD), created by MVTec Software GmbH.
- **License:** [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Strictly non-commercial research and education).
- **Download Source:** Downloaded selectively via `scripts/download_bottle.py` using the community Hugging Face mirror [`foersben/mvtec-ad`](https://huggingface.co/datasets/foersben/mvtec-ad) to avoid downloading the monolithic 4.98 GB archive onto a disk with ~8 GB free. The downloaded files (209 train normals, 83 test images, 63 masks) strictly match the official MVTec directory structure and 900x900 resolution.

### Partition Breakdown
- **Unsupervised Training Split (`train/good/`):** 209 defect-free images.
  - Used strictly for learning the normal feature manifold.
- **Held-Out Test Split (`test/`):** 83 images total:
  - `good` (pristine bottles): 20 images
  - `broken_large`: 20 images (+20 ground-truth binary masks)
  - `broken_small`: 22 images (+22 ground-truth binary masks)
  - `contamination`: 21 images (+21 ground-truth binary masks)
- **Total Ground-Truth Evaluation Masks:** 63 pixel-accurate binary masks.

> [!IMPORTANT]
> **Zero Data Leakage:** All 83 test images and 63 ground-truth masks were kept strictly isolated from model training. Training loss and model weights reflect only the 209 normal training samples.

---

## 3. Model Architecture & Training Hyperparameters

- **Backbone Feature Extractor:** Pretrained ResNet-50 (`ResNet50_Weights.DEFAULT`, 11ad3fa6), frozen weights.
- **Target Feature Layer:** `layer2` (512 feature channels, 28x28 spatial resolution for 224x224 input images).
- **Autoencoder Architecture:** 1.00M-parameter convolutional bottleneck:
  - Encoder: `Conv2d(512->256, 3x3)` -> `BatchNorm` -> `LeakyReLU` -> `Conv2d(256->128, 3x3)` -> `BatchNorm` -> `LeakyReLU`.
  - Bottleneck: 128 channels.
  - Decoder: `ConvTranspose2d(128->256, 3x3)` -> `BatchNorm` -> `LeakyReLU` -> `ConvTranspose2d(256->512, 3x3)` -> `BatchNorm` -> `Identity`.
- **Training Epochs:** 20 epochs.
- **Batch Size:** 16 images per batch.
- **Optimizer:** Adam (learning rate = $1 \times 10^{-4}$).
- **Loss Function:** Mean Squared Error (MSE) between extracted and reconstructed feature maps:
  $$\mathcal{L}_{MSE} = \frac{1}{C \cdot H_f \cdot W_f} \sum_{c, h, w} \left( \Phi(x)_{c,h,w} - \hat{\Phi}(x)_{c,h,w} \right)^2$$
- **Training Convergence:**
  - Epoch 01 Loss: 2.002540
  - Epoch 05 Loss: 1.304218
  - Epoch 10 Loss: 0.889581
  - Epoch 15 Loss: 0.648191
  - Epoch 20 Loss: 0.496971
- **Checkpoint Artifact:** `artifacts/checkpoints/reconstruction_bottle.pth` (11.5 MB).

---

## 4. Quantitative Evaluation Results

Evaluation was performed across all 83 test images using `scripts/evaluate.py`.

### Image-Level Metrics
| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Image-level AUROC** | **0.9968** (99.68%) | Near-perfect separation between defective and pristine bottles |
| **Optimal Threshold ($\tau$)** | **23.6698** | Maximizes Youden's $J = \text{TPR} - \text{FPR}$ |
| **Precision @ $\tau$** | **1.0000** (100.0%) | Zero false alarms on pristine production samples |
| **Recall @ $\tau$** | **0.9524** (95.24%) | Detects 60 of 63 defects |
| **F1-Score @ $\tau$** | **0.9756** | High harmonic balance |
| **Accuracy @ $\tau$** | **0.9639** (96.39%) | 80 out of 83 correct binary classifications |

### Confusion Matrix @ Optimal Threshold $\tau = 23.6698$
```text
                  Predicted Normal     Predicted Anomaly
Actual Normal            20 (TN)               0 (FP)
Actual Defective          3 (FN)              60 (TP)
```
- **True Negatives (TN):** 20 / 20 (100% normal acceptance rate)
- **False Positives (FP):** 0 / 20 (0% false rejection rate)
- **True Positives (TP):** 60 / 63 (95.2% anomaly capture rate)
- **False Negatives (FN):** 3 / 63 (missed subtle low-contrast contaminations: `contamination/001.png`, `012.png`, `019.png`)

### Methodological Audit: Threshold Selection & Optimism Caveat
> [!WARNING]
> **Threshold Selection Caveat:** The optimal threshold $\tau = 23.6698$ was computed post-hoc by maximizing Youden's $J$-statistic ($J = \text{TPR} - \text{FPR}$) directly on the test set ROC curve.
> In machine learning methodology, threshold-dependent metrics evaluated at this threshold (Precision: 100%, Recall: 95.24%, F1: 0.9756) are **optimistically test-set tuned** and cannot be claimed as an unbiased estimate of real-world deployment performance on unseen future products.
> 
> **Important Distinction:** **AUROC (99.68%) is threshold-independent** (it integrates over all operating points of the ROC curve) and is mathematically unaffected by threshold selection.

To establish unbiased, realistic deployment performance without relying on test-set tuning, we also evaluated thresholds calibrated **strictly unsupervised on the normal training split (`train/good/`, $N=209$)**:

| Threshold Calibration Strategy | Threshold $\tau$ | Test-Tuned? | TP | FP | TN | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Youden's $J$ (Test-Set Optimal)** | 23.6698 | **Yes (Optimistic)** | 60 | 0 | 20 | 3 | 1.0000 | 0.9524 | 0.9756 |
| **Train Normal 99th-Percentile** | 23.7207 | **No (Unbiased)** | 59 | 0 | 20 | 4 | 1.0000 | 0.9365 | 0.9672 |
| **Train Normal $\mu + 3\sigma$** | 23.8762 | **No (Unbiased)** | 57 | 0 | 20 | 6 | 1.0000 | 0.9048 | 0.9500 |

Both unsupervised training-calibrated thresholds maintain **100% Precision (zero false rejections)** while detecting **90.5% – 93.7% of all defects**, confirming robust operational viability without test-set tuning.

---

## 5. Anomaly Score Distribution by Defect Class

The reconstruction error exhibits strong class separation between normal bottles and structural damage:

| Category / Defect | Sample Count | Mean Score | Std Dev | Median (P50) | Min Score | Max Score | Separation vs Normal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`good` (Normal)** | 20 | **22.6610** | 0.3608 | 22.7386 | 21.8648 | 23.3944 | Baseline |
| **`broken_large`** | 20 | **27.3847** | 0.7934 | 27.4117 | 25.8723 | 28.6426 | **+4.72** (100% separated) |
| **`broken_small`** | 22 | **27.1680** | 1.0517 | 27.1352 | 25.1100 | 28.8325 | **+4.51** (100% separated) |
| **`contamination`** | 21 | **26.3398** | 2.7691 | 25.9791 | 23.0912 | 33.1041 | **+3.68** (90.5% separated) |

### Key Observations:
1. **Pristine Bottle Consistency:** Normal bottle reconstruction scores are tightly bounded in the range $[21.86, 23.39]$ with very low variance ($\sigma = 0.3608$).
2. **Structural Damage (`broken_large`, `broken_small`):** Minimum defect scores ($25.87$ and $25.11$) are significantly higher than the maximum normal score ($23.39$). Both structural defect classes achieve 100% detection with zero overlap.
3. **Subtle Surface Contamination:** Contamination manifests greater variance ($\sigma = 2.7691$). Severe contaminations exceed scores of $33.10$, while three faint transparent water droplet / low-contrast surface artifacts fall marginally below the decision boundary ($23.09$ to $23.66$), accounting for the 3 false negatives.

---

## 6. Pixel-Level Anomaly Localization Performance

To evaluate explainability and spatial localization accuracy, spatial reconstruction residual maps were upsampled to $224 \times 224$ and evaluated against all 63 binary ground-truth segmentation masks across the entire test set ($4,164,608$ total pixels evaluated).

- **Evaluated Pixels:** 4,164,608 pixels across 83 test images
- **Defective Pixels:** 240,575 pixels (5.78% of test area)
- **Normal Pixels:** 3,924,033 pixels (94.22% of test area)
- **Pixel-Level AUROC:** **0.9641 (96.41%)**

### Visual Explainability Overlays
Multi-panel visual explanation artifacts were generated and saved to `artifacts/eval_heatmaps/`:
- `broken_large_000_overlay.png`: Highlights jagged glass fracture along the bottle neck.
- `broken_small_000_overlay.png`: Precisely localizes small chip on bottle rim.
- `contamination_000_overlay.png`: Spots foreign particulate inside the translucent body.
- `good_000_overlay.png`: Uniform, low-residual blue heatmap across pristine glass.

Each artifact stitches:
`[ Original Input Image ] | [ Jet Anomaly Heatmap ] | [ Semi-Transparent Overlay ] | [ Ground-Truth Mask ]`

### Localization Audit & Failure Analysis
We audited spatial peak activations across representative samples:
- **High-Fidelity Localization:** On structural defects (`broken_large_000`, `001` and `broken_small_000`, `001`), the peak heatmap activation falls **directly inside the ground-truth mask (`peak_in_mask = True`)**, with inside-mask reconstruction error averaging $1.54\times$ to $1.67\times$ higher than background pixels.
- **Edge-Noise Limitation (`contamination_001`):** On `contamination_001.png`, the peak anomaly activation occurs along the boundary/contour of the bottle body (`peak_in_mask = False`) rather than on the faint surface smudge itself. This reveals that the feature reconstruction model exhibits elevated residual sensitivity along high-frequency glass refraction contours.
- **False Negative Analysis:** The 3 missed defect images (`contamination/001.png` score 23.09, `012.png` score 23.12, `019.png` score 23.12) are all faint low-contrast smudges where the maximum spatial error did not cross the threshold.

---

## 7. CPU Inference Latency & Scope Audit

Latency was benchmarked using high-resolution monotonic timing (`time.perf_counter`) on single-image CPU inference:

- **Execution Environment:** Intel(R) UHD Graphics 620 CPU (No GPU acceleration)
- **Warmup Iterations:** 5 unmeasured passes
- **Timed Repetitions:** 30 consecutive passes
- **Mean Inference Time:** **160.18 ms** (repeat run: $\pm 13.54\text{ ms}$)
- **Median Latency (P50):** **156.76 ms**
- **95th Percentile Latency (P95):** **173.53 ms**
- **Inference Throughput:** **6.24 Frames Per Second (FPS)**

### Latency Scope Breakdown (What is included vs excluded):
- **Included in Benchmark (157–160 ms):**
  1. Input tensor normalization and device preparation.
  2. ResNet-50 frozen backbone forward pass up to `layer2`.
  3. Autoencoder forward pass (reconstruction).
  4. Channel-wise $L_2$ norm error map calculation.
  5. 2D Gaussian convolution spatial smoothing ($7 \times 7$, $\sigma = 4.0$).
  6. Bilinear upsampling to $224 \times 224$ image resolution.
  7. Safe NumPy normalization ($[\min, \max]$ scaling with divide-by-zero protection).
  8. Image-level score aggregation (`max`).
- **Excluded from Benchmark:**
  1. Model initialization and loading weights from disk (performed once at application startup).
  2. Cold disk I/O when reading a file from disk. (A separate end-to-end benchmark including `Image.open()` from disk + decoding 900x900 PNG + resizing + inference measured **151.5 ms**).
  3. Disk write time of saving heatmap overlay PNGs to disk.

---

## 8. Reproducing Phase 4 Results

To reproduce these exact results from scratch:

```powershell
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Run test suite
pytest -v

# 3. Train on MVTec bottle normal samples (20 epochs)
python scripts/train.py --category bottle --epochs 20 --batch-size 16 --layer layer2

# 4. Evaluate on held-out test split
python scripts/evaluate.py --category bottle --checkpoint artifacts/checkpoints/reconstruction_bottle.pth --layer layer2 --score-aggregation max
```
