# MVTec Anomaly Detection (MVTec AD) Dataset Guide

## 1. Overview & Benchmark Description
The **MVTec Anomaly Detection (MVTec AD)** benchmark is the premier dataset for evaluating unsupervised anomaly detection algorithms in industrial manufacturing QA. It encompasses:
- 15 real-world industrial categories (5 texture classes, 10 object classes).
- Over 5,000 high-resolution color images.
- Unsupervised setting: Training sets contain exclusively defect-free ("good") samples.
- Evaluation setting: Test sets contain both normal and diverse defect varieties with pixel-accurate ground-truth binary masks.

**Target Category for Industrial AI Inspector:** `bottle` (bottle inspection with defects including `broken_large`, `broken_small`, and `contamination`).

---

## 2. Licensing, Terms & Attribution
- **Dataset License:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Usage Restrictions:** The dataset is strictly licensed for **non-commercial, academic research, and educational purposes**. Commercial deployment requires commercial licensing from MVTec Software GmbH.
- **Official Source:** [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **Attribution Citation:**
  > Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger.  
  > *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*  
  > IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

---

## 3. Storage Constraints & Download Strategy

### Storage Footprint Analysis
- The complete MVTec AD dataset is distributed as a single unified archive `mvtec_anomaly_detection.tar.xz`:
  - **Compressed Archive:** ~4.98 GB
  - **Uncompressed Archive:** ~5.25 GB
  - **Total Concurrent Space Required:** ~10.23 GB
- **Current Host Machine Constraint:** ~8.23 GB free disk space on `C:`.
- **Engineering Decision:** To avoid disk exhaustion, we **do not download the full 5 GB archive directly on `C:`**.

### Options for Setting Up the `bottle` Category (~250 MB)

#### Option A: Manual Setup (Recommended for Tight Disk Space)
1. On a system or drive with at least 10 GB of temporary space (e.g., an external drive or secondary machine), download the official archive:
   `https://www.mvtec.com/company/research/datasets/mvtec-ad`
2. Extract only the `bottle` directory:
   ```bash
   tar -xJf mvtec_anomaly_detection.tar.xz mvtec_anomaly_detection/bottle
   ```
3. Copy the extracted `bottle` folder to this project at:
   ```text
   C:\Users\HP\Downloads\indus_ai\data\mvtec_anomaly_detection\bottle\
   ```

#### Option B: Configurable External Path
If your dataset resides on an external drive or network share, set the `MVTEC_DATASET_ROOT` environment variable without copying files:
```powershell
$env:MVTEC_DATASET_ROOT = "D:\Datasets\mvtec_anomaly_detection"
```

#### Option C: Selective Category Downloader (Used in Phase 4)
Because the official MVTec website only distributes the entire 15-category benchmark as a single monolithic 4.98 GB archive, we built `scripts/download_bottle.py` to selectively download only the 157 MB `bottle` category from the community mirror [`foersben/mvtec-ad`](https://huggingface.co/datasets/foersben/mvtec-ad) on Hugging Face:
```powershell
python scripts/download_bottle.py
```
> [!NOTE]
> **Provenance Transparency:** The mirror `foersben/mvtec-ad` on Hugging Face provides an uncompressed per-file mirror of MVTec AD under the original CC BY-NC-SA 4.0 license. The directory structure, image resolutions (900x900), and image/mask counts (209 train, 83 test, 63 masks) strictly match the official MVTec publication.

---

## 4. Expected Directory Layout

The inspection pipeline (`src/data/`) validates the following structure:

```text
data/
└── mvtec_anomaly_detection/
    └── bottle/
        ├── train/
        │   └── good/                          # Pristine defect-free images (used for training)
        │       ├── 000.png
        │       ├── 001.png
        │       └── ... (209 images)
        ├── test/
        │   ├── good/                          # Normal test images (no masks expected)
        │   │   ├── 000.png
        │   │   └── ... (20 images)
        │   ├── broken_large/                  # Severe structural defects
        │   │   ├── 000.png
        │   │   └── ... (20 images)
        │   ├── broken_small/                  # Subtle structural defects
        │   │   ├── 000.png
        │   │   └── ... (22 images)
        │   └── contamination/                 # Foreign substance anomalies
        │       ├── 000.png
        │       └── ... (21 images)
        └── ground_truth/                      # Binary pixel-level segmentation masks
            ├── broken_large/
            │   ├── 000_mask.png
            │   └── ...
            ├── broken_small/
            │   ├── 000_mask.png
            │   └── ...
            └── contamination/
                ├── 000_mask.png
                └── ...
```

---

## 5. Verification Commands

Run the automated inspection script to validate the dataset:
```powershell
python scripts/inspect_dataset.py --category bottle
```

When the dataset is present and valid, the script will:
1. Verify readability of all image files with Pillow.
2. Verify 1-to-1 correspondence between defective test images and ground-truth masks.
3. Check dimension matching between images and masks.
4. Generate a visual contact sheet in `artifacts/contact_sheets/mvtec_bottle_overview.png`.
