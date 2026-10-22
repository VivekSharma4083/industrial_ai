# Industrial AI Inspector: Explainable Visual Anomaly Detection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Flask 3.0](https://img.shields.io/badge/Flask-3.0.3-black.svg)](https://flask.palletsprojects.com/)
[![PyTorch 2.14 CPU](https://img.shields.io/badge/PyTorch-2.14.1%2Bcpu-ee4c2c.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-46%20passed-brightgreen.svg)](tests/)
[![PatchCore AUROC](https://img.shields.io/badge/PatchCore--AUROC-100.0%25-success.svg)](docs/patchcore_phase5.md)
[![Recon AUROC](https://img.shields.io/badge/Reconstruction--AUROC-99.68%25-brightgreen.svg)](docs/evaluation_phase4.md)
[![Pixel AUROC](https://img.shields.io/badge/Pixel--AUROC-97.97%25-blueviolet.svg)](docs/patchcore_phase5.md)
[![License](https://img.shields.io/badge/Dataset-MVTec%20CC%20BY--NC--SA%204.0-lightgrey.svg)](docs/dataset_guide.md)

> **Placement-Grade Computer Vision & Industrial Quality Assurance System**  
> Learns representation manifolds from defect-free industrial products and identifies, localizes, and explains anomalies in unseen test images without requiring labeled defect data during training.

---

## 1. Project Objective & Core Concept

In real-world manufacturing and assembly lines, defective samples are exceedingly rare, diverse, and unpredictable. Supervised classifiers fail because anomalous failure modes cannot be exhaustively anticipated or labeled.

**Industrial AI Inspector** solves this via **unsupervised / one-class anomaly detection**:
1. **Normal Feature Learning:** Extracts deep semantic representations of pristine products using a pretrained ResNet-50 backbone.
2. **Feature Reconstruction Autoencoding:** Trains a spatial feature reconstruction network strictly on defect-free samples. When presented with an anomalous region, the model cannot faithfully reconstruct it, generating a pronounced reconstruction error.
3. **PatchCore Comparison Baseline:** Benchmarks against PatchCore, utilizing neighborhood-aware patch embeddings (`layer2` + `layer3`) and greedy coreset subsampling.
4. **Explainable Anomaly Heatmaps:** Projects spatial reconstruction residuals back to pixel space, generating high-resolution Gaussian-smoothed heatmaps and defect bounding overlays.
5. **Decoupled Full-Stack Architecture:** Features a high-performance Flask REST API coupled with a responsive, zero-framework Vanilla HTML5/CSS3/JavaScript web interface with real-time 60 FPS client-side opacity slider blending.

> [!NOTE]
> **Implementation Status (Phase 6 Complete — Interactive Full-Stack Inspection Dashboard):**  
> - **Interactive QA Interface:** Drag & drop image upload, preset sample gallery (normal, fracture, chip, contamination), model selector (Reconstruction, PatchCore, or Both), decision threshold strategy toggle (unbiased vs optimal vs custom), and real-time interactive opacity slider.
> - **Inference Engine:** Single-instance CPU singleton model service reuses checkpoints across requests.
> - **Inspection History:** Bounded in-memory cache of past 50 inspections with thumbnails and one-click workspace reloading.
> - **Dual-Model Benchmark:** Grounded in genuine MVTec AD bottle evaluation (PatchCore: 100.00% AUROC; Reconstruction: 99.68% AUROC). All 46 automated unit/integration tests passing.

---

## 2. Technology Stack

### Frontend (Zero Framework Overhead)
- **HTML5 & CSS3:** Precision dark industrial design system with custom CSS variables, responsive cards, and live telemetry.
- **Vanilla JavaScript (ES6+):** Pure Fetch API interaction with backend endpoints; no heavy framework build steps (React/Vue/Streamlit) required.

### Backend
- **Python 3.12.10:** Modern Python runtime with strict isolation.
- **Flask 3.0.3:** Lightweight, production-ready REST server serving both JSON API routes and static frontend assets from a single origin (eliminating cross-origin CORS overhead).
- **Pillow 10.4.0:** Resilient image ingestion and preprocessing.
- **PyTest 8.4.2:** Automated test discovery and endpoint verification.

### Machine Learning & Vision Foundation (Phases 2 & 3)
- **PyTorch 2.14.1+cpu & torchvision 0.29.1+cpu:** Official CPU wheel index (`https://download.pytorch.org/whl/cpu`) ensuring high efficiency on CPU hardware without requiring NVIDIA CUDA drivers.
- **NumPy 2.5.2:** Numerical tensor array operations.
- **ResNet-50 Intermediate Feature Extractor:** Frozen feature representations from `layer2` (512-dim, 28x28 grid) or `layer3` (1024-dim, 14x14 grid).
- **Spatial Autoencoder:** 1.0M-parameter convolutional bottleneck network (~4 MB on disk).
- **MVTec AD Dataset:** The industrial gold-standard benchmark. Initial target category: `bottle`.

---

## 3. Project Directory Structure

```text
industrial-ai-inspector/
├── frontend/                     # Web Client (Vanilla HTML/CSS/JS)
│   ├── index.html                # Single-page diagnostic & inspection interface
│   ├── pages/                    # Sub-views and documentation pages
│   ├── css/
│   │   └── styles.css            # Precision dark industrial stylesheet
│   ├── js/
│   │   └── app.js                # Fetch API client and UI controller
│   └── assets/                   # Icons, badges, and reference illustrations
├── backend/                      # Flask Application Server
│   ├── __init__.py               # Package marker
│   ├── app.py                    # App factory and static router (10 MB payload limit)
│   ├── routes/
│   │   ├── __init__.py           # Route blueprint registry
│   │   ├── health.py             # GET /api/health endpoint
│   │   └── inspect.py            # POST /api/inspect, GET /api/models, /api/history, /api/samples
│   └── services/                 # Backend business logic & inference services
│       ├── __init__.py
│       └── model_service.py      # Singleton model manager, base64 jet heatmaps, history cache
├── src/                          # Core ML & Vision Packages
│   ├── __init__.py
│   ├── data/                     # Dataset Pipeline (Phase 2)
│   │   ├── __init__.py           # Package exports
│   │   ├── config.py             # DatasetConfig path & category manager
│   │   ├── validator.py          # MVTecValidator & integrity verification
│   │   └── dataset.py            # PyTorch MVTecDataset implementation
│   ├── models/                   # Vision & Anomaly Models (Phases 3 & 5)
│   │   ├── __init__.py           # Package exports
│   │   ├── feature_extractor.py  # Frozen intermediate ResNet50 extractor
│   │   ├── reconstruction.py     # Lightweight convolutional autoencoder
│   │   ├── anomaly_detector.py   # Anomaly heatmap generation & scoring engine
│   │   ├── trainer.py            # Normal-only training manager & checkpointing
│   │   └── patchcore.py          # PatchCore multi-scale memory bank & coreset model
│   ├── evaluation/               # Image ROC-AUC, Pixel PRO-AUC metrics (Phases 4-5)
│   │   ├── __init__.py           # Metric exports
│   │   └── metrics.py            # AUROC, per-defect stats & latency benchmarking
│   └── visualization/            # Heatmap overlays & report generators
├── configs/                      # Experiment and hyperparameter configurations
├── scripts/                      # Operational utilities & automation
│   ├── check_environment.py     # Pre-flight environment diagnostics
│   ├── verify_live_server.py    # Live socket HTTP server validation
│   ├── inspect_dataset.py       # Dataset validation & contact-sheet generator
│   ├── smoke_test_models.py     # CPU end-to-end model smoke test
│   ├── download_bottle.py       # Selective MVTec bottle dataset downloader
│   ├── train.py                 # Training script for feature reconstruction
│   ├── evaluate.py              # Quantitative evaluation (Reconstruction model)
│   ├── train_patchcore.py       # Memory bank fitting script (PatchCore)
│   └── evaluate_patchcore.py    # Quantitative evaluation & benchmark (PatchCore)
├── tests/                        # Automated unit & integration tests (46 total passing)
│   ├── __init__.py
│   ├── test_health.py            # API route and asset delivery tests (5 tests)
│   ├── test_dataset.py           # Dataset pipeline & synthetic mock tests (8 tests)
│   ├── test_models.py            # Feature extractor, autoencoder & detector tests (9 tests)
│   ├── test_evaluation.py        # Metrics, AUROC, latency & overlay tests (8 tests)
│   ├── test_patchcore.py         # PatchCore embeddings, coreset & scoring tests (6 tests)
│   └── test_api_inspection.py    # Inspection API, model service, presets & history (10 tests)
├── artifacts/                    # Saved weights, logs, heatmaps (Git-ignored)
│   ├── checkpoints/              # Model checkpoints (*.pth)
│   ├── eval_heatmaps/            # Generated visual explanation overlays
│   └── eval_results/             # Evaluation summary JSON metrics
├── docs/                         # Technical architecture documentation
│   ├── dataset_guide.md          # MVTec AD licensing, layout & download guide
│   ├── model_architecture.md     # Feature reconstruction architecture & rationale
│   ├── evaluation_phase4.md      # Phase 4 empirical results & benchmark report
│   └── patchcore_phase5.md       # Phase 5 PatchCore technical report & comparison
├── requirements.txt              # Pinned dependency specifications (with CPU index)
├── .gitignore                    # Robust exclusion rules
└── README.md                     # Project blueprint & operating instructions
```

---

## 4. Environment Diagnostics & System Profile

- **Operating System:** Windows 64-bit (`win32`)
- **Python Runtime:** Python 3.12.10 (`py -3.12`)
- **Graphics Controller:** Intel(R) UHD Graphics 620 (integrated, no NVIDIA GPU)
- **PyTorch Execution Target:** CPU (`torch.cuda.is_available() == False`).
- **Storage Profile:** ~8.23 GB free on host drive. Checkpoints require only ~4 MB each.

---

## 5. Quickstart & Setup Guide (Windows PowerShell)

### Step 1: Activate Virtual Environment
```powershell
cd C:\Users\HP\Downloads\indus_ai
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Pre-Flight Diagnostics
```powershell
python scripts/check_environment.py
```

### Step 4: Run the Complete Test Suite
```powershell
pytest -v
```
*Expected Output: `36 passed in <21.00s`*

### Step 5: Train Feature Reconstruction Model (20 epochs)
```powershell
python scripts/train.py --category bottle --epochs 20 --batch-size 16 --layer layer2
```
*Expected Output: `[SUCCESS] Training completed! Model saved to: .../reconstruction_bottle.pth`*

### Step 6: Fit PatchCore Memory Bank (1% Coreset Subsampling)
```powershell
python scripts/train_patchcore.py --category bottle --coreset-ratio 0.01
```
*Expected Output: `[SUCCESS] PatchCore model checkpoint saved to: .../patchcore_bottle.pth`*

### Step 7: Run Quantitative Evaluation & Side-by-Side Benchmarks
```powershell
# Evaluate Feature Reconstruction Autoencoder
python scripts/evaluate.py --category bottle --checkpoint artifacts/checkpoints/reconstruction_bottle.pth --layer layer2

# Evaluate PatchCore Baseline & Generate Side-by-Side Benchmark
python scripts/evaluate_patchcore.py --category bottle --checkpoint artifacts/checkpoints/patchcore_bottle.pth
```
*Expected Output: `PatchCore: 100.0% Image AUROC, 97.97% Pixel AUROC | Reconstruction: 99.68% Image AUROC, 96.41% Pixel AUROC`*

### Step 8: Start the Flask Application
```powershell
python backend/app.py
```
Open **`http://127.0.0.1:5000/`** in your browser.

---

## 6. MVTec AD Dataset Setup Guide

### Official Source & License
- **Provider:** MVTec Software GmbH
- **Portal:** [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **License:** [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (**Non-commercial research only**).

See [`docs/dataset_guide.md`](docs/dataset_guide.md) for step-by-step extraction instructions.
See [`docs/evaluation_phase4.md`](docs/evaluation_phase4.md) for Phase 4 reconstruction benchmarks.
See [`docs/patchcore_phase5.md`](docs/patchcore_phase5.md) for Phase 5 PatchCore comparative benchmarks.

---

## 7. Planned Development Phases

| Phase | Milestone | Scope | Status |
| :---: | :--- | :--- | :---: |
| **01** | **Project Setup & Environment Verification** | Isolated venv, Flask app, Vanilla JS client, pytest suite, diagnostics | **Complete** |
| **02** | **Dataset Pipeline & PyTorch Setup** | CPU PyTorch/torchvision, MVTec dataset loader, validator, synthetic tests, dataset guide | **Complete** |
| **03** | **Feature Extraction & Reconstruction Model** | ResNet-50 intermediate extractor, spatial autoencoder, anomaly map generator, trainer | **Complete** |
| **04** | **Real MVTec Data & First Genuine Evaluation** | Selective bottle download, real training, 83-image test evaluation, AUROC (99.7%), latency benchmark | **Complete** |
| **05** | **PatchCore Comparison Baseline** | Multi-scale patch embeddings (`layer2`+`layer3`), greedy coreset (1%), dual-model benchmark (100% AUROC) | **Complete** |
| **06** | **Interactive Inspection Dashboard** | Full-stack UI, live drag-and-drop, dual-model comparison, 60 FPS opacity slider, history | **Complete** |

---

## 8. Interactive Dashboard & REST API Reference

The Phase 6 full-stack dashboard runs on `http://127.0.0.1:5000/` and provides:
- **Live Inspection Workspace:** Drag-and-drop or file upload (PNG/JPEG/BMP up to 10 MB), one-click preset test samples (Pristine normal, Fracture, Rim chip, Contamination), model selector pills (`Reconstruction`, `PatchCore`, or `Compare Both`), decision threshold selector (`Unbiased Normal Training` default vs. `Test-Tuned Optimal` vs. `Custom`), and real-time interactive opacity slider (0% to 100% alpha blend).
- **Dual-Model Comparative Panel:** Side-by-side inspection showing individual model anomaly scores, decision margins, latency, continuous Jet heatmaps, interactive overlays, and agreement reasoning (`CONCORDANT_DEFECT`, `CONCORDANT_NORMAL`, or `DISCORDANT`).
- **Inspection History:** Bounded in-memory log of recent inspections with thumbnails, status pills, and one-click workspace reloading.
- **Empirical Benchmarks Tab:** Live quantitative benchmark metrics table grounded in actual saved evaluation artifacts (`evaluation_summary.json` and `patchcore_summary.json`).

### REST API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Service status, Python runtime, platform metadata. |
| `/api/models` | `GET` | Model readiness, layers, trainable parameters, calibrated decision thresholds. |
| `/api/inspect` | `POST` | Execute anomaly inspection on uploaded `image` file or `sample_id` preset. Returns scores, thresholds, decisions, base64 heatmaps, and agreement. |
| `/api/history` | `GET` | Return list of recent inspection summaries (up to `limit=50`). |
| `/api/history/<id>` | `GET` | Retrieve detailed inspection record with full heatmaps by inspection ID. |
| `/api/history` | `DELETE` | Clear in-memory inspection history. |
| `/api/eval-summary` | `GET` | Return genuine empirical benchmark comparison metrics. |
| `/api/samples` | `GET` | List available preset test samples with metadata and thumbnails. |
| `/api/samples/<sample_id>` | `GET` | Retrieve preset sample image payload. |

---

## 9. Engineering & Security Rules

1. **Path Agility:** All paths are resolved dynamically using `pathlib.Path` relative to the project root, or via `MVTEC_DATASET_ROOT`.
2. **Exclusion Protection:** Datasets (`/data/`), model weights (`*.pth`), virtual environments (`.venv/`), and artifacts (`artifacts/*`) are strictly excluded from version control.
3. **Reproducibility:** Experiments lock random seeds (`set_seed(42)`).
4. **Scientific Integrity:** Anomaly scores represent reconstruction distances and are never claimed as calibrated probabilities without empirical validation thresholds.
5. **No Automated Git Commits:** All Git commits remain under explicit developer control.
