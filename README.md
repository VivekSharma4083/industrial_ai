# Industrial AI Inspector: Explainable Visual Anomaly Detection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Flask 3.0](https://img.shields.io/badge/Flask-3.0.3-black.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20Full--Stack-orange.svg)]()

> **Placement-Grade Computer Vision & Industrial Quality Assurance System**  
> Learns representation manifolds from defect-free industrial products and identifies, localizes, and explains anomalies in unseen test images without requiring labeled defect data during training.

---

## 1. Project Objective & Core Concept

In real-world manufacturing and assembly lines, defective samples are exceedingly rare, diverse, and unpredictable. Supervised classifiers fail because anomalous failure modes cannot be exhaustively anticipated or labeled.

**Industrial AI Inspector** solves this via **unsupervised / one-class anomaly detection**:
1. **Normal Feature Learning:** Extracts deep semantic representations of pristine products using a pretrained ResNet-50 backbone.
2. **Feature Reconstruction Autoencoding:** Trains a spatial feature reconstruction network strictly on defect-free samples. When presented with an anomalous region, the model cannot faithfully reconstruct it, generating a pronounced reconstruction error.
3. **PatchCore Comparison Baseline:** Benchmarks against PatchCore, utilizing neighborhood-aware patch embeddings and greedy coreset subsampling.
4. **Explainable Anomaly Heatmaps:** Projects spatial reconstruction and distance residuals back to pixel space, generating high-resolution Gaussian-smoothed heatmaps and defect bounding overlays.
5. **Decoupled Architecture:** Features a high-performance Flask REST API coupled with a responsive, zero-framework Vanilla HTML5/CSS3/JavaScript web interface.

---

## 2. Technology Stack

### Frontend (Zero Framework Overhead)
- **HTML5 & CSS3:** Precision dark industrial design system with custom CSS variables, responsive grids, and diagnostic telemetry.
- **Vanilla JavaScript (ES6+):** Pure Fetch API interaction with backend endpoints; no heavy framework build steps (React/Vue/Streamlit) required.

### Backend
- **Python 3.12.10:** Modern Python runtime with strict isolation.
- **Flask 3.0.3:** Lightweight, production-ready REST server serving both JSON API routes and static frontend assets from a single origin (eliminating cross-origin CORS overhead).
- **Pillow 10.4.0:** Resilient image ingestion and preprocessing.
- **PyTest 8.4.2:** Automated test discovery and endpoint verification.

### Machine Learning Roadmap (Phases 2-6)
- **PyTorch & torchvision:** Multi-scale feature extraction and reconstruction autoencoder.
- **Scikit-Learn:** Coreset reduction, ROC-AUC, and PRO-AUC (Per-Region-Overlap) metric computation.
- **MVTec AD Dataset:** The industrial gold-standard benchmark containing 15 object and texture categories.

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
│   ├── app.py                    # App factory and static router
│   ├── routes/
│   │   ├── __init__.py           # Route blueprint registry
│   │   └── health.py             # GET /api/health endpoint
│   └── services/                 # Backend business logic & inference services
│       └── __init__.py
├── src/                          # Core ML & Vision Packages (Phases 2-5)
│   ├── data/                     # Dataset loaders, transforms & split managers
│   ├── models/                   # Feature extractors, Autoencoder & PatchCore
│   ├── evaluation/               # Image ROC-AUC, Pixel PRO-AUC & confusion metrics
│   └── visualization/            # Heatmap overlays, color maps & report generators
├── configs/                      # Experiment and hyperparameter configurations (YAML)
├── scripts/                      # Operational utilities & automation
│   ├── check_environment.py     # Pre-flight environment diagnostics
│   └── verify_live_server.py    # Live socket HTTP server validation
├── tests/                        # Automated unit & integration tests
│   ├── __init__.py
│   └── test_health.py            # API route and asset delivery tests
├── artifacts/                    # Saved weights, logs, heatmaps (Git-ignored)
├── docs/                         # Technical architecture documentation
├── requirements.txt              # Phase-scoped dependency specifications
├── .gitignore                    # Robust exclusion rules
└── README.md                     # Project blueprint & operating instructions
```

---

## 4. Environment Diagnostics & System Profile

Pre-flight hardware inspection of the development machine:
- **Operating System:** Windows 64-bit (`win32`)
- **Python Runtime:** Python 3.12.10 (`py -3.12`)
- **Graphics Controller:** Intel(R) UHD Graphics 620 (integrated, no NVIDIA GPU)
- **Hardware Strategy:** Optimized CPU inference mode will be configured for PyTorch in Phase 2 to ensure maximum reproducibility, fast cold starts, and zero dependency on proprietary CUDA drivers.
- **Storage Profile:** ~9 GB free on host drive. To avoid disk exhaustion, dataset ingestion in Phase 2 will support selective category extraction (e.g., `bottle`, `carpet`).

---

## 5. Quickstart & Setup Guide (Windows PowerShell)

### Step 1: Create and Activate Virtual Environment
```powershell
# Open Windows PowerShell in the project root:
cd C:\Users\HP\Downloads\indus_ai

# Create a project-local virtual environment using Python 3.12
py -3.12 -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

> **Note on PowerShell Execution Policy:** If script execution is restricted on Windows, run:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Pre-Flight Diagnostics
```powershell
python scripts/check_environment.py
```
*Expected Output: `[SUCCESS] Environment inspection passed! Project skeleton is verified.`*

### Step 4: Run the Test Suite
```powershell
pytest -v
```
*Expected Output: `5 passed in <1.00s`*

### Step 5: Verify Live HTTP Server
```powershell
python scripts/verify_live_server.py
```

### Step 6: Start the Flask Application
```powershell
python backend/app.py
```
Open your web browser and navigate to:
**`http://127.0.0.1:5000/`**

The web interface will immediately perform a health poll against `GET /api/health` and display live system telemetry.

---

## 6. MVTec AD Dataset Setup Guide (Planned for Phase 2)

The **MVTec Anomaly Detection (MVTec AD)** benchmark is the de-facto standard for evaluating industrial visual inspection algorithms.

### Official Source
- **Provider:** MVTec Software GmbH
- **Portal:** [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **License:** Non-commercial, research & educational use.

### Expected Directory Layout
```text
data/
└── mvtec_ad/
    ├── bottle/
    │   ├── ground_truth/
    │   │   ├── broken_large/
    │   │   ├── broken_small/
    │   │   └── contamination/
    │   ├── test/
    │   │   ├── good/
    │   │   ├── broken_large/
    │   │   ├── broken_small/
    │   │   └── contamination/
    │   └── train/
    │       └── good/
    ├── cable/
    ├── capsule/
    ├── carpet/
    ...
```

> **Storage Management Best Practice:**  
> The full dataset comprises ~5 GB compressed and ~5 GB uncompressed across 15 categories. In Phase 2, the pipeline will offer a modular category loader starting with a single representative category (such as `bottle` or `carpet`), preventing local disk pressure while establishing end-to-end correctness.

---

## 7. Planned Development Phases

| Phase | Milestone | Scope | Status |
| :---: | :--- | :--- | :---: |
| **01** | **Project Setup & Environment Verification** | Isolated venv, Flask app, Vanilla JS client, pytest suite, diagnostics | **Complete** |
| **02** | **Dataset Pipeline & Category Ingestion** | PyTorch dataset loader, transforms, resolution standardization, ground-truth masks | *Pending Approval* |
| **03** | **Feature Reconstruction Model** | ResNet-50 feature pyramid, spatial autoencoder, reconstruction loss | *Pending* |
| **04** | **PatchCore Baseline & Heatmaps** | Patch-level memory bank, coreset subsampling, Gaussian anomaly maps | *Pending* |
| **05** | **Evaluation & Quantitative Metrics** | Image ROC-AUC, Pixel PRO-AUC, precision-recall analysis, comparative report | *Pending* |
| **06** | **Interactive Inspection Dashboard** | Image drag-and-drop, real-time inference, heatmap opacity slider, defect report | *Pending* |

---

## 8. Engineering & Security Rules

1. **Path Agility:** All file lookups utilize `pathlib.Path` relative to the project root, never machine-specific hardcoded paths.
2. **Credential & Artifact Security:** The `.gitignore` strictly protects datasets (`data/`), model checkpoints (`*.pth`, `*.pt`), environment variables (`.env`), and cache artifacts from accidental version control commits.
3. **Reproducibility:** Experiments in later phases will lock random seeds (`torch.manual_seed`, `np.random.seed`).
