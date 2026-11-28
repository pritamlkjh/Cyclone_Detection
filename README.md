# CycloneNet — Cross-Satellite Tropical Cyclone Detection (Kalpana-1 → INSAT-3D)

**CycloneNet** is a deep learning satellite-image classification system for detecting tropical cyclones over the Bay of Bengal using thermal infrared (`TIR`) and water vapour (`WV`) imagery. Trained exclusively on **Kalpana-1** satellite data (2008–2013) via two-stage transfer learning and evaluated zero-shot on unseen **INSAT-3D** satellite data (2013–2023), the model achieves **0.985 mean ROC-AUC** across 5-fold chronological cross-validation on Kalpana-1 and **0.925 ROC-AUC (88.50% accuracy, 0.762 F1)** on 1,183 independent INSAT-3D test frames.

> **Important Note on Model Backbone:** The actual trained checkpoint in [`outputs/final.pt`](outputs/final.pt) (`44.8 MB`) uses **ResNet-18** (`arch="resnet18"`), matching the default in `train.py` and `model.py`. `model.py` also supports `resnet34` and `resnet50` via the `--arch` CLI flag if you choose to retrain with a deeper backbone.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset Description & Local Setup](#dataset-description--local-setup)
3. [Data Preprocessing & 3-Channel Input Construction](#data-preprocessing--3-channel-input-construction)
4. [Validation & Cross-Satellite Testing Strategy](#validation--cross-satellite-testing-strategy)
5. [Architecture & Two-Stage Transfer Learning](#architecture--two-stage-transfer-learning)
6. [Class Imbalance, Loss, Optimizer & Threshold](#class-imbalance-loss-optimizer--threshold)
7. [Actual Evaluation Results](#actual-evaluation-results)
8. [Repository Structure](#repository-structure)
9. [Installation](#installation)
10. [Usage: Train, Evaluate, Plot & Run Predictions](#usage-train-evaluate-plot--run-predictions)
11. [Limitations & Future Improvements](#limitations--future-improvements)
12. [Documentation](#documentation)

---

## Problem Statement

Detecting tropical cyclones from geostationary meteorological satellites in real time is critical for early warning over the North Indian Ocean / Bay of Bengal. Key challenges include:
- **Sensor & Domain Shift:** Older satellites (Kalpana-1, operational 2002–2017) must transfer knowledge to newer satellites (INSAT-3D, launched 2013) without retraining on labeled target-satellite data.
- **Temporal Autocorrelation:** Geostationary captures occur every 3 hours; consecutive frames of the same storm look nearly identical, causing severe data leakage under naive random splits.
- **Severe Class Imbalance:** Clear-sky or non-cyclonic conditions heavily outnumber cyclone frames (~82% non-cyclone vs. ~18% cyclone in Kalpana-1).
- **Physical Symmetry Constraints:** Bay of Bengal cyclones spin counter-clockwise (Northern Hemisphere Coriolis effect); standard geometric augmentations like horizontal/vertical flips create physically impossible storms.

---

## Dataset Description & Local Setup

Raw satellite `.npz` datasets (~3.0 GB total) and the large preprocessed tensor cache `cache/X.npy` (`375 MB`) are **excluded** from this submission archive to meet the 100 MB upload limit. Only essential metadata (`cache/meta.csv`) and normalization statistics (`cache/norm.json`) are included.

### Dataset Summary (Original Data Used)

| Satellite | Split Role | Time Span | Cyclone (`.npz`) | Non-Cyclone (`.npz`) | Total | Raw Array Format |
|---|---|---|---:|---:|---:|---|
| **Kalpana-1 (`K1VHR_*`)** | Train & 5-Fold Year CV | 2008-06-25 to 2013-11-28 | 353 (18.02%) | 1,606 (81.98%) | **1,959** | `(500, 500)` `tir`, `wv` (`uint16`) |
| **INSAT-3D (`3DIMG_*`)** | Independent Zero-Shot Test | 2013-10-06 to 2023-12-05 | 266 (22.49%) | 917 (77.51%) | **1,183** | `(500, 500)` `tir` (`float64`), `wv` (`uint16`) |

*Note: In Kalpana-1, `2008` contains `213` non-cyclone images and `0` cyclone images, so `2008` is kept exclusively in training across all CV folds (`splits.py`).*

### Expected `.npz` File Schema & Naming
Each `.npz` file must contain two `500 × 500` arrays:
- `tir`: Thermal Infrared channel counts
- `wv`: Water Vapour channel counts

Filenames encode timestamps parsed by `common.parse_time()`:
- Kalpana-1: `K1VHR_DDMMMYYYY_HHMM_L1B_STD.npz` (e.g., `K1VHR_01NOV2010_0000_L1B_STD.npz`)
- INSAT-3D: `3DIMG_DDMMMYYYY_HHMM_L1B_STD_V01R00.npz` (e.g., `3DIMG_01MAY2019_0300_L1B_STD_V01R00.npz`)

### How to Provide Datasets Locally
Place your Kalpana folders inside either `data/` (recommended) or directly in the project root (`common.py` checks both automatically):
```text
CycloneNet_Submission/
├── data/                         # (or place folders at project root)
│   ├── Kalpana_Cyclones/*.npz
│   ├── Kalpana_Non_Cyclones/*.npz
│   ├── Insat_cyclone/*.npz       # optional location for INSAT testing
│   └── Insat_non_cyclone/*.npz
```

---

## Data Preprocessing & 3-Channel Input Construction

1. **NaN Sanitization (`common.load_raw`):** Converts `tir` and `wv` to `float32` and replaces non-finite values with `np.nan_to_num`.
2. **Area-Averaged Resizing (`common.resize_pair`):** Downsamples `500 × 500` frames to `224 × 224` (`IMG = 224`) using `cv2.INTER_AREA` to avoid aliasing when shrinking, returning shape `(2, 224, 224)` (cached as `float16` in `prepare_data.py`).
3. **Three-Channel Input Construction (`common.make_3ch`):**
   - **Channel 0:** `TIR` (Thermal Infrared) — captures cloud-top brightness/coldness.
   - **Channel 1:** `WV` (Water Vapour) — captures upper-tropospheric moisture circulation.
   - **Channel 2:** `TIR - WV` — physically highlights deep convective cloud tops characteristic of tropical cyclone eyewalls and dense overcasts, while providing a 3-channel tensor compatible with ImageNet-pretrained ResNet `conv1` weights without kernel surgery.
4. **Channel-Wise Z-Score Normalization (`common.normalise`):**
   Using `cache/norm.json` (computed on a fixed `400`-image Kalpana subsample with `RandomState(0)` and stored inside `outputs/final.pt`):
   - `mean`: `[666.5381, 864.8644, -198.3262]`
   - `std`: `[106.4971, 43.8071, 76.9899]`
5. **Physics-Preserving Training Augmentation (`dataset.py`):**
   Applied **only** during training:
   - Random circular spatial shift `(dx, dy) ∈ [-16, 16]` pixels (`np.roll`).
   - Radiometric gain jitter `× U(0.97, 1.03)` and count offset `+ U(-10, 10)` on `TIR` and `WV`.
   - Recomputes `x[2] = x[0] - x[1]` so the difference channel stays physically consistent.
   - **No flips or 90° rotations** are applied, preserving counter-clockwise Northern Hemisphere cyclonic spin.

---

## Validation & Cross-Satellite Testing Strategy

- **Kalpana-1 Chronological Year-Block CV (`splits.py`):**
  Every image from a given year resides in the same fold (`2009`, `2010`, `2011`, `2012`, `2013`). Each fold validates on one entire held-out year and trains on all remaining years (including `2008`). This eliminates temporal leakage between adjacent 3-hourly frames.
- **Final Model Training (`train.py`):**
  After 5-fold CV estimates generalization and finds the optimal decision threshold on out-of-fold predictions, the final model (`outputs/final.pt`) is trained from scratch (ImageNet initialization) on **all 1,959 Kalpana-1 frames**.
- **INSAT-3D Independent Testing (`test_insat.py`):**
  The Kalpana-trained model is evaluated directly on **1,183 INSAT-3D frames** using the identical preprocessing pipeline and Kalpana normalization statistics, testing genuine cross-satellite generalization.

---

## Architecture & Two-Stage Transfer Learning

- **Backbone (`model.py`):** `torchvision.models.resnet18` initialized with `ResNet18_Weights.IMAGENET1K_V1` (`11,177,025` total parameters). Uses residual skip connections (`y = F(x) + x`) across four stages (`layer1`–`layer4`) to stabilize gradient flow and retain multi-scale spatial features.
- **Classification Head:** Original 1000-way `fc` replaced with `nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))`, outputting a single binary logit.
- **Two-Stage Training Schedule (`train.py`):**
  1. **Stage 1 (`head`, 5 epochs, `lr = 1e-3`):** Only `fc` (`513` params) is trainable. `conv1`, `bn1`, `layer1`–`layer4` are frozen with BatchNorm layers forced into `.eval()` mode (`set_train_mode`) so pretrained running statistics are preserved.
  2. **Stage 2 (`finetune`, 15 epochs, `lr_ft = 1e-4`):** `layer3`, `layer4`, and `fc` (`10,493,953` params) are unfrozen with **discriminative learning rates**:
     - `layer3`: `0.3 × lr_ft = 3e-5`
     - `layer4`: `1.0 × lr_ft = 1e-4`
     - `fc`: `3.0 × lr_ft = 3e-4`
     - Early blocks (`conv1`, `bn1`, `layer1`, `layer2`) remain permanently frozen in `.eval()` mode to retain low-level edge and texture detectors.

---

## Class Imbalance, Loss, Optimizer & Threshold

- **Loss Function:** `nn.BCEWithLogitsLoss(pos_weight=pos_w)`, where `pos_w = N_non_cyclone / N_cyclone` per training split (`4.016`–`6.038` across CV folds; `4.550` on full Kalpana).
- **Optimizer & Scheduler:** `AdamW` (`weight_decay = 1e-4`) + `CosineAnnealingLR(T_max = stage_epochs)` per stage, batch size `32`, random seed `42`.
- **Decision Threshold (`0.46`):** Selected by grid search over `0.10..0.90` (step `0.01`) maximizing F1 score on the concatenated 5-fold out-of-fold Kalpana validation predictions, then saved inside `outputs/final.pt`.

---

## Actual Evaluation Results

### 1. Kalpana-1 5-Fold Chronological Cross-Validation (`outputs/cv_results.csv`)

| Validation Year | Val Samples (Cyc / Non) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| **2009** | 303 (39 / 264) | 0.9505 | 0.9286 | 0.6667 | 0.7761 | 0.9898 |
| **2010** | 312 (119 / 193) | 0.9103 | 0.8760 | 0.8908 | 0.8833 | 0.9632 |
| **2011** | 429 (48 / 381) | 0.9207 | 0.5875 | 0.9792 | 0.7344 | 0.9783 |
| **2012** | 319 (35 / 284) | 0.9843 | 0.8947 | 0.9714 | 0.9315 | 0.9993 |
| **2013** | 383 (112 / 271) | 0.9086 | 1.0000 | 0.6875 | 0.8148 | 0.9951 |
| **Mean ± Std** | **1,746 total val** | **0.9349 ± 0.0323** | **0.8574 ± 0.1581** | **0.8391 ± 0.1521** | **0.8280 ± 0.0797** | **0.9851 ± 0.0146** |

### 2. Independent INSAT-3D Test (`outputs/insat_predictions.csv`, `n = 1,183`, threshold `0.46`)

| Metric | Value | Confusion Matrix Breakdown | Count |
|---|---:|---|---:|
| **Accuracy** | **0.8850** (88.50%) | True Negatives (`TN`) | 829 |
| **Precision** | **0.7124** (71.24%) | False Positives (`FP`) | 88 |
| **Recall (Sensitivity)** | **0.8195** (81.95%) | False Negatives (`FN`) | 48 |
| **Specificity** | **0.9040** (90.40%) | True Positives (`TP`) | 218 |
| **F1 Score** | **0.7622** (76.22%) | | |
| **ROC-AUC** | **0.9249** | | |
| **PR-AUC (Avg Precision)** | **0.7892** | | |

### Visual Results (`outputs/plots/`)

- **Training Curves (5-Fold CV):** [`outputs/plots/training_curves.png`](outputs/plots/training_curves.png)
- **Kalpana-1 Cross-Validation Summary:** [`outputs/plots/kalpana_cv.png`](outputs/plots/kalpana_cv.png)
- **INSAT-3D Confusion Matrix & Metrics:** [`outputs/plots/insat_matrix_metrics.png`](outputs/plots/insat_matrix_metrics.png)
- **INSAT-3D ROC, Precision-Recall & Probability Separation:** [`outputs/plots/insat_roc_pr_probs.png`](outputs/plots/insat_roc_pr_probs.png)
- **Cross-Satellite Generalization (Kalpana CV vs INSAT Test):** [`outputs/plots/kalpana_vs_insat.png`](outputs/plots/kalpana_vs_insat.png)
- **Genuine INSAT-3D Sample Predictions:** [`outputs/plots/insat_sample_predictions.png`](outputs/plots/insat_sample_predictions.png)

---

## Repository Structure

```text
CycloneNet_Submission/
├── README.md
├── requirements.txt
├── .gitignore
├── common.py                 # Path resolution, .npz loader, 224x224 resizing, 3-ch builder, Z-score norm
├── dataset.py                # PyTorch Dataset with physics-preserving augmentations
├── model.py                  # ResNet backbone builder, stage freezing, BatchNorm eval control, LRs
├── prepare_data.py           # Preprocesses Kalpana .npz dataset into cache/ (X.npy, meta.csv, norm.json)
├── splits.py                 # Chronological year-block 5-fold cross-validation generator
├── train.py                  # Two-stage CV + final model training & threshold optimization
├── test_insat.py             # CLI batch evaluation on unseen INSAT-3D .npz folders
├── predict.py                # Single-image inference helpers used by CLI/GUI
├── predict_gui.py            # Interactive Tkinter desktop GUI with live channel preview & plot tabs
├── plot_results.py           # Generates evaluation plots from outputs/ CSVs
├── cache/
│   ├── meta.csv              # Metadata for all 1,959 Kalpana-1 frames (file, label, time, year)
│   └── norm.json             # Kalpana 3-channel normalization mean/std
├── outputs/
│   ├── final.pt              # Trained ResNet-18 checkpoint + norm stats + threshold (0.46)
│   ├── cv_results.csv        # Fold-level Kalpana CV metrics
│   ├── history_2009.csv ... history_2013.csv  # Epoch-by-epoch training/validation histories
│   ├── insat_predictions.csv # All 1,183 INSAT-3D test predictions
│   └── plots/                # Generated evaluation charts & genuine INSAT sample predictions
└── docs/
    ├── ARCHITECTURE.md       # Model, residual blocks, tensor flow & training mechanics
    └── PROJECT_REPORT.md     # Complete technical internship project report
```

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

*Note: `predict_gui.py` uses Python's standard-library `tkinter` (`TkAgg` backend).*

---

## Usage: Train, Evaluate, Plot & Run Predictions

### 1. Generate Evaluation Plots (Works Immediately Without Raw Datasets)
```bash
python plot_results.py
# Or display interactive Matplotlib windows:
python plot_results.py --show
```

### 2. Inspect Chronological Folds (Uses Included `cache/meta.csv`)
```bash
python splits.py
```

### 3. Single-Sample Python Inference (Requires Any `.npz` File with `tir` & `wv`)
```python
from predict import load_npz_pair, load_model_and_stats, predict_from_arrays

model, stats = load_model_and_stats("outputs/final.pt")
tir, wv = load_npz_pair("path/to/3DIMG_01MAY2019_0300_L1B_STD_V01R00.npz")
prob, verdict = predict_from_arrays(tir, wv, model, stats)
print(f"{verdict} (probability={prob:.4f})")
```

### 4. Launch the Desktop GUI
```bash
python predict_gui.py
```
Supports uploading `.npz` files or folders, batch-evaluating labelled INSAT cyclone/non-cyclone directories, previewing TIR & WV channels, and viewing all metrics/curves via **Show graphs**.

### 5. Evaluate on Full INSAT-3D Folders (When Raw INSAT `.npz` Files Are Provided)
```bash
python test_insat.py --cyclone_dir data/Insat_cyclone --non_cyclone_dir data/Insat_non_cyclone
```

### 6. Full Training Reproduction (When Raw Kalpana `.npz` Files Are Provided)
```bash
python prepare_data.py
python train.py --arch resnet18 --epochs_head 5 --epochs_ft 15
```
To train with ResNet-34 instead (optional experiment):
```bash
python train.py --arch resnet34 --epochs_head 5 --epochs_ft 15
```

---

## Limitations & Future Improvements

- **Cross-Satellite Precision Drop:** Precision decreases from `0.857` (Kalpana CV mean) to `0.712` on INSAT-3D (`88` false positives out of `917` non-cyclone images), driven by sensor radiometric differences (`float64` TIR range `[273.6, 923.1]` in INSAT vs. `uint16` `[529, 935]` in Kalpana) and seasonal convective cloud systems (e.g., winter/monsoon troughs).
- **Fixed Calibration Statistics:** Normalization uses Kalpana-1 global mean/std without per-image histogram matching or brightness-temperature conversion (`Kelvin`).
- **Frame-Level Classification Only:** Does not localize storm centers (bounding boxes/heatmaps) or estimate cyclone intensity (wind speed / central pressure).
- **Future Directions:** Brightness-temperature physical calibration across sensors, unsupervised domain adaptation / test-time BatchNorm calibration for INSAT-3D, temporal sequence modeling (ConvLSTM / 3D-CNN over consecutive 3-hourly frames), and Grad-CAM spatial explainability.

---

## Documentation
- [Technical Project Report](docs/PROJECT_REPORT.md)
- [Model & Training Architecture Specification](docs/ARCHITECTURE.md)
