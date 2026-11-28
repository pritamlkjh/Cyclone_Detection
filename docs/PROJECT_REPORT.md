# CycloneNet Technical Project Report

**Project Title:** Cross-Satellite Tropical Cyclone Detection Over the Bay of Bengal via Two-Stage Transfer Learning  
**Submission Target:** Labmentix Internship  
**Model Artifact:** `outputs/final.pt` (`ResNet-18`, `44,779,147` bytes, decision threshold `0.46`)  

---

## 1. Executive Summary

CycloneNet addresses automated binary detection of tropical cyclones (`Cyclone` vs. `Non-Cyclone`) from geostationary meteorological satellite imagery over the Bay of Bengal. The system is trained solely on **Kalpana-1** imagery (2008–2013, `1,959` samples) and evaluated zero-shot on **INSAT-3D** imagery (2013–2023, `1,183` samples) without ever adapting weights or normalization statistics to INSAT-3D.

Key engineering contributions:
1. **Physics-aware 3-channel representation** (`[TIR, WV, TIR - WV]`) enabling standard ImageNet-pretrained ResNet backbones to capture deep convective cloud-top contrast without modifying `conv1`.
2. **Leakage-free chronological cross-validation** holding out complete calendar years (`2009`–`2013`) to prevent near-duplicate 3-hourly frames from inflating validation scores.
3. **Physics-preserving augmentation** prohibiting horizontal/vertical flips and 90° rotations so Northern Hemisphere counter-clockwise cyclone rotation is preserved.
4. **Two-stage transfer learning** (`head` warmup → discriminative `finetune` on `layer3`, `layer4`, `fc` with frozen BatchNorm layers in early blocks).

Verified experimental performance:
- **Kalpana-1 5-Fold Year CV:** `93.49%` mean accuracy, `0.8280` mean F1, **`0.9851` mean ROC-AUC**.
- **Independent INSAT-3D Test (`n = 1,183`):** `88.50%` accuracy, `71.24%` precision, `81.95%` recall, `90.40%` specificity, `0.7622` F1, **`0.9249` ROC-AUC**, and `0.7892` PR-AUC.

---

## 2. Problem Formulation & Domain Challenges

Geostationary meteorological satellites provide continuous 3-hourly snapshots across Thermal Infrared (`TIR`) and Water Vapour (`WV`) bands. Converting these observations into an automated cyclone detector introduces four non-trivial ML challenges:

| Challenge | Manifestation in Dataset | CycloneNet Solution |
|---|---|---|
| **Temporal Autocorrelation** | Adjacent `.npz` frames are 3 hours apart; random k-fold splits leak nearly identical cloud structures across train/val. | Chronological year-block cross-validation (`splits.py`). |
| **Severe Class Imbalance** | Only `353 / 1,959` (`18.02%`) Kalpana frames are cyclones (`4.55:1` non-cyclone ratio). | Dynamically weighted `BCEWithLogitsLoss(pos_weight=N_neg/N_pos)` + OOF F1 threshold tuning (`0.46`). |
| **Cross-Satellite Domain Shift** | Kalpana-1 (`K1VHR`, `uint16` counts) vs. INSAT-3D (`3DIMG`, `float64` TIR / `uint16` WV) have distinct calibration curves and sensor distributions. | Shared preprocessing pipeline (`common.py`), radiometric gain/offset jitter during training (`dataset.py`), and frozen low-level ResNet blocks. |
| **Coriolis Spin Asymmetry** | Bay of Bengal cyclones spin counter-clockwise; flipping an image creates physically invalid clockwise storms. | Spatial translation (`±16 px`) + radiometric jitter only; flips/rotations excluded. |

---

## 3. Dataset Analysis

### 3.1 Kalpana-1 Training & Cross-Validation Corpus (`cache/meta.csv`)

- **Date Range:** `2008-06-25 00:00` to `2013-11-28 21:00`
- **Total Files:** `1,959` `.npz` archives (`353` Cyclone, `1,606` Non-Cyclone)

| Year | Non-Cyclone (`0`) | Cyclone (`1`) | Total | Cyclone Share | Role in 5-Fold CV |
|---:|---:|---:|---:|---:|---|
| **2008** | 213 | 0 | 213 | 0.0% | Always in training (no positive validation frames) |
| **2009** | 264 | 39 | 303 | 12.9% | Fold 1 validation (`train pos_weight = 4.274`) |
| **2010** | 193 | 119 | 312 | 38.1% | Fold 2 validation (`train pos_weight = 6.038`) |
| **2011** | 381 | 48 | 429 | 11.2% | Fold 3 validation (`train pos_weight = 4.016`) |
| **2012** | 284 | 35 | 319 | 11.0% | Fold 4 validation (`train pos_weight = 4.157`) |
| **2013** | 271 | 112 | 383 | 29.2% | Fold 5 validation (`train pos_weight = 5.539`) |
| **Total** | **1,606** | **353** | **1,959** | **18.0%** | Full-dataset `pos_weight = 4.550` |

### 3.2 INSAT-3D Independent Test Corpus

- **Total Files:** `1,183` `.npz` archives (`266` Cyclone across 2013–2023; `917` Non-Cyclone across 2014–2019)
- **Cyclone Year Distribution:** 2013 (`33`), 2014 (`15`), 2015 (`15`), 2016 (`45`), 2018 (`30`), 2019 (`22`), 2020 (`36`), 2021 (`26`), 2022 (`11`), 2023 (`33`).
- **Non-Cyclone Year Distribution:** 2014 (`340`), 2015 (`365`), 2016 (`117`), 2017 (`13`), 2019 (`82`).
- **Zero-Shot Isolation:** Neither INSAT statistics nor INSAT samples were used for normalization (`cache/norm.json`), hyperparameter selection, or threshold optimization (`0.46`).

---

## 4. Methodology

### 4.1 Preprocessing & Feature Engineering
1. Raw `500 × 500` `tir` and `wv` arrays are loaded as `float32`, sanitized with `np.nan_to_num`, and downsampled to `224 × 224` using `cv2.INTER_AREA`.
2. Channel 3 is computed as `TIR - WV`, creating `[TIR, WV, TIR - WV]` (`(3, 224, 224)`). Cold, optically thick convective cyclone tops exhibit distinct brightness differences relative to mid/upper-level water vapour absorption, making `TIR - WV` a strong physical discriminator.
3. Inputs are standardized using Kalpana subsample statistics (`n = 400`, seed `0`):
   - `mean = [666.5381, 864.8644, -198.3262]`
   - `std  = [106.4971, 43.8071, 76.9899]`

### 4.2 Backbone & Two-Stage Training Configuration
- **Model:** `ResNet-18` (`IMAGENET1K_V1`, `11,177,025` params) with `Dropout(0.3) -> Linear(512, 1)` head. *(Note: `model.py` supports `resnet18`, `resnet34`, and `resnet50`; the submitted trained checkpoint `outputs/final.pt` is `resnet18`.)*
- **Stage 1 (`head`, 5 epochs):** Freeze `conv1`, `bn1`, `layer1..4` (keeping BatchNorm in `.eval()` mode). Train only `fc` (`513` params) with `AdamW(lr=1e-3, weight_decay=1e-4)` and `CosineAnnealingLR`.
- **Stage 2 (`finetune`, 15 epochs):** Unfreeze `layer3`, `layer4`, `fc` (`10,493,953` params) with discriminative learning rates (`3e-5`, `1e-4`, `3e-4`) and `CosineAnnealingLR`; keep `conv1`, `bn1`, `layer1`, `layer2` permanently frozen in `.eval()` mode.

---

## 5. Experimental Results & Analysis

### 5.1 Stage 1 vs. Stage 2 Convergence Dynamics (`outputs/history_*.csv`)

Across all five chronological folds, Stage 1 (`head` training on frozen ImageNet features) reaches only modest separation before Stage 2 (`finetune`) transforms performance:

| Val Fold | End of Stage 1 (Epoch 5 `head`) | End of Stage 2 (Epoch 15 `finetune`) | Peak Validation AUC (Epoch) |
|---:|---|---|---|
| **2009** | loss `0.790`, F1 `0.209`, AUC `0.477` | loss `0.0045`, F1 `0.776`, AUC `0.990` | `0.9934` (finetune ep 6) |
| **2010** | loss `0.805`, F1 `0.518`, AUC `0.582` | loss `0.0023`, F1 `0.883`, AUC `0.963` | `0.9645` (finetune ep 13) |
| **2011** | loss `0.810`, F1 `0.280`, AUC `0.793` | loss `0.0019`, F1 `0.734`, AUC `0.978` | `0.9812` (finetune ep 6) |
| **2012** | loss `0.801`, F1 `0.267`, AUC `0.833` | loss `0.0033`, F1 `0.932`, AUC `0.999` | `0.9994` (finetune ep 13) |
| **2013** | loss `0.893`, F1 `0.638`, AUC `0.905` | loss `0.0025`, F1 `0.815`, AUC `0.995` | `0.9964` (finetune ep 8) |

**Observation:** Natural-image high-level features (`layer3`/`layer4` from ImageNet) are suboptimal for meteorological infrared fields on their own, but unfreezing `layer3` and `layer4` immediately drives validation AUC above `0.96` in every fold within 1–2 fine-tuning epochs while retaining generic low-level edge detectors in `layer1`/`layer2`.

### 5.2 Kalpana-1 5-Fold Cross-Validation Summary (`outputs/cv_results.csv`)

- **Mean Accuracy:** `0.9349 ± 0.0323`
- **Mean Precision:** `0.8574 ± 0.1581`
- **Mean Recall:** `0.8391 ± 0.1521`
- **Mean F1:** `0.8280 ± 0.0797`
- **Mean ROC-AUC:** `0.9851 ± 0.0146`

Year-level variance is expected in meteorological datasets: `2011` experienced lower precision (`0.5875`) at high recall (`0.9792`), whereas `2013` achieved `1.0000` precision at `0.6875` recall (`0.9951` AUC), reflecting threshold sensitivity across different cyclone seasons. Maximizing F1 on concatenated out-of-fold predictions produced the global decision threshold **`0.46`**.

### 5.3 Independent INSAT-3D Evaluation (`outputs/insat_predictions.csv`)

On `1,183` unseen INSAT-3D frames (`266` cyclone, `917` non-cyclone) at threshold `0.46`:

- **Confusion Matrix:** `TN = 829`, `FP = 88`, `FN = 48`, `TP = 218`
- **Accuracy:** `0.8850` | **Precision:** `0.7124` | **Recall:** `0.8195` | **Specificity:** `0.9040` | **F1:** `0.7622` | **ROC-AUC:** `0.9249` | **PR-AUC:** `0.7892`

**Probability Calibration & Confidence:**
- **True Non-Cyclone (`n = 917`):** median probability `0.0003`, 75th percentile `0.0087`, mean `0.1003`
- **True Cyclone (`n = 266`):** median probability `0.9985`, 25th percentile `0.9226`, mean `0.8159`

**Threshold Sensitivity on INSAT-3D (Informational — not tuned on test data):**

| Threshold | Accuracy | Precision | Recall | F1 | Confusion (`TN / FP / FN / TP`) |
|---:|---:|---:|---:|---:|---|
| 0.30 | 0.8833 | 0.6988 | 0.8459 | 0.7653 | 820 / 97 / 41 / 225 |
| 0.40 | 0.8850 | 0.7083 | 0.8308 | 0.7647 | 826 / 91 / 45 / 221 |
| **0.46 (Kalpana OOF)** | **0.8850** | **0.7124** | **0.8195** | **0.7622** | **829 / 88 / 48 / 218** |
| 0.50 | 0.8859 | 0.7162 | 0.8158 | 0.7627 | 831 / 86 / 49 / 217 |
| 0.60 | 0.8901 | 0.7297 | 0.8120 | 0.7687 | 837 / 80 / 50 / 216 |

Even without tuning on INSAT-3D, the Kalpana OOF threshold (`0.46`) operates within `0.0065` F1 of the post-hoc peak, confirming clean probability separation across satellites.

### 5.4 Genuine INSAT-3D Prediction Examples (`outputs/plots/insat_sample_predictions.png`)

Verified directly using `predict.py` and `outputs/final.pt`:

| Filename | Timestamp (UTC) | True Label | Predicted Prob | Verdict | Category |
|---|---|---|---:|---|---|
| `3DIMG_01MAY2019_0300_L1B_STD_V01R00.npz` | 2019-05-01 03:00 | Cyclone (`1`) | `1.0000` | `CYCLONE` | True Positive |
| `3DIMG_17MAY2020_0300_L1B_STD_V01R00.npz` | 2020-05-17 03:00 | Cyclone (`1`) | `1.0000` | `CYCLONE` | True Positive |
| `3DIMG_11MAY2023_1200_L1B_STD_V01R00.npz` | 2023-05-11 12:00 | Cyclone (`1`) | `1.0000` | `CYCLONE` | True Positive |
| `3DIMG_09OCT2013_2100_L1B_STD_V01R00.npz` | 2013-10-09 21:00 | Cyclone (`1`) | `0.9962` | `CYCLONE` | True Positive |
| `3DIMG_01APR2015_1500_L1B_STD_V01R00.npz` | 2015-04-01 15:00 | Non-Cyclone (`0`) | `0.0023` | `NO CYCLONE` | True Negative |
| `3DIMG_25JAN2014_0000_L1B_STD_V01R00.npz` | 2014-01-25 00:00 | Non-Cyclone (`0`) | `0.0000` | `NO CYCLONE` | True Negative |
| `3DIMG_01DEC2015_0000_L1B_STD_V01R00.npz` | 2015-12-01 00:00 | Non-Cyclone (`0`) | `1.0000` | `CYCLONE` | False Positive (winter deep convection) |
| `3DIMG_26JUL2015_0300_L1B_STD_V01R00.npz` | 2015-07-26 03:00 | Cyclone (`1`) | `0.0000` | `NO CYCLONE` | False Negative (weak/early disturbance) |

---

## 6. Conclusion & Recommendations

CycloneNet demonstrates that a lightweight `ResNet-18` backbone, combined with physically motivated `[TIR, WV, TIR - WV]` feature construction, strict chronological validation, physics-preserving augmentations, and two-stage discriminative fine-tuning, generalizes effectively from Kalpana-1 (`0.985` CV AUC) to unseen INSAT-3D observations (`0.925` test AUC).

Recommended next steps for production deployment:
1. Convert raw satellite counts to calibrated **brightness temperatures (K)** before normalization to remove sensor-specific scaling drift between INSAT-3D/3DR/3DS and Kalpana-1.
2. Add **Grad-CAM** overlays in `predict_gui.py` so forecasters can visually confirm the model focuses on the cyclonic eyewall/spiral bands.
3. Evaluate **ResNet-34** (`python train.py --arch resnet34`) and **temporal smoothing** across consecutive 3-hourly captures to reduce isolated false positives (`FP = 88`) and missed early-stage depressions (`FN = 48`).
