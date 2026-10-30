# AGP 10 — EXP-B (Geometry-Refitted Baseline) Validation Report

## 1. Executive Summary

EXP-B evaluated the direct impact of geometry-refitted annotations (`cv2.minAreaRect`) using [`datasets/geoagri_obb_v3`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/datasets/geoagri_obb_v3). All hyperparameters, architecture (`yolo11n-obb.pt`), seeds, and data-loaders were held identical to the AGP 08 baseline.

Training converged with early stopping at **epoch 30** (best model saved at epoch 18).
* **Validation mAP@50 reached 0.4306**, an absolute gain of **+3.04 percentage points** over the clamped baseline (0.4002).
* **Validation mAP@50–95 reached 0.2567**, an absolute gain of **+2.78 percentage points** over the baseline (0.2289).
* **Overall Precision surged to 0.6025** (+13.67 percentage points over baseline 0.4658), driven by sharp reductions in boundary false positives.
* **Minority-Class (`structures`) Performance improved dramatically on validation:**
  * Precision: **0.9239** (up from 0.6350)
  * Recall: **0.6389** (up from 0.5830)
  * mAP@50: **0.7339** (up from 0.5690)
  * mAP@50–95: **0.4940** (up from 0.3700)

Per non-negotiable rules, EXP-B was evaluated strictly on the validation split. No evaluation on the held-out test split was performed.

---

## 2. Quantitative Comparison: Baseline vs. EXP-B

Evaluated on the 109 validation images (19 independent scenes) under identical Ultralytics `OBBValidator` settings (`conf=0.001`, `iou=0.7`, `imgsz=640`, GPU 0).

| Metric | AGP 08 Baseline (v1, Clamped) | AGP 10 EXP-B (v2 Refit, minAreaRect) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Epochs Trained** | 60 (full budget) | 30 (early stopped, best at 18) | -30 epochs |
| **Training Duration** | 824.4s (13.7 min) | 601.9s (10.0 min) | -222.5s |
| **Peak GPU VRAM** | 2,810 MB | 2,727.8 MB | -82.2 MB |
| **Overall Precision (P)** | 0.4658 | **0.6025** | **+0.1367** |
| **Overall Recall (R)** | **0.3800** | 0.3731 | -0.0069 |
| **Overall mAP@50** | 0.4002 | **0.4306** | **+0.0304** |
| **Overall mAP@50–95** | 0.2289 | **0.2567** | **+0.0278** |

### Per-Class Detailed Breakdown (Validation Split)

| Class | Baseline P | EXP-B P | Baseline R | EXP-B R | Baseline mAP50 | EXP-B mAP50 | Baseline mAP50-95 | EXP-B mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `boulders` | 0.3820 | 0.3725 | **0.2300** | 0.2079 | **0.2520** | 0.2235 | **0.1170** | 0.0976 |
| `others` | 0.4850 | **0.5964** | 0.7650 | **0.7941** | 0.7780 | **0.8153** | 0.4430 | **0.4985** |
| `ruins` | 0.3840 | **0.5312** | **0.1610** | 0.1254 | **0.1830** | 0.1536 | **0.1030** | 0.0897 |
| `structures` | 0.6350 | **0.9239** | 0.5830 | **0.6389** | 0.5690 | **0.7339** | 0.3700 | **0.4940** |
| `vegetation` | 0.4430 | **0.5885** | **0.1610** | 0.0993 | 0.2190 | **0.2270** | 0.1120 | 0.1038 |

---

## 3. Analysis & Key Insights

1. **Why Precision Rose by 13.7 Points:**
   In the clamped baseline dataset (v2), coordinates were arbitrarily truncated at $0.0$ and $1.0$, creating non-orthogonal trapezoids. The OBB angle regression head was penalized when fitting true rectangles to deformed trapezoids. Replacing clamped boundary boxes with true minimum-area bounding rectangles eliminated artificial aspect-ratio distortion, allowing the network to make cleaner, higher-confidence predictions.
2. **Impact on Minority Class (`structures`):**
   Structural features benefit most from clean geometric borders because architectural foundations have strict rectilinear geometries. Eliminating edge truncation raised structural validation precision to $92.39\%$ and validation recall to $63.89\%$, driving structural mAP@50 from $0.569$ to $0.734$.
3. **Inference Latency Profile:**
   * Preprocess: 1.44 ms
   * Inference: 9.79 ms
   * Postprocess: 22.38 ms
   * Total latency: ~33.6 ms per image (~30 FPS), well within operational limits for real-time edge survey on the RTX 3050.

---

## 4. Visual Verification

The visual contact sheet combining validation batch predictions was generated and saved to:
[`artifacts/agp10/exp_b/contact_sheet.jpg`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/artifacts/agp10/exp_b/contact_sheet.jpg)

Full training curves, normalized confusion matrix, and PR curves are preserved under:
[`runs/obb/geoagri_yolo11n_obb_v2_refit/`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/runs/obb/geoagri_yolo11n_obb_v2_refit)

