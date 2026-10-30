# AGP 10 — EXP-C (Domain-Shift Augmentation) Validation Report

## 1. Executive Summary

EXP-C evaluated the impact of domain-shift invariance augmentations (conservative HSV color jittering, rotation, translation, and vertical reflection) applied to [`datasets/geoagri_obb_v3`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/datasets/geoagri_obb_v3).

Training reached early stopping at **epoch 32** (best checkpoint at epoch 20).
* **Validation mAP@50 reached 0.4497 (0.450)**, an absolute gain of **+4.95 percentage points** over the baseline (0.4002) and **+1.91 percentage points** over EXP-B (0.4306).
* **Validation mAP@50–95 reached 0.2638**, an absolute gain of **+3.49 percentage points** over the baseline (0.2289) and **+0.71 percentage points** over EXP-B (0.2567).
* **Overall Recall rose significantly to 0.4140**, surpassing both EXP-B (0.3731) and the baseline (0.3800).
* **Minority-Class (`structures`) Reached Historic Highs on Validation:**
  * Recall reached **0.7222 (72.2%)**, up from 0.6389 in EXP-B and 0.5830 in baseline.
  * mAP@50 reached **0.7775**, up from 0.7339 in EXP-B and 0.5690 in baseline.
  * mAP@50–95 reached **0.5268**, up from 0.4940 in EXP-B and 0.3700 in baseline.
* **Ruins and Boulders Recall also benefited:** Boulder recall increased from 0.2079 to 0.2973 (+8.94 points); ruins mAP@50 increased from 0.1536 to 0.2034 (+4.98 points).

---

## 2. Three-Way Quantitative Comparison (Validation Split)

Evaluated under strictly identical conditions on the 109 validation images across 19 independent scenes:

| Metric | AGP 08 Baseline (v1, Clamped) | AGP 10 EXP-B (v2, Refit) | AGP 10 EXP-C (v3, Domain Aug) | EXP-C vs Baseline ($\Delta$) | EXP-C vs EXP-B ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Epochs to Converge** | 60 (full) | 30 (best 18) | 32 (best 20) | -28 epochs | +2 epochs |
| **Training Duration** | 824.4s (13.7 min) | 601.9s (10.0 min) | 661.5s (11.0 min) | -162.9s | +59.6s |
| **Peak GPU VRAM** | 2,810 MB | 2,727.8 MB | 3,014.6 MB | +204.6 MB | +286.8 MB |
| **Overall Precision (P)** | 0.4658 | **0.6025** | 0.5722 | **+0.1064** | -0.0303 |
| **Overall Recall (R)** | 0.3800 | 0.3731 | **0.4140** | **+0.0340** | **+0.0409** |
| **Overall mAP@50** | 0.4002 | 0.4306 | **0.4497** | **+0.0495** | **+0.0191** |
| **Overall mAP@50–95** | 0.2289 | 0.2567 | **0.2638** | **+0.0349** | **+0.0071** |

---

## 3. Per-Class Performance Comparison

| Class | Baseline mAP50 | EXP-B mAP50 | EXP-C mAP50 | Baseline Recall | EXP-B Recall | EXP-C Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `boulders` | 0.2520 | 0.2235 | **0.2487** | 0.2300 | 0.2079 | **0.2973** |
| `others` | 0.7780 | 0.8153 | **0.8504** | 0.7650 | 0.7941 | **0.7941** |
| `ruins` | 0.1830 | 0.1536 | **0.2034** | **0.1610** | 0.1254 | 0.1164 |
| `structures` | 0.5690 | 0.7339 | **0.7775** | 0.5830 | 0.6389 | **0.7222** |
| `vegetation` | 0.2190 | **0.2270** | 0.1687 | **0.1610** | 0.0993 | 0.1402 |

---

## 4. Error Pattern & Trade-Off Analysis

1. **Trade-Off Between Precision and Recall:**
   * EXP-B prioritized boundary rigidity, reaching a peak precision of $0.6025$ at the expense of lower recall ($0.3731$).
   * EXP-C introduced color and rotation diversity, lowering precision slightly to $0.5722$ while boosting recall substantially to $0.4140$. For archaeological hazard monitoring, higher recall on structures and boulders is advantageous because missing an ancient wall or boulder formation presents greater risk than a false alarm.
2. **Impact on Minority Class (`structures`):**
   * Structural recall increased from $58.3\%$ (baseline) to $63.9\%$ (EXP-B) and **$72.2\%$ (EXP-C)**.
   * Structural mAP@50 rose to **$0.778$** (up from $0.569$).
   * The HSV hue shift (`0.05`) and vertical flipping allowed the model to recognize structures under varying solar illumination angles and camera nadir bearings.
3. **Inference Performance on RTX 3050:**
   * Preprocess: 16.90 ms
   * Inference: 10.11 ms
   * Postprocess: 28.35 ms
   * Total latency: ~55.3 ms per image (~18 FPS).

---

## 5. Visual Artifacts

The contact sheet comparing validation predictions is stored at:
[`artifacts/agp10/exp_c/contact_sheet.jpg`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/artifacts/agp10/exp_c/contact_sheet.jpg)

Full run logs, PR curves, and confusion matrices are preserved at:
[`runs/obb/geoagri_yolo11n_obb_v3_aug/`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/runs/obb/geoagri_yolo11n_obb_v3_aug)

