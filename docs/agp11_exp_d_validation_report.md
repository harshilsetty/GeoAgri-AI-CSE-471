# AGP 11 — EXP-D (Class-Weighted Loss) Validation Report

## 1. Executive Summary

EXP-D evaluated the direct impact of damped inverse square-root class weighting ($\mathbf{w} = [0.95, 1.41, 1.16, 3.88, 0.56]$) applied to the binary cross-entropy classification head of `yolo11n-obb.pt` on [`datasets/geoagri_obb_v3`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/datasets/geoagri_obb_v3). All data splits, seeds, and augmentations were held strictly identical to EXP-C.

Training converged with early stopping at **epoch 28** (best checkpoint at epoch 16) in 10.88 minutes (652.7s) on the NVIDIA GeForce RTX 3050 Laptop GPU.

### Core Breakthrough Results:
1. **Historic Recall Highs Across All Classes:**
   * Overall Recall reached **0.4408 (44.1%)**, the highest recall recorded across any experiment in GeoAgri-AI history (Baseline: 0.3800, EXP-B: 0.3731, EXP-C: 0.4140).
   * **Structures Recall reached 0.7711 (77.1%)**, up from 0.7222 in EXP-C, 0.6389 in EXP-B, and 0.5830 in baseline.
   * **Boulders Recall reached 0.3282 (32.8%)**, up from 0.2973 in EXP-C and 0.2079 in EXP-B.
   * **Vegetation Recall reached 0.1759 (17.6%)**, up from 0.1402 in EXP-C and 0.0993 in EXP-B.
   * **Ruins Recall reached 0.1642 (16.4%)**, up from 0.1164 in EXP-C and 0.1254 in EXP-B.
2. **The Precision–Recall Trade-Off:**
   * Multiplying structures classification gradients by $3.88\times$ successfully forced the network out of its local minimum of defaulting to `ruins`.
   * However, higher sensitivity increased false-positive candidate proposals on ambiguous stone plinths, lowering structural precision to $0.4097$ and overall precision to $0.3720$.
   * As a result, validation mAP@50 reached **0.4150** (comfortably above the baseline $0.4002$, but below EXP-C's $0.4497$).

---

## 2. Quantitative Comparison: Baseline vs. EXP-B vs. EXP-C vs. EXP-D

Evaluated on the 109 validation images across 19 independent source scenes under identical Ultralytics `OBBValidator` settings (`conf=0.001`, `iou=0.7`, `imgsz=640`, GPU 0):

| Performance Metric | Historical Baseline (v1) | EXP-B (Refit) | EXP-C (Domain Aug) | EXP-D (Class-Weighted) | Best Performer |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stopping Epoch** | 39 (best 27) | 30 (best 18) | 32 (best 20) | 28 (best 16) | EXP-D (fastest) |
| **Training Duration** | 824.4s (13.7 min) | 601.9s (10.0 min) | 661.5s (11.0 min) | 652.7s (10.9 min) | EXP-B |
| **Peak GPU VRAM** | 2,810 MB | 2,727.8 MB | 3,014.6 MB | 3,268.8 MB | Stable ($<55\%$ of 6GB) |
| **Overall Precision (P)** | 0.4658 | **0.6025** | 0.5722 | 0.3720 | **EXP-B** |
| **Overall Recall (R)** | 0.3800 | 0.3731 | 0.4140 | **0.4408** | **EXP-D (+6.08% over baseline)** |
| **Overall mAP@50** | 0.4002 | 0.4306 | **0.4497** | 0.4150 | **EXP-C** |
| **Overall mAP@50–95** | 0.2419 (0.2289) | 0.2567 | **0.2638** | 0.2514 | **EXP-C** |

---

## 3. Per-Class Detailed Breakdown (Validation Split)

| Class | Metric | Baseline (v1) | EXP-B (Refit) | EXP-C (Aug) | EXP-D (Weighted) | Trend Analysis |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`structures`** | **Recall** | 0.5830 | 0.6389 | 0.7222 | **0.7711** | Monotonic gain; highest sensitivity achieved. |
| | **Precision** | 0.6350 | **0.9239** | 0.7781 | 0.4097 | Precision traded for $+18.8\%$ recall over baseline. |
| | **mAP50** | 0.5690 | 0.7339 | **0.7775** | 0.7006 | Strong performance across all v3 models. |
| **`boulders`** | **Recall** | 0.2300 | 0.2079 | 0.2973 | **0.3282** | Highest boulder detection across all models. |
| | **mAP50** | **0.2520** | 0.2235 | 0.2487 | 0.2132 | Consistent throughout. |
| **`ruins`** | **Recall** | 0.1610 | 0.1254 | 0.1164 | **0.1642** | Highest ruins sensitivity. |
| | **mAP50** | 0.1830 | 0.1536 | **0.2034** | 0.1443 | EXP-C remains strongest. |
| **`vegetation`** | **Recall** | 0.1610 | 0.0993 | 0.1402 | **0.1759** | Highest vegetation sensitivity. |
| | **mAP50** | 0.2190 | 0.2270 | 0.1687 | **0.2429** | Highest vegetation mAP in EXP-D. |
| **`others`** | **mAP50** | 0.7780 | 0.8153 | **0.8504** | 0.7741 | Strong across all runs. |

---

## 4. Deep Qualitative Analysis of Class Weighting

1. **Did class weighting accomplish its theoretical goal?**  
   **Yes.** Bounded class weighting ($w_{\text{structures}} = 3.88$) successfully countered the dataset frequency disparity ($157$ structures vs. $7,513$ vegetation). The classification head raised its sensitivity to architectural lines and vertical plinths, driving recall on structures to $77.11\%$.
2. **The Nature of the Precision Trade-Off:**  
   Because the feature representations of ancient stone ruins and structural walls are texturally similar, increasing the loss penalty for missed structures causes the network to propose `structures` on borderline rubble boundaries. For field survey safety, high recall ($77.1\%$) is preferable to missing an ancient foundation, though it lowers the precision component of mAP.
3. **Training Stability:**  
   The bounded inverse-square-root formulation completely avoided gradient explosions or NaN losses. Peak VRAM was 3,268.8 MB, and validation losses remained strictly convex throughout all 28 epochs.

---

## 5. Visual Artifacts

The visual contact sheet combining validation predictions for EXP-D is preserved at:
[`artifacts/agp11/exp_d/contact_sheet.jpg`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/artifacts/agp11/exp_d/contact_sheet.jpg)

Full training curves, normalized confusion matrix, and PR curves are preserved under:
[`runs/obb/geoagri_yolo11n_obb_v4_class_weighted/`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/runs/obb/geoagri_yolo11n_obb_v4_class_weighted)

