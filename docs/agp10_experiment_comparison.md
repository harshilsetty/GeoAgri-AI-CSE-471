# AGP 10 — Controlled Experiments Comparative Analysis

## 1. Executive Summary

This document provides a systematic, multi-dimensional comparison of the three experimental iterations of YOLO11 Nano OBB within the GeoAgri-AI framework:
1. **Historical Baseline (`geoagri_yolo11n_obb_v1`):** Trained on `datasets/geoagri_obb_v2` (naive boundary coordinate clamping).
2. **EXP-B (`geoagri_yolo11n_obb_v2_refit`):** Trained on `datasets/geoagri_obb_v3` (refitted boundary boxes via `cv2.minAreaRect`, identical baseline hyperparameters).
3. **EXP-C (`geoagri_yolo11n_obb_v3_aug`):** Trained on `datasets/geoagri_obb_v3` with domain-shift invariance augmentations (HSV jitter, minor angle rotation, vertical reflection).

All validation evaluations were conducted on the 109 validation images (19 independent source scenes) using identical Ultralytics `OBBValidator` settings (`conf=0.001`, `iou=0.7`, `imgsz=640`, GPU 0).

---

## 2. Experimental Configurations Comparison

| Configuration Dimension | Baseline (v1) | EXP-B (Refit) | EXP-C (Domain Aug) | Controlled Variables & Differences |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset Version** | `datasets/geoagri_obb_v2` | `datasets/geoagri_obb_v3` | `datasets/geoagri_obb_v3` | v3 refits 2,701 boundary-crossing instances with `cv2.minAreaRect`. |
| **Pretrained Weights** | `yolo11n-obb.pt` | `yolo11n-obb.pt` | `yolo11n-obb.pt` | Identical backbone initialization. |
| **Input Resolution** | 640 | 640 | 640 | Identical across all runs. |
| **Batch Size** | 4 | 4 | 4 | Identical across all runs. |
| **Max Epochs / Patience**| 60 / 12 | 60 / 12 | 60 / 12 | Identical early-stopping policy. |
| **Random Seed** | 42 | 42 | 42 | Deterministic execution. |
| **Hue (`hsv_h`)** | 0.015 | 0.015 | **0.050** | Increased in EXP-C to bridge sandstone vs. red brick spectrum. |
| **Saturation (`hsv_s`)** | 0.70 | 0.70 | **0.80** | Increased in EXP-C for mineral weathering variance. |
| **Value (`hsv_v`)** | 0.40 | 0.40 | **0.50** | Increased in EXP-C for solar intensity variance. |
| **Rotation (`degrees`)** | 0.0 | 0.0 | **5.0** | Small rotational perturbation for angle head invariance. |
| **Translation (`translate`)**| 0.10 | 0.10 | **0.15** | Enhanced perimeter object boundary invariance. |
| **Vertical Flip (`flipud`)**| 0.0 | 0.0 | **0.20** | Enabled in EXP-C for aerial/oblique perspective invariance. |

---

## 3. Quantitative Validation Comparison

Evaluated strictly on the 109 validation images across 19 independent scenes:

| Performance Metric | Baseline (v1, Clamped) | EXP-B (Refit, minAreaRect) | EXP-C (Domain Aug) | Best Performer |
| :--- | :---: | :---: | :---: | :---: |
| **Stopping Epoch** | 39 (stopped) / 27 (best) | 30 (stopped) / 18 (best) | 32 (stopped) / 20 (best) | EXP-B (fastest convergence) |
| **Training Duration** | 824.4s (13.7 min) | 601.9s (10.0 min) | 661.5s (11.0 min) | EXP-B |
| **Peak GPU VRAM** | 2,810 MB | 2,727.8 MB | 3,014.6 MB | Stable across all ($<50\%$ of 6GB) |
| **Overall Precision (P)** | 0.4658 | **0.6025** | 0.5722 | **EXP-B** (+13.67% over baseline) |
| **Overall Recall (R)** | 0.3800 | 0.3731 | **0.4140** | **EXP-C** (+3.40% over baseline) |
| **Overall mAP@50** | 0.4002 | 0.4306 | **0.4497** | **EXP-C** (+4.95% over baseline) |
| **Overall mAP@50–95** | 0.2289 (0.2419 at ep39)| 0.2567 | **0.2638** | **EXP-C** (+3.49% over baseline) |

### Per-Class Performance Breakdown (Validation Split)

#### 1. Precision (P)
| Class | Baseline (v1) | EXP-B (Refit) | EXP-C (Domain Aug) | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| `boulders` | 0.3820 | 0.3725 | 0.3293 | Color jitter increased candidate proposals on rubble. |
| `others` | 0.4850 | 0.5964 | **0.8484** | Sharp reduction in false positive others. |
| `ruins` | 0.3840 | 0.5312 | **0.5823** | Clean boundary refitting + orientation jitter boosted precision. |
| `structures` | 0.6350 | **0.9239** | 0.7781 | EXP-B achieved near-perfect structural precision ($92.4\%$). |
| `vegetation` | 0.4430 | **0.5885** | 0.3230 | High vegetation density caused lower precision under color jitter. |

#### 2. Recall (R)
| Class | Baseline (v1) | EXP-B (Refit) | EXP-C (Domain Aug) | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| `boulders` | 0.2300 | 0.2079 | **0.2973** | EXP-C captured $+8.94\%$ more boulders than EXP-B. |
| `others` | 0.7650 | **0.7941** | **0.7941** | Maintained high sensitivity. |
| `ruins` | **0.1610** | 0.1254 | 0.1164 | Remains difficult due to rubble ambiguity. |
| `structures` | 0.5830 | 0.6389 | **0.7222** | **Historic high:** EXP-C recovered $72.2\%$ of structural footprints. |
| `vegetation` | **0.1610** | 0.0993 | 0.1402 | Modest recovery in EXP-C. |

#### 3. Mean Average Precision (mAP@50)
| Class | Baseline (v1) | EXP-B (Refit) | EXP-C (Domain Aug) | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| `boulders` | 0.2520 | 0.2235 | 0.2487 | Broadly stable across experiments. |
| `others` | 0.7780 | 0.8153 | **0.8504** | Consistent monotonic improvement. |
| `ruins` | 0.1830 | 0.1536 | **0.2034** | Highest mAP50 achieved in EXP-C. |
| `structures` | 0.5690 | 0.7339 | **0.7775** | Massive gain ($+20.85\%$ over baseline). |
| `vegetation` | 0.2190 | **0.2270** | 0.1687 | Peak in EXP-B. |

---

## 4. Strengths, Weaknesses, and Trade-Offs

### Candidate 1: Historical Baseline (v1, Clamped)
* **Strengths:** Established the leak-free, scene-stratified benchmark ($0.4002$ val mAP50).
* **Weaknesses:** Naive coordinate clamping created non-orthogonal trapezoids on 2,701 instances, penalizing the OBB angle head and capping structural precision at $63.5\%$.

### Candidate 2: EXP-B (Geometry-Refitted Baseline)
* **Strengths:** 
  * Reached the highest overall precision (**$0.6025$**).
  * Exceptional structural precision (**$0.9239$**), proving that geometric fidelity at image boundaries directly stabilizes the classification and regression heads.
  * Faster convergence (epoch 18 vs. 27).
* **Weaknesses:** Lower overall recall ($0.3731$), as the model became conservative in proposing bounding boxes for low-contrast instances.

### Candidate 3: EXP-C (Domain-Shift Augmentation)
* **Strengths:** 
  * Reached the highest overall validation mAP@50 (**$0.4497$**) and mAP@50–95 (**$0.2638$**).
  * Highest overall recall (**$0.4140$**) and highest structural recall (**$0.7222$**).
  * Color and rotational jittering effectively bridged mineral weathering patterns across the validation scenes.
* **Weaknesses:** Slightly reduced precision ($0.5722$ vs. $0.6025$) due to additional background false positives in dense vegetation areas.

---

## 5. Candidate Selection for Final Test Evaluation

Under the Phase 8 decision gates:
1. **Val mAP@50 $\ge 0.40$:** EXP-C ($0.4497$) > EXP-B ($0.4306$) > Baseline ($0.4002$).
2. **Structures Recall $> 0.05$:** EXP-C ($0.7222$) > EXP-B ($0.6389$) > Baseline ($0.5830$).
3. **No Unexplained Class Collapse:** EXP-C achieved the highest mAP across `structures`, `others`, and `ruins`, and highest recall across `boulders` and `structures`.

**Conclusion:** **EXP-C (`geoagri_yolo11n_obb_v3_aug`) was selected as the superior validation model** and qualified for the single, quarantined evaluation on the held-out test split.

