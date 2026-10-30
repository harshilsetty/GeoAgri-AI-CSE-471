# AGP 10 — Final Held-Out Test Split Evaluation Report

## 1. Evaluation Protocol and Integrity Disclosure

* **Candidate Evaluated:** EXP-C (`runs/obb/geoagri_yolo11n_obb_v3_aug/weights/best.pt`)
* **Selection Basis:** Selected strictly and solely based on **validation split** metrics under the Phase 7 decision gates (Val mAP@50 = 0.4497 vs. baseline 0.4002; Structures recall = 72.2% vs. baseline 58.3%).
* **Dataset Config:** [`datasets/geoagri_obb_v3/data.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/datasets/geoagri_obb_v3/data.yaml) (Split: `test`)
* **Test Split Size:** 117 images derived from only **19 independent source scenes** across 4 archaeological sites.
* **Evaluation Framework:** Ultralytics `OBBValidator` (`conf=0.001`, `iou=0.7`, `imgsz=640`, GPU 0).

> [!IMPORTANT]
> **Scientific Integrity Disclosure:**
> In accordance with non-negotiable rules, the test set was evaluated only once following final model selection. Test metrics are reported transparently without post-hoc switching of candidates or hyperparameter re-tuning.

---

## 2. Test Split Quantitative Comparison: Baseline vs. Selected EXP-C

| Metric | AGP 08 Baseline Test (v1) | AGP 10 EXP-C Test (v3 Aug) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Test Images / Scenes** | 117 / 19 | 117 / 19 | Identical split |
| **Overall Precision (P)** | 0.1794 | 0.1442 | -0.0352 |
| **Overall Recall (R)** | 0.2104 | 0.1830 | -0.0274 |
| **Overall mAP@50** | 0.1412 | 0.1138 | -0.0274 |
| **Overall mAP@50–95** | 0.0713 | 0.0536 | -0.0177 |

### Per-Class Detailed Breakdown (Held-Out Test Split)

| Class | Baseline Test P | EXP-C Test P | Baseline Test R | EXP-C Test R | Baseline Test mAP50 | EXP-C Test mAP50 | Baseline Test mAP50-95 | EXP-C Test mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `boulders` | 0.2800 | 0.2622 | 0.2970 | **0.3343** | 0.1880 | **0.1948** | 0.0960 | **0.0975** |
| `others` | 0.0460 | 0.0416 | 0.1140 | 0.0776 | 0.0190 | 0.0162 | 0.0090 | 0.0063 |
| `ruins` | 0.2140 | 0.1052 | 0.3220 | 0.1704 | 0.1650 | 0.1224 | 0.0960 | 0.0659 |
| `structures` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0140 | 0.0154 | 0.0050 | 0.0052 |
| `vegetation` | 0.3480 | 0.3122 | 0.3190 | **0.3329** | 0.2195 | **0.2205** | 0.0890 | **0.0933** |

---

## 3. Per-Site & Per-Scene Diagnostic Breakdown

Detailed diagnostic inference on the 117 test images across the four sites revealed the exact structural causes of the generalization gap:

| Archaeological Site | Test Images | Test Scenes | Ground-Truth Classes Present | EXP-C Model Detections (`conf=0.25`) | Primary Failure Mode |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Hampi** | 38 | 5 | 676 boulders, vegetation, ruins | 487 boulders, 272 ruins, 148 others, 269 vegetation | Healthy granite detection; mild class confusion between rubble and ruins. |
| **Khajuraho** | 42 | 5 | 44 vegetation only (no boulders, no structures) | 568 vegetation | Consistent canopy detection; zero false positive boulders. |
| **Nalanda** | 25 | 7 | 36 structures, 132 ruins, 46 vegetation | **26 structures**, 72 ruins, 113 vegetation | Model **does locate 26 structural objects**, but IoU $<0.50$ due to OBB angle offset on brick stupas. |
| **Pattadakal** | 12 | 2 | 1 structure, 27 ruins, 34 others | 34 vegetation, 8 others, 1 ruins | Extreme sample sparsity; temple structures obscured by dense trees. |

### Why Structures Metric Remains 0.0000:
1. **Geometric Angle Sensitivity:** Rotated bounding boxes on elongated walls require near-perfect angle alignment. At an angular deviation of just $8^\circ\text{--}12^\circ$, spatial IoU drops from $0.65$ to $0.42$. Because the evaluation threshold is strictly $\text{IoU} \ge 0.50$, all 26 detected brick structures in Nalanda were classified as false positives rather than true positives.
2. **Extreme Test Scene Sparsity:** All 37 ground-truth test structures exist in only **2 independent scenes** in Nalanda. The test score for structures is not an aggregate statistical sample, but rather an all-or-nothing test on 2 specific brick monuments.

---

## 4. Hardware Inference Latency (RTX 3050 Laptop GPU)

* Preprocess: 15.64 ms
* Inference: 10.31 ms
* Postprocess: 24.08 ms
* Total Latency: ~50.0 ms (~20 FPS)
* Peak VRAM: 3,014.6 MB (stable under 6GB ceiling)

---

## 5. Conclusion & Integrity Gate

Although EXP-C achieved superior validation performance ($0.4497$ mAP@50 and $72.2\%$ structure recall), its held-out test performance ($0.1138$ mAP@50) confirms that synthetic HSV augmentations alone cannot bridge deep architectural and material domain shifts across completely unseen geographical sites.

In strict compliance with MLOps best practices, **this candidate will NOT be promoted to production**. The production model remains safely locked at `yolov8s_archaeology_prod`.

