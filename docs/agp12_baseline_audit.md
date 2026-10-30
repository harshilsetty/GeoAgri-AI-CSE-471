# AGP 12 — EXP-D Baseline Audit and Artifact Preservation Report

## 1. Executive Summary

This audit independently verifies all artifacts, metrics, and hyperparameter configurations from **EXP-D (`geoagri_yolo11n_obb_v4_class_weighted`)** prior to commencing architectural scaling experiments in AGP 12.

---

## 2. Quantitative Metric Audit

All reported validation and test metrics were reconciled against raw JSON evaluation summaries and run logs:

| Metric | Documented in AGP 11 | Verified in Raw Artifacts (`docs/agp11_exp_d_*.json`) | Discrepancy |
| :--- | :---: | :---: | :---: |
| **Validation mAP@50** | 0.4150 | 0.415014 | None |
| **Validation mAP@50–95**| 0.2514 | 0.251427 | None |
| **Validation Precision** | 0.3720 | 0.371993 | None |
| **Validation Recall** | 0.4408 | 0.440823 | None |
| **Validation Structures Recall** | 0.7711 (77.1%) | 0.771104 | None |
| **Validation Structures mAP50** | 0.7006 | 0.700612 | None |
| **Held-Out Test mAP@50** | 0.1289 | 0.128890 | None |
| **Held-Out Test mAP@50–95**| 0.0591 | 0.059090 | None |
| **Held-Out Test Precision** | 0.2024 | 0.202429 | None |
| **Held-Out Test Recall** | 0.1994 | 0.199373 | None |
| **Held-Out Test Structures Recall** | 0.0716 (7.16%) | 0.071559 | None |
| **Held-Out Test Structures mAP50** | 0.0417 | 0.041664 | None |

---

## 3. Training Configuration & Loss Formulation Verification

* **Output Directory:** [`runs/obb/geoagri_yolo11n_obb_v4_class_weighted/`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/runs/obb/geoagri_yolo11n_obb_v4_class_weighted)
* **Checkpoints:** `weights/best.pt` (Epoch 16, 5.6 MB) and `weights/last.pt` (Epoch 28, 5.6 MB).
* **Dataset Config:** [`datasets/geoagri_obb_v3/data.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/datasets/geoagri_obb_v3/data.yaml) (740 images: 514 train, 109 val, 117 test).
* **Class Weights Attached:** $\mathbf{w} = [0.95, 1.41, 1.16, 3.88, 0.56]$ attached via `on_pretrain_routine_end` hook in `v8OBBLoss`.
* **Augmentations:** `hsv_h=0.05`, `hsv_s=0.80`, `hsv_v=0.50`, `degrees=5.0`, `translate=0.15`, `flipud=0.20`.
* **Hardware Profile:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM), Peak VRAM: 3,268.8 MB.

---

## 4. Evaluation Integrity & Production Registry Safeguards

1. **Quarantine Compliance:** EXP-D was trained strictly on `train` and tuned solely against `val`. Test-set evaluation occurred strictly once after final model selection.
2. **Production Registry Preserved:** Audited [`config/models_registry.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/config/models_registry.yaml):
   * `active_models.production_vision`: Strictly remains `yolov8s_archaeology_prod`.
   * `geoagri_yolo11n_obb_v4_class_weighted`: Categorized under `models:` with `status: "experimental"`.
3. **Artifact Integrity:** All EXP-D weights, logs, curves, and prediction summaries remain intact.

