# AGP 09 — Controlled Experiment Matrix & Hypothesis Verification Plan

**Author:** Senior Machine Learning Research Engineer & Experimentation Lead  
**Date:** October 28, 2026  
**Project:** GeoAgri-AI  
**Framework:** Ultralytics YOLO11 (Task: OBB)  
**Compute Budget:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)  

---

## 1. Experiment Governance & Evaluation Protocol

To prevent test-set contamination and guarantee scientific validity:
1. **Validation Split as Development Benchmark:** All iterative experimentation (hyperparameter tuning, augmentation, loss reweighting) will be evaluated exclusively on `datasets/geoagri_obb_v2/val` (109 images).
2. **Held-Out Test Set Quarantine:** The 117 test images (19 unseen historical scenes) remain strictly quarantined until the final winning candidate is selected.
3. **Reproducibility Mandate:** Fixed random seeds (`seed=42`), versioned dataset paths, and preserved configuration YAMLs.

---

## 2. Structured Experiment Matrix

| Exp ID | Experiment Name | Primary Hypothesis | Dataset Version | Training Parameters | Compute / Time | Acceptance Gate |
| :---: | :--- | :--- | :---: | :--- | :---: | :--- |
| **EXP-A** | **Current Baseline (Control)** | Establishes reproducible baseline on verified dataset. | `geoagri_obb_v2` | `yolo11n-obb.pt`, 60 ep, batch 4, imgsz 640, AdamW | ~13.7 min (RTX 3050) | Baseline: Val mAP50 = 0.400, Test mAP50 = 0.141 |
| **EXP-B** | **Geometric Refit Remediation** | Eliminates boundary distortion in 360 flagged border instances via minAreaRect refitting. | `geoagri_obb_v3` | Identical to EXP-A | ~13.7 min (RTX 3050) | Val mAP50 $\ge 0.420$, corner IoU improved |
| **EXP-C** | **Multi-Site Cross Augmentation** | Synthetic color-jitter and perspective rotation bridge geological domain gaps across sites. | `geoagri_obb_v3` | EXP-A + `hsv_h=0.03, hsv_s=0.8, degrees=15.0, perspective=0.0005` | ~15.0 min (RTX 3050) | Val mAP50 $\ge 0.440$, Nalanda brick recall $+15\%$ |
| **EXP-D** | **Class-Frequency Reweighted Loss** | Inverse class weighting compensates for `structures` (1.25%) and `ruins` (12.9%) sample scarcity. | `geoagri_obb_v3` | EXP-C + `cls_pw` reweighting ($w_{\text{struct}} \approx 4.0, w_{\text{ruins}} \approx 2.0$) | ~15.0 min (RTX 3050) | `structures` Val Recall $\ge 0.50$, Val mAP50 $\ge 0.450$ |
| **EXP-E** | **Model Capacity Exploration (YOLO11s-OBB)** | Higher backbone capacity (9.4M params) improves fine-grained feature discrimination. | `geoagri_obb_v3` | `yolo11s-obb.pt`, 60 ep, batch 2 (to fit 6GB VRAM), imgsz 640 | ~26.0 min (RTX 3050) | Val mAP50 $\ge 0.480$, P95 Latency $\le 33\text{ ms}$ |

---

## 3. In-Depth Technical Justifications

### 3.1 EXP-B: Geometric Refit Remediation
* **Rationale:** In AGP 09 Phase 2, 360 instances (1.95% of annotations) experienced $>15\%$ area change from naive coordinate clamping. By replacing vertex clamping with rotated polygon refitting (`cv2.minAreaRect` on the intersection with the unit square), exact 4-point OBB compliance is maintained without unnatural angle shifts.

### 3.2 EXP-D: Class-Frequency Reweighting vs. Blind Focal Loss
* **Focal Loss Technical Compatibility:** Ultralytics natively employs BCEWithLogitsLoss for classification, with an optional `cls_pw` parameter representing positive class weights. Applying uncalibrated focal loss $\gamma > 2.0$ risks suppressing minority class gradients during early epochs when class probabilities are low.
* **Controlled Strategy:** We compute effective class inverse frequencies:
  $$w_c = \sqrt{\frac{N_{\max}}{N_c}}$$
  For `structures`: $w_3 = \sqrt{\frac{10620}{230}} \approx 6.79$ (clamped to 4.0 to prevent training instability).

### 3.3 EXP-E: Capacity Scaling vs. VRAM Ceiling
* **Why YOLO11s-OBB?** YOLO11n has 2.65M parameters and 6.6 GFLOPs. Its compact feature maps struggle to distinguish crumbled brick from natural rock texture. YOLO11s increases parameters to 9.4M and GFLOPs to 21.5.
* **VRAM Safety:** Benchmarking indicates YOLO11s with `batch=2` and `imgsz=640` requires ~4.6 GB VRAM during training, remaining safely below the 6.0 GB ceiling of the RTX 3050 Laptop GPU.

---

## 4. Rollback and Promotion Procedure

If any experiment fails to exceed baseline validation metrics:
1. Halt execution immediately.
2. Purge checkpoint weights from staging.
3. Revert `config/models_registry.yaml` active slot to `yolov8s_archaeology_prod`.
4. Document the negative result in experiment logs.

