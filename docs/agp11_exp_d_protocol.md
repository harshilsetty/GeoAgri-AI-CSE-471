# AGP 11 — EXP-D Experiment Protocol Specification

## 1. Experiment Overview

* **Experiment Identifier:** `geoagri_yolo11n_obb_v4_class_weighted`
* **Objective:** Determine whether bounded, damped inverse square-root class-weighted optimization ($w_{\text{structures}} = 3.88$) forces the classification head to overcome the Bayesian prior favoring `ruins`, thereby improving structural detection without causing gradient collapse or excessive false positives.
* **Control Principle:** Holds every data split, augmentation parameter, network architecture, batch size, learning rate schedule, and seed identical to **EXP-C**, varying **only** the class-weighting vector.

---

## 2. Experimental Configuration Matrix

| Parameter | EXP-C (Augmentation Control) | EXP-D (Class-Weighted Target) | Control Notes |
| :--- | :--- | :--- | :--- |
| **Dataset** | `datasets/geoagri_obb_v3/data.yaml` | `datasets/geoagri_obb_v3/data.yaml` | Identical v3 dataset |
| **Model** | `yolo11n-obb.pt` | `yolo11n-obb.pt` | Identical pretrained backbone |
| **Input Resolution** | $640 \times 640$ | $640 \times 640$ | Identical |
| **Batch Size** | 4 | 4 | Conservative for 6GB VRAM |
| **Precision** | AMP (`amp=True`) | AMP (`amp=True`) | Mixed precision enabled |
| **Max Epochs / Patience** | 60 / 12 | 60 / 12 | Identical early stopping |
| **Seed** | 42 | 42 | Deterministic execution |
| **Hue (`hsv_h`)** | 0.05 | 0.05 | Preserved from EXP-C |
| **Saturation (`hsv_s`)** | 0.80 | 0.80 | Preserved from EXP-C |
| **Value (`hsv_v`)** | 0.50 | 0.50 | Preserved from EXP-C |
| **Rotation (`degrees`)** | 5.0 | 5.0 | Preserved from EXP-C |
| **Translation (`translate`)** | 0.15 | 0.15 | Preserved from EXP-C |
| **Vertical Flip (`flipud`)** | 0.20 | 0.20 | Preserved from EXP-C |
| **Class Weights ($\mathbf{w}$)** | **None (1.0 across all)** | **`[0.95, 1.41, 1.16, 3.88, 0.56]`** | **Sole experimental variable** |

---

## 3. Pre-Flight Verification & Safety Checks

Prior to executing full training, the following safety gates were validated:
1. **Unit Test Gate:** `tests/test_agp11_class_weighting.py` verified that default unweighted loss is unmodified, class weights broadcast to `(1, 1, 5)`, and gradients remain strictly finite when a batch contains 0 structures (**Passed**).
2. **GPU Memory Availability:** NVIDIA GeForce RTX 3050 Laptop GPU verified at 0.0 MB active allocated memory, 0% utilization, and temperature 54°C (**Passed**).
3. **Directory Isolation:** Target run path `runs/obb/geoagri_yolo11n_obb_v4_class_weighted/` does not conflict with existing EXP-B or EXP-C runs (**Passed**).
4. **Registry Protection:** Production model `yolov8s_archaeology_prod` remains locked and quarantined in [`config/models_registry.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/config/models_registry.yaml) (**Passed**).

