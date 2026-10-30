# GeoAgri-AI: YOLO Training Status and Artifact Verification Report (Final)

**Author:** Senior MLOps Engineer & ML Evaluator  
**Date:** October 28, 2026 (22:10 IST)  
**Project:** GeoAgri-AI  
**Target Architecture:** YOLO11 Nano Oriented Bounding Box (YOLO11n-OBB)  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)  

---

## 1. Process Status & Execution State

* **Active Background Task:** `e1fbed33-83cf-44f8-afb5-d63dd2d35f91/task-955`
* **Process Status:** **COMPLETED (Exit Code: 0)**
* **Termination Reason:** Early stopping triggered at **Epoch 39** (no fitness improvement after patience window of 12 epochs beyond best epoch 27).
* **Total Runtime:** **13.74 minutes** (824.4 seconds).
* **Peak GPU VRAM Allocated:** **3,796.67 MB** (~3.8 GB out of 6.0 GB, ~61.8% peak VRAM utilization).
* **Thermal Performance:** GPU operated stably at ~51°C to 54°C under high-performance power state `P0`.
* **Execution Errors:** **0 errors**. All 39 training epochs, validation cycles, optimizer updates, and checkpoint serialization steps completed successfully.

---

## 2. Verified Output Artifacts

| Artifact | Absolute Path | Size | Verification Status |
| :--- | :--- | :---: | :---: |
| **Run Directory** | `runs/obb/geoagri_yolo11n_obb_v1` | — | **Verified** |
| **Best Checkpoint** | `runs/obb/geoagri_yolo11n_obb_v1/weights/best.pt` | 5,888,862 bytes (~5.89 MB stripped) | **Verified & Loadable** |
| **Last Checkpoint** | `runs/obb/geoagri_yolo11n_obb_v1/weights/last.pt` | 5,888,862 bytes (~5.89 MB stripped) | **Verified & Loadable** |
| **Results CSV** | `runs/obb/geoagri_yolo11n_obb_v1/results.csv` | 6,248 bytes (39 epochs logged) | **Verified** |
| **Run Arguments** | `runs/obb/geoagri_yolo11n_obb_v1/args.yaml` | 2,430 bytes | **Verified** |
| **Validation Plots** | `runs/obb/geoagri_yolo11n_obb_v1/val_batch0_labels.jpg`, `val_batch0_pred.jpg`, `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png` | Various | **Generated on disk** |

---

## 3. Best Checkpoint Selection & Validation Results

* **Best Epoch:** **Epoch 27**
  - **Ultralytics Fitness:** **0.2577** ($0.1 \times \text{mAP50} + 0.9 \times \text{mAP50-95}$)
  - **Validation mAP@50:** **0.4002** (40.02%)
  - **Validation mAP@50-95:** **0.2419** (24.19%)
  - **Precision:** 0.4710
  - **Recall:** 0.3920
* **Per-Class Validation Performance (at Best Checkpoint):**
  - `structures`: **mAP@50 = 0.687**, Recall = 0.639, Precision = 0.701
  - `others`: **mAP@50 = 0.784**, Recall = 0.853, Precision = 0.553
  - `boulders`: **mAP@50 = 0.240**, Recall = 0.192, Precision = 0.480
  - `ruins`: **mAP@50 = 0.162**, Recall = 0.125, Precision = 0.365
  - `vegetation`: **mAP@50 = 0.128**, Recall = 0.152, Precision = 0.254
* **Validation Inference Latency:** **7.1 ms** inference per image on RTX 3050 GPU.

---

## 4. Conclusion

The training phase of the candidate model `runs/obb/geoagri_yolo11n_obb_v1/weights/best.pt` is completely verified. We proceed directly to Phase 2 (Annotation Quality Report) and Phase 3 (Independent Test Split Evaluation).
