# GeoAgri-AI: Final YOLO11n-OBB Evaluation & Performance Report

**Author:** Senior Computer Vision Engineer & ML Evaluator  
**Date:** October 28, 2026  
**Project:** GeoAgri-AI  
**Evaluated Checkpoint:** `runs/obb/geoagri_yolo11n_obb_v1/weights/best.pt`  
**Dataset Config:** `datasets/geoagri_obb_v2/data.yaml`  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)  
**Evaluation Split:** Strictly held-out test split (19 unseen source scenes, 117 images, 2,799 instances)  

---

## 1. Executive Summary

This report documents the rigorous, independent evaluation of the candidate model `YOLO11n-OBB` on the leak-free test partition of `datasets/geoagri_obb_v2`. 

While the model achieved an apparent **0.4002 mAP@50** on the validation split, independent evaluation on the untouched test split yielded an overall **0.1412 mAP@50** (14.12%) and **0.0713 mAP@50-95** (7.13%). Crucially, the minority class `structures` dropped to **0.0266 mAP@50** with **0.0000 recall**. This stark delta uncovers a significant domain-shift challenge across historical archaeological sites, underscoring the critical necessity of leak-free group-aware splitting and demonstrating why validation metrics alone cannot justify production deployment.

On hardware efficiency, however, the model demonstrated outstanding inference speed: **14.49 ms median end-to-end latency** (**63.1 FPS**) and **54.23 MB allocated VRAM**, comfortably meeting all real-time requirements.

---

## 2. Independent Test Split Evaluation Metrics

### 2.1 Overall Performance Summary

$$\text{Test Split: 117 images, 19 independent historical scenes, 2,799 annotated instances}$$

| Metric | Measured Test Result | Validation Split (for comparison) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Precision (P)** | **0.1794** | 0.4710 | -0.2916 |
| **Recall (R)** | **0.2104** | 0.3920 | -0.1816 |
| **mAP@50** | **0.1412** (14.12%) | 0.4002 (40.02%) | -0.2590 |
| **mAP@50-95** | **0.0713** (7.13%) | 0.2419 (24.19%) | -0.1706 |

### 2.2 Per-Class Performance Breakdown

| Class ID | Class Name | Test Instances | Test Precision | Test Recall | Test mAP@50 | Test mAP@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | boulders | 676 | 0.3481 | 0.3089 | **0.2452** | 0.1405 |
| **1** | others | 123 | 0.1575 | 0.1702 | **0.0920** | 0.0431 |
| **2** | ruins | 311 | 0.1180 | 0.2283 | **0.1418** | 0.0711 |
| **3** | structures | 37 | **0.0000** | **0.0000** | **0.0266** | 0.0113 |
| **4** | vegetation | 1,652 | 0.2736 | 0.3444 | **0.2005** | 0.0906 |

### 2.3 Analysis of the Minority Class (`structures`)
In the test split, `structures` is represented by only 37 instances across 3 distinct scenes. In these unseen scenes, the architectural features (carved granite pillars and brick stupa bases) differ visually from the Khajuraho temples in the training set. Consequently, the model fails to detect `structures` with high confidence, frequently confusing them with `ruins` ($mAP50 = 0.0266$, $Recall = 0.0000$).

---

## 3. Visual Diagnostic Inspection & Failure Modes

Visual diagnostic predictions were generated across test images and compiled into:
* **Diagnostic Contact Sheet:** [`artifacts/yolo_vision_v2/evaluation/predictions/prediction_contact_sheet.jpg`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/artifacts/yolo_vision_v2/evaluation/predictions/prediction_contact_sheet.jpg)
* **Prediction Manifest:** [`artifacts/yolo_vision_v2/evaluation/predictions/visual_predictions_manifest.json`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/artifacts/yolo_vision_v2/evaluation/predictions/visual_predictions_manifest.json)

### Key Qualitative Observations:
1. **Accurate Detections:**
   - Large boulders in Hampi rocky landscapes are accurately detected with well-oriented bounding boxes aligned to the granite contours.
   - Dense vegetation patches are consistently captured with rotated polygons matching tree canopy boundaries.
2. **False Positives & Over-Segmentation:**
   - Soil texture and shadow patterns are occasionally misidentified as `others` or small `boulders`.
3. **Class Confusion (`ruins` vs `structures`):**
   - Temple columns and standing brick walls are frequently detected as `ruins` rather than `structures`. The visual boundary between collapsed stone and preserved stonework requires multi-scale contextual features that YOLO11n's compact receptive field struggles to distinguish.
4. **Small Object Misses:**
   - Tiny vegetation tufts and distant rubble fragments ($<20 \times 20$ pixels) are frequently omitted below the 0.25 confidence threshold.

---

## 4. Hardware Inference Latency & VRAM Benchmarks

Inference performance was profiled using 100 timed iterations on the NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) using native mixed precision (FP16 / AMP) at $640 \times 640$ resolution:

| Metric | Measured Value | Real-Time Target (<33.3 ms / 30 FPS) |
| :--- | :---: | :---: |
| **Model-Only Inference (Median)** | **9.40 ms** | PASS |
| **Model-Only Inference (Mean)** | **10.25 ms** | PASS |
| **Model-Only Inference (P95)** | **15.84 ms** | PASS |
| **End-to-End Latency (Median)** | **14.49 ms** | PASS |
| **End-to-End Latency (Mean)** | **15.84 ms** | PASS |
| **End-to-End Latency (P95)** | **21.99 ms** | PASS |
| **End-to-End Latency (P99)** | **28.75 ms** | PASS |
| **Throughput (FPS)** | **63.1 FPS** | PASS (>2x real-time margin) |
| **GPU Memory Allocated** | **54.23 MB** | PASS (<1% of 6GB VRAM) |
| **GPU Memory Peak** | **77.66 MB** | PASS |

---

## 5. Dataset Limitations

1. **Source Scene Scarcity:** Although the dataset contains 740 images, they originate from only **128 base photographs**. The test split comprises only **19 independent scenes**.
2. **Site-Level Domain Shift:** Each archaeological site has distinctive rock geology (Hampi granite vs Khajuraho sandstone vs Nalanda brick). An unseen site presents substantial visual domain shift.
3. **Label Sparsity for Structures:** With only 230 total instances across the entire dataset (and 37 in the test set), statistical confidence in `structures` detection is limited.

---

## 6. Reproducible Evaluation Command

To independently reproduce this evaluation at any time:

```powershell
.venv\Scripts\python.exe scripts/eval_geoagri_yolo11n_obb.py `
    --weights runs/obb/geoagri_yolo11n_obb_v1/weights/best.pt `
    --data datasets/geoagri_obb_v2/data.yaml `
    --device 0 `
    --imgsz 640 `
    --batch 4
```
Raw metrics output saved to [`docs/final_yolo_test_metrics.json`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/docs/final_yolo_test_metrics.json).

