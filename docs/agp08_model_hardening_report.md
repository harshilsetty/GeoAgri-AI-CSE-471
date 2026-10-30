# AGP 08 — GeoAgri-AI CV Model Hardening & Acceptance Criteria Report

**Author:** Senior Computer Vision Engineer & MLOps Lead  
**Date:** October 28, 2026  
**Project:** GeoAgri-AI  
**Evaluated Candidate:** YOLO11n-OBB (`runs/obb/geoagri_yolo11n_obb_v1/weights/best.pt`)  
**Production Baseline:** YOLOv8s Detection (`runs/detect/yolov8s_archaeology2/weights/best.pt`)  

---

## 1. Executive Summary

This report defines the explicit model acceptance criteria, architectural hardening safeguards, and production boundary controls for computer vision models within GeoAgri-AI. 

Following the completion of YOLO11n-OBB training (39 epochs, early stopped, peak VRAM 3.8 GB) and rigorous evaluation on the leak-free test split (117 images across 19 unseen historical source scenes), the candidate model demonstrated outstanding hardware throughput (**14.49 ms median latency**, **63.1 FPS**, **54.2 MB VRAM**), but exhibited severe domain-shift degradation on unseen scenes (**0.1412 mAP@50**, and **0.0000 recall on structures**).

Under our explicit acceptance criteria, the candidate model is designated as a **VALIDATION CANDIDATE (NOT PRODUCTION READY)**. The production baseline is safely preserved via an explicit Model Registry.

---

## 2. Explicit Model Acceptance Criteria

Before any computer vision model can be promoted to active production in GeoAgri-AI, it must satisfy all of the following seven gates:

| Gate | Evaluation Domain | Threshold / Standard | Candidate Model Status | Gate Verdict |
| :---: | :--- | :--- | :---: | :---: |
| **G1** | **Overall Test Detection** | Test $mAP@50 \ge 0.35$ on leak-free unseen scenes | Measured: **0.1412** | **FAIL** |
| **G2** | **Minority Class Recall** | `structures` Test Recall $\ge 0.30$ | Measured: **0.0000** | **FAIL** |
| **G3** | **Hardware Latency** | End-to-End P95 Latency $\le 33.3\text{ ms}$ ($\ge 30\text{ FPS}$) | Measured: **21.99 ms (63.1 FPS)** | **PASS** |
| **G4** | **GPU VRAM Ceiling** | Peak VRAM during inference $\le 1.0\text{ GB}$ (6GB budget) | Measured: **77.66 MB** | **PASS** |
| **G5** | **Input Hardening** | 100% rejection of corrupt, empty, tiny, or invalid inputs | 8/8 regression tests pass | **PASS** |
| **G6** | **API Contract Preservation** | Zero breaking changes to existing response schemas | Verified backward-compatible | **PASS** |
| **G7** | **Zero-Detection Stability** | No crash / division-by-zero on images with 0 detections | Tested on pure black images | **PASS** |

**Summary Gate Verdict:** **FAIL (2 critical gates failed: G1 and G2)**. The model cannot be promoted to production without targeted retraining on multi-site structural architecture.

---

## 3. Implemented Inference Hardening Safeguards

The inference bridge ([`ui/server/predict_bridge.py`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/ui/server/predict_bridge.py)) was hardened with defensive safeguards:

1. **Path & Type Safety:** Rejects non-string/PathLike inputs with `TypeError` and non-existent files with `FileNotFoundError`.
2. **Zero-Byte File Guard:** Detects and halts execution on empty (0-byte) image files with clear `ValueError`.
3. **Format Whitelist:** Restricts allowable file extensions to `{".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}`.
4. **Corrupt Buffer Guard:** Verifies OpenCV image decoding (`image_bgr is not None and image_bgr.size > 0`).
5. **Dimension Floor Guard:** Rejects images smaller than $16 \times 16$ pixels to prevent CNN stride underflow.
6. **Zero-Detection Resilience:** Guarded box extraction `detection_boxes = yolo_results[0].boxes if (yolo_results and yolo_results[0].boxes is not None) else []`, preventing `TypeError: 'NoneType' object is not iterable` when an image has zero detections.
7. **Orientation-Preserving Polygon Drawing:** Detects whether YOLO output contains `.obb` or `.boxes` dynamically, rendering 4-point rotated polylines (`cv2.polylines`) for OBBs and axis-aligned rectangles for standard detections.

---

## 4. Prioritized Remediation Plan

To advance the model from **VALIDATION CANDIDATE** to **PRODUCTION READY**:

* **P0 (Critical Blocker):** Augment the minority `structures` class. Collect at least 150 additional independent photographs of South Indian temple architecture (Hampi, Pattadakal, Badami, Aihole) with varied focal lengths.
* **P1 (High Priority):** Implement cross-site domain adaptation using mosaic augmentation tuned to balance boulder and ruin texture similarities.
* **P2 (Medium Priority):** Add test-time augmentation (TTA) with multiscale inference ($[512, 640, 768]$) to boost recall on distant rubble and small vegetation clusters.

