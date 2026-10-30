# AGP 12B — EXP-E Validation-Only Evaluation Report

## 1. Executive Summary & Evaluation Configuration

This report details the standalone, independent validation of **EXP-E** (`geoagri_yolo11s_obb_v5_class_weighted/weights/best.pt`), evaluated strictly against the **held-in validation partition**. 

In strict adherence to evaluation integrity principles, the 117-scene held-out test split was **not accessed, evaluated, or consulted** during this evaluation.

* **Model Architecture:** YOLO11s-OBB (109 fused layers, 9,700,722 parameters, 22.4 GFLOPs)
* **Weights Checkpoint:** [`runs/obb/geoagri_yolo11s_obb_v5_class_weighted/weights/best.pt`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/runs/obb/geoagri_yolo11s_obb_v5_class_weighted/weights/best.pt)
* **Dataset Config:** [`datasets/geoagri_obb_v3/data.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/datasets/geoagri_obb_v3/data.yaml)
* **Split Partition:** Validation split (`val`)
* **Sample Units:**
  - **Tile / Image Count:** **109 images**
  - **Unique Source Scenes:** **19 scenes** (Hampi: 6, Khajuraho: 6, Nalanda: 3, Pattadakal: 4)
  - **Total Annotated Instances:** **2,442 instances**
* **Raw JSON Metrics:** [`docs/agp12_exp_e_val_metrics.json`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/docs/agp12_exp_e_val_metrics.json)

---

## 2. Overall Validation Metrics

| Metric | Score | Scientific Interpretation |
| :--- | :---: | :--- |
| **Overall Precision (P)** | **0.4680** | Moderate precision across all 5 classes; 53.2% false discovery rate |
| **Overall Recall (R)** | **0.3717** | Detects roughly 37.2% of all ground-truth target instances |
| **mAP@50** | **0.3797** | Modest overall detection at standard IoU threshold |
| **mAP@50-95** | **0.2297** | Stringent boundary and orientation alignment score |

---

## 3. Per-Class Performance Breakdown

| Class Name | Target Images (Scenes) | Ground Truth Instances | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`boulders`** | 24 (6 Hampi scenes) | 582 | 0.4828 | 0.2509 | 0.2761 | 0.1290 |
| **`others`** | 22 (6 Khajuraho scenes) | 34 | 0.4530 | 0.7059 | 0.6475 | 0.4546 |
| **`ruins`** | 37 (6 Hampi, 3 Nalanda) | 335 | 0.3907 | 0.1224 | 0.1665 | 0.0863 |
| **`structures`** | 21 (4 Pattadakal, 3 Nalanda, 6 Khajuraho) | 36 | **0.5868** | **0.5556** | **0.5404** | **0.3515** |
| **`vegetation`** | 109 (All 19 scenes) | 1,455 | 0.4267 | 0.2241 | 0.2679 | 0.1272 |

---

## 4. Failure Mode & Confusion Pattern Analysis

1. **`ruins` Detection Collapse (Recall: 0.1224):**
   * Only 41 of 335 ruins were detected. 
   * In Hampi granite terrain, collapsed masonry walls share identical spectral reflectance and texture with natural talus debris and granitic tors, causing severe background confusion.
2. **`structures` Precision vs Recall Trade-Off:**
   * Precision reached **0.5868**, which is substantially higher than EXP-D (0.4097), but recall dropped from **0.7711 (EXP-D)** to **0.5556**.
   * The higher capacity model became overly conservative, rejecting partially obscured temple walls or heavily weathered structures unless high-confidence structural corners were visible.
3. **`boulders` Spatial Boundary Fragmentation:**
   * While mAP@50 improved to 0.2761, recall remained low at 0.2509 due to dense boulder clusters in Hampi merging into single polygon predictions rather than individual oriented boxes.

---

## 5. Hardware Latency Profile

Benchmarked across 109 validation images on the NVIDIA RTX 3050 Laptop GPU (batch size = 4, imgsz = 640):

| Stage | Latency per Image | Share of Frame Budget | Effective Throughput |
| :--- | :---: | :---: | :---: |
| **Pre-process (letterbox, format)** | 1.20 ms | 3.3% | — |
| **Neural Inference (forward pass)** | **13.84 ms** | 38.5% | $\approx 72.3$ FPS |
| **Post-process (rotated NMS)** | **20.88 ms** | 58.1% | — |
| **Total End-to-End Latency** | **35.92 ms** | 100.0% | **$\approx 27.8$ FPS** |

* **Latency Assessment:** At 35.92 ms/image, EXP-E satisfies the real-time threshold (< 50 ms) for local tile rendering, but introduces a **36% latency increase** in forward pass over YOLO11n (10.2 ms vs 13.8 ms).

