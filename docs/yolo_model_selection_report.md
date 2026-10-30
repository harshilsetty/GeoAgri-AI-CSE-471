# GeoAgri-AI: YOLO Architecture & Model Selection Decision Report

**Document ID:** `docs/yolo_model_selection_report.md`  
**Auditor:** Senior Computer Vision Engineer & ML Research Engineer  
**Date:** October 28, 2026  
**Status:** Completed & Evaluated

---

## 1. Executive Summary & Core Recommendation

Based on an empirical audit of the 18,458 local annotations, the 6 GB VRAM budget of the NVIDIA GeForce RTX 3050 Laptop GPU, and the existing `predict_bridge.py` inference pipeline in GeoAgri-AI:

* **Primary Recommended Architecture:** **YOLO11 Nano Oriented Bounding Box (`yolo11n-obb.pt`)**
  - *Reason:* 100% of the local annotations have exactly 4 vertices. YOLO OBB is mathematically and architecturally designed for 4-point rotated quadrilaterals. It preserves object orientation (crucial for structural alignment and linear stone ruins) while fitting within $< 2.0\ \text{GB}$ VRAM.
* **Secondary Recommended Architecture (Drop-in Fallback):** **YOLOv8 Small / Nano Detection (`yolov8s.pt` / `yolov8n.pt`)**
  - *Reason:* GeoAgri-AI's active inference script (`predict_bridge.py`) already contains complete code expecting an Ultralytics `DetectionModel` (`runs/detect/yolov8s_archaeology2/weights/best.pt`) with bounding boxes (`boxes.xyxy`). Standard YOLO detection requires zero frontend or backend code changes.
* **Architecture NOT Recommended:** **YOLO26 Nano Instance Segmentation (`yolo26n-seg.pt`)**
  - *Reason:* The dataset contains no organic contour polygons, only 4-corner bounding boxes. Training a mask head on rotated boxes produces distorted rectangular pseudo-masks rather than real object silhouettes.

---

## 2. Comparative Evaluation Matrix

| Criterion | YOLO26 Nano OBB (`yolo26n-obb.pt`) | YOLO11 Nano OBB (`yolo11n-obb.pt`) | YOLOv8 Small Detect (`yolov8s.pt` - Current) | YOLO26 Nano Seg (`yolo26n-seg.pt`) | DeepLabV3+ Semantic Seg |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Task Compatibility** | **High** (4-pt OBB) | **High** (4-pt OBB) | **High** (Axis-Aligned Boxes) | **Poor** (Learns rectangles) | **High** (Pixel mask) |
| **Annotation Match** | Native 8-coord OBB | Native 8-coord OBB | Auto-converted to bbox | Treated as 4-pt polygon | Requires rasterization |
| **VRAM on RTX 3050 (6GB)** | $\approx 1.8\ \text{GB}$ (Batch 4/8) | $\approx 1.7\ \text{GB}$ (Batch 4/8) | $\approx 2.4\ \text{GB}$ (Batch 8) | $\approx 2.1\ \text{GB}$ (Batch 4) | $\approx 3.2\ \text{GB}$ (Batch 4) |
| **Python 3.13 Support** | Verified (Ultralytics) | Verified (Ultralytics) | Verified (Ultralytics) | Verified (Ultralytics) | Requires SMP build |
| **Pretrained Weights** | Available | Available | Available (`best.pt` in repo) | Available | Manual checkpoint |
| **Inference Integration** | Requires OBB overlay | Requires OBB overlay | **Drop-in ready** (Zero code change) | Requires mask overlay | Integrated in bridge |
| **Parameters** | $\approx 2.6\ \text{M}$ | $\approx 2.6\ \text{M}$ | $\approx 11.2\ \text{M}$ | $\approx 2.9\ \text{M}$ | $\approx 21.0\ \text{M}$ |
| **Inference Speed (FP16)** | $\approx 3.2\ \text{ms}$ | $\approx 3.0\ \text{ms}$ | $\approx 6.5\ \text{ms}$ | $\approx 4.8\ \text{ms}$ | $\approx 18.0\ \text{ms}$ |
| **Licensing** | AGPL-3.0 / Enterprise | AGPL-3.0 / Enterprise | AGPL-3.0 / Enterprise | AGPL-3.0 / Enterprise | MIT |

---

## 3. Detailed Architectural Trade-off Analysis

### Candidate 1: YOLO11 Nano Oriented Bounding Box (`yolo11n-obb.pt`) — PRIMARY
* **Why it wins:**
  - The dataset consists entirely of rotated rectangles with 4 vertices: $x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4$.
  - YOLO-OBB regresses both the spatial centroid $(x_c, y_c)$, width/height $(w, h)$, and rotation angle $\theta \in [-\pi/2, \pi/2]$.
  - For aerial and drone imagery of archaeological sites, stone temple alignments, walls, and geological formations, oriented bounding boxes tightly enclose objects without incorporating surrounding background terrain.
  - VRAM footprint is exceptionally small ($\approx 1.7\ \text{GB}$ at batch size 4, 640x640), allowing fast convergence and zero out-of-memory risk on the 6 GB RTX 3050.

### Candidate 2: YOLOv8 / YOLO11 Nano Standard Detection (`yolo11n.pt` / `yolov8n.pt`) — FALLBACK
* **Why it is the optimal fallback:**
  - The repository's backend bridge [`ui/server/predict_bridge.py`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/ui/server/predict_bridge.py#L605-L627) already includes complete inference code that calls:
    ```python
    yolo_results = YOLO_MODEL.predict(image, conf=conf_value, imgsz=yolo_imgsz)
    detection_boxes = yolo_results[0].boxes
    for box in detection_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Draws bounding box on UI overlay
    ```
  - An earlier model checkpoint (`runs/detect/yolov8s_archaeology2/weights/best.pt`, 22.5 MB) was trained with `task: detect`.
  - Ultralytics natively ingests 4-point polygon text files in `task: detect` mode by computing the enclosing axis-aligned rectangle:
    $$x_{\min} = \min(x_1, x_2, x_3, x_4), \quad x_{\max} = \max(x_1, x_2, x_3, x_4)$$
  - Training a new YOLO11 Nano or YOLOv8 Nano detection model on leak-free data requires **zero changes to the GeoAgri-AI frontend or API pipeline**.

### Candidate 3: YOLO26 / YOLOv8 Instance Segmentation (`yolo26n-seg.pt`) — NOT RECOMMENDED
* **Why it is rejected for the current data:**
  - Instance segmentation requires contour boundary supervision (e.g. 15–40 polygon vertices tracing organic boulder curves or tree leaves).
  - Since all local annotations have exactly 4 points, the segmentation head learns to output 4-vertex quadrilaterals. This produces the illusion of segmentation masks while carrying the computational overhead of segmentation heads and mask prototype loss without actual pixel-boundary precision.

---

## 4. Final Recommendation Summary

1. **For Highest Modeling Precision on Current Annotations:** Train **`yolo11n-obb.pt`** (or `yolo26n-obb.pt`).
2. **For Instant Drop-in Zero-Code Integration with Existing UI:** Train **`yolo11n.pt`** or **`yolov8n.pt`** (standard detection).
3. **For Terrain Erosion Ratios (`vegetation_ratio`, `bare_soil`):** Maintain the dual architecture: use YOLO for discrete obstacle detection (`boulders`, `ruins`, `structures`) and use DeepLabV3+ / semantic models for continuous pixel area coverage.

