# GeoAgri-AI: YOLO Annotation Format and Geometry Audit Report

**Document ID:** `docs/yolo_annotation_format_audit.md`  
**Auditor:** Senior Computer Vision Engineer & Dataset Quality Auditor  
**Dataset Inspected:** `..\Archaeological-Site-Mapping-AI-V2\Archaeological-Site-Mapping-AI-V2\dataset`  
**Configuration File:** `..\Archaeological-Site-Mapping-AI-V2\Archaeological-Site-Mapping-AI-V2\data.yaml`  
**Date:** October 28, 2026  
**Status:** Completed & Verified

---

## 1. Executive Summary

An exhaustive coordinate-by-coordinate geometric audit of all 740 label files (18,458 object instances) in the local dataset was conducted.

### Primary Audit Findings
1. **Annotation Format Verification:**
   - **100.0% of instances (18,458 of 18,458)** consist of **exactly 4 coordinate pairs (8 float values)** preceded by an integer class ID:
     $$\text{Label Line:} \quad \langle \text{class\_id} \rangle \quad x_1 \quad y_1 \quad x_2 \quad y_2 \quad x_3 \quad y_3 \quad x_4 \quad y_4$$
   - Zero instances contain arbitrary polygon masks ($N > 4$ points).
   - Zero instances contain standard 4-token axis-aligned bounding boxes ($\langle \text{class} \rangle\ x_c\ y_c\ w\ h$).
2. **Geometric Reality — Oriented Bounding Boxes (OBB):**
   - Analysis of consecutive edge vectors demonstrates that opposite sides are parallel and adjacent sides are perpendicular ($90^\circ \pm 1^\circ$).
   - The annotations represent **Oriented Bounding Boxes (OBBs) / rotated rectangles**, exported as 4-vertex quadrilaterals.
3. **Instance Segmentation Incompatibility:**
   - The dataset was reportedly created under the Roboflow project `archaeological-site-segmentation` (`seg-v1`).
   - However, because the annotations do not trace amorphous object boundaries (such as irregular boulder silhouettes or winding tree canopies), passing these labels to a YOLO instance segmentation model (`yolo-seg`) will train the segmentation head to predict rotated rectangular polygons rather than real masks.
4. **Out-of-Bounds Artifacts:**
   - **198 instances** contain normalized coordinates outside $[0.0, 1.0]$ (e.g., $-0.072$ or $1.068$), caused by unclipped rotational augmentations applied by Roboflow.

---

## 2. Dataset Configuration and Class Schema

From `data.yaml`:
```yaml
path: dataset
train: train/images
val: valid/images
test: test/images

names:
  0: boulders
  1: others
  2: ruins
  3: structures
  4: vegetation
```

* **Number of Classes:** 5
* **Class ID Verification:** Every class ID in all 18,458 lines strictly falls within $\{0, 1, 2, 3, 4\}$. Zero non-integer or out-of-schema class IDs exist.
* **Negative Images:** Exactly 0 empty label files. Every image contains at least one annotation.

---

## 3. Coordinate Geometry and Vertex Count Distribution

| Vertex Count ($N$) | Coordinate Count ($2N$) | Number of Instances | Percentage | Geometry Type |
| :---: | :---: | :---: | :---: | :--- |
| 2 points | 4 floats | 0 | 0.0% | Standard axis-aligned box ($x_c, y_c, w, h$) |
| 3 points | 6 floats | 0 | 0.0% | Triangle |
| **4 points** | **8 floats** | **18,458** | **100.0%** | **Oriented Bounding Box / Quadrilateral** |
| 5–10 points | 10–20 floats | 0 | 0.0% | Coarse Polygon |
| $>10$ points | $>20$ floats | 0 | 0.0% | Fine Contour Polygon |

### Representative Instance Vector Analysis (`hampi_01_jpg.rf.3383badc8b7180cbd82c152fd6e4c24e.txt`)

* **Instance 0 (Class 2 - Ruins):**
  - $P_1 = (0.5840, 0.4869)$
  - $P_2 = (0.5736, 0.1894)$
  - $P_3 = (0.3206, 0.1983)$
  - $P_4 = (0.3310, 0.4957)$
  - Vector $\vec{v}_{12} = (-0.0104, -0.2975)$, Length $= 0.2977$
  - Vector $\vec{v}_{23} = (-0.2530, +0.0089)$, Length $= 0.2532$
  - Vector $\vec{v}_{34} = (+0.0104, +0.2974)$, Length $= 0.2976$
  - Vector $\vec{v}_{41} = (+0.2530, -0.0088)$, Length $= 0.2532$
  - Dot product $\vec{v}_{12} \cdot \vec{v}_{23} \approx 0.0000$ (Angle: $90.0^\circ$)
  - **Verdict:** True mathematical rectangle rotated at $\approx 88^\circ$.

---

## 4. Boundary and Normalization Audit

* **Coordinate Range Expected:** $[0.0, 1.0]$
* **Extreme Min Coordinate Found:** $-0.0721$
* **Extreme Max Coordinate Found:** $+1.0684$
* **Total Instances with Vertices Outside $[0.0, 1.0]$:** **198 instances (1.07%)**
  - All 198 cases occur in augmented images where rotation or affine shearing pushed a corner outside the canvas.
  - Full itemized log available at: [`docs/local_image_dataset_annotation_issues.csv`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/docs/local_image_dataset_annotation_issues.csv).

---

## 5. Compatibility Matrix Across YOLO Tasks

| YOLO Task | Target Head | Compatibility Status | Technical Assessment |
| :--- | :--- | :---: | :--- |
| **YOLO Oriented Bounding Box (`obb`)** | `yolo-obb` (e.g. `yolo11n-obb`, `yolov8n-obb`) | **EXCELLENT (Native Match)** | 100% of labels are 4-vertex rotated boxes. Natively supported by Ultralytics OBB loader. |
| **YOLO Standard Object Detection (`detect`)** | `yolo-detect` (e.g. `yolo11n`, `yolov8s`) | **EXCELLENT (Automatic Conversion)** | Ultralytics converts 4-point polygons to bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ automatically during dataset loading. Matches existing `best.pt` checkpoint. |
| **YOLO Instance Segmentation (`segment`)** | `yolo-segment` (e.g. `yolo26n-seg`, `yolov8n-seg`) | **POOR / MISALIGNED** | Model will ingest the 4 vertices as a polygon, but will learn to output rotated rectangles rather than masks. |
| **Semantic Segmentation (`semantic`)** | DeepLabV3+ / UNet | **REQUIRES CONVERSION** | Requires rasterizing OBBs to pixel masks via `cv2.fillPoly` before training. |

---

## 6. Recommendations

1. **Do not force a segmentation model onto rotated bounding boxes** unless a rectangular mask output is acceptable to downstream modules.
2. **Primary Recommended Architecture:** **YOLO11 Nano Oriented Bounding Box (`yolo11n-obb.pt`)** or **YOLOv8 Nano Detection (`yolov8n.pt`)**.
3. **Pre-Training Label Hygiene:** Clamp the 198 coordinate instances to $[0.0, 1.0]$ before training to avoid dataloader warnings.

