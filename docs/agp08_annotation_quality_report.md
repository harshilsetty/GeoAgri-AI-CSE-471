# AGP 08 — Annotation Quality & OBB Geometric Integrity Report

**Author:** Senior Computer Vision Engineer & Dataset Quality Auditor  
**Date:** October 28, 2026  
**Project:** GeoAgri-AI  
**Audited Dataset:** `datasets/geoagri_obb_v2`  
**Sample Artifacts Directory:** `artifacts/yolo_vision_v2/annotation_audit/`  
**Contact Sheet:** `artifacts/yolo_vision_v2/annotation_audit/obb_annotation_contact_sheet.jpg`  

---

## 1. Audit Methodology & Scope

To rigorously verify that coordinate clamping did not introduce geometric collapse, self-intersection, or label distortion, a systematic sampling audit was performed across all three splits:
* **Sample Size:** **90 unique images** (30 Train, 30 Validation, 30 Test) selected via deterministic pseudo-random sampling (`seed=42`).
* **Total Instances Inspected:** **1,868 object instances**.
* **Geometric Analysis:** Each 4-point quadrilateral was evaluated using OpenCV contour geometry:
  - Surface area via Green's theorem / Shoelace formula: $A = \frac{1}{2} |\sum (x_i y_{j} - x_j y_i)|$.
  - Convexity verification comparing contour area to its convex hull area: $|A - A_{\text{hull}}| < 1.0 \text{ px}^2$.
  - Border proximity detection: Coordinates within $\delta = 0.005$ (3.2 pixels) of image borders.
  - Class distribution and label correctness inspection.

---

## 2. Quantitative Geometric Audit Results

| Metric | Measured Value | Percentage (%) | Engineering Interpretation |
| :--- | :---: | :---: | :--- |
| **Total Reviewed Images** | 90 | 100.0% | 30 train, 30 val, 30 test |
| **Total Ground-Truth Instances** | 1,868 | — | 20.76 instances per image average |
| **Border-Touching OBBs** | 88 images | **97.8%** | Expected due to high-altitude landscape framing where terrain features intersect boundaries. |
| **Distorted / Non-Convex OBBs** | 1 image | **1.1%** | Negligible distortion; 98.9% of reviewed images have 100% strictly convex quadrilaterals. |
| **Degenerate / Collapsed OBBs ($A \le 10\text{ px}^2$)** | 0 instances | **0.0%** | Zero polygons collapsed into lines or points after clamping. |
| **Images with `others` Class** | 33 images | **36.7%** | Broad miscellaneous class; captured non-archaeological modern elements. |

### Sampled Class Breakdown:
* **Vegetation:** 1,101 instances (58.9%)
* **Boulders:** 357 instances (19.1%)
* **Ruins:** 256 instances (13.7%)
* **Others:** 115 instances (6.2%)
* **Structures:** 39 instances (2.1%)

---

## 3. Visual & Geometric Findings

1. **Border Clamping Fidelity:**
   The 3,185 coordinate occurrences originally generated slightly outside $[0.0, 1.0]$ by Roboflow's rotation augmentations were clamped to $[0.0, 1.0]$. Visual inspection confirms that the resulting boxes are cleanly truncated at the image border without polygon twisting or inversion.
2. **Convexity Preservation:**
   Only 1 instance out of 1,868 exhibited a slight non-convex irregularity ($<1.1\%$ of images), which was caused by an ambiguous complex multi-faceted ruin fragment originally annotated with close vertices. No systematic deformation was introduced.
3. **Class Label Ambiguity (`ruins` vs `structures`):**
   - The distinction between `ruins` and `structures` is primarily functional: `structures` encompasses preserved architectural elements (pillars, lintels, roofs, intact temples), while `ruins` represents crumbling stone walls and collapsed rubble.
   - In 4 out of 90 images, partial temple foundations could plausibly be labeled as either class. This semantic ambiguity explains why model confidence is slightly dispersed between classes 2 and 3.

---

## 4. Visual Evidence Artifacts

* High-resolution contact sheet: [`artifacts/yolo_vision_v2/annotation_audit/obb_annotation_contact_sheet.jpg`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/artifacts/yolo_vision_v2/annotation_audit/obb_annotation_contact_sheet.jpg)
* 90 individual annotated ground-truth images saved in [`artifacts/yolo_vision_v2/annotation_audit/`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/artifacts/yolo_vision_v2/annotation_audit/)

---

## 5. Geometric Integrity Verdict

**PASS**: The OBB geometry is sound, non-degenerate, and verified intact. No retraining or transformation rollback is required.

