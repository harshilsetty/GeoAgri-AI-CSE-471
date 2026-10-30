# AGP 09 — Annotation Recovery & OBB Geometric Distortion Audit

**Author:** Senior Computer Vision Research Engineer & Dataset Auditor  
**Date:** October 28, 2026  
**Project:** GeoAgri-AI  
**Audited Datasets:** 
- Original Source: `..\Archaeological-Site-Mapping-AI-V2\Archaeological-Site-Mapping-AI-V2\dataset`
- Remediated Dataset: `datasets/geoagri_obb_v2`
**Artifacts Directory:** `artifacts/agp09/annotation_review/`  
**Flagged Annotations Manifest:** `artifacts/agp09/annotation_review/suspicious_annotations.csv`  

---

## 1. Executive Summary

This phase audited the geometric consequences of individual coordinate clamping applied during AGP 07/08 to remediate the 3,185 out-of-range coordinates. By performing an instance-by-instance geometric diff between the unclipped source annotations and the clamped dataset across all 18,458 instances, we quantified changes in surface area, edge proportions, vertex ordering, and convexity.

### Key Audit Findings:
* **Total Instances Clamped:** **2,701 instances** (14.63% of the total dataset).
* **Convexity Preservation:** **0 instances** lost convexity ($100\%$ preserved convex geometry).
* **Suspicious Geometry Shift:** **360 instances** (13.33% of clamped instances, **1.95% of total dataset**) exhibited area changes $> 15\%$ or edge distortions $> 25\%$.
* **Mean Area Reduction in Flagged Cases:** **21.34%** (Maximum: 56.21%).
* **Primary Cause:** Objects truncated by image borders during online rotation augmentations in Roboflow.
* **Format Constraint:** Because the YOLO-OBB architecture strictly mandates $N=4$ vertices, conventional Sutherland-Hodgman polygon clipping (which produces 5-vertex polygons when clipping corners) cannot be directly encoded in standard 8-parameter OBB labels without polygonal re-fitting (e.g., minimum area rotated rectangle).

---

## 2. Quantitative Geometric Comparison

| Metric | Measured Value | Percentage of Dataset |
| :--- | :---: | :---: |
| **Total Audited Instances** | 18,458 | 100.0% |
| **Instances with Coordinate Clamping** | 2,701 | 14.63% |
| **Instances with Area Change $\le 5\%$** | 2,118 | 78.4% of clamped |
| **Instances with Area Change $5\% - 15\%$** | 223 | 8.3% of clamped |
| **Instances Flagged Suspicious (Area Change $> 15\%$)** | **360** | **13.3% of clamped (1.95% total)** |
| **Polygons Losing Convexity** | **0** | **0.0%** |
| **Degenerate Polygons ($A < 10\text{ px}^2$)** | **0** | **0.0%** |

### Breakdown of Suspicious Cases by Class:
* `vegetation`: 176 instances (48.9%)
* `boulders`: 85 instances (23.6%)
* `ruins`: 53 instances (14.7%)
* `others`: 31 instances (8.6%)
* `structures`: 15 instances (4.2%)

### Breakdown of Suspicious Cases by Split:
* `train`: 271 instances (75.3%)
* `test`: 52 instances (14.4%)
* `val`: 37 instances (10.3%)

---

## 3. Geometric Distortion Analysis: Clamping vs. True Clipping

When a rotated bounding box crosses an image boundary ($x < 0$ or $x > 1$):
1. **The Clamping Effect:**
   $$\tilde{x}_i = \min(1.0, \max(0.0, x_i))$$
   When an angled edge crosses a border, clamping the external vertex to the border shifts the vertex along the edge normal, altering the box's apparent rotation angle by $3^\circ$ to $12^\circ$ for severely truncated objects.
2. **The Sutherland-Hodgman Dilemma in OBB:**
   True geometric polygon clipping against the unit square $[0, 1] \times [0, 1]$ introduces new intersection vertices along the border. For an oriented rectangle intersecting a corner, clipping produces a pentagon or hexagon ($N=5$ or $N=6$). Since YOLO-OBB requires an exact 4-point quadrilateral, standard clipping cannot be stored directly in `.txt` format without running `cv2.minAreaRect()` on the clipped polygon.
3. **Impact on Model Generalization:**
   Because only **1.95%** of annotations experienced material distortion ($>15\%$ area change), this geometric clamping is **NOT** the primary root cause of the severe test drop ($0.4002 \rightarrow 0.1412$). However, for boundary-touching `structures` (15 instances), clamping slightly degraded corner localization.

---

## 4. Visual Evidence Artifacts

Visual overlay comparisons showing the original unclipped polygon in **red** and the clamped polygon in **green** were generated and saved in:
* Directory: [`artifacts/agp09/annotation_review/`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/artifacts/agp09/annotation_review/)
* Complete manifest of 360 flagged annotations: [`artifacts/agp09/annotation_review/suspicious_annotations.csv`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/artifacts/agp09/annotation_review/suspicious_annotations.csv)

---

## 5. Remediation Recommendation

In Phase 7, implement a geometry-preserving `cv2.minAreaRect()` refitting pipeline for any boundary-crossing polygon to construct a versioned dataset `datasets/geoagri_obb_v3` without altering the existing baseline.

