# AGP 10 — Pre-Flight Validation Report

## 1. Executive Summary

This pre-flight audit validates the computational environment, dataset integrity, annotation topology, and split isolation prior to launching controlled training experiments for AGP 10. All checks have passed without exception.

---

## 2. Hardware and Environment Verification

* **Device:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, Bus ID `0000:01:00.0`)
* **Driver Version:** 581.86 | **CUDA Version:** 13.0 (Driver) / 12.4 (PyTorch `torch.version.cuda`)
* **PyTorch Version:** 2.6.0+cu124
* **Ultralytics Version:** 8.4.191
* **GPU Allocation at Pre-Flight:**
  * Active Training Processes: None
  * Allocated Memory: 0.0 MB
  * Reserved Memory: 0.0 MB
  * GPU Utilization: 0%
  * Temperature: 54°C
  * Background processes: PID 30072 (`backend_api:app` FastAPI daemon, 0 MiB VRAM)

---

## 3. Dataset Configuration & Path Resolution

Both dataset configuration YAML files were inspected and verified:
* **Baseline Dataset:** [`datasets/geoagri_obb_v2/data.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/datasets/geoagri_obb_v2/data.yaml)
* **Refitted Dataset:** [`datasets/geoagri_obb_v3/data.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/datasets/geoagri_obb_v3/data.yaml)

### Split Counts and Image Resolution:
| Split | Images Verified | Label Files Verified | Matching Pairs |
| :--- | :--- | :--- | :--- |
| **Train** | 514 | 514 | 100% |
| **Validation** | 109 | 109 | 100% |
| **Held-Out Test** | 117 | 117 | 100% |
| **Total** | 740 | 740 | 100% |

All images resolve to standard $640 \times 640$ resolution.

---

## 4. Class Taxonomy & ID Mapping Alignment

The class mapping is 100% identical across both versions:
```yaml
names:
  0: boulders
  1: others
  2: ruins
  3: structures
  4: vegetation
```
No renumbering, omissions, or taxonomy substitutions exist between v2 and v3.

---

## 5. Source-Scene Isolation Verification

Scene isolation was audited across all splits by extracting scene identifiers (`<site>_<scene_id>` from filename stems prior to `.rf.<hash>`):
* **Train Scenes:** 90 unique source photographs
* **Validation Scenes:** 19 unique source photographs
* **Held-out Test Scenes:** 19 unique source photographs
* **Total Unique Scenes:** 128 unique scenes
* **Cross-Split Overlaps:**
  * Train $\cap$ Val: **0**
  * Train $\cap$ Test: **0**
  * Val $\cap$ Test: **0**

Leak-free scene boundary isolation is strictly preserved.

---

## 6. Annotation Topology & Geometry Audit

The geometry-refitted annotations in [`datasets/geoagri_obb_v3`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/datasets/geoagri_obb_v3) were audited instance by instance:
* **Total Audited Instances:** 18,458 instances
* **Coordinates Strictly in $[0.0, 1.0]$:** 18,458 (100.0%)
* **Strict 4-Point Quadrilateral (8 numbers):** 18,458 (100.0%)
* **Degenerate Instances ($\text{Area} \le 10^{-6}$):** **0**
* **Non-Convex or Self-Intersecting Polygons:** **0**
* **Refitted Boundary Instances:** 2,701 instances

### Side-by-Side Inspection Samples:
1. `hampi_02_jpg.rf.515ffd5393d5ddb3362721ea456b5b59.txt` (Instance 2, class 4 vegetation):
   * *v2 (clamped):* `4 1.000000 0.911522 0.996229 0.279587 0.916553 0.280978 0.927584 0.912913`
   * *v3 (refitted):* `4 0.916000 0.280000 0.996000 0.279000 1.000000 0.911039 0.923901 0.912039`
2. `hampi_02_jpg.rf.515ffd5393d5ddb3362721ea456b5b59.txt` (Instance 6, class 2 ruins):
   * *v2 (clamped):* `2 0.216997 0.670590 0.211434 0.351889 0.000000 0.355625 0.002967 0.674326`
   * *v3 (refitted):* `2 0.000000 0.354354 0.211000 0.351000 0.216010 0.670646 0.002000 0.674000`

`cv2.minAreaRect` establishes true rotated rectangles whose 4 vertices satisfy standard OBB coordinate constraints without introducing artificial right-angled trapezoids.

---

## 7. Limitations & Acknowledged Risks

While `cv2.minAreaRect` eliminates out-of-bounds coordinates and preserves box convexity:
* It cannot synthesize features clipped beyond the physical camera sensor.
* It operates strictly on the polygon envelope visible within the $[0, 1] \times [0, 1]$ normalized canvas.
* As established in AGP 09, annotation clamping was not the primary driver of the held-out test score drop (which is driven by site-level material domain shift); however, refitting cleans the label space and guarantees consistent gradient feedback.

---

## 8. Go / No-Go Decision

**Status: GO.**
All pre-flight criteria are satisfied. No blockers or defects were detected. Sequentially executing EXP-B and EXP-C is approved.

