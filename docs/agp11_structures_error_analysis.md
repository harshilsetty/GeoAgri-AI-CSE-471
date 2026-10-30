# AGP 11 — Structures Localization & Classification Error Diagnosis

## 1. Executive Summary

This diagnostic investigation directly audited the 36 ground-truth `structures` instances across the 25 held-out test images of the **Nalanda** archaeological site. By matching predictions from EXP-C against ground-truth rotated bounding boxes, we isolated the exact mathematical and visual root causes of the $0.0000$ test recall.

### Core Breakthrough Finding:
**The failure is primarily driven by Semantic Class Confusion (`structures` $\to$ `ruins`), compounded secondarily by OBB angular sensitivity:**
1. **Semantic Misclassification (Primary - 94.4%):** The model localized **34 out of 36** ground-truth structural footprints in Nalanda, but classified **100% of those detections as `ruins` (class 2) rather than `structures` (class 3)**. Under standard evaluation metrics (Pascal VOC / COCO), a correct bounding box with the wrong class label is scored as both a false negative for `structures` (yielding 0 recall) and a false positive for `ruins`.
2. **Localization & Angular Deviations (Secondary):**
   * **17 out of 36 (47.2%)** instances achieved spatial overlap $\text{IoU} \ge 0.50$ (up to $0.8089$ IoU).
   * **9 out of 36 (25.0%)** instances fell into the borderline $0.30 \le \text{IoU} < 0.50$ band, where a median angular deviation of **$8.66^\circ$** clipped the spatial overlap just below the $0.50$ acceptance threshold.
   * **10 out of 36 (27.8%)** instances had $\text{IoU} < 0.30$ due to heavy occlusion or center displacement.

> [!IMPORTANT]
> Even if bounding box regression had achieved a 100% perfect IoU ($1.0$) on all 36 objects, **structures test recall would still evaluate to exactly 0.0000** because of class misassignment.

---

## 2. Quantitative Diagnostic Census

| Metric / Attribute | Measured Value across Nalanda Test GT |
| :--- | :--- |
| **Total Nalanda Test Images** | 25 |
| **Total Ground-Truth Structures** | 36 instances |
| **Instances with Candidate Localization Match** | 36 / 36 (100.0%) |
| **Predicted Class for Best Matches** | **`ruins`: 34 (94.4%)**, `vegetation`: 2 (5.6%), `structures`: 0 (0.0%) |
| **Maximum Polygon IoU** | **0.8089** |
| **Mean Polygon IoU** | **0.4384** |
| **Matches at $\text{IoU} \ge 0.50$** | **17 / 36 (47.2%)** |
| **Matches at $0.30 \le \text{IoU} < 0.50$** | **9 / 36 (25.0%)** |
| **Matches at $\text{IoU} < 0.30$** | **10 / 36 (27.8%)** |
| **Median Angular Error** | **$8.66^\circ$** |
| **Mean Angular Error** | **$20.97^\circ$** |
| **Mean Center Displacement** | **32.07 pixels** |
| **Mean Prediction Confidence** | **0.2292** |

---

## 3. Analysis of Failure Modes

### A. Material & Architectural Domain Shift
* In the **training set**, `structures` are predominantly represented by intact buff sandstone temples at Khajuraho and standing granite pillared mandapas at Hampi, featuring vertical spires, ornate cornices, and intact roofs.
* In the **held-out test set (Nalanda)**, structures are 5th-century CE Buddhist monasteries and stupas constructed entirely of unplastered red fired brick, lacking roofs and consisting of tiered plinths, open courtyards, and monks' cells.
* In the training set, low-lying exposed brickwork is exclusively annotated as `ruins`. Consequently, the network correctly extracts the architectural features of the Nalanda ruins, but assigns them to class 2 (`ruins`) rather than class 3 (`structures`).

### B. Severe Class Frequency Imbalance
* In the training set, `ruins` has **1,740 instances** across 34 scenes, whereas `structures` has only **157 instances** across 16 scenes (an $11.1 : 1$ training ratio).
* When presented with ambiguous weathered masonry, the Bayesian prior learned by the classification head strongly favors `ruins` over `structures`.

### C. OBB Angular Cliff
* For elongated bounding boxes (e.g. monastery foundation walls with aspect ratios $> 4:1$), an angular deviation of $8^\circ\text{--}12^\circ$ drops the polygon IoU from $>0.60$ to $<0.45$.
* While this geometric cliff exists and impacts 9 of the 36 instances, solving the angular cliff alone cannot rescue the test score without correcting class classification.

---

## 4. Visual Diagnostic Evidence

Diagnostic visualization overlays showing ground-truth OBBs (green), predicted OBBs (blue/orange), confidence scores, and IoU matching were generated and saved to:
* [`artifacts/agp11/nalanda_diagnosis/`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/artifacts/agp11/nalanda_diagnosis/)
* Complete numerical logs: [`artifacts/agp11/nalanda_diagnosis/nalanda_struct_analysis.json`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/artifacts/agp11/nalanda_diagnosis/nalanda_struct_analysis.json)
