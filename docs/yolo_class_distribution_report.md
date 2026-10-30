# GeoAgri-AI: YOLO Class Distribution, Diversity & Imbalance Audit Report

**Document ID:** `docs/yolo_class_distribution_report.md`  
**Auditor:** Senior Computer Vision Engineer & Dataset Quality Auditor  
**Dataset Inspected:** `..\Archaeological-Site-Mapping-AI-V2\Archaeological-Site-Mapping-AI-V2\dataset`  
**Date:** October 28, 2026  
**Status:** Completed & Verified

---

## 1. Executive Summary

An audit of class representations across the 18,458 labeled instances was conducted. In addition to measuring raw instance counts, this analysis computed the **number of independent source photographs ($N=128$) containing each class**, which is the true measure of visual diversity and generalization potential.

### Critical Takeaways
1. **Severe Class Imbalance:**
   - `vegetation` accounts for **57.5% of all labeled objects** (10,620 instances) and appears in 100% of images.
   - `structures` accounts for only **1.2% of all labeled objects** (230 instances).
   - Raw imbalance ratio: **46.2 to 1**.
2. **Effective Visual Diversity vs. Raw Instances:**
   - While `structures` has 230 bounding boxes, those 230 boxes originate from **only 21 unique physical camera setups** across Nalanda, Pattadakal, and Khajuraho.
   - In the test split, the 15 structure annotations come from **only 3 unique scenes**. A single misclassification will drastically skew test mAP for this class.
3. **Geographic Class Specialization:**
   - `boulders` is almost entirely concentrated in **Hampi** (granitic landscape).
   - `structures` is concentrated in **Nalanda** (monastic brick complexes) and **Pattadakal** (stone temples).

---

## 2. Comprehensive Class Distribution Table

| Class ID | Class Name | Total Instances | % of All Instances | Unique Source Scenes ($N=128$) | % of All Source Scenes | Primary Archaeological Site |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | `boulders` | 3,876 | 21.00% | 30 | 23.44% | Hampi |
| **1** | `others` | 1,346 | 7.29% | 54 | 42.19% | All Sites (Pathways, roads, modern debris) |
| **2** | `ruins` | 2,386 | 12.93% | 49 | 38.28% | Hampi, Khajuraho, Nalanda |
| **3** | `structures` | **230** | **1.25%** | **21** | **16.41%** | Nalanda, Pattadakal, Khajuraho |
| **4** | `vegetation` | **10,620** | **57.54%** | **128** | **100.00%** | All Sites (Lawns, scrub, trees, wild canopy) |
| **Total** | — | **18,458** | **100.0%** | **128** | — | — |

---

## 3. Split-by-Split Class Breakdown (Current vs. Source Diversity)

```
Class 0: boulders
  Train:  3,530 instances (from 24 unique scenes)
  Valid:    200 instances (from  4 unique scenes)
  Test:     146 instances (from  2 unique scenes)

Class 1: others
  Train:  1,138 instances (from 46 unique scenes)
  Valid:    137 instances (from 11 unique scenes)
  Test:      71 instances (from  6 unique scenes)

Class 2: ruins
  Train:  2,116 instances (from 42 unique scenes)
  Valid:    169 instances (from  9 unique scenes)
  Test:     101 instances (from  6 unique scenes)

Class 3: structures  <-- SEVERE MINORITY
  Train:    186 instances (from 15 unique scenes)
  Valid:     29 instances (from  3 unique scenes: nalanda_08, pattadakal_06, pattadakal_07)
  Test:      15 instances (from  3 unique scenes: nalanda_04, pattadakal_04, pattadakal_09)

Class 4: vegetation  <-- SUPER DOMINANT
  Train:  8,790 instances (from 108 unique scenes)
  Valid:  1,060 instances (from  51 unique scenes)
  Test:     770 instances (from  27 unique scenes)
```

---

## 4. Risks & Mitigations

### Risk 1: Structure Detector Underfitting
* **Symptom:** The loss function is dominated by the 10,620 vegetation instances, causing the model to prioritize background foliage over rare masonry structures.
* **Mitigation:**
  - Increase classification loss weight `cls` in YOLO training (e.g. `cls=1.5`).
  - Apply online mosaic and copy-paste augmentation (`mosaic=1.0`, `copy_paste=0.3`) to artificially paste structure instances onto different background scenes without creating misleading duplicate disk files.

### Risk 2: Semantic Ambiguity of `others`
* **Symptom:** Class 1 (`others`) has an undefined visual boundary. In some images it labels tourist pathways, in others it labels stone fences, modern concrete benches, or river sand.
* **Mitigation:** Evaluate whether `others` is necessary for GeoAgri-AI. In erosion modeling, only physical obstacles (`boulders`, `ruins`, `structures`) and vegetation matter. `others` could be silenced or filtered during downstream feature extraction.

### Risk 3: Small Test Sample for Structures
* **Symptom:** Only 15 instances in the current test set. A single missed detection drops Recall from $100\%$ to $93.3\%$.
* **Mitigation:** During group-aware re-splitting, ensure at least 5 structure scenes (approx. 35 instances) are allocated to the holdout test split.

