# GeoAgri-AI: Source-Scene Decomposition & Cross-Split Data Leakage Audit

**Document ID:** `docs/yolo_scene_leakage_audit.md`  
**Auditor:** Senior Computer Vision Engineer & MLOps Auditor  
**Dataset Inspected:** `..\Archaeological-Site-Mapping-AI-V2\Archaeological-Site-Mapping-AI-V2\dataset`  
**Date:** October 28, 2026  
**Status:** Completed & Verified (Severe Leakage Detected)

---

## 1. Executive Summary

An audit of image provenance across the 740 local images was conducted to verify whether the train, validation, and test splits represent independent observations.

### Critical Findings
1. **Severe Cross-Split Contamination:**
   - The reported 740 images originate from only **128 unique real-world photographs**.
   - **77.8% of the Test split (21 of 27 unique scenes)** is already present in the Training split as augmented crops/rotations.
   - **70.6% of the Validation split (36 of 51 unique scenes)** is already present in the Training split.
   - **5 physical scenes appear simultaneously in all three splits (Train, Valid, and Test).**
2. **Root Cause of Leakage:**
   - Roboflow augmented the 128 original photographs $\approx 5.8\times$ (generating 740 images) and subsequently performed a **random image-level split** rather than a **group-aware scene split**.
3. **Evaluation Impact:**
   - Any evaluation metrics reported on the current validation or test splits (such as mAP50 $> 90\%$) are heavily contaminated by landmark memorization.
   - The model is being tested on rotated variants of photographs it has already seen during gradient updates.

---

## 2. Source-Scene Decomposition

By stripping Roboflow's cryptographic hash suffix (`<base_id>_jpg.rf.<32-char-hash>.jpg`), each image was traced to its original captured photograph.

### Breakdown by Archaeological Site ($N = 128$ Unique Scenes)

| Heritage Site | State / Geography | Unique Scenes | Primary Visual Features | Dominant Classes |
| :--- | :--- | :---: | :--- | :--- |
| **Hampi** | Karnataka (Deccan Plateau) | **41** | Granitic boulder hills, Vijayanagara temples, arid scrub | `boulders`, `ruins`, `vegetation` |
| **Khajuraho** | Madhya Pradesh (Bundelkhand) | **40** | Nagara-style sandstone temples, manicured heritage lawns | `ruins`, `vegetation`, `structures` |
| **Pattadakal** | Karnataka (Malaprabha River) | **30** | Chalukya stone temples, monolithic structures, riverbanks | `structures`, `ruins`, `vegetation` |
| **Nalanda** | Bihar (Gangetic Plain) | **17** | Monastic brick ruins, excavated walls, university stupas | `structures`, `ruins`, `vegetation` |
| **Total** | — | **128** | — | — |

---

## 3. Quantification of Cross-Split Scene Leakage

```
                          Total Images    Unique Scenes    Scenes Shared with Train    Leakage Percentage
Training Split (train)         648             108                    —                         —
Validation Split (valid)        61              51                    36                      70.59%
Testing Split (test)            31              27                    21                      77.78%
```

### Specific Scenes Present Across Multiple Splits
* **Scenes in All Three Splits (Train, Valid, AND Test simultaneously):**
  1. `hampi_34`
  2. `hampi_41`
  3. `khajuraho_07`
  4. `khajuraho_09`
  5. `pattadakal_06`
* **Additional Test Scenes Leaking into Training:**
  `hampi_16`, `hampi_22`, `hampi_28`, `hampi_30`, `hampi_35`, `hampi_36`, `khajuraho_03`, `khajuraho_06`, `khajuraho_11`, `khajuraho_14`, `khajuraho_17`, `khajuraho_20`, `khajuraho_24`, `nalanda_04`, `nalanda_08`, `pattadakal_04`.

---

## 4. Bitwise Exact Duplicate Files

In addition to scene leakage, exactly one pair of **bitwise identical files** exists within the training partition:
* `hampi_31_jpg.rf.ba380e45e6cfa45e0735a79c2797317a.jpg`
* `hampi_31_jpg.rf.c848f2d5beff05ad9b931b9ac75ca8dc.jpg`
* *SHA-256:* `46e39664f981c69c0f03e15bb27b3874068cc969fa5faa3464e7ae933768dd63`
* *Remediation:* Remove one redundant file during dataset preparation.

---

## 5. Proposed Leak-Free Group-Stratified Split Strategy

To ensure scientific validity and genuine out-of-distribution evaluation, the dataset must be partitioned at the **source scene level** ($N=128$), ensuring:
1. Zero scenes overlap between partitions.
2. All 4 archaeological sites are represented in all splits.
3. The minority class `structures` (which appears in only 21 scenes across Nalanda, Pattadakal, and Khajuraho) is stratified across all partitions.

### Proposed Group Partition

| Split | Target Scene Count | Target Image Count | Site Distribution | Structure Scenes Allocated |
| :--- | :---: | :---: | :--- | :---: |
| **Train** | **90 scenes (70.3%)** | $\approx 520$ images | 29 Hampi, 28 Khajuraho, 21 Pattadakal, 12 Nalanda | 15 scenes |
| **Validation** | **20 scenes (15.6%)** | $\approx 115$ images | 6 Hampi, 6 Khajuraho, 5 Pattadakal, 3 Nalanda | 3 scenes |
| **Holdout Test** | **18 scenes (14.1%)** | $\approx 105$ images | 6 Hampi, 6 Khajuraho, 4 Pattadakal, 2 Nalanda | 3 scenes |
| **Total** | **128 scenes (100%)** | **740 images** | 41 Hampi, 40 Khajuraho, 30 Pattadakal, 17 Nalanda | 21 scenes |

---

## 6. Execution Rules
* In compliance with safety guidelines, **no files have been moved, renamed, or modified** during this audit.
* The original directory structure in `dataset/` remains strictly intact.
* When training is authorized, a script will generate the new group-aware manifest or create a separate `dataset_group_split/` directory.

