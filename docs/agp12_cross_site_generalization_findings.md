# AGP 12B — Cross-Site Generalization Findings & Shortcut Learning Analysis

## 1. Executive Summary

A core objective of AGP 12B is to diagnose the mechanisms underlying the cross-site performance gap in GeoAgri-AI. By analyzing the ground-truth distribution and validation errors across the 4 primary archaeological sites (Hampi, Khajuraho, Nalanda, and Pattadakal), this report exposes severe geographic and geomorphological confounding, identifies site-specific shortcut learning, and establishes why increasing model capacity cannot resolve dataset-level structural deficiencies.

---

## 2. Validation Partition Census by Site

The validation partition contains 109 cropped/augmented image tiles derived from **only 19 unique parent source scenes**:
* **Hampi (Granite Koppal Substrate):** 34 image tiles across **6 unique scenes** (`hampi_13, 15, 27, 29, 35, 36`)
* **Khajuraho (Vindhyan Sandstone Substrate):** 42 image tiles across **6 unique scenes** (`khajuraho_01, 08, 16, 19, 26, 31`)
* **Nalanda (Alluvial Gangetic Plain / Brick):** 17 image tiles across **3 unique scenes** (`nalanda_05, 06, 11`)
* **Pattadakal (Malaprabha Sandstone Basin):** 16 image tiles across **4 unique scenes** (`pattadakal_09, 13, 22, 24`)

### Ground-Truth Instance Matrix by Site (Validation Split)
| Site Name | Dominant Geology | `boulders` | `others` | `ruins` | `structures` | `vegetation` | Site Total |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hampi** | Archaean Granite / Tors | **582** (100.0%) | 3 (8.8%) | **296** (88.4%) | **0** (0.0%) | 586 (40.3%) | 1,467 |
| **Khajuraho** | Buff Vindhyan Sandstone | **0** (0.0%) | **31** (91.2%) | **0** (0.0%) | 7 (19.4%) | **759** (52.2%) | 797 |
| **Nalanda** | Clay Alluvium / Terracotta Brick | **0** (0.0%) | 0 (0.0%) | 39 (11.6%) | 9 (25.0%) | 58 (4.0%) | 106 |
| **Pattadakal** | Sandstone Terraces | **0** (0.0%) | 0 (0.0%) | **0** (0.0%) | **20** (55.6%) | 52 (3.6%) | 72 |
| **Total Validation** | — | **582** | **34** | **335** | **36** | **1,455** | **2,442** |

---

## 3. Identification of Severe Site-Class Confounding

The census reveals that the GeoAgri-AI dataset does not contain an independent, balanced representation of semantic classes across geographic domains. Instead, **class identity is almost completely confounded with geographic site identity**:

1. **`boulders` Confounding (100.0% Hampi):**
   * Every single boulder in the validation split resides in Hampi. There are **zero boulders** in Khajuraho, Nalanda, or Pattadakal.
   * *Consequence:* The detector cannot learn generic semantic features of boulders; it learns the granitic radiometric profile and arid terrain signature of Hampi.
2. **`structures` Geographic Fragmentation:**
   * Over $80\%$ of validation structures are split between Pattadakal ($20$) and Nalanda ($9$), with $0$ in Hampi.
   * *Consequence:* In Hampi, where ancient monolithic temple structures exist, the dataset has zero annotated structures, penalizing any detector that correctly identifies architecture in granite settings.
3. **`ruins` Polarization (88.4% Hampi):**
   * Ruins are virtually absent in Khajuraho and Pattadakal ($0$ instances). 
   * A model exposed to ruins only learns granitic rubble heaps in Hampi, completely failing to detect sandstone architectural ruin rubble at other sites.

---

## 4. Shortcut Learning vs Causal Feature Extraction

### 4.1 Observed Association vs Causal Mechanism
* **Observed Empirical Phenomenon:** When YOLO11 models are evaluated on unseen scenes from a new site, detection recall drops precipitously (e.g., test split mAP drops to $0.1289$).
* **False Hypothesis:** "The model lacks capacity to capture complex rotated geometries." (Disproven by EXP-E, which quadrupled capacity but worsened generalization).
* **True Causal Mechanism (Shortcut Learning):**
  - Convolutional networks exploit the easiest statistical shortcuts that minimize training loss.
  - In GeoAgri-AI, the easiest discriminator between `boulders` and `vegetation` or `structures` is **background radiometric color and soil texture**, not the geometric orientation of walls.
  - The model associates **gray Archaean granite pixels** with `boulders`/`ruins`, **buff sandstone pixels** with `others`/`vegetation`, and **red alluvial brick pixels** with `structures`.
  - When presented with a brick structure against a novel terrain background, or sandstone ruins that lack surrounding red brick, the feature activations fail to reach detection thresholds.

---

## 5. Methodological Limitations of the Current Partition

1. **The "109 Images" Illusion:**
   * While the validation set contains 109 image tiles, they originate from **only 19 unique aerial scenes**.
   * Tiles sliced from the same scene share identical solar azimuth, illumination angle, soil moisture, and camera sensor calibration. Thus, the effective statistical degrees of freedom is $N=19$, not $N=109$.
2. **Sample Size Insufficiency for Minority Classes:**
   * `structures` contains only 36 instances across the entire validation split. A single misclassified building shifts class recall by nearly $3\%$.
3. **Impossibility of In-Distribution Model Scaling:**
   * Scaling model capacity (EXP-E) on 19 validation scenes merely allows the larger network to memorize idiosyncratic sensor noise and terrain artifacts faster, triggering early stopping at Epoch 8.

---

## 6. Strategic Takeaway for Future Research

**Model architecture tweaks (loss weighting, backbone capacity, TTA) have reached their theoretical ceiling on the v3 dataset.** 
The only scientifically defensible path to genuine cross-site generalization is **breaking the site-class correlation at the data level** by collecting multi-class annotations across geographically diverse archaeological complexes as specified in `docs/agp12_data_collection_protocol.md`.

