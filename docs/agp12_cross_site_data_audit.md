# AGP 12 — Cross-Site Data Coverage and Geographic Confounding Audit

## 1. Executive Summary

This audit conducted an exhaustive cross-site census across all 740 images and 18,458 instances in [`datasets/geoagri_obb_v3`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/datasets/geoagri_obb_v3).

### Key Discovery: Severe Class-Site Confounding
The class taxonomy is severely entangled with geography, building materials, and local lithology:
1. **`boulders` is 99.8% confounded with Hampi:** 3,869 of 3,876 boulder instances reside in Hampi. No boulders exist in Khajuraho or Nalanda.
2. **`ruins` is 92.0% confounded with Hampi:** 2,196 of 2,386 ruins instances are Hampi granite rubble.
3. **`structures` is 95.7% concentrated in Nalanda and Pattadakal:** Hampi has **zero (0)** annotated structures despite hundreds of standing mandapas; Khajuraho has only 10 annotated structures across 40 scenes.
4. **Generalization Bottleneck:** Because classes are bound to specific geological sites, splitting by scene alone cannot evaluate true class generalization across unseen sites without cross-site architectural diversity.

---

## 2. Complete Cross-Site Census Matrix

| Site | Unique Scenes | Total Images | Train / Val / Test | Boulders | Others | Ruins | Structures | Vegetation | Total Instances | Material & Lithology |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Hampi** | 41 | 255 | 183 / 34 / 38 | **3,869** | 678 | **2,196** | **0** | 4,590 | 11,333 | Gray granite tors, boulder fields, dry savanna |
| **Khajuraho** | 40 | 238 | 154 / 42 / 42 | **0** | 533 | **0** | **10** | **5,213** | 5,756 | Buff sandstone temples, manicured lawns, tall trees |
| **Nalanda** | 17 | 117 | 75 / 17 / 25 | **0** | 44 | 145 | **126** | 472 | 787 | Weathered unplastered red fired brick stupas & cells |
| **Pattadakal**| 30 | 130 | 102 / 16 / 12 | 7 | 91 | 45 | **94** | 345 | 582 | Red/yellow sandstone vimanas, plinths, river plains |
| **Total** | **128** | **740** | **514 / 109 / 117** | **3,876** | **1,346** | **2,386** | **230** | **10,620** | **18,458** | Multi-site heritage domain |

---

## 3. Generalization Implications

1. **Why Nalanda Structures were Predicted as Ruins:**
   * In Hampi (the dominant training source for masonry), all fragmented stonework is labeled `ruins`.
   * In Nalanda, exposed ancient brick stupas have textured, step-like layers that visually resemble the weathered courses of Hampi ruins.
   * Because the training split contains $2,196$ ruins from Hampi and only $75$ structural images from Nalanda, gradient updates heavily biased the network toward predicting `ruins`.
2. **Evaluation Scene Scarcity:**
   * Although Nalanda has 126 structural instances, they originate from only **17 independent scenes**.
   * In the held-out test split, the 37 structures exist in only **2 independent source scenes**. Any evaluation on test structures is an all-or-nothing test on two specific monuments.

