# AGP 09 — Architectural Taxonomy & Structures Class Strategy

**Author:** Senior Computer Vision Research Engineer & Archaeological AI Specialist  
**Date:** October 28, 2026  
**Project:** GeoAgri-AI  
**Target:** Remediation of Minority Class Starvation and Semantic Ambiguity (`structures` vs. `ruins`)  

---

## 1. Problem Definition: The Minority Class Dilemma

In `datasets/geoagri_obb_v2`, the `structures` class accounts for only **230 of 18,458 instances (1.25%)** across the entire dataset:
* **Train Split:** 157 instances across 15 scenes
* **Validation Split:** 36 instances across 3 scenes
* **Test Split:** 37 instances across 3 scenes (all in Nalanda except 1 in Pattadakal)
* **Test Recall:** **0.0000** (under standard evaluation benchmark)

This extreme imbalance creates dual failure modes:
1. **Sample Starvation:** The model receives insufficient gradient signal during training to learn a robust representation of multi-style architectural features.
2. **Semantic Ambiguity:** In archaeological terrain, where does an intact wall end and a ruin begin?

---

## 2. Taxonomic Analysis: `structures` vs. `ruins`

### 2.1 The Semantic Continuum
Archaeological heritage exists along a continuous physical decay spectrum rather than a clean discrete binary:

```
[ Intact Temple / Mandapa ] ---> [ Standing Colonade / Wall ] ---> [ Collapsed Wall / Rubble ] ---> [ Scattered Stone Blocks ]
       (Class 3: structures)            (Ambiguous Boundary)              (Class 2: ruins)                (Class 0: boulders / ruins)
```

### 2.2 Semantic Overlap Evidence
In the test set (Nalanda Monastery complex):
* Ground truth contains 36 `structures` and 48 `ruins`.
* Model predictions at `conf=0.25`: **90 `ruins`** and **38 `structures`**.
* The model repeatedly predicts `ruins` on ancient brick perimeter walls because the eroded mortar and crumbled top courses resemble collapsed masonry.

### 2.3 Trade-Off Analysis: Should Classes Be Merged?

| Strategy | Description | Advantages | Disadvantages / Risks | Recommendation |
| :--- | :--- | :--- | :--- | :---: |
| **Option A: Merge `structures` and `ruins` into `built_heritage`** | Combine classes 2 & 3 into a single cultural heritage class. | Eliminates semantic boundary ambiguity; immediately boosts minority class representation to 2,616 instances (14.2%). | Loses the engineering distinction between standing monuments (low erosion risk, preservation priority) and crumbling rubble (high erosion risk). | **REJECT FOR PRODUCTION** (Preserve distinction) |
| **Option B: Hierarchical Multi-Head Classification** | High-level detection of `built_heritage` followed by a secondary classifier distinguishing `intact` vs `collapsed`. | Preserves fine-grained classification while allowing the detector to localize masonry boundaries with high recall. | Adds architectural complexity to the lightweight YOLO11n backbone. | **LONG-TERM RESEARCH** |
| **Option C: Targeted Multi-Site Data Augmentation & Protocol Re-annotation (Recommended)** | Keep classes separate, but expand `structures` with 150+ diverse independent scenes and formalize strict boundary rules. | Directly fixes root cause (sample starvation) without compromising domain specificity for cultural preservation. | Requires targeted image collection and rigorous annotation. | **RECOMMENDED FOR AGP 10** |

---

## 3. Targeted Multi-Site Data-Collection Protocol

To overcome sample starvation without introducing augmented duplicates, the following field collection protocol is established:

### 3.1 Target Archaeological Sites & Geological Diversity
* **Group 1 (Granite Monoliths & Pillars):** Hampi (Vijayanagara), Lepakshi, Mahabalipuram.
* **Group 2 (Sandstone Sanctuaries & Shikharas):** Pattadakal, Aihole, Badami Cave Temples, Khajuraho.
* **Group 3 (Terracotta & Fired-Brick Complexes):** Nalanda, Vikramashila, Sarnath.
* **Target Count:** **Minimum 150 independent photographs** (at least 35 unique scenes per architectural group).

### 3.2 Canonical Annotation Decision Rules
To eliminate annotator subjectivity, labelers must adhere to the following deterministic rules:

1. **Rule of Architectural Integrity:**
   - If a monument possesses an intact roof, standing load-bearing pillars, or recognizable architectural capitals $\rightarrow$ Label as `structures`.
   - If a wall has collapsed below 50% of its original height with unbonded stone rubble $\rightarrow$ Label as `ruins`.
2. **Rule of Discrete Colonades:**
   - Long colonnades (e.g. Hampi bazaar corridors) must be annotated as cohesive rectangular blocks along the colonnade orientation, rather than splitting individual 1-meter pillar spans.
3. **Rule of Background Terrain Exclusion:**
   - Never include surrounding soil, grass, or talus slope inside a structural OBB. Boxes must tightly enclose the masonry perimeter.

---

## 4. Multi-View & Environmental Diversity Guidelines

Every newly collected monument scene must include:
* **Distance Variations:** Long-range landscape overview ($>50\text{ m}$), mid-range facade ($15 - 30\text{ m}$), and close-up masonry ($5 - 10\text{ m}$).
* **Elevation Angles:** Eye-level horizontal ($0^\circ$), low-angle upward perspective ($+20^\circ$), and elevated hillside / drone perspective ($-30^\circ$).
* **Lighting Conditions:** Harsh midday sun (high-contrast shadows), golden hour (warm directional light), and overcast diffuse lighting.
* **Vegetation Intrusion:** Document overgrown temples (Ficus tree roots, creeper vines) to train the model to separate vegetation occlusion from structural decay.

