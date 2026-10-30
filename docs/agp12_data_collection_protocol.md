# AGP 12 — Field Data Collection & Multi-Site Annotation Protocol

## 1. Objective and Scope

To resolve the verified geographic and material confounding identified in [`docs/agp12_cross_site_data_audit.md`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/docs/agp12_cross_site_data_audit.md), this protocol specifies the acquisition and labeling standards for **120+ new independent archaeological scenes** across underrepresented architectural styles and materials.

---

## 2. Target Site Allocation & Sampling Quotas

| Target Site | Target Independent Scenes | Target Images | Primary Target Class | Target Material & Architectural Features |
| :--- | :---: | :---: | :--- | :--- |
| **Nalanda (Expansion)** | 40 scenes | 50 images | `structures` & `ruins` | Excavated Monastery 1, Stupa 3 plinths, brick chaityas, votive stupas, weathered kiln-fired red brick. |
| **Pattadakal (Expansion)**| 30 scenes | 35 images | `structures` & `others` | Virupaksha, Mallikarjuna, and Papanatha complexes; tiered sandstone vimanas, outer perimeter prakara walls. |
| **Aihole (New Cross-Site)**| 30 scenes | 35 images | `structures` & `ruins` | Durga temple, Lad Khan, Konti Gudi; transitional Chalukyan rock-cut and structural sandstone temples. |
| **Hampi (Re-balancing)** | 20 scenes | 25 images | `structures` (standing) | Krishna Temple mandapas, Achyutaraya bazaar columns, Hazara Rama plinths (currently 0 annotated structures in Hampi). |
| **Total** | **120 scenes** | **145 images** | Multi-class balance | Strict 1:1 camera-to-monument independence. |

---

## 3. Flight & Photographic Capture Specifications

1. **Camera Altitude and GSD:**
   * Low-altitude oblique survey: $20\text{--}40\text{ m}$ AGL (Ground Sampling Distance $< 1.5\text{ cm/px}$).
   * Medium-altitude overview: $60\text{--}80\text{ m}$ AGL (GSD $< 3.5\text{ cm/px}$).
2. **Camera Angles:**
   * $45^\circ$ Oblique: Captures both vertical architectural relief (walls, columns) and horizontal floor plinths.
   * $90^\circ$ Nadir: Orthomosaic mapping for boundary spatial alignment.
3. **Solar & Weather Constraints:**
   * Diffuse overcast or early morning/late afternoon ($15^\circ\text{--}35^\circ$ solar elevation) to avoid harsh vertical shadows that obscure foundation boundaries.
4. **Anti-Leakage Independence Rule:**
   * No two images may share $>30\%$ visual overlap unless designated as distinct multi-scale views of an expansive complex ($>100\text{ m}$ separation). Near-duplicate crops from single sensor bursts are strictly barred.

---

## 4. Annotation Standards for YOLO-OBB

1. **Structures vs. Ruins Semantic Decision Rule:**
   * **`structures` (Class 3):** Any architectural feature retaining recognizable geometric integrity (standing walls $> 1.0\text{ m}$ in height, intact doorways, stupa bases with defined circumambulation paths, covered shrines, and intact pillared halls).
   * **`ruins` (Class 2):** Fallen rubble, displaced masonry blocks, collapsed wall fragments, broken plinths, and eroded earthen mounds lacking straight wall headers.
2. **Geometric Bounding:**
   * Rotated bounding boxes must strictly enclose the physical perimeter of the foundation without clipping decorative corners.
   * Quadrilaterals must maintain strict convexity with coordinates normalized to $[0.0, 1.0]$.
   * Multi-wall perimeter courtyards must be broken into constituent rectangular wings rather than enclosing vast open courtyards within a single box.

