# GeoAgri-AI: Computer Vision Task Architecture & Downstream Integration

**Document ID:** `docs/geoagri_cv_task_architecture.md`  
**Auditor:** Senior Computer Vision Engineer & System Architect  
**Date:** October 28, 2026  
**Status:** Completed & Validated

---

## 1. Architectural Overview & Separation of Concerns

A critical finding of this audit is that **GeoAgri-AI's computer vision requirements must be separated into two distinct visual tasks**, rather than forcing one model to perform incompatible duties:

```
+--------------------------------------------------------------------------------------------------+
|                                    AERIAL / DRONE INPUT IMAGE                                    |
+--------------------------------------------------------------------------------------------------+
                                  |                                     |
                                  v                                     v
+----------------------------------------------------+   +-----------------------------------------+
|      TASK A: DISCRETE OBSTACLE DETECTION (YOLO)    |   | TASK B: TERRAIN COVERAGE (SEMANTIC SEG) |
|   Architecture: YOLO11-OBB or YOLOv8/11 Detect     |   | Architecture: DeepLabV3+ (ResNet-34)    |
|   Target Output: Rotated / Standard Bounding Boxes |   | Target Output: Pixel-wise Mask (0..5)   |
|   Classes: Boulders, Ruins, Structures             |   | Classes: Vegetated, Bare Ground, Rock   |
+----------------------------------------------------+   +-----------------------------------------+
                          |                                                   |
                          | (Object Counts & Spatial Locations)               | (Pixel Area Ratios)
                          v                                                   v
          +-------------------------------------------------------------------------------+
          |                        FEATURE AGGREGATION & BRIDGE                           |
          |               (ui/server/predict_bridge.py: build_erosion_features)           |
          |  - boulders_ratio   = sum(mask == 1) / total_pixels                            |
          |  - vegetation_ratio = sum(mask == 5) / total_pixels                            |
          |  - ruins_ratio      = sum(mask == 3) / total_pixels                            |
          |  - structures_ratio = sum(mask == 4) / total_pixels                            |
          +-------------------------------------------------------------------------------+
                                                  |
                                                  | + Ingest GIS / API:
                                                  |   slope, elevation, rainfall, soil
                                                  v
          +-------------------------------------------------------------------------------+
          |                  DOWNSTREAM EROSION CLASSIFIER (XGBOOST V2)                   |
          |             Output: P(erosion risk) -> LOW / MODERATE / HIGH                  |
          +-------------------------------------------------------------------------------+
                                                  |
                                                  v
          +-------------------------------------------------------------------------------+
          |               GEOAGRI-AI ANALYST DASHBOARD & VISUAL OVERLAYS                  |
          |              Detection Boxes + Segmentation Heatmap + TreeSHAP                |
          +-------------------------------------------------------------------------------+
```

---

## 2. Distinction Between Vision Tasks

### Task A: Discrete Object Detection / Oriented Bounding Box (YOLO)
* **Goal:** Detect, count, and localize physical objects (monolithic granite boulders, archaeological ruins, masonry walls, buildings).
* **Input Data:** The local archaeological dataset (`dataset/`).
* **Why YOLO is optimal here:** YOLO operates with extreme speed ($\approx 3\ \text{ms}$ on RTX 3050), accurately outputting counts and coordinates for discrete objects.
* **Appropriate Formulation:** **YOLO-OBB (Oriented Bounding Box)** or **YOLO Standard Detection**.

### Task B: Continuous Terrain Coverage Estimation (Semantic Segmentation)
* **Goal:** Quantify the exact percentage of surface area covered by vegetation ($V_{\text{ratio}}$) and bare erodible soil ($S_{\text{ratio}}$).
* **Why Bounding Boxes CANNOT Perform This Task:**
  - Bounding boxes (both standard and rotated) enclose significant background terrain within their rectangular perimeter.
  - For irregular tree foliage, calculating `sum(box_area) / total_area` inflates vegetation coverage by **$30\% - 65\%$**, leading the downstream erosion model to severely underestimate erosion risk.
* **Appropriate Formulation:** **Pixel-wise Semantic Segmentation (DeepLabV3+)**.

---

## 3. Data Flow into Downstream XGBoost Model

The erosion classifier (`models/erosion_model_v2.pkl`) requires an 8-feature vector:
$$\mathbf{x} = [\text{slope}, V_{\text{ratio}}, \text{elevation}, \text{rainfall}, \text{soil}, B_{\text{ratio}}, R_{\text{ratio}}, S_{\text{ratio}}]$$

* **From GIS / Remote Sensing APIs (`src/geospatial/`):**
  - `slope`: Extracted from Open-Meteo elevation gradient / SRTM DEM.
  - `elevation`: Extracted from elevation API.
  - `rainfall`: Extracted from weather API.
  - `soil`: Extracted from ISRIC SoilGrids.
* **From Computer Vision Pipeline (When Drone/Aerial Photo is Supplied):**
  - $V_{\text{ratio}}$ (`vegetation`): Pixel ratio from DeepLabV3+ semantic mask.
  - $B_{\text{ratio}}$ (`boulders`): Pixel ratio or density from YOLO detection mask.
  - $R_{\text{ratio}}$ (`ruins`): Surface ratio of historic masonry.
  - $S_{\text{ratio}}$ (`structures`): Surface ratio of built environment.
* **Point-Mode Fallback (When No Image is Supplied):**
  - In point mode, $V_{\text{ratio}}, B_{\text{ratio}}, R_{\text{ratio}}, S_{\text{ratio}}$ fall back to empirical regional priors defined in `_FEATURE_PRIORS` (`predict_bridge.py`).

---

## 4. Key Takeaways for System Integration

1. The local archaeological dataset directly powers **Task A** (Obstacle & Ruin Detection).
2. It should **not** be used to claim full agricultural crop classification or bare soil erosion modeling without introducing genuine agricultural imagery.
3. The existing architecture in `predict_bridge.py` is sound: it renders YOLO detection boxes for visual clarity and overlays semantic heatmaps for risk representation.

