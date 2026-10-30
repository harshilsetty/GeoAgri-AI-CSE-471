# GeoAgri-AI — CSE 471 Research & Viva Presentation Slides

**Academic Course:** CSE 471 — Deep Learning for Computer Vision  
**Assessment Type:** Continuous Assessment (CA) Project / Final Viva Presentation  
**Presentation Theme:** Premium Light Mode (Apple Keynote / Google Material Design inspired)  
**Deliverable File:** [`presentation/GeoAgri_AI_CSE471_Presentation.pptx`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/presentation/GeoAgri_AI_CSE471_Presentation.pptx)  

---

## Slide 1 — Cover
* **Header Tag:** `ACADEMIC CA PROJECT • SEM 5`
* **Title:** GEOAGRI-AI
* **Subtitle:** An Integrated Geospatial AI Decision-Support Framework for Agricultural Intelligence & Terrain Analysis
* **Core Triad (3 Interconnected Intelligence Cards):**
  1. **🌱 Crop Intelligence:** XGBoost Multiclass Classifier, 26 agro-climatic dimensions, FAO EcoCrop / ICAR physiological guardrails.
  2. **🏔️ Terrain Intelligence:** XGBoost Regressor/Classifier, 8 environmental features, SRTM elevation, TreeSHAP local attributions.
  3. **🛰️ Vision Intelligence:** YOLO11n-OBB, 5 rotated architectural & terrain classes, $640 \times 640$ aerial resolution, damped class weighting.

---

## Slide 2 — Problem and Motivation: One Landscape. Multiple Intelligence Needs.
* **Lead In:** Agricultural landscapes operate under overlapping pedological, climatic, and heritage constraints that fragmented single-purpose tools fail to address.
* **Three Domain Challenges:**
  1. **Agronomic Uncertainty (Soil + Climate → Crop Selection):** Traditional static tables fail to account for rainfall deficits. Non-linear multi-nutrient interactions require biophysical constraints to prevent unviable crop selection.
  2. **Land Degradation (Terrain + Rainfall → Soil Erosion):** Steep slopes combined with monsoon rainfall accelerate irreversible topsoil loss; requires continuous spatial risk assessment.
  3. **Heritage & Physical Risk (Aerial Imagery → Feature Detection):** Agrarian lands contain ancient structures and stone ruins lying at arbitrary angles. Unsupervised development risks destroying archaeological heritage.

---

## Slide 3 — Research Gap: Integration, Interpretability and Generalisation
* **Core Research Position:** The project contribution is **system-level integration and rigorous empirical evaluation** across heterogeneous modalities — not the speculative invention of a new ML algorithm.
* **Three Distinct Research Gaps:**
  1. **Fragmented Systems:** Agronomic, erosion, and drone vision exist in isolated software silos. GeoAgri-AI bridges subsurface chemistry with aerial computer vision.
  2. **Interpretability Barrier:** Complex deep models act as black boxes. GeoAgri-AI integrates TreeSHAP game-theoretic Shapley values and ICAR physiological gating.
  3. **Generalisation Shift:** Aerial detectors tuned on one site fail on unseen geographies. GeoAgri-AI evaluates domain augmentation and damped class weighting under cross-site distribution shifts.

---

## Slide 4 — Six-Layer Architecture: Six Logical Layers Connect Data to Decisions
* **Layer 1 (Data Acquisition):** ISRIC SoilGrids (0-30cm N, P, K, pH), Open-Meteo Climatology, SRTM 30m Elevation, Aerial Imagery.
* **Layer 2 (Data Processing):** Multivariate cleaning, 26D agronomic engineering, OBB 4-point coordinate normalization, spatial indexing.
* **Layer 3 (ML Intelligence):** XGBoost Crop (120 trees), XGBoost Erosion (300 trees), YOLO11n-OBB (196 layers, 6.7 GFLOPs).
* **Layer 4 (Backend / API Tier):** Asynchronous FastAPI gateway, Python prediction bridge, strict JSON schema validation.
* **Layer 5 (Visualisation Engine):** OpenCV OBB polygon overlays, Gaussian feature density heatmap, Leaflet geospatial risk halos, TreeSHAP bar plots.
* **Layer 6 (User Interface):** Next.js 14 App Router, TypeScript, Tailwind CSS, interactive map selection, real-time layer toggles.

---

## Slide 5 — Three AI Models: Three Engines. Three Prediction Tasks.
* **Model 1: Crop Intelligence (Multiclass Classification):**
  - Algorithm: XGBoost Classifier (120 boosting trees).
  - Features: 26 engineered agro-climatic dimensions.
  - Gating: FAO EcoCrop & ICAR biophysical feasibility rules.
  - Evaluation: 546 strictly chronological holdout queries.
* **Model 2: Terrain Erosion (Probability Regression & Classification):**
  - Algorithm: XGBoost Classifier / Regressor (300 trees, max depth 6).
  - Features: 8 terrain and environmental features (slope, rainfall, soil, elevation, feature ratios).
  - Explainability: Exact TreeSHAP local attributions.
  - Risk Thresholds: Low ($<30\%$), Moderate ($30-69\%$), High ($\ge 70\%$).
* **Model 3: Aerial Computer Vision (Oriented Object Detection - OBB):**
  - Algorithm: YOLO11n-OBB (196 layers, 344 PyTorch modules).
  - Parameters: 2.66M parameters ($6.7\text{ GFLOPs}$, $10.2\text{ ms}$ latency).
  - Target: Rotated 4-point bounding polygons and confidence scores.

---

## Slide 6 — Dataset: 740 Aerial Images. 18,458 Annotations.
* **Key Numerical KPIs:**
  - **740** Total Aerial Tiles ($640 \times 640$ resolution).
  - **18,458** Oriented Bounding Boxes.
  - **5** Object Classes.
* **Dataset Partitions & Scene Distribution:**
  - **Training Set (69.5%):** 514 images across 88 unique aerial scenes.
  - **Validation Set (14.7%):** 109 images across 19 unique aerial scenes.
  - **Held-Out Test Set (15.8%):** 117 images across 18 unique aerial scenes.
* **Class Imbalance:**
  - Boulders and Vegetation represent $>80\%$ of annotations.
  - **Structures (Critical Minority):** $<3\%$ of total annotations. Models trained without class weighting scored **0.00% recall** on test scenes.

---

## Slide 7 — Experimental Comparison: YOLO11 OBB Experimental Results

| Experiment | Training Strategy | Precision | Recall | mAP@50 | mAP@50–95 | Structures Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EXP-B** | Geometry Refitting | **60.25%** | 37.31% | 43.06% | 25.67% | 63.89% |
| **EXP-C** | Domain Augmentation | 57.22% | 41.40% | **44.97%** | **26.38%** | 72.22% |
| **EXP-D** | Damped Class Weights | 37.20% | **44.08%** | 41.50% | 25.14% | **77.11%** |
| **EXP-E** | Capacity Scaling (YOLO11s) | 46.80% | 37.17% | 37.97% | 22.97% | 55.56% |

* **EXP-B:** Highest precision (60.25%) by eliminating flipped normals and invalid geometries.
* **EXP-C:** Highest overall mAP@50 (44.97%) via HSV jittering and $360^\circ$ rotation invariance.
* **EXP-D:** Highest overall recall (44.08%) and structures recall (77.11%) using damped inverse-frequency weights ($w = [0.95, 1.41, 1.16, 3.88, 0.56]$).

---

## Slide 8 — Cross-Site Generalisation: Validation Performance Did Not Transfer to Unseen Sites
* **Structures Recall Transfer Gap:**
  - **EXP-C:** Validation: **72.22%** $\to$ Test Split: **0.00%** (Complete Blackout).
  - **EXP-D:** Validation: **77.11%** $\to$ Test Split: **7.16%** (Broke 0% Barrier).
* **EXP-D Held-Out Test Audit:**
  - Test Precision: **20.24%**
  - Test Recall: **19.94%**
  - Test mAP@50: **12.89%**
  - Test mAP@50–95: **5.91%**
* **Root Cause (Geographic Confounding):** 100% of boulders are located in Hampi, while structures reside in Nalanda/Pattadakal. The model learned local background lithology shortcuts.
* **Methodological Disclosure:** The test partition was consulted during earlier baseline audits; it confirms real-world cross-site degradation, not an untouched external benchmark.

---

## Slide 9 — Model Architecture and Selection: Why YOLO11n-OBB?
* **Comparative Evaluation:**
  - **YOLO11n-OBB (EXP-D):** 2.66M parameters, 6.7 GFLOPs, 10.2 ms latency, 77.11% structures recall. **Selected as Experimental Leader.**
  - **YOLO11s-OBB (EXP-E):** 9.72M parameters, 22.5 GFLOPs, 13.8 ms latency, 55.56% structures recall. **Rejected due to overfitting on 88 training scenes.**
  - **YOLOv8s Baseline:** 11.14M parameters, 28.7 GFLOPs, 16.4 ms latency. **Retained in Production.**
* **Production Isolation Policy:** EXP-D is strictly quarantined in research and was NOT promoted to production in `config/models_registry.yaml` because cross-site generalization remains unproven.

---

## Slide 10 — Dashboard Visualisation: Transparent Visual Analytics
* **Four Core Visual Components:**
  1. **OBB Polygon Footprints:** Rotated 4-point bounding polygons with vibrant class borders and confidence scores.
  2. **Feature Density Heatmap:** Gaussian kernel density map over detected entity centroids.
  3. **Erosion Risk Heatmap:** Multi-tier radial heat halos on the Leaflet GIS map dynamically colored by erosion probability ($<30\%$ Green, $30-69\%$ Amber, $\ge 70\%$ Red).
  4. **Interactive Layer Controls:** Independent toggles for Heatmap Halos, Markers, and Bounding Overlays.
* **Academic Disclaimer:**
  > *"Pixel-level segmentation is currently unavailable. The active computer vision model supports oriented object detection."*

---

## Slide 11 — Technology Stack and Testing: Full-Stack Integration and Verification
* **Decoupled Stack:** Next.js 14, React 18, TypeScript, Tailwind CSS, FastAPI, Uvicorn, XGBoost, PyTorch 2.6 CUDA, OpenCV, Leaflet, TreeSHAP.
* **Reported Software Test Audit:**
  - **API Endpoint Suite:** **4 / 4 PASSED (100%)** in 61.11s.
  - **YOLO OBB Hardening Suite:** **8 / 8 PASSED (100%)** in 107.15s.
  - **Frontend ESLint:** **0 ERRORS (PASS)**.
  - **Production Next.js Build:** **4 / 4 ROUTES COMPILED (PASS)**.

---

## Slide 12 — Limitations and Future Work: Critical Reflection and Future Research
* **Current System Limitations:**
  - Low cross-site detection generalization.
  - Extreme class imbalance ($<3\%$ structures).
  - Test partition was examined during baseline development.
  - Pixel-level semantic segmentation is unavailable.
  - Standalone crop and erosion precision/recall metrics require field agronomic trials.
* **Future Research Roadmap:**
  - **AGB-120 Multi-Site Expansion:** 120-scene field collection across 6 new Indian archaeological sites.
  - Truly untouched external blind test benchmark.
  - Targeted minority-class tile-level mosaic oversampling.
  - Dense pixel annotation for U-Net / DeepLabV3+ masonry mortar segmentation.
  - Multi-site regional ICAR field sensor calibration.

---

## Slide 13 — Conclusion: An Integrated Prototype. A Measurable Research Lesson.
* **Three Central Contributions:**
  1. **System Integration:** Successfully unified 26D agronomic intelligence, 8D terrain erosion risk, and aerial OBB detection into an interactive platform.
  2. **Empirical Rigor:** Conducted 4 controlled experiments (EXP-B to EXP-E), elevating structures recall to 77.11% with 6.7 GFLOPs efficiency.
  3. **Core Scientific Insight:** Proved that boosting validation recall on a single-site dataset does NOT guarantee cross-site generalization, and that capacity scaling overfits on limited aerial scenes.
* **Viva Closing:** *Thank You — Questions & Defense Discussion*.

---

## Slide 14 — Selected Technical and Research References
1. **Archaeological Object Detection:** Olivier, M., & Verschoof-van der Vaart, W. B. (2021). *Detecting Archaeological Features in Airborne Laser Scanning Data Using Automated Deep Learning*. Journal of Computer Applications in Archaeology, 4(1), 1-18. DOI: 10.5334/jcaa.78
2. **Real-Time Object Detection Framework:** Jocher, G., Chaurasia, A., & Qiu, J. (2024). *YOLO by Ultralytics (Version 11.0.0)*. Ultralytics Inc.
3. **Gradient Boosted Tree Architecture:** Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. In Proc. 22nd ACM SIGKDD (pp. 785-794).
4. **Explainable AI & Feature Attribution:** Lundberg, S. M., et al. (2020). *From local explanations to global understanding with explainable AI for trees*. Nature Machine Intelligence, 2(1), 56-67.
5. **Soil Pedology Database:** ISRIC — World Soil Information (2021). *SoilGrids 250m 2.0: Global gridded soil information*.
6. **Meteorological Reanalysis & Forecasts:** Zippenfenig, P. (2023). *Open-Meteo.com Weather API*.
7. **Academic Course Repository:** *GeoAgri-AI — CSE 471 Deep Learning for Computer Vision Project Repository*. https://github.com/harshilsetty/GeoAgri-AI-CSE-471
