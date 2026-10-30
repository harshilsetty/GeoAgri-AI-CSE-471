# Technical Audit: GeoAgri AI (Crop Intelligence V5)

**Course Assessment:** INT373 — Agile Driven Development and Project Management  
**Audit Date:** October 2026  
**Auditor:** Senior ML Engineer & Software Quality Auditor  
**Repository State:** `harshilsetty/GeoAgri-AI` (formerly `Archaeological-Site-Mapping-AI-V4`)  
**Audited Runtime Environment:** Windows 11 Enterprise x64 · Python 3.13.3 · Node.js v20.x · NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.0)

---

## 1. Executive Summary & Verification Methodology

This technical audit provides an empirical, code-level verification of the GeoAgri AI software repository. Every technical claim in this document has been verified against active source code, binary model artifacts, serialised metadata, test suites, or live runtime execution. 

In strict adherence to academic integrity and project management standards, system capabilities are classified under six mutually exclusive verification tiers:
- **`IMPLEMENTED`**: Fully implemented in source code and operational within the runtime application.
- **`TESTED`**: Validated by reproducible test execution, automated test suites, or live pipeline execution.
- **`REPOSITORY-REPORTED`**: Documented in historical repository logs/benchmarks but not independently re-trained during this audit.
- **`PARTIALLY IMPLEMENTED`**: Module scaffolds or partial functions exist, but end-to-end integration is incomplete or coupled to deprecated legacy code.
- **`PROPOSED`**: Roadmap items, speculative design ideas, or future grant objectives without operational implementation.
- **`UNVERIFIED`**: Inconclusive or absent evidence to validate the capability.

---

## 2. Complete Evidence Inventory

| Claim or Capability | Evidence Path | Relevant Function / Class / Object | Verification Method | Status | Technical Limitation / Risk |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Multiclass Crop Classification (16 crops)** | `models/crop/v5/crop_model.pkl`<br>`src/models/agri_inference.py` | `agri_inference.load_crop_artifacts()`<br>`xgb.XGBClassifier.predict_proba()` | Python bytecode load & inference test on 546 held-out samples | **`TESTED`** | Multiclass top-1 accuracy is 18.86%; minority classes (Cotton, Soybean) suffer 0.0% precision without domain fusion. |
| **Candidate-Level Learning-to-Rank (LambdaMART)** | `models/crop/v5/ranking_model.pkl`<br>`scripts/train_crop_v5.py` | `xgb.XGBRanker(objective="rank:ndcg")`<br>`run_learning_to_rank_benchmark()` | Model loading and evaluation in `ranking_model_comparison.csv` | **`TESTED`** | Falsified central hypothesis: LambdaMART achieved NDCG@5 = 0.3580 vs Multiclass NDCG@5 = 0.3899. |
| **Yield Regressor (t/ha)** | `models/crop/v5/yield_model.pkl`<br>`src/models/agri_inference.py` | `agri_inference.estimate_yield_and_potential()`<br>`xgb.XGBRegressor` | Loading binary, evaluating on conditioned test set | **`TESTED`** | Conditioned $R^2 = 0.954$, but MAPE is high (76.73%) due to low-yield crop denominators (pulses/oilseeds $\le 0.5$ t/ha). |
| **26-Feature Agronomic Preprocessing** | `src/preprocessing/agri_features.py`<br>`agri_features.py` | `engineer_features_row()`<br>`engineer_dataframe()` | Unit test execution (`test_feature_engineering.py`) | **`TESTED`** | Static ICAR benchmark scoring; assumes neutral optimal pH = 7.0 for all crops in SFI. |
| **Physiological Compatibility Engine** | `src/models/agri_inference.py` | `CROP_ECOLOGICAL_RULES`<br>`calculate_domain_suitability()` | Test execution (`test_crop_v3_knowledge.py`) | **`TESTED`** | Hard-coded heuristic rules from FAO EcoCrop; does not account for local cultivar/micro-climate variance. |
| **Multi-Objective Pareto Engine** | `models/crop/v5/feature_metadata.json`<br>`artifacts/crop_v5/pareto_analysis.csv` | `supported_decision_profiles`<br>`compute_pareto_frontier()` | Code audit and artifact verification | **`IMPLEMENTED`** | Computed over 5 decision objectives; front filtering is done offline/post-inference. |
| **Live Soil Profile Ingestion** | `src/geospatial/fetch_soil.py` | `fetch_unified_soil_profile()` | Live API call with real GPS (15.335, 76.46) | **`TESTED`** | Relies on external SoilGrids REST API; falls back to static district priors if network fails. |
| **Live Weather & Climatology Ingestion** | `src/geospatial/fetch_weather.py` | `fetch_live_weather()` | Live API call with Open-Meteo REST service | **`TESTED`** | Dependent on free-tier rate limits and availability of Open-Meteo servers. |
| **Live Terrain & Slope Ingestion** | `src/geospatial/fetch_terrain.py` | `fetch_terrain_profile()` | Live API call to Open-Elevation / local SRTM fallback | **`TESTED`** | High latency on public Open-Elevation endpoint (~1.5s per request). |
| **Soil Erosion Risk Predictor** | `terrain_model/erosion_model.pkl` | `joblib.load(MODEL_PATH)` | Exact test execution (`test_erosion_regression.py`) | **`TESTED`** | Preserved from archaeological erosion dataset (synthetic 1000 rows); 90.50% holdout accuracy verified. |
| **Local Explainability (TreeSHAP)** | `src/models/agri_inference.py` | `agri_inference._compute_shap_explanations()` | Module inspection & test execution | **`TESTED`** | High CPU overhead when computing full KernelSHAP; relies on TreeSHAP fast path. |
| **LLM Natural Language Reasoning** | `src/explainability/ai_explainer.py` | `generate_groq_from_prompt_with_status()` | Code inspection and mock test validation | **`IMPLEMENTED`** | Requires `GROQ_API_KEY` environment variable; gracefully falls back to deterministic rule template if absent. |
| **FastAPI REST API Service** | `src/api/backend_api.py`<br>`backend_api.py` | `app = FastAPI()`<br>`/api/agriculture/recommend` | Integration test and code audit | **`PARTIALLY IMPLEMENTED`** | Crashes on startup if `cv2` or `torch` are missing due to eager import of legacy `predict_bridge.py`. |
| **Next.js 14 Interactive Web Dashboard** | `ui/app/page.tsx`<br>`ui/components/` | React client components (`CropIntelligencePanel`) | UI source code audit & dependency check | **`IMPLEMENTED`** | Subprocess bridge `predict_worker.py` fails if python environment lacks deep vision modules. |
| **Automated Testing Suite** | `tests/` (17 test modules) | 75 collected automated tests | `pytest` CLI execution in `.venv` | **`TESTED`** | 75/75 executed tests passed (100%); 2 test modules require `matplotlib` (now verified). |
| **Computer Vision (Satellite/Drone)** | `src/vision/README.md`<br>`ui/server/predict_bridge.py` | `ultralytics.YOLO`, `smp.DeepLabV3Plus` | Code inspection of deprecation notice | **`PROPOSED`** | Roadmap target for KPIT Sparkle 2027; operational vision weights are legacy archaeological site models. |
| **Automated Volumetric Irrigation** | `assets/irrigation-advisory-preview.png` | Mentioned in markdown documentation | Asset review and search in `src/` | **`PROPOSED`** | UI preview card exists in `assets/`; no operational calculation engine in `src/`. |
| **Canopy Moisture Stress Zonation** | `assets/stress-map-preview.png` | Mentioned in markdown documentation | Asset review and search in `src/` | **`PROPOSED`** | Roadmap concept; Copernicus Sentinel-2 live ingestion pipeline not implemented in `src/geospatial/`. |

---

## 3. End-to-End Execution Flow

```
[User GPS (lat, lon) & Season]
             │
             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 1. Geospatial Data Ingestion                           │
  │    • SoilGrids API -> pH, N, P, K, OC, Sand, Silt, Clay│
  │    • Open-Meteo API -> Temp, Humidity, Rain, ET0       │
  │    • Open-Elevation -> Elevation, Calculated Slope     │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2. Feature Engineering Pipeline (26 Dimensions)        │
  │    • Soil Fertility Index (SFI) [ICAR Ratings]         │
  │    • Water Stress Index (WSI)                          │
  │    • Rainfall Deviation Anomaly                        │
  │    • Categorical Texture & Season Mappings             │
  │    • Legacy Erosion Risk Score (XGBoost 8-Feature)     │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 3. ML Inference & Knowledge Fusion (V5 Engine)         │
  │    • Multiclass XGBoost Softmax -> 16 Probabilities    │
  │    • Crop Ecological Rules -> Compatibility [0, 1]     │
  │    • Hybrid Fusion: Score = 0.85 * ML + 0.15 * KB      │
  │    • XGBoost Yield Regressor -> Potential (t/ha)       │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 4. Multi-Objective Decision & Explainability Layer     │
  │    • 5-Objective Pareto Frontier Filtering             │
  │    • Decision Profile Tuning (Yield/Water/Risk)        │
  │    • TreeSHAP Attribution (Top Drivers & Limitations)  │
  │    • Counterfactual Simulation (Rainfall/pH Shifts)    │
  │    • Downstream Groq Llama-3.1 Synthesis (or Fallback) │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
               [Structured JSON Response]
```

---

## 4. Model Artifact Inventory

All models are serialized using `joblib` and located in `models/crop/` and `terrain_model/`:

| Model Identifier | Relative Path | File Size | Algorithm / Architecture | Input Dimension | Output Target | Verified Status |
| :--- | :--- | :---: | :--- | :---: | :--- | :---: |
| **Crop Model V5** | `models/crop/v5/crop_model.pkl` | 4.24 MB | XGBoost Classifier (`multi:softprob`) | 26 features | 16-class probability distribution | `TESTED` |
| **Label Encoder V5**| `models/crop/v5/label_encoder.pkl` | 638 B | Scikit-Learn `LabelEncoder` | N/A | 16 discrete crop string labels | `TESTED` |
| **Ranking Model V5**| `models/crop/v5/ranking_model.pkl` | 250 KB | XGBoost Ranker (`rank:ndcg`) | 43 features (query+cand+KB) | Continuous ranking relevance score | `TESTED` |
| **Yield Model V5** | `models/crop/v5/yield_model.pkl` | 552 KB | XGBoost Regressor (`reg:squarederror`)| 42 features (one-hot candidate) | Yield estimate in metric tonnes/ha | `TESTED` |
| **Erosion Model** | `terrain_model/erosion_model.pkl` | 520 KB | XGBoost Classifier (`binary:logistic`)| 8 geomorphic features | Binary soil erosion risk probability | `TESTED` |
| **Crop Model V4** | `models/crop/v4/crop_model.pkl` | 4.24 MB | XGBoost Classifier (`multi:softprob`) | 26 features | 16-class probability distribution | `TESTED` |
| **Crop Model V3** | `models/crop/v3/crop_model.pkl` | 4.24 MB | XGBoost Classifier (`multi:softprob`) | 26 features | 16-class probability distribution | `TESTED` |
| **Crop Model V2** | `models/crop/v2/crop_model.pkl` | 3.81 MB | XGBoost Classifier (`multi:softprob`) | 26 features | 16-class probability distribution | `TESTED` |
| **Crop Model V1** | `models/crop/v1/crop_model.pkl` | 3.12 MB | XGBoost Classifier (`multi:softprob`) | 20 features | 16-class probability distribution | `TESTED` |

---

## 5. Dataset Inventory & Provenance

| Dataset Title | Storage Path | Rows | Columns | Source Provenance | Target Variable | Data Quality Finding |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **Indian Agro-Climatic Multi-Season Dataset** | `data/agriculture/v1.0/processed/crop_suitability_dataset.csv` | 10,091 | 31 | Ministry of Agriculture (DES), ICRISAT, ICAR-IISS, Open-Meteo | `crop` (16 classes) & `yield` (t/ha) | Complete records (0 missing); 4 rows violate soil particle texture conservation ($\text{sand}+\text{silt}+\text{clay} \ne 100\%$). |
| **Historical District Coordinates** | `data/agriculture/v1.0/raw/lat_lon_india_district.csv` | 640 | 5 | Survey of India / Census 2011 district centroid mapping | `latitude`, `longitude` | Static spatial coordinate lookup. |
| **Crop Knowledge Base V1** | `data/agriculture/knowledge/v1/crop_knowledge_base.json` | 16 crops | Detailed rules | FAO EcoCrop database and ICAR package of practices | Agronomic thresholds | Deterministic bounds for pH, rainfall, temperature, and slope tolerance. |
| **Erosion Dataset** | `terrain_model/erosion_dataset.csv` | 1,000 | 9 | Synthetic geomorphic simulation (`generate_dataset_v3.py`) | `erosion` (binary) | Balanced (500:500); purely synthetic, derived from archaeological terrain simulation. |

---

## 6. Testing Inventory & Verification Results

The automated test suite in `tests/` was executed using `.venv\Scripts\python.exe -m pytest`:

| Test Module | Executed Test Count | Tests Passed | Tests Failed | Primary Verification Focus |
| :--- | :---: | :---: | :---: | :--- |
| `tests/test_crop_v5_validation.py` | 25 | 25 | 0 | V5 model schema, Pareto engine, decision profiles, calibration, immutability |
| `tests/test_crop_v4_validation.py` | 13 | 13 | 0 | V4 baseline verification, knowledge fusion stability, constraint bounds |
| `tests/test_crop_inference.py` | 2 | 2 | 0 | End-to-end `generate_crop_recommendation()` pipeline execution |
| `tests/test_feature_engineering.py` | 3 | 3 | 0 | 26-feature transformation, SFI, WSI, and rainfall deviation calculations |
| `tests/test_crop_model_registry.py` | 4 | 4 | 0 | Version resolution (V1 through V5) and fallback handling |
| `tests/test_crop_v3_counterfactual.py` | 2 | 2 | 0 | Counterfactual climate/soil perturbation scenario responses |
| `tests/test_crop_v3_knowledge.py` | 3 | 3 | 0 | FAO EcoCrop rule boundaries and penalty gating |
| `tests/test_crop_v3_ranking.py` | 3 | 3 | 0 | Top-K accuracy, NDCG@K, MRR, and MAP calculation accuracy |
| `tests/test_crop_v3_regression.py` | 4 | 4 | 0 | Yield prediction bounds, non-negativity, and yield potential normalization |
| `tests/test_crop_v3_uncertainty.py` | 2 | 2 | 0 | Decision margin and ranking stability metrics |
| `tests/test_soil_weather_terrain.py` | 4 | 4 | 0 | Geospatial API response schemas and district prior fallbacks |
| `tests/test_erosion_regression.py` | 3 | 3 | 0 | Holdout verification of legacy 90.50% erosion model accuracy |
| `tests/test_crop_v2_model.py` | 3 | 3 | 0 | V2 legacy model artifact preservation and prediction shape |
| `tests/test_crop_calibration.py` | 2 | 2 | 0 | Expected Calibration Error (ECE) and Brier score evaluation |
| `tests/test_crop_ranking.py` | 2 | 2 | 0 | Synthetic ranked query testing and tie-breaking |
| **Total Test Execution** | **75** | **75** | **0** | **100% Pass Rate across all operational modules** |

*Note on Collection Errors:* `tests/test_api_endpoints.py` failed test collection when executed in an environment lacking computer-vision modules (`cv2`, `torch`), because `src/api/backend_api.py` imports `ui/server/predict_bridge.py` at top-level. When tested independently via direct function calls, all API endpoints succeed.

---

## 7. Contradictions Between Documentation and Implementation

1. **Repository Identity & Scope Contradiction:**  
   The project README and assets represent GeoAgri AI as an advanced multi-modal agricultural remote sensing platform integrating Sentinel-2 satellite imagery, drone orthomosaics, and computer vision canopy segmentation for KPIT Sparkle 2027. However, the active codebase in `src/` is an offline tabular XGBoost agronomic decision-support engine. The vision subsystem (`src/vision/README.md`) is documented as "In design / roadmap", and operational vision weights in `terrain_model/` are legacy archaeological site detection models.
2. **Dataset Validation Contradiction:**  
   In `data/agriculture/v1.0/dataset_validation_report.json`, the sub-check for `soil_ranges` reports `"status": "FAIL"` with `texture_sum_violations: 4`. Yet the overarching report status on line 127 is marked `"status": "PASS"`. This contradiction must be explicitly disclosed: 4 sample records violate physical soil texture conservation.
3. **Scoring Formulation Discrepancy:**  
   In `scripts/train_crop_v5.py`, model evaluation benchmarks candidate ranking using the formula $\text{Score} = 0.85 \cdot P_{\text{ML}} + 0.15 \cdot S_{\text{KB}}$. Conversely, the operational runtime pipeline in `src/models/agri_inference.py` calculates composite domain suitability via a 5-factor weighted sum: $\text{Weight}_{\text{Model}} = 0.45$, $\text{Weight}_{\text{Soil}} = 0.20$, $\text{Weight}_{\text{Season}} = 0.15$, $\text{Weight}_{\text{Climate}} = 0.10$, $\text{Weight}_{\text{Erosion}} = 0.10$.
4. **Promotion Justification vs Empirical Metrics:**  
   `models/crop/v5/metrics_summary.json` declares Model V5 as `PROMOTED`. However, the underlying empirical data in `v4_vs_v5.csv` and `ranking_model_comparison.csv` reveals that V5 achieved slightly lower random-split Top-5 accuracy (58.24% vs 59.34% in V4) and that LambdaMART underperformed multiclass softmax. V5 was promoted on architectural grounds (Pareto multi-objective engine and spatiotemporal holdout robustness), not pure classification superiority.

---

## 8. Claims That Must NOT Be Made in the Academic Report

To preserve academic integrity in Continuous Assessment (INT373), the following unsupported claims must be strictly prohibited:
1. **DO NOT claim operational satellite or drone computer vision:** The system does not currently ingest live Sentinel-2 bands or execute CNN/Transformer canopy segmentation.
2. **DO NOT claim high Top-1 classification accuracy:** The model's Top-1 accuracy is 18.86%. It is an exploratory recommendation system designed for Top-3 (42.31%) and Top-5 (58.24%) candidate screening, not a definitive single-crop classifier.
3. **DO NOT claim that LambdaMART proved superior to Softmax:** The empirical data conclusively falsified this hypothesis; multiclass softmax outperformed LambdaMART across all metrics.
4. **DO NOT claim real-world farmer validation or yield improvement:** No field trials, randomized control trials (RCTs), or direct farmer deployments have been conducted.
5. **DO NOT claim the LLM validates the agronomic predictions:** The Groq Llama-3.1 model provides natural-language narrative formatting for human readability; it does not independently verify the numerical predictions.
