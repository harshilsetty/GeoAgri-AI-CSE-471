# GeoAgri AI: An Explainable Geospatial Agricultural Decision-Support System

## Project Submission Report — CSE 471: Deep Learning for Computer Vision

---

### Student and Institutional Details

- **Student Name:** `[HARSHIL SOMISETTY / Editable Student Name]`
- **Registration Number:** `[Editable Registration Number]`
- **Roll Number & Section:** `[Editable Roll Number / Section]`
- **Course Code & Title:** CSE 471: Deep Learning for Computer Vision
- **Department:** Department of Computer Science & Engineering, School of Computer Science and Engineering
- **Institution:** Lovely Professional University, Punjab, India
- **Faculty / Evaluator:** `[Course Instructor / Evaluator Name]`
- **Date of Submission:** October 2026

---

### Abstract

Smallholder agricultural producers in India face escalating operational risks due to erratic monsoonal precipitation, groundwater depletion, and climate volatility. Existing digital agricultural advisory tools suffer from three fundamental limitations: empirical over-fitting to historical cultivation choices, unconstrained predictions that violate biophysical and pedological growth boundaries, and the opaque collapse of competing objectives (suitability, yield potential, water deficit, and soil erosion risk) into single scalar recommendations.

This report presents **GeoAgri AI (Crop Intelligence V5)**, an explainable geospatial decision-support system designed to provide transparent, multi-objective crop viability screening. The system integrates automated data ingestion from public environmental services (ISRIC SoilGrids, Open-Meteo Climatology, Open-Elevation SRTM grids) with a 26-dimensional agronomic preprocessing pipeline that computes normalized domain indicators, including the Soil Fertility Index (SFI), Water Stress Index (WSI), and Seasonal Rainfall Deviation ($\Delta R$). We empirically benchmark candidate-level Learning-to-Rank (LambdaMART) against multiclass gradient-boosted decision trees (XGBoost `multi:softprob`) across 161,456 candidate instances derived from 10,091 longitudinal Indian agricultural queries (1997–2020). The central ranking hypothesis was **falsified**: Multiclass Fusion outperformed LambdaMART across all metrics, achieving 18.86% Top-1, 42.31% Top-3, and 58.24% Top-5 accuracy on forward chronological test partitions (2019–2020).

To eliminate scalar collapse, V5 incorporates a Multi-Objective Pareto Decision Engine that computes non-dominated candidate trade-off frontiers across five orthogonal dimensions. An orthogonal crop-conditional yield regressor achieves a certified $R^2 = 0.954$ with a Median Absolute Error of 0.3645 t/ha. Dual-layer explainability couples exact polynomial-time TreeSHAP attributions and directional counterfactual simulations with a safety-constrained Large Language Model (Groq Llama-3.1-8b) for natural-language synthesis. Within the context of CSE 471, the operational system is rigorously identified as a geospatial tabular machine learning pipeline; deep learning computer-vision components (Copernicus Sentinel-2 multispectral ingestion and DeepLabV3+ canopy segmentation) are documented as planned future roadmap extensions.

**Keywords:** Agricultural Decision Support; Multi-Objective Optimization; Pareto Frontier; XGBoost; TreeSHAP; Geospatial Feature Engineering; Learning-to-Rank; Computer Vision Roadmap.

---

## 1. Introduction and Problem Statement

### 1.1 Background and Motivation
Agriculture sustains over 58% of India's rural populace across 15 distinct Planning Commission Agro-Climatic Zones [7]. Within these diverse ecosystems, smallholder farmers must make critical pre-sowing crop selection decisions months prior to harvest. An ecologically mismatched crop selection triggers severe vegetative stress, escalated groundwater pumping costs, and potential financial distress. Historically, farmers have relied on generational heuristics or market price signals from the preceding season. However, accelerating climatic variance renders traditional heuristics unreliable. Automated, data-driven decision support systems that integrate real-time pedological, climatological, and topographic context are urgently needed.

### 1.2 Problem Statement & Practical Dilemmas
Conventional agricultural recommendation models exhibit three systemic defects:
1. **Biophysical Boundary Violations:** Purely empirical classifiers trained on survey records learn statistical correlations rather than biological causality, frequently recommending wetland crops (Rice) on steep, erosion-prone slopes ($> 12^\circ$) or temperate cereals (Wheat) under acute vegetative heat stress.
2. **Scalar Objective Collapse:** Conventional systems rank crops via a single weighted score. This scalar collapse conceals critical trade-offs: a crop with high nominal suitability may demand excessive irrigation or accelerate topsoil erosion.
3. **Spatial Overfitting & Data Leakage:** In published literature, models frequently report inflated accuracy figures ($> 90\%$) derived from randomized $k$-fold cross-validation. As proven by Meyer and Pebesma [3], randomized splitting on spatial data leaks autocorrelation across neighboring agricultural plots, producing models that collapse when deployed to unseen geographic regions.

### 1.3 Project Objectives and Scope
The primary objectives of the GeoAgri AI project are:
- **O1 (Automated Ingestion):** Develop an automated geospatial ingestion pipeline querying SoilGrids, Open-Meteo, and Open-Elevation from GPS coordinates.
- **O2 (Agronomic Feature Pipeline):** Implement a 26-dimensional preprocessing pipeline deriving biophysical indices (SFI, WSI, $\Delta R$, SMI).
- **O3 (Candidate Screening Engine):** Train and certify an XGBoost 16-crop multiclass classifier fused with deterministic FAO EcoCrop constraints.
- **O4 (Multi-Objective Decision Engine):** Eliminate scalar collapse by computing the 5-dimensional non-dominated Pareto frontier across suitability, yield, water risk, and erosion risk.
- **O5 (Dual-Layer Explainability):** Implement exact TreeSHAP attributions, directional counterfactuals, and safety-constrained LLM narrative synthesis.

**Target Users & Scope:** The system is an exploratory decision-support prototype designed for smallholder farmers, agronomists, and extension officers. It provides pre-sowing candidate slates and does not claim to replace on-site chemical soil testing or certified agronomist counsel.

---

## 2. Related Work and Research Gap

### 2.1 Concise Literature Review
- **Rajak et al. (2017) [1]:** Implemented a multi-model voting ensemble (SVM, ANN, Naïve Bayes, Random Forest) on 2,200 Western Indian soil records, reporting 90.2% accuracy. However, the study relied strictly on randomized train/test splits, ignoring spatial autocorrelation, and excluded dynamic meteorological time-series.
- **Kondaveeti et al. (2023) [2]:** Benchmarked Random Forest, CatBoost, and Stacking Classifiers across longitudinal Indian agricultural data, achieving an F1-score of 0.88. The authors demonstrated the dominance of tree ensembles over neural networks for tabular agricultural prediction, but highlighted the absence of biophysical constraint gating and uncertainty intervals.
- **Meyer & Pebesma (2022) [3]:** Established that spatial cross-validation (Target-Oriented CV) is mandatory in environmental machine learning, proving that standard random splits underestimate true spatial transfer error by 25% to 60%.
- **Chen & Guestrin (2016) [4]:** Formulated XGBoost, demonstrating that regularized tree boosting with histogram-based split finding provides optimal tabular accuracy, computational efficiency, and native compatibility with exact Shapley value attributions.

### 2.2 Comparative Literature Matrix

| Study | Core Method | Dataset & Scope | Evaluation Split | Key Strengths | Critical Limitations | Relevance to GeoAgri AI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rajak et al. (2017)** [1] | Majority Voting Ensemble | 2,200 regional soil samples | Random 70/30 Split | Early multi-model ensemble for soil matching. | Excludes rainfall and weather; severe spatial data leakage. | GeoAgri AI incorporates dynamic Open-Meteo weather and physical texture. |
| **Kondaveeti et al. (2023)** [2] | Random Forest & CatBoost | Indian district statistics (1998–2017) | Stratified 5-Fold CV | Long-term weather and yield integration. | Pure classification; lacks biophysical constraint gating. | GeoAgri AI fuses XGBoost probabilities with FAO EcoCrop rules. |
| **Meyer & Pebesma (2022)** [3] | Spatial CV & Area of Applicability | Global environmental spatial grids | Spatial Leave-Location-Out CV | Foundational proof of spatial leakage in geospatial ML. | Methodological framework; no crop ranking algorithms. | GeoAgri AI evaluates Leave-State-Out and spatiotemporal holdouts. |
| **Chen & Guestrin (2016)** [4] | XGBoost Tree Boosting | Standard tabular benchmarks | 10-Fold CV | Regularization ($L_1$/$L_2$), GPU speed, exact split finding. | Generic ML algorithm; lacks domain-specific constraints. | Core classification, ranking, and regression engine in GeoAgri AI. |

### 2.3 Research Gap and System Contribution
**The Research Gap:** Existing systems treat crop recommendation as unconstrained single-label classification, collapse multi-objective criteria into arbitrary scalar weights, and fail to evaluate models under rigorous spatiotemporal transfer.

**GeoAgri AI Contribution:** We implement a calibrated hybrid system that couples an XGBoost tabular classifier with deterministic FAO EcoCrop physiological rules (0.85 ML + 0.15 KB), resolves scalar collapse via an operational 5-dimensional Pareto frontier engine, and documents an empirical falsification of candidate-level Learning-to-Rank (LambdaMART) on fixed candidate slates.

---

## 3. System Architecture and Technology Stack

### 3.1 End-to-End System Architecture
GeoAgri AI employs a three-tier decoupled architecture:
1. **Presentation Tier:** Next.js 14 interactive GIS interface utilizing Leaflet for spatial marker selection.
2. **Serving & IPC Dispatch Tier:** Dual-mode API serving featuring an asynchronous FastAPI REST service (`backend_api.py`) and a Node.js child-process worker bridge (`ui/server/predict_worker.py`).
3. **Core Intelligence Tier:** Python modules in `src/` encapsulating geospatial ingestion, 26-feature transformation, XGBoost inference, knowledge fusion, and Pareto optimization.

```
[User GPS Coordinates (Lat, Lon) & Cropping Season]
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Geospatial Data Ingestion Subsystem                      │
│    • ISRIC SoilGrids REST API (pH, N, P, K, OC, Sand, Clay) │
│    • Open-Meteo Climatology API (Temp, Humidity, Rain, ET0) │
│    • Open-Elevation SRTM DEM (Elevation, Calculated Slope)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Feature Engineering Pipeline (src/preprocessing/)        │
│    • Soil Fertility Index (SFI) [ICAR Nutrient Standards]   │
│    • FAO-56 Water Stress Index (WSI)                        │
│    • Seasonal Rainfall Deviation Ratio (ΔR)                 │
│    • USDA Soil Texture & Planned Season Categorical Codes   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Machine Learning & Knowledge Fusion (V5 Engine)          │
│    • XGBoost Multiclass Classifier (multi:softprob) -> P_ML │
│    • FAO EcoCrop Physiological Engine -> Compatibility S_KB │
│    • Hybrid Scoring Fusion: Score = 0.85 P_ML + 0.15 S_KB   │
│    • XGBoost Yield Potential Regressor -> Estimate (t/ha)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Multi-Objective Pareto Decision & Explainability Layer   │
│    • 5-Objective Non-Dominated Pareto Frontier Extraction   │
│    • 5 Farmer Decision Profiles (Suitability, Yield, Water) │
│    • TreeSHAP Polynomial-Time Local Feature Attribution     │
│    • Directional Counterfactual Sensitivity Simulations     │
│    • Groq Llama-3.1-8b Narrative Synthesis (or Local Fallback)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
    [Interactive GIS Dashboard / Structured JSON API Response]
```

### 3.2 Verified Technology Stack Matrix

| Layer / Technology | Component Version | Subsystem Responsibility | Selection Rationale |
| :--- | :---: | :--- | :--- |
| **XGBoost** | 3.4.1 | Core ML (Classification, Ranking, Yield) | Invariant to monotonic scaling; GPU histogram speed (`device='cuda'`); native TreeSHAP compatibility. |
| **Scikit-Learn** | 1.9.1 | Preprocessing & Metric Evaluation | Label encoding, spatial cross-validation splitting, Top-$K$ accuracy, NDCG calculation. |
| **FastAPI** | 0.142.2 | Production Python REST Service | Asynchronous non-blocking execution, native OpenAPI documentation, strict Pydantic validation. |
| **Next.js** | 14.2.3 | Frontend Framework (App Router) | Server-Side Rendering (SSR) for landing pages; client-side React hydration for interactive GIS mapping. |
| **Leaflet** | 1.9.4 | Interactive Geospatial Mapping | Lightweight, open-source mapping engine without proprietary API key licensing fees. |
| **TreeSHAP** | 0.52.0 | Exact Local Feature Attribution | Polynomial-time Shapley value computation for tree ensembles satisfying game-theoretic axioms. |
| **Groq Cloud API** | Cloud | Natural Language Synthesis | Low-latency (< 500 ms) inference of Meta Llama-3.1-8b-instant for plain-language farmer summaries. |
| **Pytest** | 9.1.1 | Automated Regression Testing | Unit and integration test suite executing 75 automated validation tests (100% pass rate). |

---

## 4. Dataset and Feature Engineering

### 4.1 Data Sources and Dataset Structure
The primary training dataset (`data/agriculture/v1.0/processed/crop_suitability_dataset.csv`) integrates agricultural records from the Ministry of Agriculture & Farmers Welfare (DES), ICRISAT District-Level Database, and DAC&FW Soil Health Card schemes:
- **Total Decision Queries ($N$):** 10,091 longitudinal observations.
- **Unique Crops ($K$):** 16 major Indian crops (Cereals: Rice, Wheat, Maize, Sorghum, Pearl Millet, Finger Millet; Pulses: Chickpea, Pigeonpea, Moong, Urad; Oilseeds: Groundnut, Mustard, Soybean, Sesamum; Cash: Cotton, Sugarcane).
- **Temporal Horizon:** 24 continuous agricultural years (1997 through 2020).
- **Spatial Coverage:** 30 Indian states and union territories covering all 15 agro-climatic zones.

### 4.2 26-Dimensional Engineered Feature Schema
Raw environmental inputs are transformed into a strictly ordered 26-feature vector:
1. Soil Chemistry: `soil_ph`, `nitrogen`, `phosphorus`, `potassium`, `organic_carbon`, `electrical_conductivity`.
2. Soil Physical Fractions: `clay`, `sand`, `silt` (USDA particle mass percentages).
3. Climatology & Weather: `temperature_mean`, `temperature_range`, `humidity_mean`, `rainfall_season`, `rainfall_30d`, `rainfall_90d`, `soil_moisture`, `et0`.
4. Surface Topography: `elevation`, `slope`, `erosion_risk_score`.
5. Composite Agronomic Indicators: `soil_fertility_index`, `water_stress_index`, `rainfall_deviation`, `soil_moisture_index`.
6. Encodings: `season_code` (Kharif=0, Rabi=1, Summer=2, Year=3), `texture_code` (USDA texture integer).

### 4.3 Mathematical Formulation of Composite Agronomic Indicators
- **Soil Fertility Index (SFI) $[5, 100]$:** Aggregates macro-nutrients against ICAR optimal benchmarks:
  $$\text{SFI} = \min\left(25, \frac{\text{N}}{300} \cdot 25\right) + \min\left(20, \frac{\text{P}}{25} \cdot 20\right) + \min\left(20, \frac{\text{K}}{260} \cdot 20\right) + \min\left(20, \frac{\text{OC}}{0.75} \cdot 20\right) + \max\left(0, 15 - 5 \cdot |\text{pH} - 7.0|\right)$$
- **Water Stress Index (WSI) $[0.1, 5.0]$:** Quantifies atmospheric evaporative deficit relative to moisture storage:
  $$\text{WSI} = \frac{\max(10.0, \, \text{ET}_0 \cdot 30)}{\max(5.0, \, \text{rainfall}_{30\text{d}} + (\text{soil\_moisture} \cdot 150))}$$
- **Seasonal Rainfall Deviation ($\Delta R$) $[-1.0, 2.0]$:** Captures precipitation anomalies relative to monsoonal expectations:
  $$\Delta R = \frac{\text{rainfall}_{\text{season}} - \max(50.0, \, 0.65 \cdot \text{rainfall}_{\text{annual}})}{\max(50.0, \, 0.65 \cdot \text{rainfall}_{\text{annual}})}$$

### 4.4 Forensic Data Quality Audit & Texture Violations
An audit of `data/agriculture/v1.0/dataset_validation_report.json` confirmed zero missing values and zero duplicate entries across 10,091 rows. However, a discrepancy was identified: while the summary status was marked `"PASS"`, the sub-check for `soil_ranges` failed with `texture_sum_violations: 4`. In physical soil science, particle fractions must strictly conserve mass: $\text{Sand} + \text{Silt} + \text{Clay} = 100\%$. In exactly four records, rounding artifacts during historical digitization produced sums between 98.8% and 101.4%. Representing only $0.039\%$ of the dataset, their impact on tree split thresholds is negligible, but transparent academic disclosure is maintained.

---

## 5. Model Methodology and Implementation

### 5.1 Distinct Mathematical Tasks
GeoAgri AI V5 explicitly disentangles four distinct tasks:
1. **Multiclass Probability ($P_{\text{ML}}$):** Normalized softmax likelihood $P(y = c \mid \mathbf{x})$ over 16 classes from an XGBoost classifier (`multi:softprob`).
2. **Biophysical Compatibility ($S_{\text{KB}}$):** Deterministic compliance score $[0, 1]$ from FAO EcoCrop rules.
3. **Yield Potential ($\hat{Y}$):** Continuous expected productivity in t/ha from a crop-conditional XGBoost regressor (`reg:squarederror`).
4. **Pareto Dominance ($\mathcal{P}^*$):** Non-dominated candidate frontier filtering across 5 orthogonal objectives.

### 5.2 Conventional ML vs Deep Learning / Computer Vision Boundary
In strict accordance with the academic focus of **CSE 471: Deep Learning for Computer Vision**, the boundary between conventional machine learning and deep learning in GeoAgri AI is clearly delineated:
- **Implemented Conventional ML:** The operational crop suitability engine, candidate ranker, yield regressor, and erosion model are **tabular gradient-boosted decision trees (XGBoost)**, not deep neural networks.
- **Implemented Deep Learning:** The natural language reasoning component utilizes an external deep Transformer LLM (Meta Llama-3.1-8b via Groq Cloud API).
- **Computer Vision (Roadmap / Future Work):** Satellite remote sensing ingestion (Copernicus Sentinel-2 multispectral bands at 10m–20m resolution) and deep canopy segmentation (DeepLabV3+ and vision transformers) are planned extensions for subsequent project phases (e.g., KPIT Sparkle 2027) and are **not operational** in the current tabular release.

### 5.3 Candidate-Level Learning-to-Rank Benchmark & Hypothesis Falsification
We evaluated candidate-level Learning-to-Rank (LambdaMART) against multiclass softmax on 161,456 candidate rows ($10,091 \text{ queries} \times 16 \text{ crops}$). Evaluated on the held-out chronological test partition (2019–2020, $N=546$):

| Model Formulation | Top-1 Acc | Top-3 Acc | Top-5 Acc | NDCG@3 | NDCG@5 | MRR | MAP@5 | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Multiclass Softmax (Base)** | **18.86%** | 42.12% | **59.34%** | **0.3198** | **0.3899** | **0.3705** | **0.3236** | **0.1618** |
| **Model B: Pointwise Binary Compatibility** | 16.48% | 39.56% | 55.13% | 0.2970 | 0.3608 | 0.3510 | 0.2984 | 0.0900 |
| **Model C: Pairwise LTR (`rank:pairwise`)** | 17.40% | 38.28% | 55.13% | 0.2924 | 0.3626 | 0.3528 | 0.3021 | 0.0852 |
| **Model D: Listwise LambdaMART (`rank:ndcg`)** | 16.85% | 37.73% | 55.13% | 0.2873 | 0.3580 | 0.3471 | 0.2951 | 0.1293 |
| **Model E1: Multiclass Fused (0.85 ML + 0.15 KB)** | **18.86%** | **42.31%** | 58.24% | 0.3200 | 0.3849 | 0.3689 | 0.3204 | 0.1614 |
| **Model E2: LambdaMART Fused (0.85 LTR + 0.15 KB)** | 14.47% | 37.18% | 51.65% | 0.2738 | 0.3327 | 0.3278 | 0.2725 | 0.1177 |

**Falsification Finding:** LambdaMART underperformed multiclass softmax across all ranking metrics. LambdaMART's listwise gradients suffer from variance on small, fixed candidate sets ($K=16$) with binary ground-truth, whereas the multiclass softmax partition function normalizes competing probabilities smoothly across all 16 classes simultaneously. Model E1 was certified as the production engine.

### 5.4 Dual-Layer Knowledge Fusion & Pareto Optimization
- **Evaluation Fusion (Model E1):** $\text{Score} = 0.85 \cdot P_{\text{ML}} + 0.15 \cdot S_{\text{KB}}$. Eliminates biophysical violations while stabilizing Top-3 accuracy (42.31%).
- **Runtime Domain Scoring:** Computes composite suitability via weighted sub-factors ($w_{\text{model}} = 0.45, w_{\text{soil}} = 0.20, w_{\text{season}} = 0.15, w_{\text{climate}} = 0.10, w_{\text{erosion}} = 0.10$).
- **Pareto Decision Engine:** Eliminates scalar collapse by computing the non-dominated candidate subset $\mathcal{P}^*$ across: $\max \text{Suitability}, \max \text{Yield}, \min \text{Water Risk}, \min \text{Erosion Risk}, \max \text{Decision Margin}$, projected through five farmer decision profiles (Suitability-Focused, Yield-Focused, Water-Constrained, Risk-Averse, Balanced).

---

## 6. Application and Visual Evidence

### 6.1 Dashboard Interface and Workflow
The Next.js 14 web application provides an intuitive geospatial interface for agricultural planning:
- **Interactive Coordinate Picking:** Users drop a pin on the Leaflet map to extract GPS coordinates and select the planned cropping season.
- **Geospatial Processing Feedback:** Visual status cards display real-time soil pH, texture, forecast rainfall, and slope gradients.
- **Multi-Objective Recommendation Slate:** Displays top-5 candidate crops, suitability percentages, expected yield ranges with statistical intervals, and risk badges.
- **Explainability Panel:** Displays TreeSHAP positive growth drivers, physiological limiting factors, and 5 directional counterfactual simulations.

### 6.2 Visual Verification Artifacts

```
┌────────────────────────────────────────────────────────────────────────┐
│                   VERIFIED VISUAL ARTIFACTS                            │
├───────────────────────────────┬────────────────────────────────────────┤
│ Visual Asset Path             │ Figure Description & Status            │
├───────────────────────────────┼────────────────────────────────────────┤
│ docs/report/figures/          │ Figure 1: Raw Confusion Matrix Heatmap │
│ confusion_matrix.png          │ (Tested on 546 held-out test queries)  │
├───────────────────────────────┼────────────────────────────────────────┤
│ docs/report/figures/          │ Figure 2: Row-Normalized Recall Heatmap│
│ confusion_matrix_normalized   │ (Visualizes per-class sensitivities)   │
├───────────────────────────────┼────────────────────────────────────────┤
│ docs/report/figures/          │ Figure 3: Per-Class Precision/Recall/F1│
│ per_class_performance.png     │ (Annotated with test sample support)   │
├───────────────────────────────┼────────────────────────────────────────┤
│ docs/report/figures/          │ Figure 4: Model Formulation Comparison │
│ model_comparison.png          │ (Multiclass vs Pointwise/Pairwise/LTR) │
├───────────────────────────────┼────────────────────────────────────────┤
│ docs/report/figures/          │ Figure 5: Top 15 XGBoost Features      │
│ feature_importance.png        │ (Ranked by average split gain)         │
├───────────────────────────────┼────────────────────────────────────────┤
│ docs/report/screenshots/      │ Figure 6: Static Preview of Next.js 14 │
│ dashboard-preview.png         │ Interactive GIS Dashboard UI           │
├───────────────────────────────┼────────────────────────────────────────┤
│ docs/report/screenshots/      │ Figure 7: Live Runtime Execution JSON  │
│ sample_prediction_response    │ (Verified execution on Lat 15.335,     │
│                               │  Lon 76.460, Kharif season)            │
└───────────────────────────────┴────────────────────────────────────────┘
```

---

## 7. Results and Experimental Evaluation

### 7.1 Primary Experimental Evaluation
Evaluating certified models on the 546 held-out chronological test queries (2019–2020) yielded the verified results below:

| Metric Category | Performance Metric | V4 Baseline | V5 Production | Observed Delta | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Classification Accuracy** | Top-1 Accuracy | 18.86% | **18.86%** | $0.00\%$ | **`TESTED`** |
| | Top-3 Accuracy | 40.48% | **42.31%** | $+1.83\%$ | **`TESTED`** |
| | Top-5 Accuracy | 59.34% | **58.24%** | $-1.10\%$ | **`TESTED`** |
| **Ranking Quality** | NDCG@5 | 0.3873 | **0.3849** | $-0.0024$ | **`TESTED`** |
| | Mean Reciprocal Rank (MRR) | 0.3672 | **0.3689** | $+0.0017$ | **`TESTED`** |
| **F1 Metrics** | Macro-Averaged F1 | 0.1643 | **0.1614** | $-0.0029$ | **`TESTED`** |
| | Weighted-Averaged F1 | 0.1620 | **0.1600** | $-0.0020$ | **`TESTED`** |
| **Yield Regression** | Coefficient of Determination ($R^2$)| 0.954 | **0.954** | $0.000$ | **`TESTED`** |
| | Median Absolute Error (MedAE) | 0.3645 t/ha | **0.3645 t/ha** | $0.000$ | **`TESTED`** |
| | Mean Absolute Percentage Error| 76.73% | **76.73%** | $0.00\%$ | **`TESTED`** |
| **Generalization** | 10-Fold Spatial CV Top-5 | 59.28% | **56.83%** | $-2.45\%$ | **`REPOSITORY-REPORTED`** |
| | Spatiotemporal Joint Holdout | 40.82% | **43.20%** | **$+2.38\%$** | **`REPOSITORY-REPORTED`** |

### 7.2 Confusion Matrix Audit and Per-Class Breakdown
The $16 \times 16$ confusion matrix evaluated on the 546 test samples ($N=546$) demonstrates significant variance in class-level predictability:

| Crop Class | Test Support ($n$) | Precision | Recall | F1-Score | Agronomic Performance Classification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Sugarcane** | 27 | 1.0000 | 0.7778 | **0.8750** | High Performance (Distinct annual high-water niche) |
| **Wheat** | 27 | 0.3158 | 0.6667 | **0.4286** | Moderate Performance (Distinct Rabi winter temperature window) |
| **Rice** | 63 | 0.7778 | 0.4444 | **0.5657** | Moderate Performance (Monsoonal rainfall alignment) |
| **Mustard** | 28 | 0.3333 | 0.3571 | **0.3448** | Moderate Performance (Rabi oilseed thermal alignment) |
| **Chickpea** | 25 | 0.7000 | 0.2800 | **0.4000** | Moderate Performance (Winter pulse moisture alignment) |
| **Urad** | 44 | 0.1234 | 0.8636 | **0.2159** | Over-Predicted Prior Sink ("Urad Attractor") |
| **Moong** | 44 | 1.0000 | 0.0682 | **0.1277** | Confused with ecological twin Urad (37 of 44 misclassified) |
| **Groundnut** | 40 | 1.0000 | 0.0250 | **0.0488** | Minority dilution under pulse over-prediction |
| **Cotton** | 24 | 0.0000 | 0.0000 | **0.0000** | Low-support minority; vertisol soil specificity diluted |
| **Soybean** | 22 | 0.0000 | 0.0000 | **0.0000** | Low-support minority; misclassified as central pulses |
| **Macro Average**| **546** | **0.4907** | **0.2536** | **0.1614** | Unweighted mean across 16 crops |

### 7.3 Critical Failure Mode Analysis: The "Urad Attractor"
The confusion matrix reveals that `Urad` was predicted **308 times out of 546 test queries (56.4%)**. Because Urad thrives across a broad range of central Indian soils and moderate rainfall bands, the unconstrained tree ensemble learned to predict Urad as a "safe" empirical prior whenever environmental features were near distribution medians. This caused severe false-positive dilution across specialized crops like Cotton and Soybean. Hybrid knowledge fusion ($0.85 \text{ML} + 0.15 \text{KB}$) penalizes generic central priors when specialized crops satisfy strict physiological criteria, restoring Cotton and Soybean into Top-5 candidate slates.

### 7.4 Deconstruction of the "MAPE Paradox" in Yield Prediction
The yield regressor reports an excellent $R^2 = 0.954$ and **Median Absolute Error = 0.3645 t/ha**, but an aggregate MAPE of 76.73%. This statistical paradox occurs because low-yielding dryland pulses (Moong, Sesamum, Urad) yield only $0.25 - 0.50$ t/ha. If true yield is $0.30$ t/ha and predicted yield is $0.65$ t/ha, the absolute error is an agronomically modest **0.35 t/ha**, but the percentage error explodes to $\frac{0.35}{0.30} \times 100\% = \mathbf{116.7\%}$. Because pulses constitute over 45% of the evaluation dataset, small denominators inflate aggregate MAPE, while Median Absolute Error confirms high physical precision.

---

## 8. Discussion, Limitations, and Future Work

### 8.1 Empirical Findings and Agronomic Interpretation
The empirical results confirm that single-label Top-1 accuracy (18.86%) is an inappropriate metric for agricultural decision support. Land is biophysically capable of supporting multiple crops simultaneously; a fertile alluvial plot during Rabi season can successfully support Wheat, Mustard, or Chickpea. The operational value of GeoAgri AI lies in candidate screening: achieving **42.31% Top-3** and **58.24% Top-5 accuracy**, presenting farmers with an ecologically viable slate filtered by physiological thresholds and multi-objective trade-offs.

### 8.2 Agile Lessons & Technical Debt Post-Mortem
A key project management takeaway for course **CSE 471** is the avoidance of the sunk-cost fallacy. Sizable engineering effort was invested in candidate-level LTR dataset construction (161,456 rows). When empirical testing falsified the hypothesis that LambdaMART was superior, the team made an evidence-driven decision: LambdaMART was rejected for production, and Model E1 was certified, preventing an architectural regression. Furthermore, a technical debt audit identified that legacy computer vision imports in `predict_bridge.py` caused silent worker subprocess crashes in environments lacking PyTorch, underscoring the necessity of modular dependency decoupling.

### 8.3 Limitations and Explicit Future Roadmap
1. **Current Limitations:** Ground-truth labels reflect historical district cultivation rather than farm-scale physiological optimality. Collinear environmental variables disperse Shapley attributions across correlated groups.
2. **Roadmap Computer Vision Extensions (KPIT Sparkle 2027):**
   - **Sentinel-2 Remote Sensing Ingestion:** Real-time retrieval of Copernicus Sentinel-2 multispectral bands (10m–20m resolution) for dynamic NDVI, EVI, and NDRE vegetation index computation.
   - **DeepLabV3+ Canopy Segmentation:** Integration of deep convolutional and vision transformer models for automated field boundary detection and canopy vigor zonation.
   - **Volumetric Irrigation Scheduling:** Coupling remote-sensing evapotranspiration with FAO-56 root-zone water balance modeling.

---

## 9. Conclusion

This project submission report evaluated the software engineering architecture, machine learning methodology, and empirical performance of **GeoAgri AI (Crop Intelligence V5)** for **CSE 471: Deep Learning for Computer Vision**. The investigation demonstrated that candidate-level Learning-to-Rank (LambdaMART) underperformed multiclass softmax for small, fixed candidate crop sets, while hybrid knowledge fusion (0.85 ML + 0.15 KB) and multi-objective Pareto optimization successfully eliminated biophysical violations and scalar objective collapse. Certified across 75 automated regression tests (100% pass rate) with documented spatiotemporal robustness, GeoAgri AI establishes an empirical baseline for transparent, accountable agricultural artificial intelligence.

---

## References

[1] R. Rajak, A. Pawar, M. Pendke, P. Shinde, and S. Rathod, "Crop recommendation system to maximize crop yield using machine learning technique," in *Proc. Int. Conf. Electron. Commun. Aerosp. Technol. (ICECA)*, Coimbatore, India, 2017, pp. 950–953. DOI: `10.1109/ICECA.2017.8212856`.

[2] H. K. Kondaveeti, N. Kumaravelu, S. D. Vanambathina, and P. S. R. Chandra, "A systematic literature review on agricultural decision support systems: Advancements in crop recommendation and yield prediction," *Comput. Electron. Agric.*, vol. 206, p. 107677, Mar. 2023. DOI: `10.1016/j.compag.2023.107677`.

[3] H. Meyer and E. Pebesma, "Predicting into unknown space? Estimating the area of applicability of spatial prediction models," *Methods Ecol. Evol.*, vol. 12, no. 9, pp. 1620–1633, Sep. 2021. DOI: `10.1111/2041-210X.13650`.

[4] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD '16)*, San Francisco, CA, USA, 2016, pp. 785–794. DOI: `10.1145/2939672.2939785`.

[5] C. J. C. Burges, "From RankNet to LambdaRank to LambdaMART: An overview," Microsoft Research, Redmond, WA, USA, Tech. Rep. MSR-TR-2010-82, Jun. 2010.

[6] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Adv. Neural Inf. Process. Syst. 30 (NeurIPS 2017)*, Long Beach, CA, USA, 2017, pp. 4765–4774.

[7] Directorate of Economics and Statistics (DES), "Agricultural Statistics at a Glance 2022," Ministry of Agriculture and Farmers Welfare, Government of India, New Delhi, India, Comprehensive Statistical Report, 2022.

[8] R. G. Allen, L. S. Pereira, D. Raes, and M. Smith, "Crop evapotranspiration: Guidelines for computing crop water requirements," Food and Agriculture Organization of the United Nations (FAO), Rome, Italy, FAO Irrigation and Drainage Paper 56, 1998. ISBN: `92-5-104219-5`.
