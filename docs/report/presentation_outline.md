# Academic Presentation Outline: GeoAgri AI (Crop Intelligence V5)

**Course Assessment:** INT373 — Agile Driven Development and Project Management  
**Assessment Format:** University Continuous Assessment (CA) Presentation & Defense  
**Allotted Time:** 15 Minutes (10 min presentation + 5 min Q&A/Viva)  
**Deliverable File:** `docs/report/presentation_outline.md`

---

## Slide 1: Title & Course Context (0:00 – 1:00)
- **Title:** GeoAgri AI — Crop Intelligence V5: Robust Multi-Objective Agronomic Decision Intelligence & Empirical Learning-to-Rank Audit
- **Subtitle:** An Empirical Software Engineering & Machine Learning Assessment
- **Student Details:** [Student Name] · [Registration / Roll Number]
- **Course & Faculty:** INT373: Agile Driven Development and Project Management · Department of Computer Science & Engineering
- **Key Message:** This assessment presents an empirical, evidence-grounded evaluation of the actual implemented codebase. It is an honest academic critique of an engineering system, explicitly separating operational capabilities from future roadmap aspirations.

---

## Slide 2: Agricultural Problem Statement & Motivation (1:00 – 2:00)
- **The Core Problem:** Smallholder and commercial farmers make crop selection decisions under high climatic, pedological, and market uncertainty.
- **The Failure of Conventional Tools:**
  1. Static zoning tables cannot account for dynamic seasonal rainfall deficits.
  2. Pure machine learning models predict based on historical correlation, recommending biologically unviable crops in extreme terrain.
  3. Single-scalar recommendation models conceal critical trade-offs between crop suitability, water depletion, and land degradation risks.
- **Visual:** Map of India displaying the 15 heterogeneous agro-climatic zones and rainfall variance.

---

## Slide 3: Evidence-Backed Research Gap (2:00 – 3:30)
- **Gap 1 (Empirical vs Biophysical):** Pure ML models lack physiological guardrails; pure rule engines lack multi-nutrient interaction modeling.
- **Gap 2 (The Ranking Hypothesis):** Does candidate-level Learning-to-Rank (LambdaMART) outperform standard multiclass softmax on a fixed 16-crop slate?
- **Gap 3 (The Pareto Gap):** How can agricultural decision support expose orthogonal multi-objective trade-offs (Suitability vs Yield vs Water vs Erosion) without uncalibrated scalar collapse?
- **Gap 4 (Spatiotemporal Transferability):** Standard random $k$-fold cross-validation inflates accuracy by leaking spatial autocorrelations across neighboring agricultural plots.
- **Visual:** Research Gap Venn Diagram contrasting pure empirical ML, rigid expert heuristics, and GeoAgri AI's hybrid Pareto architecture.

---

## Slide 4: End-to-End System Architecture (3:30 – 5:00)
- **Multimodal Geospatial Ingestion:** Live REST interfaces to ISRIC SoilGrids (0–30 cm soil chemistry), Open-Meteo (10-year climatology + 7-day NWP forecasts), and Open-Elevation (DEM topography & slope).
- **26-Feature Preprocessing Pipeline:** Formulation of ICAR-benchmarked Soil Fertility Index (SFI), FAO-56 Water Stress Index (WSI), and Seasonal Rainfall Deviation ($\Delta R$).
- **Core Decoupled Stack:** Next.js 14 interactive GIS frontend (Leaflet) paired with an asynchronous FastAPI serving tier and an XGBoost CUDA-accelerated inference backend.
- **Visual:** System Architecture Diagram (Figure from `docs/report/architecture.md`).

---

## Slide 5: The Empirical Falsification of LambdaMART (5:00 – 6:30)
- **The Central Engineering Experiment:** Evaluated candidate-level Learning-to-Rank on 161,456 candidate instances ($10,091 \times 16$ crops) grouped atomically.
- **Hypothesis Tested:** LambdaMART (`rank:ndcg`) would superiorly prioritize crop slates compared to Multiclass Softmax.
- **The Empirical Result:** **Falsified**.
  - Multiclass Softmax: Top-1 = 18.86%, Top-5 = 59.34%, NDCG@5 = 0.3899, MRR = 0.3705.
  - LambdaMART Ranker: Top-1 = 16.85%, Top-5 = 55.13%, NDCG@5 = 0.3580, MRR = 0.3471.
- **Agile Project Management Takeaway:** Data-driven falsification prevented deploying an inferior, complex ranking model to production. Multiclass Fusion (Model E1) was certified as the production core.
- **Visual:** Bar chart comparison from `docs/report/figures/model_comparison.png`.

---

## Slide 6: Multi-Objective Pareto Decision Engine (6:30 – 7:30)
- **Eliminating Scalar Collapse:** Rather than blending yield and water risk into a single scalar number, V5 computes the 5-dimensional non-dominated Pareto frontier:
  $$\max \text{Suitability}, \quad \max \text{Yield}, \quad \min \text{Water Risk}, \quad \min \text{Erosion Risk}, \quad \max \text{Decision Margin}$$
- **Supported Farmer Decision Profiles:**
  1. Suitability-Focused (Default)
  2. Yield-Focused (Commercial / Irrigated)
  3. Water-Constrained (Drought-prone drylands)
  4. Risk-Averse (Steep topography & erosion control)
  5. Balanced
- **Visual:** 2D projection of the Pareto frontier showing Cotton vs Sesamum vs Sorghum trade-offs.

---

## Slide 7: Evaluation, Confusion Matrix & The "Urad Attractor" (7:30 – 8:30)
- **Evaluation Dataset:** 546 strictly held-out future queries (2019–2020 chronological holdout).
- **Primary Metrics:** Top-1 Acc = 18.86%, Top-3 Acc = 42.31%, Top-5 Acc = 58.24%, Macro F1 = 0.1614.
- **Failure Mode Discovery ("Urad Attractor"):** Standalone XGBoost over-predicted Urad (308 of 546 predictions) as a central broad prior, starving Cotton and Soybean.
- **Knowledge Fusion Impact:** 0.85 ML + 0.15 KB fusion suppressed false pulse predictions and elevated specialized crops.
- **Visual:** Heatmaps from `docs/report/figures/confusion_matrix.png` and `confusion_matrix_normalized.png`.

---

## Slide 8: Yield Regression, The "MAPE Paradox" & Explainability (8:30 – 9:30)
- **Yield Potential Regressor:** Conditioned test set $R^2 = 0.954$, Median Absolute Error = **0.3645 t/ha**.
- **The "MAPE Paradox" Explained:** Aggregate MAPE is 76.73% because dryland pulses yield only $0.3 - 0.5$ t/ha. A tiny $0.35$ t/ha absolute error yields an explosive $> 100\%$ percentage error. MedAE is the true agronomic metric.
- **Explainability:** Exact TreeSHAP attributions identify top drivers. Directional counterfactuals simulate $+30\%$ rain or pH shifts. Groq Llama-3.1 synthesizes plain-language summaries with deterministic local fallbacks.
- **Visual:** Feature importance chart from `docs/report/figures/feature_importance.png`.

---

## Slide 9: Agile Quality Assurance & Technical Debt Post-Mortem (9:30 – 10:00)
- **Automated Testing Suite:** 75/75 automated pytest tests passed (100% pass rate).
- **Technical Debt Disclosure:**
  - Audit revealed `predict_bridge.py` eagerly imported `cv2` and `torch` from a legacy archaeological project, causing silent worker failure when deep vision libraries were missing.
  - Transparently disclosed that satellite computer vision and automated irrigation are roadmap targets, not operational code.
- **Conclusion:** GeoAgri AI V5 demonstrates rigorous Agile project management by establishing verified baselines, testing hypotheses honestly, and providing explainable multi-objective agronomic decision intelligence.
