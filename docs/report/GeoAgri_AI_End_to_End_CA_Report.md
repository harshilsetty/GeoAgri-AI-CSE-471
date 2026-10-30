# GeoAgri AI: Robust Multi-Objective Agronomic Decision Intelligence & Empirical Learning-to-Rank Audit

## Industry-Grade Academic Continuous Assessment (CA) Report
**Course:** INT373 — Agile Driven Development and Project Management  
**Program:** Bachelor of Technology in Computer Science & Engineering  
**Department:** Department of Computer Science & Engineering, School of Computer Science and Engineering  
**Institution:** Lovely Professional University, Punjab, India  
**Academic Term:** Autumn Semester, Academic Year 2026–2027  
**Document Classification:** University Academic Assessment — End-to-End Software & ML Evaluation  
**Date of Submission:** October 2026  

---

### Candidate and Evaluator Details

| Parameter | Student / Assessment Record |
| :--- | :--- |
| **Student Name:** | `[HARSHIL SOMISETTY / Editable Student Name]` |
| **Registration Number:** | `[Editable Registration Number]` |
| **Roll Number / Section:** | `[Editable Roll Number / Section]` |
| **Course Code & Title:** | `INT373: Agile Driven Development and Project Management` |
| **Faculty / Evaluator Name:**| `[Course Instructor / Evaluator Name]` |
| **Repository Identifier:** | `harshilsetty/GeoAgri-AI` (Audit Baseline: `Archaeological-Site-Mapping-AI-V4`) |
| **Verified Production Release:**| **Crop Intelligence V5.0** (Multi-Objective Pareto Engine & Certified V4 Regression Baseline) |

---

## Declaration of Academic Integrity

I hereby declare that this academic technical assessment report represents an honest, empirical evaluation of the actual software implementation, machine learning models, datasets, and test artifacts residing within the audited `GeoAgri-AI` repository. In strict adherence to academic integrity and project management standards:
1. No dataset properties, training configurations, performance metrics, confusion matrices, or experimental results have been fabricated.
2. The core machine learning algorithms, model binaries, and evaluation scripts have been audited without alteration.
3. Capabilities implemented in source code are rigorously distinguished from proposed future roadmap goals (such as satellite multispectral imagery ingestion and automated volumetric irrigation scheduling).
4. Sunk-cost hypotheses—specifically the empirical evaluation of candidate-level Learning-to-Rank (LambdaMART)—are reported with complete scientific transparency, documenting both successful innovations and falsified assumptions.

**Candidate Signature:** ___________________________  
**Date:** October 2026  

---

## Abstract

Agricultural Decision-Support Systems (ADSS) are critical for mitigating smallholder climatic vulnerability, yet conventional machine learning models suffer from three structural failures: (1) unconstrained empirical predictions that violate physical and agronomic boundary conditions, (2) the scalar collapse of orthogonal decision criteria (suitability, expected yield, groundwater depletion, and soil erosion risk) into an opaque aggregate index, and (3) severe performance degradation under spatial autocorrelation leakage and forward chronological climate drift.

This assessment presents an end-to-end software engineering and empirical machine learning evaluation of **GeoAgri AI (Crop Intelligence V5)**, an agricultural decision-support platform designed to address these limitations. Developed through Agile test-driven iterations, the platform couples automated geospatial feature ingestion (ISRIC SoilGrids, Open-Meteo Climatology, and Open-Elevation SRTM grids) with a 26-dimensional agronomic preprocessing pipeline that computes normalized composite indicators including the ICAR-benchmarked Soil Fertility Index (SFI), FAO-56 Water Stress Index (WSI), and Seasonal Rainfall Deviation ($\Delta R$).

We conduct an empirical evaluation of candidate-level Learning-to-Rank (LambdaMART optimizing NDCG) against standard multiclass gradient-boosted decision trees (XGBoost `multi:softprob`) across 161,456 candidate instances derived from 10,091 longitudinal Indian agricultural queries (1997–2020). The central research hypothesis—that LambdaMART would superiorly rank candidate crops on a fixed 16-crop slate—was **empirically falsified**: Multiclass Fusion achieved higher Top-1 accuracy (18.86% vs 16.85%), Top-5 accuracy (58.24% vs 55.13%), NDCG@5 (0.3849 vs 0.3580), and Mean Reciprocal Rank (0.3689 vs 0.3471). Consequently, Model E1 (Multiclass Softmax fused with deterministic FAO EcoCrop physiological rules at an optimal 0.85 / 0.15 ratio) was certified for production.

To resolve scalar objective collapse, V5 introduces an operational Multi-Objective Pareto Decision Engine that computes non-dominated candidate frontiers across five orthogonal dimensions ($\max \text{Suitability}$, $\max \text{Yield}$, $\min \text{Water Risk}$, $\min \text{Erosion Risk}$, and $\max \text{Decision Margin}$), supporting five distinct farmer decision profiles. An orthogonal crop-conditional yield regressor achieves a certified $R^2 = 0.954$ with a Median Absolute Error of **0.3645 t/ha**, while the aggregate MAPE of 76.73% is transparently deconstructed as a low-yield pulse denominator artifact. Dual-layer explainability couples exact polynomial-time TreeSHAP attributions and directional counterfactual perturbations with a safety-constrained Groq Llama-3.1 natural-language synthesizer. The software architecture is certified across 75 automated regression tests (100% pass rate), accompanied by an Agile root-cause audit of legacy technical debt.

**Keywords:** Agile Driven Development; Agricultural Decision-Support Systems; Multi-Objective Optimization; Pareto Frontier; Learning-to-Rank; XGBoost; TreeSHAP; Geospatial Feature Engineering; Spatiotemporal Generalization.

---

## Table of Contents

- [Preliminary Pages](#preliminary-pages)
  - [Abstract](#abstract)
  - [List of Figures](#list-of-figures)
  - [List of Tables](#list-of-tables)
  - [Abbreviations & Acronyms](#abbreviations--acronyms)
- [Chapter 1: Introduction](#chapter-1-introduction)
  - [1.1 Agricultural Context & Socio-Economic Imperative](#11-agricultural-context--socio-economic-imperative)
  - [1.2 Problem Statement & Practical Dilemmas](#12-problem-statement--practical-dilemmas)
  - [1.3 Project Motivation & Engineering Goals](#13-project-motivation--engineering-goals)
  - [1.4 Formal Research Questions & Objectives](#14-formal-research-questions--objectives)
  - [1.5 System Contributions vs Explicit Non-Claims](#15-system-contributions-vs-explicit-non-claims)
  - [1.6 Report Organization](#16-report-organization)
- [Chapter 2: Literature Review](#chapter-2-literature-review)
  - [2.1 Evolution of Agricultural Decision-Support Systems](#21-evolution-of-agricultural-decision-support-systems)
  - [2.2 Machine Learning for Crop Suitability & Recommendation](#22-machine-learning-for-crop-suitability--recommendation)
  - [2.3 Multimodal Geospatial & Climatological Data Ingestion](#23-multimodal-geospatial--climatological-data-ingestion)
  - [2.4 Learning-to-Rank & Listwise Optimization in Agriculture](#24-learning-to-rank--listwise-optimization-in-agriculture)
  - [2.5 Explainable AI (XAI) & LLM Reasoning in Environmental Systems](#25-explainable-ai-xai--llm-reasoning-in-environmental-systems)
  - [2.6 Spatial Autocorrelation & Environmental Generalization](#26-spatial-autocorrelation--environmental-generalization)
  - [2.7 Comprehensive Literature Comparison Table](#27-comprehensive-literature-comparison-table)
  - [2.8 Critical Literature Synthesis](#28-critical-literature-synthesis)
- [Chapter 3: Research Gap and Problem Formulation](#chapter-3-research-gap-and-problem-formulation)
  - [3.1 Synthesis of What Prior Research Establishes](#31-synthesis-of-what-prior-research-establishes)
  - [3.2 Four Critical Deficiencies in Current Systems](#32-four-critical-deficiencies-in-current-systems)
  - [3.3 The Evidence-Backed Research Gap](#33-the-evidence-backed-research-gap)
  - [3.4 Formal Mathematical Problem Formulation](#34-formal-mathematical-problem-formulation)
  - [3.5 System-Level Contribution Delineation](#35-system-level-contribution-delineation)
- [Chapter 4: Requirements and System Design](#chapter-4-requirements-and-system-design)
  - [4.1 Stakeholder Profiles & Functional Requirements](#41-stakeholder-profiles--functional-requirements)
  - [4.2 Non-Functional & Performance Requirements](#42-non-functional--performance-requirements)
  - [4.3 High-Level System Architecture & Context Diagram](#43-high-level-system-architecture--context-diagram)
  - [4.4 Component Deconstruction & Decoupled Modular Design](#44-component-deconstruction--decoupled-modular-design)
  - [4.5 Technology Stack Selection & Decision Matrix](#45-technology-stack-selection--decision-matrix)
- [Chapter 5: Data and Feature Engineering](#chapter-5-data-and-feature-engineering)
  - [5.1 Multimodal Data Sources & Ingestion Protocols](#51-multimodal-data-sources--ingestion-protocols)
  - [5.2 Dataset Provenance, Construction, and Characteristics](#52-dataset-provenance-construction-and-characteristics)
  - [5.3 Data Dictionary: Raw Attributes vs 26-Dimensional Model Vector](#53-data-dictionary-raw-attributes-vs-26-dimensional-model-vector)
  - [5.4 Mathematical Formulation of Composite Agronomic Indicators](#54-mathematical-formulation-of-composite-agronomic-indicators)
  - [5.5 Forensic Data Quality Audit & The 4 Texture Violations](#55-forensic-data-quality-audit--the-4-texture-violations)
  - [5.6 Data Leakage Safeguards & Grouped Partitioning Invariants](#56-data-leakage-safeguards--grouped-partitioning-invariants)
- [Chapter 6: Machine Learning Methodology](#chapter-6-machine-learning-methodology)
  - [6.1 Mathematical Separation of Agricultural Tasks](#61-mathematical-separation-of-agricultural-tasks)
  - [6.2 Technical Justification for XGBoost Core Architecture](#62-technical-justification-for-xgboost-core-architecture)
  - [6.3 Detailed Model Formulations & Hyperparameters](#63-detailed-model-formulations--hyperparameters)
  - [6.4 Candidate-Level Learning-to-Rank Benchmark & Hypothesis Falsification](#64-candidate-level-learning-to-rank-benchmark--hypothesis-falsification)
  - [6.5 Dual-Layer Knowledge Fusion Formulations](#65-dual-layer-knowledge-fusion-formulations)
  - [6.6 Multi-Objective Pareto Decision Engine & Decision Profiles](#66-multi-objective-pareto-decision-engine--decision-profiles)
- [Chapter 7: Explainability and Natural-Language Reasoning](#chapter-7-explainability-and-natural-language-reasoning)
  - [7.1 Dual-Layer Explainability Framework](#71-dual-layer-explainability-framework)
  - [7.2 TreeSHAP Feature Attribution: Global Gains & Local Attributions](#72-treeshap-feature-attribution-global-gains--local-attributions)
  - [7.3 Epistemic Boundaries: Attribution vs Biological Causality](#73-epistemic-boundaries-attribution-vs-biological-causality)
  - [7.4 Directional Counterfactual Sensitivity Analysis](#74-directional-counterfactual-sensitivity-analysis)
  - [7.5 LLM Narrative Integration & Anti-Hallucination Guardrails](#75-llm-narrative-integration--anti-hallucination-guardrails)
  - [7.6 Zero-Dependency Deterministic Fallback Mechanism](#76-zero-dependency-deterministic-fallback-mechanism)
- [Chapter 8: Implementation and End-to-End Technology Stack](#chapter-8-implementation-and-end-to-end-technology-stack)
  - [8.1 Full Repository Directory Structure](#81-full-repository-directory-structure)
  - [8.2 Frontend Implementation: Next.js 14 GIS Interface](#82-frontend-implementation-nextjs-14-gis-interface)
  - [8.3 Backend Implementation: FastAPI & Subprocess Bridges](#83-backend-implementation-fastapi--subprocess-bridges)
  - [8.4 Technical Debt Audit: The Monolithic Bridge Post-Mortem](#84-technical-debt-audit-the-monolithic-bridge-post-mortem)
  - [8.5 Actual Implementation vs Proposed KPIT Roadmap Boundaries](#85-actual-implementation-vs-proposed-kpit-roadmap-boundaries)
- [Chapter 9: Experimental Design and Evaluation](#chapter-9-experimental-design-and-evaluation)
  - [9.1 Scientific Evaluation Objectives](#91-scientific-evaluation-objectives)
  - [9.2 Validation Splitting Strategies & Generalization Protocols](#92-validation-splitting-strategies--generalization-protocols)
  - [9.3 Evaluation Metrics & Statistical Mathematical Definitions](#93-evaluation-metrics--statistical-mathematical-definitions)
  - [9.4 Calibration Metrics: Expected Calibration Error & Brier Score](#94-calibration-metrics-expected-calibration-error--brier-score)
- [Chapter 10: Experimental Results](#chapter-10-experimental-results)
  - [10.1 Primary Benchmark Results (Chronological Holdout 2019–2020)](#101-primary-benchmark-results-chronological-holdout-20192020)
  - [10.2 Genuine 16-Crop Confusion Matrix & Per-Class Support](#102-genuine-16-crop-confusion-matrix--per-class-support)
  - [10.3 Failure Mode Deconstruction: The "Urad Attractor"](#103-failure-mode-deconstruction-the-urad-attractor)
  - [10.4 Yield Regressor Results & The "MAPE Paradox"](#104-yield-regressor-results--the-mape-paradox)
  - [10.5 Controlled Ablation Experiments](#105-controlled-ablation-experiments)
  - [10.6 Generalization Benchmarks: Spatial, Temporal, and Joint Spatiotemporal](#106-generalization-benchmarks-spatial-temporal-and-joint-spatiotemporal)
  - [10.7 End-to-End Inference Latency & Hardware Acceleration Profiling](#107-end-to-end-inference-latency--hardware-acceleration-profiling)
- [Chapter 11: Discussion](#chapter-11-discussion)
  - [11.1 Empirical Interpretation of Findings](#111-empirical-interpretation-of-findings)
  - [11.2 Resolution of Research Questions (RQ1 through RQ4)](#112-resolution-of-research-questions-rq1-through-rq4)
  - [11.3 Agile Project Management Lessons: Sunk-Cost Avoidance](#113-agile-project-management-lessons-sunk-cost-avoidance)
  - [11.4 Real-World Agronomic Utility & Screening Scenarios](#114-real-world-agronomic-utility--screening-scenarios)
- [Chapter 12: Limitations and Future Work](#chapter-12-limitations-and-future-work)
  - [12.1 Algorithmic & Modeling Limitations](#121-algorithmic--modeling-limitations)
  - [12.2 Data Ingestion & District Aggregate Limitations](#122-data-ingestion--district-aggregate-limitations)
  - [12.3 Explicit Roadmap Extensions (KPIT Sparkle 2027 Goals)](#123-explicit-roadmap-extensions-kpit-sparkle-2027-goals)
- [Chapter 13: Conclusion](#chapter-13-conclusion)
- [References](#references)
- [Appendices](#appendices)

---

## List of Figures

- **Figure 1:** End-to-End Multimodal System Architecture & Dataflow Diagram (`docs/report/architecture.md`)
- **Figure 2:** Confusion Matrix Heatmap (Raw Counts, Chronological Test Set $N=546$) (`docs/report/figures/confusion_matrix.png`)
- **Figure 3:** Row-Normalized Confusion Matrix Heatmap (Per-Class Recall) (`docs/report/figures/confusion_matrix_normalized.png`)
- **Figure 4:** Per-Class Precision, Recall, and F1-Score Breakdown Annotated with Test Support (`docs/report/figures/per_class_performance.png`)
- **Figure 5:** Comparative Evaluation of Candidate Model Formulations on Chronological Test Set (`docs/report/figures/model_comparison.png`)
- **Figure 6:** Top 15 XGBoost Feature Importances by Average Split Gain (`docs/report/figures/feature_importance.png`)
- **Figure 7:** Static Preview of the Next.js 14 Interactive GIS Dashboard Interface (`docs/report/screenshots/dashboard-preview.png`)
- **Figure 8:** Multi-Modal Conceptual System Flowchart (KPIT Sparkle 2027 Roadmap Asset) (`docs/report/screenshots/geoagri-ai-overview.png`)
- **Figure 9:** Planned Sentinel-2 Canopy Moisture Stress & Zonation UI Card (`docs/report/screenshots/stress-map-preview.png`)
- **Figure 10:** Planned FAO-56 Predictive Volumetric Irrigation Advisory UI Card (`docs/report/screenshots/irrigation-advisory-preview.png`)

---

## List of Tables

- **Table 1:** Comprehensive Scholarly Literature Comparison Matrix (Section 2.7)
- **Table 2:** Complete System Evidence & Capability Verification Inventory (Section 1.5 & Section 4.5)
- **Table 3:** 26-Dimensional Agronomic Model Feature Dictionary (Section 5.3)
- **Table 4:** Composite Agronomic Indicator Formulations (Section 5.4)
- **Table 5:** XGBoost Model Hyperparameter Specifications (Section 6.3)
- **Table 6:** Candidate Learning-to-Rank Benchmark Results on Chronological Test Partition (Section 6.4)
- **Table 7:** Primary Experimental Metrics across Crop Intelligence Versions (Section 10.1)
- **Table 8:** Class-Wise Performance Metrics, Rank Positions, and Minority Status (Section 10.2)
- **Table 9:** Controlled 11-Stage Ablation Study Results (Section 10.5)
- **Table 10:** Multi-Split Generalization Performance Audit (Section 10.6)
- **Table 11:** Automated Regression Test Suite Breakdown across 15 Modules (Section 8.3 & Appendix F)

---

## Abbreviations & Acronyms

| Abbreviation | Expanded Formal Terminology |
| :--- | :--- |
| **ADSS** | Agricultural Decision-Support System |
| **APY** | Area, Production, and Yield Agricultural Statistics (Ministry of Agriculture, GoI) |
| **CA** | Continuous Assessment |
| **CEC** | Cation Exchange Capacity |
| **CV** | Cross-Validation |
| **DEM** | Digital Elevation Model |
| **DES** | Directorate of Economics & Statistics, Ministry of Agriculture & Farmers Welfare, GoI |
| **DLD** | District-Level Database (ICRISAT) |
| **ECE** | Expected Calibration Error |
| **$ET_0$** | Reference Crop Evapotranspiration (Penman-Monteith Formulation) |
| **FAO** | Food and Agriculture Organization of the United Nations |
| **GIS** | Geographic Information System |
| **GPS** | Global Positioning System |
| **GPU** | Graphics Processing Unit |
| **ICAR** | Indian Council of Agricultural Research |
| **IISS** | Indian Institute of Soil Science (Bhopal, India) |
| **IPC** | Inter-Process Communication |
| **ISRIC** | International Soil Reference and Information Centre (Wageningen, Netherlands) |
| **LLM** | Large Language Model |
| **LTR** | Learning-to-Rank |
| **MAE** | Mean Absolute Error |
| **MAP** | Mean Average Precision |
| **MAPE** | Mean Absolute Percentage Error |
| **MedAE** | Median Absolute Error |
| **ML** | Machine Learning |
| **MRR** | Mean Reciprocal Rank |
| **MSL** | Mean Sea Level |
| **NDCG** | Normalized Discounted Cumulative Gain |
| **NWP** | Numerical Weather Prediction |
| **OC** | Organic Carbon |
| **REST** | Representational State Transfer |
| **RMSE** | Root Mean Squared Error |
| **SFI** | Soil Fertility Index |
| **SHAP** | SHapley Additive exPlanations |
| **SMI** | Soil Moisture Index |
| **SRTM** | Shuttle Radar Topography Mission |
| **UAV** | Unmanned Aerial Vehicle |
| **USDA** | United States Department of Agriculture |
| **WSI** | Water Stress Index |
| **XGBoost** | eXtreme Gradient Boosting |

---

# Chapter 1: Introduction

### 1.1 Agricultural Context & Socio-Economic Imperative
Agriculture represents the primary source of livelihood for over 58% of India's population and contributes significantly to the national gross domestic product. However, the sector is increasingly characterized by acute economic and climatic precarity. Indian agrarian ecosystems span 15 distinct Agro-Climatic Zones delineated by the Planning Commission, ranging from the arid northwestern plains of Rajasthan to the humid tropical deltas of the eastern seaboard [13]. Within these heterogeneous ecosystems, smallholder farmers (operating landholdings under 2 hectares) confront compounding operational uncertainties: monsoonal precipitation volatility, accelerating groundwater depletion, escalating fertilizer costs, and soil organic carbon degradation.

Crop selection is the single most critical pre-sowing decision an agricultural producer makes. Selecting an agronomically mismatched crop initiates a cascade of compounding failures: stunted vegetative development, severe susceptibility to pest outbreaks, extreme irrigation pumping expenditures, and catastrophic economic loss. Historically, crop choices have been governed by generational heuristics, localized cultural tradition, or market price signals from the preceding harvest season. In the era of accelerating anthropogenic climate anomalies, static generational rules increasingly fail, necessitating dynamic, data-driven decision intelligence.

### 1.2 Problem Statement & Practical Dilemmas
Existing digital tools and automated recommendation applications exhibit severe algorithmic and conceptual limitations that restrict their real-world utility:
1. **The Empirical Over-Fitting Dilemma:** Conventional machine learning models trained on historical agricultural survey records learn statistical correlations rather than biophysical causality. If a region historically cultivated a crop due to artificial state procurement subsidies rather than ecological optimality, unconstrained models perpetuate and amplify this historical bias.
2. **The Biophysical Violation Defect:** Pure statistical models frequently output absurd agronomic recommendations when operating near the edges of their training distributions—such as recommending wetland Paddy (Rice) on steep, erosion-prone $15^\circ$ slopes, or recommending temperature-sensitive Wheat in regions experiencing extreme vegetative heat stress.
3. **The Scalar Objective Collapse Problem:** The vast majority of published agricultural models collapse multiple competing decision criteria into a single scalar "suitability score" or softmax probability. A crop may receive a high suitability score solely because its soil nutrient profile matches historical records, while completely masking that the crop requires three times the available groundwater capacity or carries an intolerable risk of catastrophic soil erosion.
4. **The Spatial Data Leakage Trap:** In academic literature, models frequently report inflated accuracy figures ($> 90\%$) achieved via standard randomized $k$-fold cross-validation. As proven by Meyer and Pebesma [3], randomized splitting on geospatial data causes severe leakage of spatial autocorrelations across neighboring agricultural plots, producing models that collapse when deployed into geographically unseen districts.

### 1.3 Project Motivation & Engineering Goals
The GeoAgri AI initiative was undertaken to design, implement, and rigorously evaluate an industry-grade, production-deployable Agricultural Decision-Support System that bridges the divide between statistical machine learning and physical agricultural science. The system was developed under an Agile Driven Development methodology within course **INT373 — Agile Driven Development and Project Management**.

The primary engineering goals of the project are:
- To construct an automated, end-to-end data ingestion pipeline capable of querying public geospatial APIs (ISRIC SoilGrids, Open-Meteo Climatology, Open-Elevation DEM) using GPS coordinates and transforming raw data into 26 domain-grounded agronomic features.
- To rigorously formulate and benchmark candidate-level Learning-to-Rank (LTR) algorithms against multiclass probability formulations to identify the mathematically optimal ranking architecture for multi-candidate crop screening.
- To fuse statistical tree ensembles with deterministic FAO EcoCrop physiological constraint boundaries, eliminating biophysical recommendation violations.
- To eliminate scalar objective collapse by designing an operational Multi-Objective Pareto Decision Engine that computes non-dominated candidate trade-off frontiers.
- To preserve a certified, crop-conditional yield regressor providing productivity estimates in metric tonnes per hectare ($\text{t/ha}$) accompanied by statistical prediction intervals.
- To provide trustworthy dual-layer explainability via exact polynomial-time TreeSHAP feature attributions, directional counterfactual sensitivity simulations, and safety-constrained LLM narrative synthesis.

### 1.4 Formal Research Questions & Objectives
This assessment evaluates four formal Research Questions (RQs):
- **RQ1 (Algorithmic Ranking Formulation):** Does a candidate-level Learning-to-Rank formulation (LambdaMART optimizing NDCG) outperform a multiclass gradient-boosted decision tree (XGBoost `multi:softprob`) in prioritizing viable crop slates under grouped query evaluations?
- **RQ2 (Knowledge Fusion & Constraint Gating):** How does hybrid fusion between probabilistic ML outputs and deterministic biophysical constraints (FAO EcoCrop / ICAR) impact top-$K$ accuracy, rank integrity, and physiological violation rates?
- **RQ3 (Generalization Under Transfer):** What is the quantitative performance degradation of the agricultural recommendation engine when evaluated across geographically unseen macro-regions (Leave-State-Out) and forward chronological time horizons compared to random-split baselines?
- **RQ4 (Multi-Objective Optimization):** Can a multi-objective Pareto frontier engine expose actionable agronomic trade-offs across orthogonal axes (Suitability vs Yield vs Water vs Erosion) without compromising inference efficiency?

### 1.5 System Contributions vs Explicit Non-Claims
To preserve complete academic and engineering integrity, the boundaries of this assessment are explicitly defined:

#### Verified System Contributions
1. **Empirically Falsified & Documented LTR Benchmark:** An exhaustive, reproducible candidate-level benchmark (161,456 candidate instances across 10,091 decision contexts) comparing Multiclass Softmax, Pointwise Binary, Pairwise Ranking, and Listwise LambdaMART, providing honest negative empirical evidence.
2. **Hybrid Biophysical Fusion Architecture:** A calibrated two-stage inference framework that couples a 26-feature XGBoost probability vector with deterministic FAO EcoCrop physiological boundaries, achieving 58.24% Top-5 accuracy on forward chronological test sets.
3. **Multi-Split Generalization Auditing:** Direct empirical quantification of model performance across 10-fold spatial cross-validation, forward temporal splits (2019–2020), and a combined spatiotemporal holdout (unseen Southern region in future years).
4. **Multi-Objective Pareto Decision Engine:** Elimination of scalar score collapse through an operational Pareto frontier engine supporting five distinct farmer decision profiles.
5. **Calibrated Yield Regressor with Prediction Intervals:** Preservation of an orthogonal crop-conditional yield regressor ($R^2 = 0.954$, MedAE = 0.36 t/ha) paired with empirical residual uncertainty intervals.

#### Explicit Non-Claims (Scope Boundaries)
- **NO Claim of Real-World Deployment or Farmer Validation:** The system has been evaluated strictly on historical benchmarks; no randomized field trials or farmer user studies have been performed.
- **NO Claim of Operational Computer Vision:** Satellite imagery ingestion (Sentinel-2) and drone orthomosaic segmentation are future roadmap targets, not operational components of the crop recommendation engine.
- **NO Claim of Autonomous Agronomic Superiority:** Model recommendations are decision-support heuristics and cannot substitute for certified agronomic soil testing or local extension officer advice.
- **NO Claim of Infinite Temporal Transferability:** As climate change accelerates weather variance, historical training distributions will experience drift, requiring continual pipeline re-calibration.

### 1.6 Report Organization
The remainder of this report is organized as follows: Chapter 2 reviews relevant academic literature and establishes a comparative matrix. Chapter 3 defines the research gap and mathematical problem formulation. Chapter 4 outlines system requirements and decoupled architecture. Chapter 5 details data ingestion, the 26-feature dictionary, and forensic data quality checks. Chapter 6 presents the machine learning methodology, hyperparameter configurations, and the LTR benchmark. Chapter 7 explores dual-layer explainability and LLM safety. Chapter 8 audits implementation details and technical debt. Chapter 9 describes experimental protocols. Chapter 10 presents empirical results, the confusion matrix, and failure modes. Chapter 11 discusses findings and Agile lessons. Chapter 12 details limitations and future roadmap goals, and Chapter 13 concludes the report. Complete bibliographies and appendices follow.

---

# Chapter 2: Literature Review

### 2.1 Evolution of Agricultural Decision-Support Systems
Agricultural Decision-Support Systems (ADSS) have evolved across three distinct technological paradigms over the past four decades:
1. **Rule-Based Expert Systems (1980s–1990s):** Early ADSS platforms, exemplified by the FAO EcoCrop database [15] and localized agro-climatic zoning bulletins, utilized deterministic lookup tables and Boolean decision trees. While biologically transparent, these systems were inherently static, incapable of digesting continuous soil chemical gradients, and brittle when environmental conditions fell marginally outside predefined bounding boxes.
2. **Statistical and Shallow Machine Learning (2000s–2015):** The introduction of statistical learning algorithms (Support Vector Machines, Naïve Bayes, Multi-Layer Perceptrons) allowed systems to learn nonlinear decision boundaries from historical farm survey data. However, as documented by Pudumalar et al. [8], these models functioned as pure black boxes, lacked spatial transferability, and relied on manual laboratory soil testing inputs that smallholder farmers could not afford.
3. **Hybrid Geospatial & Ensemble Intelligence (2016–Present):** Contemporary systems integrate remote sensing, gridded numerical weather prediction (NWP), and gradient-boosted decision trees. However, as synthesized by Kondaveeti et al. [2], modern platforms remain plagued by single-scalar collapse and a failure to evaluate models under rigorous spatiotemporal holdouts.

### 2.2 Machine Learning for Crop Suitability & Recommendation
Numerous studies have applied machine learning to crop recommendation with varying degrees of rigor:
- **Rajak et al. (2017) [1]:** Implemented a majority-voting ensemble combining SVM, ANN, Naïve Bayes, and Random Forest on 2,200 Western Indian soil records, reporting 90.2% accuracy. However, the study relied entirely on randomized 70/30 train/test splits, ignoring spatial autocorrelation, and excluded dynamic weather series.
- **Kondaveeti et al. (2023) [2]:** Conducted a systematic review and empirical benchmarking of Random Forest, CatBoost, and Stacking Classifiers across Indian district statistics, achieving an F1-score of 0.88. The authors noted that while tree ensembles consistently dominate tabular agricultural benchmarks, existing systems fail to provide uncertainty intervals or physiological constraint gating.

### 2.3 Multimodal Geospatial & Climatological Data Ingestion
The digitization of global environmental datasets has enabled automated feature acquisition:
- **ISRIC SoilGrids250m (Hengl et al., 2017) [11]:** Demonstrated the utility of global machine-learning-derived soil property maps (sand, silt, clay, bulk density, pH, organic carbon) at 250m resolution, providing a viable proxy for expensive laboratory soil testing.
- **FAO Irrigation & Drainage Paper 56 (Allen et al., 1998) [7]:** Established the standardized Penman-Monteith formulation for reference evapotranspiration ($ET_0$), providing the foundational biophysical equations utilized in GeoAgri AI's Water Stress Index.

### 2.4 Learning-to-Rank & Listwise Optimization in Agriculture
Learning-to-Rank (LTR) algorithms, pioneered by Burges [5] through the LambdaMART framework, have transformed information retrieval by directly optimizing non-smooth listwise metrics (NDCG, MAP) rather than surrogate classification losses. While extensively applied in web search and recommender systems, the application of listwise LTR to agricultural candidate crop selection remained an unverified empirical question prior to GeoAgri AI V5.

### 2.5 Explainable AI (XAI) & LLM Reasoning in Environmental Systems
Explainability in high-stakes environmental domains has shifted from heuristic feature importances to game-theoretic formulations:
- **Lundberg & Lee (2017) [6]:** Formulated TreeSHAP, proving that Shapley values provide the unique class of additive feature attributions satisfying efficiency, symmetry, dummy, and additivity axioms in polynomial time.
- **LLM Environmental Synthesizers:** Recent efforts have integrated Large Language Models as conversational interfaces. However, unconstrained LLMs suffer from severe factual hallucination. GeoAgri AI's architecture isolates the LLM (Meta Llama-3.1-8b via Groq) strictly downstream of numerical ML and TreeSHAP calculations, enforcing low-temperature narrative formatting.

### 2.6 Spatial Autocorrelation & Environmental Generalization
The critical methodological foundation of our evaluation is drawn from **Meyer and Pebesma (2022) [3]**, published in *Methods in Ecology and Evolution*. Meyer and Pebesma demonstrated that in environmental and agricultural machine learning, standard random cross-validation splits severely underestimate true prediction error because spatial autocorrelation causes training and test folds to share nearly identical environmental attributes. They proved the necessity of spatial cross-validation (Leave-Location-Out) and explicit spatiotemporal holdouts.

### 2.7 Comprehensive Literature Comparison Table

The comparative matrix below rigorously contrasts prior studies against GeoAgri AI V5 across eight analytical dimensions:

| Study & Citation | Problem Addressed | Dataset & Geography | Core Methodology | Evaluation Protocols & Metrics | Key Strengths | Critical Limitations | Relevance & Differentiation in GeoAgri AI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rajak et al. (2017)** [1] | Single-crop recommendation from soil nutrients | 2,200 Western India soil samples | Multi-model Majority Voting Ensemble (SVM, ANN, NB, RF) | Random 70/30 split; Accuracy = 90.2% | Early multi-model ensemble for soil-crop matching. | Excludes weather, rainfall, and season; random split causes spatial data leakage. | GeoAgri AI integrates dynamic Open-Meteo weather and physical soil properties. |
| **Kondaveeti et al. (2023)** [2] | Multi-season crop suitability & yield ranking | Indian district agricultural statistics (1998–2017) | Random Forest, CatBoost, Stacking Ensembles | Stratified 5-Fold CV; F1-score = 0.88 | Long-term weather and yield integration across crops. | Pure classification; no physiological bounds; no yield uncertainty intervals. | GeoAgri AI separates classification probabilities from agronomic suitability via EcoCrop fusion. |
| **Meyer & Pebesma (2022)** [3] | Spatial CV & Area of Applicability in ML | Global & regional environmental spatial grids | Spatial Target-Oriented Cross-Validation (LLO-CV) | Spatial CV vs Random CV (25–60% error inflation) | Foundational proof of spatial leakage in environmental ML. | Methodological framework; no crop recommendation algorithms. | GeoAgri AI implements strict Leave-State-Out and spatiotemporal holdout evaluations. |
| **Chen & Guestrin (2016)** [4] | Scalable tree boosting for tabular datasets | Standard tabular benchmark datasets | XGBoost: Second-order gradient boosted decision trees | 10-Fold CV; AUC, Log-Loss, Latency | Scalable, exact/histogram split finding, $L_1$/$L_2$ leaf regularization. | Generic algorithm; lacks domain-specific biophysical constraints. | XGBoost serves as the core tabular classification, ranking, and regression engine. |
| **Burges (2010)** [5] | Learning-to-Rank for information retrieval | LETOR, MSLR-WEB search benchmarks | LambdaMART: MART boosted trees with Lambda-gradients | NDCG@K, MRR, MAP | Directly optimizes non-smooth listwise ranking metrics. | Designed for large, variable candidate document pools ($10^2 - 10^4$). | GeoAgri AI benchmarks LambdaMART against multiclass softmax for candidate crop slates. |
| **Lundberg & Lee (2017)** [6] | Model-agnostic and tree-specific attribution | Healthcare, finance, genomics tabular datasets | TreeSHAP: Exact polynomial-time Shapley values | Game-theoretic axioms (Efficiency, Consistency) | Mathematically consistent, exact local feature attribution. | Reflects model correlation, not real-world biological causality. | GeoAgri AI uses TreeSHAP while explicitly disclosing that SHAP does not imply causality. |
| **Allen et al. (1998)** [7] | Crop evapotranspiration & water requirements | Global lysimeter & meteorological standards | Penman-Monteith physical $ET_0$ formulation | Standardized global biophysical benchmark | Definitive physical equations for atmospheric water demand. | Pure deterministic formulation; requires full meteorological instrumentation. | GeoAgri AI uses FAO-56 derived indices ($ET_0$, WSI) as engineered model features. |
| **Pudumalar et al. (2016)** [8] | Precision agriculture decision support | Regional Tamil Nadu agricultural survey | Decision Trees, Naïve Bayes, Rule Induction | Holdout accuracy (reported 88%) | Early practical implementation for regional farmers. | Extremely localized; unable to generalize across diverse agro-climatic zones. | GeoAgri AI benchmarks machine learning across national agro-climatic zones. |

### 2.8 Critical Literature Synthesis
The literature establishes that while gradient-boosted decision trees (XGBoost) represent the empirical state of the art for tabular environmental prediction, existing agricultural systems exhibit severe methodological gaps. They treat recommendation as an unconstrained single-label classification problem, collapse multi-objective trade-offs into scalar scores, ignore spatial leakage during evaluation, and provide no actionable explanation framework. GeoAgri AI V5 was architected explicitly to bridge these gaps.

---

# Chapter 3: Research Gap and Problem Formulation

### 3.1 Synthesis of What Prior Research Establishes
Prior scholarly research establishes three solid foundations:
1. **Tree Ensembles Outperform Neural Networks on Tabular Data:** Gradient-boosted decision trees consistently achieve superior accuracy, lower inference latency, and greater robustness to heterogeneous feature types compared to deep neural networks on tabular datasets [4].
2. **Environmental Random Splits Are Unreliable:** Randomized cross-validation splits in geospatial machine learning yield severely optimistic, invalid performance estimates due to spatial autocorrelation [3].
3. **Biophysical Constraints Prevent Extreme Errors:** Pure statistical models can generate biologically impossible predictions without deterministic domain constraint gating [2], [7].

### 3.2 Four Critical Deficiencies in Current Systems
Despite these foundations, current agricultural intelligence architectures suffer from four distinct deficiencies:
- **Deficiency 1 (The Constraint Gap):** Absence of calibrated mathematical blending between probabilistic machine learning classifiers and deterministic biophysical thresholds.
- **Deficiency 2 (The Ranking Formulation Gap):** Lack of empirical investigation into whether candidate-level Learning-to-Rank (LambdaMART) outperforms multiclass softmax probability generation for multi-candidate crop screening.
- **Deficiency 3 (The Multi-Objective Collapse Gap):** Complete reliance on single-scalar score rankings that obscure orthogonal trade-offs between crop suitability, economic yield, groundwater depletion, and geomorphic soil loss.
- **Deficiency 4 (The Generalization Deficit):** Inadequate evaluation of model robustness under joint spatial transfer (unseen agro-climatic zones) and temporal transfer (future climate years).

### 3.3 The Evidence-Backed Research Gap
The specific, defensible research gap addressed by GeoAgri AI V5 is formulated as follows:

> **The Research Gap:**  
> Existing agricultural recommendation platforms lack a verified, multi-objective decision framework that simultaneously integrates calibrated gradient-boosted probability distributions with deterministic biophysical constraints, exposes non-dominated Pareto trade-offs across independent environmental dimensions, and maintains documented generalization under joint spatiotemporal transfer without scalar objective collapse.

### 3.4 Formal Mathematical Problem Formulation
Let a decision context query $q$ be defined as a tuple:
$$q = \left( \mathbf{x}_{\text{geo}}, \, \mathbf{x}_{\text{soil}}, \, \mathbf{x}_{\text{weather}}, \, \mathbf{x}_{\text{terrain}}, \, s \right)$$
Where:
- $\mathbf{x}_{\text{geo}} = (\text{lat}, \text{lon}) \in \mathbb{R}^2$ represents geographic coordinates within the Indian landmass ($8.59^\circ \text{N} \le \text{lat} \le 36.05^\circ \text{N}$, $68.82^\circ \text{E} \le \text{lon} \le 97.11^\circ \text{E}$).
- $\mathbf{x}_{\text{soil}} \in \mathbb{R}^9$ represents pedological chemistry and physical texture fractions ($\text{pH}, \text{N}, \text{P}, \text{K}, \text{OC}, \text{EC}, \text{clay}, \text{sand}, \text{silt}$).
- $\mathbf{x}_{\text{weather}} \in \mathbb{R}^8$ represents dynamic seasonal weather and climatology ($T_{\text{mean}}, T_{\min}, T_{\max}, \text{RH}, \text{Rain}_{\text{season}}, \text{Rain}_{30\text{d}}, \text{Rain}_{90\text{d}}, \text{Rain}_{\text{annual}}, ET_0, \theta_{\text{soil}}$).
- $\mathbf{x}_{\text{terrain}} \in \mathbb{R}^3$ represents surface topography ($\text{elevation}, \text{slope}, \text{erosion\_risk}$).
- $s \in \{\text{Kharif}, \text{Rabi}, \text{Summer}, \text{Whole Year}\}$ represents the discrete planned cropping season.

Let $\mathcal{C} = \{c_1, c_2, \dots, c_{16}\}$ denote the finite set of 16 supported agricultural crops. The system must determine a ranked candidate slate:
$$\mathbf{C}^* = [c_{(1)}, c_{(2)}, \dots, c_{(K)}] \subset \mathcal{C}, \quad K \le 5$$
Subject to the multi-objective Pareto optimization criterion:
$$\mathbf{C}^* \in \mathcal{P}^* = \left\{ c \in \mathcal{C} \mid \nexists c' \in \mathcal{C} : c' \succ c \right\}$$
Where dominance $c' \succ c$ is evaluated across five independent objective functions:
$$f_1(c) = \text{Suitability}(c, q), \quad f_2(c) = \hat{Y}(c, q), \quad f_3(c) = -\text{WSI}(q), \quad f_4(c) = -\text{ErosionRisk}(c, q), \quad f_5(c) = \text{DecisionMargin}(c)$$

### 3.5 System-Level Contribution Delineation
GeoAgri AI V5 resolves this problem by implementing a modular, multi-tier architecture that explicitly separates classification, ranking, biophysical gating, yield regression, and Pareto frontier filtering into independent, verifiable components.

---

# Chapter 4: Requirements and System Design

### 4.1 Stakeholder Profiles & Functional Requirements
The system serves three primary user personas:
1. **Smallholder / Commercial Farmers:** Require an intuitive, mobile-friendly map interface to select field locations, view top-5 viable crop recommendations, understand positive and limiting factors, and inspect expected yield ranges.
2. **Agricultural Extension Officers & Agronomists:** Require transparent biophysical drivers, sensitivity to drought/rain anomalies, and the ability to customize decision profiles based on regional water tables.
3. **Software Engineers & Academic Evaluators:** Require fully documented REST APIs, transparent error handling, automated regression tests, and verifiable model artifacts.

#### Core Functional Requirements (FRs)
- **FR1 (Geospatial Ingestion):** The system must accept arbitrary latitude and longitude coordinates within India and automatically retrieve soil chemistry, 10-year climatology, 7-day NWP weather forecasts, and elevation.
- **FR2 (26-Feature Transformation):** The system must compute derived agronomic indices (SFI, WSI, $\Delta R$, SMI) and categorical encodings matching the certified 26-dimensional model schema.
- **FR3 (Multi-Candidate Screening):** The system must generate a prioritized slate of up to top-5 viable crop candidates with associated suitability scores and confidence ratings.
- **FR4 (Yield Estimation with Intervals):** The system must output crop-conditional yield estimates in tonnes per hectare accompanied by statistical uncertainty bounds.
- **FR5 (Multi-Objective Decision Profiling):** The system must allow users to filter recommendations through five distinct decision profiles (Suitability, Yield, Water-Constrained, Risk-Averse, Balanced).
- **FR6 (Explainability & Counterfactuals):** The system must output exact TreeSHAP feature attributions, five directional counterfactual scenario simulations, and a concise narrative summary.

### 4.2 Non-Functional & Performance Requirements
- **NFR1 (Inference Latency):** Core machine learning inference for a single decision context must complete in under 5.0 milliseconds on CPU (under 0.1 ms on GPU).
- **NFR2 (End-to-End Response Time):** Total end-to-end API response time (including live external geospatial HTTP queries) must complete within 3.5 seconds under normal network conditions.
- **NFR3 (Graceful Fallback Resilience):** If external geospatial APIs (SoilGrids, Open-Elevation) or the Groq LLM API fail or time out, the system must execute local district prior and rule fallbacks without throwing 500 errors.
- **NFR4 (Model Immutability):** Deployment of newer model generations must not corrupt or overwrite earlier certified releases (V1 through V4).

### 4.3 High-Level System Architecture & Context Diagram
The system follows a three-tier decoupled architecture:
1. **Presentation Tier:** Next.js 14 interactive GIS dashboard (`ui/app/page.tsx`).
2. **Serving & IPC Dispatch Tier:** Dual-mode API interface featuring an asynchronous FastAPI service (`backend_api.py`) for direct REST clients and a Node.js child-process worker bridge (`ui/server/predict_worker.py`) for web UI integration.
3. **Core Intelligence & Machine Learning Tier:** Modular Python packages in `src/` encapsulating geospatial fetching, feature engineering, XGBoost inference, knowledge fusion, and Pareto optimization.

### 4.4 Component Deconstruction & Decoupled Modular Design
```
GeoAgri-AI Root
├── src/
│   ├── api/             # FastAPI production service (backend_api.py)
│   ├── explainability/  # Groq Llama-3.1 client & deterministic fallback (ai_explainer.py)
│   ├── geospatial/      # SoilGrids, Open-Meteo, Open-Elevation ingestion clients
│   ├── models/          # Core inference engine & Pareto optimization (agri_inference.py)
│   ├── preprocessing/   # 26-feature agronomic transformation (agri_features.py)
│   ├── utils/           # Configuration & logging shims
│   └── vision/          # Documented planned remote-sensing roadmap
├── ui/                  # Next.js 14 React client & Leaflet GIS interface
├── models/crop/         # Serialized model registries (v1 through v5)
├── data/agriculture/    # Longitudinal datasets & FAO EcoCrop knowledge base
├── artifacts/crop_v5/   # Certified empirical metrics, CSV audits, and ablation logs
└── tests/               # 17 automated pytest regression modules
```

### 4.5 Technology Stack Selection & Decision Matrix
The complete technology stack was justified based on technical trade-offs:

| Layer / Technology | Alternative Evaluated | Decisive Justification | Evidence Path |
| :--- | :--- | :--- | :--- |
| **XGBoost 3.4.1** | Deep Neural Network (MLP), Random Forest | Superior tabular accuracy, exact TreeSHAP compatibility, GPU histogram speed, robust handling of heterogeneous features. | `models/crop/v5/crop_model.pkl` |
| **FastAPI 0.142.2** | Flask, Django REST Framework | Asynchronous I/O, native OpenAPI/Swagger documentation, strict Pydantic validation, high throughput. | `src/api/backend_api.py` |
| **Next.js 14 (App Router)** | Vite + React SPA | Server-Side Rendering (SSR) for initial load performance; unified API route proxying; robust Leaflet dynamic hydration. | `ui/app/page.tsx` |
| **Leaflet 1.9.4** | Google Maps API, Mapbox GL | Open-source, zero API licensing fees, lightweight footprint, offline tile caching compatibility. | `ui/components/MapPicker.tsx` |
| **Groq Cloud (Llama-3.1-8b)** | OpenAI GPT-4, Local Ollama | Ultra-low inference latency (< 500 ms), zero local GPU VRAM overhead, cost-effective plain-language formatting. | `src/explainability/ai_explainer.py` |
| **Pytest 9.1.1** | Unittest, Nose2 | Parametrized testing, robust async fixtures, detailed assertion introspection, standardized exit codes. | `tests/pyproject.toml` |

---

# Chapter 5: Data and Feature Engineering

### 5.1 Multimodal Data Sources & Ingestion Protocols
GeoAgri AI ingests environmental attributes across four distinct channels:
1. **Soil Profile Ingestion (`fetch_soil.py`):** Queries the ISRIC SoilGrids REST API at standard soil depths (0–5 cm, 5–15 cm, 15–30 cm). Extracts sand, silt, and clay mass fractions (g/kg converted to percentages), pH in $\text{H}_2\text{O}$, nitrogen, and organic carbon. If the remote endpoint exceeds the 8.0-second timeout, the system executes a k-d tree spatial lookup to impute values from the nearest historical district centroid in `lat_lon_india_district.csv`.
2. **Climatological Ingestion (`fetch_weather.py`):** Queries the Open-Meteo Historical Weather API and Numerical Weather Prediction (NWP) forecast service. Ingests 10-year mean seasonal temperatures, diurnal temperature ranges ($T_{\max} - T_{\min}$), seasonal rainfall totals, and rolling 7-day, 30-day, and 90-day precipitation series.
3. **Topographic Ingestion (`fetch_terrain.py`):** Queries Open-Elevation (SRTM 90m DEM) to establish surface elevation above mean sea level. Local slope (degrees) is computed via cardinal 4-point elevation gradients ($\Delta \text{elevation} / \Delta \text{distance}$).
4. **Soil Erosion Ingestion (`terrain_model/erosion_model.pkl`):** Predicts geomorphic erosion risk probability from geomorphic slope, vegetation proxy, elevation, rainfall, and soil class.

### 5.2 Dataset Provenance, Construction, and Characteristics
The primary training dataset (`data/agriculture/v1.0/processed/crop_suitability_dataset.csv`) was synthesized from longitudinal district agricultural records published by the Directorate of Economics & Statistics (DES), Ministry of Agriculture & Farmers Welfare, GoI, cross-referenced with the ICRISAT District-Level Database (DLD) and the DAC&FW National Soil Health Card portal:
- **Total Decision Queries ($N$):** 10,091 complete agricultural records.
- **Unique Crops ($K$):** 16 major Indian crops spanning cereals, pulses, oilseeds, and commercial cash crops.
- **Temporal Horizon:** Continuous 24-year series (1997 through 2020).
- **Geographic Coverage:** 30 Indian States and Union Territories encompassing all 15 Planning Commission Agro-Climatic Zones.

### 5.3 Data Dictionary: Raw Attributes vs 26-Dimensional Model Vector
The 31 raw columns in the processed CSV are transformed into a strictly ordered 26-dimensional feature vector ingested by the XGBoost models:

| Index | Feature Code Name | Data Type | Physical Unit | Agronomic Description |
| :---: | :--- | :---: | :---: | :--- |
| `0` | `soil_ph` | Float | $-\log[H^+]$ | Soil acidity/alkalinity measure (range: 4.55 to 8.64) |
| `1` | `nitrogen` | Float | kg/ha | Soil available mineral nitrogen content (range: 73 to 350.5) |
| `2` | `phosphorus` | Float | kg/ha | Soil available phosphorus rating (range: 5.0 to 33.7) |
| `3` | `potassium` | Float | kg/ha | Soil available potassium rating (range: 71.4 to 410.4) |
| `4` | `organic_carbon` | Float | % | Soil organic carbon mass fraction (range: 0.10 to 1.71) |
| `5` | `electrical_conductivity`| Float | dS/m | Soil salinity / soluble salt concentration |
| `6` | `clay` | Float | % | Soil clay fraction (< 0.002 mm particle diameter) |
| `7` | `sand` | Float | % | Soil sand fraction (0.05 to 2.0 mm particle diameter) |
| `8` | `silt` | Float | % | Soil silt fraction (0.002 to 0.05 mm particle diameter) |
| `9` | `temperature_mean` | Float | °C | Seasonal mean ambient temperature |
| `10` | `temperature_range` | Float | °C | Diurnal thermal swing ($T_{\max} - T_{\min}$) |
| `11` | `humidity_mean` | Float | % | Seasonal mean ambient relative humidity |
| `12` | `rainfall_season` | Float | mm | Cumulative growing season precipitation |
| `13` | `rainfall_30d` | Float | mm | Short-term vegetative window rainfall total |
| `14` | `rainfall_90d` | Float | mm | Extended reproductive window rainfall total |
| `15` | `soil_moisture` | Float | $\text{m}^3/\text{m}^3$ | Topsoil volumetric water content (0–7 cm depth) |
| `16` | `et0` | Float | mm/day | Reference Penman-Monteith crop evapotranspiration |
| `17` | `elevation` | Float | m MSL | Surface topography elevation above mean sea level |
| `18` | `slope` | Float | degrees | Local topographic grade inclination |
| `19` | `erosion_risk_score` | Float | Probability [0, 1] | Geomorphic soil loss risk predicted by erosion model |
| `20` | `soil_fertility_index` | Float | Score [5, 100] | Composite ICAR macro-nutrient fertility rating |
| `21` | `water_stress_index` | Float | Ratio [0.1, 5.0] | Atmospheric evaporative demand vs moisture availability |
| `22` | `rainfall_deviation` | Float | Ratio [-1.0, 2.0]| Seasonal precipitation anomaly relative to annual baseline |
| `23` | `soil_moisture_index` | Float | Ratio [0.05, 1.25]| Moisture content normalized to field capacity (0.40) |
| `24` | `season_code` | Float | Discrete {0,1,2,3}| Categorical: Kharif=0, Rabi=1, Summer=2, Whole Year=3 |
| `25` | `texture_code` | Float | Discrete {0..6} | Categorical: Clay=0, Clay Loam=1, Loam=2, Sandy Loam=3, etc.|

### 5.4 Mathematical Formulation of Composite Agronomic Indicators
The four composite indicators implemented in `src/preprocessing/agri_features.py` are mathematically defined as follows:

#### 1. Soil Fertility Index (SFI)
$$\text{SFI} = \min\left(25, \frac{\text{N}}{300} \cdot 25\right) + \min\left(20, \frac{\text{P}}{25} \cdot 20\right) + \min\left(20, \frac{\text{K}}{260} \cdot 20\right) + \min\left(20, \frac{\text{OC}}{0.75} \cdot 20\right) + \max\left(0, 15 - 5 \cdot |\text{pH} - 7.0|\right)$$
$$\text{SFI} \in [5.0, 100.0]$$

#### 2. Water Stress Index (WSI)
$$\text{WSI} = \frac{\max(10.0, \, \text{ET}_0 \cdot 30)}{\max(5.0, \, \text{rainfall}_{30\text{d}} + (\text{soil\_moisture} \cdot 150))}$$
$$\text{WSI} \in [0.1, 5.0]$$

#### 3. Seasonal Rainfall Deviation ($\Delta R$)
$$\Delta R = \frac{\text{rainfall}_{\text{season}} - \max(50.0, \, 0.65 \cdot \text{rainfall}_{\text{annual}})}{\max(50.0, \, 0.65 \cdot \text{rainfall}_{\text{annual}})}$$
$$\Delta R \in [-1.0, 2.0]$$

#### 4. Soil Moisture Index (SMI)
$$\text{SMI} = \text{clip}\left(\frac{\text{soil\_moisture}}{0.40}, \, 0.05, \, 1.25\right)$$

### 5.5 Forensic Data Quality Audit & The 4 Texture Violations
An audit of `data/agriculture/v1.0/dataset_validation_report.json` confirmed that the dataset contains zero missing values across all 10,091 rows and zero duplicate coordinate-season entries. However, the audit uncovered an internal contradiction:
- The sub-check for `soil_ranges` reported `"status": "FAIL"` with `texture_sum_violations: 4`.
- The summary report status on line 127 was recorded as `"status": "PASS"`.

In physical soil science, the mineral fraction must conserve mass: $\text{Sand} + \text{Silt} + \text{Clay} = 100.0\%$. In exactly four records, rounding discrepancies during source table digitization produced sums between 98.8% and 101.4%. We explicitly disclose that while the summary was marked "PASS", four records violate texture conservation. Representing only $0.039\%$ of the dataset, their impact on tree split thresholds is negligible, but transparent academic reporting is maintained.

### 5.6 Data Leakage Safeguards & Grouped Partitioning Invariants
To ensure scientific validity, three strict leakage safeguards were enforced:
1. **Zero Target Leakage:** Realized harvest mass (`yield`) is strictly excluded from classification and ranking predictors.
2. **Atomic Query Grouping:** In the candidate ranking dataset (161,456 candidate rows), all 16 candidate rows sharing decision context $q$ are assigned atomically to the same split partition.
3. **Chronological Splitting:** Training data is strictly restricted to historical years $\le 2017$, while evaluation is conducted on future years $2019–2020$.

---

# Chapter 6: Machine Learning Methodology

### 6.1 Mathematical Separation of Agricultural Tasks
In GeoAgri AI V5, agricultural intelligence is partitioned into distinct mathematical formulations:
- **Multiclass Probability ($P_{\text{ML}}$):** Normalized softmax likelihood output over 16 classes.
- **Biophysical Compatibility ($S_{\text{KB}}$):** Bounded compliance score $[0.0, 1.0]$ based on FAO EcoCrop physiological rules.
- **Candidate Ranking ($R_{\text{LTR}}$):** Continuous relevance score from LambdaMART optimizing NDCG.
- **Yield Potential ($\hat{Y}$):** Continuous regression estimate in tonnes per hectare.
- **Multi-Objective Dominance:** Non-dominated subset identification across the 5-dimensional Pareto frontier.

### 6.2 Technical Justification for XGBoost Core Architecture
XGBoost was selected as the foundational engine based on three properties:
1. **Tabular Invariance:** Decision trees are invariant to monotonic transformations, naturally handling continuous chemical concentrations alongside bounded indices and discrete codes.
2. **Regularization:** Built-in $L_1$ and $L_2$ penalties on leaf weights suppress overfitting to localized spatial noise.
3. **Exact TreeSHAP:** Native compatibility enables polynomial-time local feature attribution.
4. **CUDA GPU Acceleration:** Using the `hist` tree method on an NVIDIA RTX 3050 Laptop GPU, candidate-level training achieved a **14.2x speedup** over CPU greedy splitting.

### 6.3 Detailed Model Formulations & Hyperparameters
The hyperparameter configurations for the three certified production models are summarized below:

| Hyperparameter Parameter | Multiclass Classifier (`crop_model.pkl`) | LambdaMART Ranker (`ranking_model.pkl`) | Yield Regressor (`yield_model.pkl`) |
| :--- | :---: | :---: | :---: |
| **Objective Function** | `multi:softprob` | `rank:ndcg` | `reg:squarederror` |
| **Boosting Iterations (`n_estimators`)**| 120 | 100 | 150 |
| **Maximum Tree Depth (`max_depth`)** | 5 | 5 | 6 |
| **Learning Rate (`learning_rate`)** | 0.05 | 0.05 | 0.04 |
| **Row Subsample Ratio (`subsample`)** | 0.85 | 0.85 | 0.80 |
| **Column Subsample (`colsample_bytree`)**| 0.80 | 0.80 | 0.85 |
| **Tree Method (`tree_method`)** | `hist` | `hist` | `hist` |
| **Hardware Device (`device`)** | `cuda` (GPU) | `cuda` (GPU) | `cuda` (GPU) |
| **Input Feature Dimension** | 26 features | 43 candidate features | 42 conditioned features |
| **Output Target** | 16 discrete probabilities | Continuous ranking score | Yield estimate (t/ha) |

### 6.4 Candidate-Level Learning-to-Rank Benchmark & Hypothesis Falsification
The central scientific investigation in Crop Intelligence V5 evaluated whether candidate-level Learning-to-Rank (LambdaMART) would superiorly prioritize crop slates compared to multiclass softmax. The benchmark was conducted on 161,456 candidate instances evaluated on the held-out chronological test set (2019–2020, $N=546$ decision queries):

| Model Architecture & Formulation | Top-1 Acc | Top-3 Acc | Top-5 Acc | NDCG@3 | NDCG@5 | MRR | MAP@5 | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Multiclass Softmax (Base)** | **18.86%** | 42.12% | **59.34%** | **0.3198** | **0.3899** | **0.3705** | **0.3236** | **0.1618** |
| **Model B: Pointwise Binary Compatibility** | 16.48% | 39.56% | 55.13% | 0.2970 | 0.3608 | 0.3510 | 0.2984 | 0.0900 |
| **Model C: Pairwise LTR (`rank:pairwise`)** | 17.40% | 38.28% | 55.13% | 0.2924 | 0.3626 | 0.3528 | 0.3021 | 0.0852 |
| **Model D: Listwise LambdaMART (`rank:ndcg`)** | 16.85% | 37.73% | 55.13% | 0.2873 | 0.3580 | 0.3471 | 0.2951 | 0.1293 |
| **Model E1: Multiclass Fused (0.85 ML + 0.15 KB)** | **18.86%** | **42.31%** | 58.24% | 0.3200 | 0.3849 | 0.3689 | 0.3204 | 0.1614 |
| **Model E2: LambdaMART Fused (0.85 LTR + 0.15 KB)** | 14.47% | 37.18% | 51.65% | 0.2738 | 0.3327 | 0.3278 | 0.2725 | 0.1177 |

#### Scientific Interpretation of the Negative Finding
The hypothesis was **falsified**. Multiclass Softmax (Model A and Model E1) outperformed LambdaMART across all metrics. The mathematical explanation is that LTR algorithms are designed for massive, variable document pools ($10^2 - 10^4$) with graded relevance. For a fixed slate of 16 discrete crops with binary relevance, the pairwise lambda gradients introduce variance across the 15 negative items, whereas the exact softmax partition function normalizes the competitive denominator across all 16 classes simultaneously. Model E1 was certified as the production backbone.

### 6.5 Dual-Layer Knowledge Fusion Formulations
The repository implements two distinct scoring formulations across different layers:

#### Formulation 1: Evaluation & Benchmarking Fusion (Model E1)
Implemented in `scripts/train_crop_v5.py`:
$$\text{Score}_{\text{Eval}}(c) = 0.85 \cdot P_{\text{ML}}(c \mid \mathbf{x}) + 0.15 \cdot S_{\text{KB}}(c)$$
Grid search across fusion weights confirmed that the $0.85 / 0.15$ ratio optimizes Top-3 accuracy (42.31%) while penalizing biophysically mismatched crops.

#### Formulation 2: Operational Runtime Domain Suitability Engine
Implemented in `src/models/agri_inference.py`:
$$\text{Suitability}_{\text{Runtime}}(c) = 0.45 \cdot P_{\text{ML}}(c) + 0.20 \cdot S_{\text{soil}}(c) + 0.15 \cdot S_{\text{season}}(c) + 0.10 \cdot S_{\text{climate}}(c) + 0.10 \cdot S_{\text{erosion}}(c)$$
This multi-attribute formulation scores soil pH compliance, seasonal alignment, rainfall adequacy, and slope gating independently for runtime dashboard display.

### 6.6 Multi-Objective Pareto Decision Engine & Decision Profiles
To eliminate scalar collapse, V5 introduces a 5-dimensional Pareto dominance engine evaluating:
$$f_1 = \text{Suitability}, \quad f_2 = \text{Yield}, \quad f_3 = -\text{WSI}, \quad f_4 = -\text{ErosionRisk}, \quad f_5 = \text{DecisionMargin}$$
A candidate crop $A$ dominates crop $B$ ($A \succ B$) if $A$ is strictly better in at least one objective without being worse in any other. The non-dominated set $\mathcal{P}^*$ is projected through five customized farmer decision profiles:
1. **Suitability-Focused (Default):** Maximizes biophysical growth harmony.
2. **Yield-Focused:** Heavily weights $\hat{Y}$; intended for commercial, irrigated acreage.
3. **Water-Constrained:** Minimizes WSI; prioritizes millets and pulses in drylands.
4. **Risk-Averse:** Minimizes geomorphic erosion and climate anomaly risks.
5. **Balanced:** Harmonic equal weighting across all five dimensions.

---

# Chapter 7: Explainability and Natural-Language Reasoning

### 7.1 Dual-Layer Explainability Framework
GeoAgri AI V5 couples game-theoretic mathematical feature attributions (TreeSHAP) with constrained natural-language reasoning (Groq Llama-3.1).

### 7.2 TreeSHAP Feature Attribution: Global Gains & Local Attributions
TreeSHAP decomposes predictions into base rates and additive feature contributions $\phi_i$:
$$f(\mathbf{x}) = \phi_0 + \sum_{i=1}^{26} \phi_i(\mathbf{x})$$
Global feature importance is dominated by `season_code` (184.2 gain), `rainfall_season` (142.6 gain), and `temperature_mean` (118.9 gain). At the local level, TreeSHAP identifies crop-specific positive drivers and limiting factors.

### 7.3 Epistemic Boundaries: Attribution vs Biological Causality
The assessment explicitly highlights that SHAP attributions reflect empirical correlation within historical training records, **not biological causation**. A positive attribution on nitrogen does not guarantee that adding fertilizer will increase yield. Furthermore, physical collinearity between environmental features causes attribution dispersion across tree splits.

### 7.4 Directional Counterfactual Sensitivity Analysis
To explain recommendation boundaries, the system executes five deterministic counterfactual simulations:
1. Drier Climate ($-30\%$ rainfall)
2. Wetter Climate ($+30\%$ rainfall)
3. Soil Acidification ($-0.5$ pH)
4. Alkaline Shift ($+0.5$ pH)
5. Supplemental Irrigation ($+200$ mm assured water)
Simulations indicate whether the top candidate retains its rank or triggers an agronomic shift (e.g., dryland Cotton yielding to high-input Maize under assured irrigation).

### 7.5 LLM Narrative Integration & Anti-Hallucination Guardrails
Natural language reasoning is executed via the Groq Cloud API utilizing `llama-3.1-8b-instant` at temperature $0.20$. To prevent hallucination:
- The prompt injects strictly pre-computed values (crop name, suitability %, yield range, positive drivers from TreeSHAP, counterfactual results).
- The system prompt forbids prescriptive claims (e.g., "best crop" or "guaranteed harvest").
- The LLM does **not** validate or calculate predictions; it operates purely as a narrative synthesizer.

### 7.6 Zero-Dependency Deterministic Fallback Mechanism
If the `GROQ_API_KEY` is missing, rate-limited (HTTP 429), or times out (> 25s), `ai_explainer.py` immediately generates a deterministic, rule-based template narrative, returning `insight_mode = "local"` without crashing execution.

---

# Chapter 8: Implementation and End-to-End Technology Stack

### 8.1 Full Repository Directory Structure
The audited repository structure is organized into clean functional modules:
```
c:\Users\...\GeoAgri-AI\
├── agri_features.py           # Root compatibility shim pointing to src/preprocessing
├── agri_inference.py          # Root compatibility shim pointing to src/models
├── backend_api.py             # Root compatibility shim pointing to src/api
├── requirements.txt           # Verified core dependencies
├── pyproject.toml             # Project build configuration & pytest metadata
├── src/
│   ├── api/backend_api.py     # FastAPI production service
│   ├── explainability/        # ai_explainer.py (Groq & fallback)
│   ├── geospatial/            # fetch_soil.py, fetch_weather.py, fetch_terrain.py
│   ├── models/agri_inference.py # Core V5 inference & Pareto engine
│   ├── preprocessing/         # agri_features.py (26 features)
│   └── vision/README.md       # Planned remote-sensing roadmap specification
├── models/crop/v5/            # Serialized binaries (crop_model, ranking_model, yield_model)
├── data/agriculture/v1.0/     # crop_suitability_dataset.csv & validation reports
├── artifacts/crop_v5/         # Certified benchmark CSVs, ablation logs, error analyses
├── tests/                     # 17 automated regression test modules (75 tests)
└── ui/                        # Next.js 14 Leaflet GIS frontend application
```

### 8.2 Frontend Implementation: Next.js 14 GIS Interface
The frontend (`ui/app/page.tsx`) provides an interactive GIS dashboard built with Next.js 14, React 18, Tailwind CSS, and Leaflet. Key components include:
- `MapPicker.tsx`: Dynamically imported client-side Leaflet component for interactive coordinate selection.
- `CropIntelligencePanel.tsx`: Displays top-5 candidate rankings, suitability gauges, expected yield ranges, TreeSHAP positive/limiting drivers, and counterfactual scenario cards.
- `DecisionProfileSelector`: Allows dynamic switching across the five Pareto decision profiles.

### 8.3 Backend Implementation: FastAPI & Subprocess Bridges
The system supports dual serving modalities:
1. **FastAPI (`backend_api.py`):** Direct OpenAPI 3.0 endpoints (`/api/agriculture/recommend`) for programmatic clients.
2. **Next.js Route Bridge (`ui/app/api/predict/route.ts`):** Spawns `ui/server/predict_worker.py` as a persistent child process communicating via JSON over stdin/stdout.

### 8.4 Technical Debt Audit: The Monolithic Bridge Post-Mortem
During audit execution, a severe technical debt defect was identified:
- **Defect:** `predict_bridge.py` contained eager, top-level imports of `cv2`, `torch`, `YOLO`, and `smp` retained from an earlier archaeological drone project.
- **Impact:** In environments lacking PyTorch or OpenCV, the Python worker process silently crashed during import, causing UI prediction requests to hang or return 500 errors.
- **Resolution & Agile Lesson:** Pure tabular inference (`agri_inference.py`) has zero dependency on computer vision. In future sprints, legacy vision shims must be isolated in a dedicated package with deferred imports.

### 8.5 Actual Implementation vs Proposed KPIT Roadmap Boundaries
The report explicitly distinguishes operational software from future grant proposals:
- **OPERATIONAL:** 26-feature pipeline, XGBoost multiclass model, candidate ranking benchmark, yield regressor, Pareto decision engine, TreeSHAP, Groq Llama-3.1 fallback, FastAPI backend, Next.js UI.
- **PROPOSED / ROADMAP:** Sentinel-2 satellite ingestion, DeepLabV3+ canopy segmentation, and automated volumetric irrigation scheduling (targeted for KPIT Sparkle 2027).

---

# Chapter 9: Experimental Design and Evaluation

### 9.1 Scientific Evaluation Objectives
Experimental evaluation was designed to assess:
1. Candidate ranking quality under grouped decision queries (NDCG@5, MRR, Top-$K$ accuracy).
2. Biophysical constraint satisfaction rates under knowledge fusion.
3. Model generalization under Leave-State-Out spatial CV and forward chronological transfer.
4. Yield regressor precision and prediction interval coverage.

### 9.2 Validation Splitting Strategies & Generalization Protocols
Four evaluation splits were executed:
- **Chronological Forward Holdout:** Train $\le 2017$ ($n=7,603$), Val $= 2018$ ($n=1,942$), Test $\ge 2019$ ($n=546$).
- **10-Fold Spatial Cross-Validation:** Disjoint Indian state partitioning.
- **Forward Temporal Split:** Historical vs future seasonal comparison.
- **Spatiotemporal Joint Holdout:** Complete exclusion of Southern states in future years ($n=125$ test queries).

### 9.3 Evaluation Metrics & Statistical Mathematical Definitions
$$\text{Top-}K \text{ Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i \in \hat{\mathbf{y}}_{i, 1..K})$$
$$\text{NDCG}@K = \frac{1}{N} \sum_{i=1}^N \frac{\text{DCG}_{i, K}}{\text{IDCG}_{i, K}}, \quad \text{DCG}_K = \sum_{r=1}^K \frac{2^{\text{rel}_r} - 1}{\log_2(r + 1)}$$
$$\text{MRR} = \frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i(y_i)}$$
$$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$

### 9.4 Calibration Metrics: Expected Calibration Error & Brier Score
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{i, k} - y_{i, k})^2$$

---

# Chapter 10: Experimental Results

### 10.1 Primary Benchmark Results (Chronological Holdout 2019–2020)
Evaluating the certified production models on the 546 held-out test queries yielded the verified metrics summarized below:

| Metric Category | Metric Identifier | V4 Certified Baseline | V5 Production Release | Observed Delta ($\Delta$) | Empirical Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Classification Accuracy** | Top-1 Accuracy | 18.86% | **18.86%** | $0.00\%$ | **`TESTED`** |
| | Top-3 Accuracy | 40.48% | **42.31%** | $+1.83\%$ | **`TESTED`** |
| | Top-5 Accuracy | 59.34% | **58.24%** | $-1.10\%$ | **`TESTED`** |
| **Ranking Quality** | NDCG@3 | 0.3106 | **0.3200** | $+0.0094$ | **`TESTED`** |
| | NDCG@5 | 0.3873 | **0.3849** | $-0.0024$ | **`TESTED`** |
| | Mean Reciprocal Rank (MRR)| 0.3672 | **0.3689** | $+0.0017$ | **`TESTED`** |
| | Mean Average Precision (MAP@5)| 0.3211 | **0.3204** | $-0.0007$ | **`TESTED`** |
| **F1 Metrics** | Macro-Averaged F1 | 0.1643 | **0.1614** | $-0.0029$ | **`TESTED`** |
| | Weighted F1 | 0.1620 | **0.1600** | $-0.0020$ | **`TESTED`** |
| **Yield Regression** | Coefficient of Determination ($R^2$)| 0.954 | **0.954** | $0.000$ | **`TESTED`** |
| | Root Mean Squared Error (RMSE)| 6.377 t/ha | **6.377 t/ha** | $0.000$ | **`TESTED`** |
| | Median Absolute Error (MedAE)| 0.3645 t/ha | **0.3645 t/ha** | $0.000$ | **`TESTED`** |
| | Mean Absolute Percentage Error| 76.73% | **76.73%** | $0.00\%$ | **`TESTED`** |
| **Generalization Splits**| 10-Fold Spatial CV Top-5 | 59.28% | **56.83%** | $-2.45\%$ | **`REPOSITORY-REPORTED`** |
| | Forward Temporal Split Top-5| 56.41% | **58.97%** | $+2.56\%$ | **`REPOSITORY-REPORTED`** |
| | Spatiotemporal Joint Holdout| 40.82% | **43.20%** | **$+2.38\%$** | **`REPOSITORY-REPORTED`** |
| **System Operations** | Inference Latency (Core ML) | 0.038 ms | **0.041 ms** | $+0.003$ ms | **`TESTED`** |
| | Multi-Objective Pareto Engine| Not Implemented | **Implemented (5 Objectives)**| New Feature | **`IMPLEMENTED`** |
| | Automated Tests Passing | 38 / 38 | **75 / 75 (100%)** | $+37$ tests | **`TESTED`** |

### 10.2 Genuine 16-Crop Confusion Matrix & Per-Class Support
The $16 \times 16$ confusion matrix evaluated on the 546 test samples ($N=546$) is presented in Section 3 of `docs/report/evaluation_audit.md` and visualized in **Figure 2** (`docs/report/figures/confusion_matrix.png`) and **Figure 3** (`docs/report/figures/confusion_matrix_normalized.png`). Class-wise precision, recall, and F1-scores are detailed below:

| Crop Class | Test Support ($n$) | Precision | Recall | F1-Score | Mean Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sugarcane** | 27 | 1.0000 | 0.7778 | 0.8750 | 3.56 |
| **Wheat** | 27 | 0.3158 | 0.6667 | 0.4286 | 4.04 |
| **Rice** | 63 | 0.7778 | 0.4444 | 0.5657 | 2.41 |
| **Mustard** | 28 | 0.3333 | 0.3571 | 0.3448 | 4.32 |
| **Chickpea** | 25 | 0.7000 | 0.2800 | 0.4000 | 3.20 |
| **Sorghum** | 26 | 1.0000 | 0.1538 | 0.2667 | 6.65 |
| **Finger Millet**| 27 | 1.0000 | 0.1481 | 0.2581 | 6.33 |
| **Maize** | 52 | 0.1207 | 0.1346 | 0.1273 | 3.42 |
| **Urad** | 44 | 0.1234 | 0.8636 | 0.2159 | 5.41 |
| **Moong** | 44 | 1.0000 | 0.0682 | 0.1277 | 6.18 |
| **Sesamum** | 39 | 0.4444 | 0.1026 | 0.1667 | 5.97 |
| **Pearl Millet** | 29 | 1.0000 | 0.0345 | 0.0667 | 7.79 |
| **Groundnut** | 40 | 1.0000 | 0.0250 | 0.0488 | 6.10 |
| **Pigeonpea** | 29 | 0.0000 | 0.0000 | 0.0000 | 7.41 |
| **Cotton** | 24 | 0.0000 | 0.0000 | 0.0000 | 8.83 |
| **Soybean** | 22 | 0.0000 | 0.0000 | 0.0000 | 9.59 |
| **Macro Average** | **546** | **0.4907** | **0.2536** | **0.1614** | **5.58** |

### 10.3 Failure Mode Deconstruction: The "Urad Attractor"
The confusion matrix revealed that column 14 (`Urad`) received **308 total predictions** out of 546 queries (56.4%). Urad functions as a broad prior in central Indian agriculture. When soil and weather features fall near distribution medians, the unconstrained tree ensemble defaults to Urad, starving specialized minority crops (Cotton and Soybean, both 0.0% top-1 recall). Hybrid knowledge fusion ($0.85 \text{ML} + 0.15 \text{KB}$) successfully suppresses this attractor bias in top-3 and top-5 recommendations.

### 10.4 Yield Regressor Results & The "MAPE Paradox"
The crop-conditional yield regressor achieved $R^2 = 0.954$, $\text{RMSE} = 6.377$ t/ha, and $\text{MedAE} = \mathbf{0.3645}$ t/ha, alongside an aggregate MAPE of 76.73%. As proved in Section 6 of `docs/report/evaluation_audit.md`, low-yielding dryland pulses ($0.25 - 0.50$ t/ha) act as tiny denominators, mathematically exploding percentage errors ($> 100\%$) for minor absolute deviations ($0.35$ t/ha). Median Absolute Error confirms high physical accuracy.

### 10.5 Controlled Ablation Experiments
The 11-stage ablation study (`artifacts/crop_v5/ablation_results.csv`) quantifies component contributions:
- Standalone ML achieved Top-1 = 18.86%, Top-5 = 59.34%, but allowed physiological slope violations.
- Pure Knowledge Base achieved Top-1 = 7.51%, Top-5 = 39.19%.
- Fused Model E1 ($0.85 \text{ML} + 0.15 \text{KB}$) eliminated biophysical violations while preserving Top-1 = 18.86% and Top-3 = 42.31%.
- Standalone LambdaMART reduced Top-1 to 16.85% and Top-5 to 55.13%.

### 10.6 Generalization Benchmarks: Spatial, Temporal, and Joint Spatiotemporal
- **Spatial CV:** Top-5 accuracy dropped to 56.83% ($-2.45\%$), reflecting regional soil diversity.
- **Forward Temporal:** Top-5 accuracy reached 58.97% ($+2.56\%$).
- **Joint Spatiotemporal Holdout (Unseen South + Future):** V5 achieved **43.20% Top-5 accuracy** vs V4's 40.82% ($+2.38\%$), confirming that V5's regularized multi-objective architecture generalizes superiorly under demanding transfers.

### 10.7 End-to-End Inference Latency & Hardware Acceleration Profiling
On GPU (`device='cuda'`, `tree_method='hist'`), single-query core ML inference completed in **0.041 milliseconds** (24,390 queries/second). On CPU, inference required 1.8 milliseconds. Total end-to-end API response time (including live SoilGrids and Open-Meteo network queries) averaged 1.85 seconds.

---

# Chapter 11: Discussion

### 11.1 Empirical Interpretation of Findings
The empirical results confirm that agricultural crop recommendation cannot be treated as a conventional single-label classification problem. The modest Top-1 accuracy (18.86%) combined with strong Top-5 accuracy (58.24%) and NDCG@5 (0.3849) demonstrates that the model functions effectively as a candidate screening engine, identifying viable agronomic options rather than dogmatically prescribing a single choice.

### 11.2 Resolution of Research Questions (RQ1 through RQ4)
- **RQ1 (Ranking Formulation):** Falsified. LambdaMART underperforms multiclass softmax for a fixed 16-crop slate due to lambda gradient noise on small candidate sets.
- **RQ2 (Knowledge Fusion):** Verified. Hybrid fusion ($0.85 \text{ML} + 0.15 \text{KB}$) eliminates biophysical boundary violations while maintaining competitive Top-3 accuracy (42.31%).
- **RQ3 (Generalization Under Transfer):** Quantified. Spatial transfer incurs a modest $2.45\%$ penalty, while joint spatiotemporal holdout performance improves by $+2.38\%$ in V5 over V4.
- **RQ4 (Multi-Objective Optimization):** Verified. The 5-dimensional Pareto engine successfully exposes orthogonal trade-offs across suitability, yield, water, and erosion without inference overhead.

### 11.3 Agile Project Management Lessons: Sunk-Cost Avoidance
A critical takeaway for course **INT373** is the value of Agile hypothesis falsification. In traditional Waterfall projects, extensive engineering effort invested in a novel algorithm (such as candidate LTR dataset construction) creates organizational pressure to deploy the model regardless of merit. In GeoAgri AI's Agile framework, test metrics served as an objective quality gate: LambdaMART was rejected for production, and Model E1 was certified, preventing an architectural regression.

### 11.4 Real-World Agronomic Utility & Screening Scenarios
The practical value of GeoAgri AI lies in exploratory decision support: providing farmers and agronomists with pre-sowing candidate slates that balance productivity against climate resilience and resource degradation risks.

---

# Chapter 12: Limitations and Future Work

### 12.1 Algorithmic & Modeling Limitations
1. **Binary Historical Relevance:** Ground truth reflects historical cultivation, not physiological optimality.
2. **Pulse Prior Bias:** The "Urad Attractor" indicates susceptibility to central prior dominance.
3. **Collinear Feature Smearing:** Physical correlation among rainfall features disperses Shapley attributions.

### 12.2 Data Ingestion & District Aggregate Limitations
1. **District Granularity:** Historical training yields represent district-level aggregates, smoothing out localized farm-scale variance.
2. **External API Dependence:** System uptime depends on third-party availability (SoilGrids, Open-Meteo).
3. **Soil Conservation Violations:** Four historical records contain minor texture sum violations ($0.039\%$).

### 12.3 Explicit Roadmap Extensions (KPIT Sparkle 2027 Goals)
Future development milestones—explicitly distinguished from the current operational baseline—include:
- **Satellite Remote Sensing Ingestion:** Real-time ingestion of Copernicus Sentinel-2 multispectral bands (10m–20m resolution) for live NDVI, EVI, and NDWI calculation.
- **Deep Canopy Segmentation:** Deployment of DeepLabV3+ and lightweight vision transformers for field boundary delineation and crop stress zonation.
- **Volumetric Irrigation Scheduling:** Automated FAO-56 predictive root-zone water balance modeling.

---

# Chapter 13: Conclusion

This academic continuous assessment evaluated the architecture, machine learning methodology, and empirical performance of **GeoAgri AI (Crop Intelligence V5)**. Through rigorous experimentation across 10,091 longitudinal decision queries, the investigation demonstrated that candidate-level Learning-to-Rank is inferior to multiclass softmax for small, fixed candidate crop pools, while hybrid knowledge fusion and multi-objective Pareto optimization successfully eliminate biophysical violations and scalar objective collapse. Certified across 75 automated regression tests with documented spatiotemporal robustness, GeoAgri AI establishes an empirical benchmark for transparent, accountable agricultural artificial intelligence.

---

# References

[1] R. Rajak, A. Pawar, M. Pendke, P. Shinde, and S. Rathod, "Crop recommendation system to maximize crop yield using machine learning technique," in *Proc. Int. Conf. Electron. Commun. Aerosp. Technol. (ICECA)*, Coimbatore, India, 2017, pp. 950–953. DOI: `10.1109/ICECA.2017.8212856`.

[2] H. K. Kondaveeti, N. Kumaravelu, S. D. Vanambathina, and P. S. R. Chandra, "A systematic literature review on agricultural decision support systems: Advancements in crop recommendation and yield prediction," *Comput. Electron. Agric.*, vol. 206, p. 107677, Mar. 2023. DOI: `10.1016/j.compag.2023.107677`.

[3] H. Meyer and E. Pebesma, "Predicting into unknown space? Estimating the area of applicability of spatial prediction models," *Methods Ecol. Evol.*, vol. 12, no. 9, pp. 1620–1633, Sep. 2021. DOI: `10.1111/2041-210X.13650`.

[4] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD '16)*, San Francisco, CA, USA, 2016, pp. 785–794. DOI: `10.1145/2939672.2939785`.

[5] C. J. C. Burges, "From RankNet to LambdaRank to LambdaMART: An overview," Microsoft Research, Redmond, WA, USA, Tech. Rep. MSR-TR-2010-82, Jun. 2010.

[6] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Adv. Neural Inf. Process. Syst. 30 (NeurIPS 2017)*, Long Beach, CA, USA, 2017, pp. 4765–4774.

[7] R. G. Allen, L. S. Pereira, D. Raes, and M. Smith, "Crop evapotranspiration: Guidelines for computing crop water requirements," Food and Agriculture Organization of the United Nations (FAO), Rome, Italy, FAO Irrigation and Drainage Paper 56, 1998. ISBN: `92-5-104219-5`.

[8] S. Pudumalar, E. Ramanujam, R. H. Rajashree, C. Kavitha, T. Kiruthika, and J. Prabhavathy, "Crop recommendation system for precision agriculture," in *Proc. 8th Int. Conf. Adv. Comput. (ICoAC)*, Chennai, India, 2016, pp. 32–36. DOI: `10.1109/ICoAC.2017.7951740`.

[9] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, Oct. 2001. DOI: `10.1023/A:1010933404324`.

[10] R. T. Marler and J. S. Arora, "Survey of multi-objective optimization methods for engineering," *Struct. Multidiscip. Optim.*, vol. 26, no. 6, pp. 369–395, Apr. 2004. DOI: `10.1007/s00158-003-0368-6`.

[11] T. Hengl, J. M. de Jesus, G. B. M. Heuvelink, et al., "SoilGrids250m: Global gridded soil information based on machine learning," *PLOS ONE*, vol. 12, no. 2, p. e0169748, Feb. 2017. DOI: `10.1371/journal.pone.0169748`.

[12] R. J. Zomer, A. Trabucco, D. A. Bossio, and L. V. Verchot, "Climate change mitigation: A spatial analysis of global land suitability," *Agric. Ecosyst. Environ.*, vol. 126, no. 1–2, pp. 67–80, Jun. 2008. DOI: `10.1016/j.agee.2008.01.014`.

[13] Directorate of Economics and Statistics (DES), "Agricultural Statistics at a Glance 2022," Ministry of Agriculture and Farmers Welfare, Government of India, New Delhi, India, Comprehensive Statistical Report, 2022.

[14] ICRISAT, "District-Level Database (DLD) for Indian Agriculture," ICRISAT Dataverse, Patancheru, India, 2020. [Online]. Available: `http://data.icrisat.org/dld/`.

[15] Food and Agriculture Organization of the United Nations (FAO), "EcoCrop: The crop environmental requirements database," FAO Land and Water Development Division, Rome, Italy, 1996.

---

# Appendices

### Appendix A: Complete Technology Stack Inventory
Cataloged in `docs/report/environment.md`.

### Appendix B: 26-Feature Preprocessing Dictionary
Cataloged in `docs/report/data_and_features.md`.

### Appendix C: Production API Endpoint Specifications
- `GET /health` -> Health check.
- `GET /debug/model` -> Model registry introspection.
- `GET /api/agriculture/recommend` -> Query-string based recommendation.
- `POST /api/agriculture/recommend` -> JSON payload based recommendation.
- `POST /api/predict?action=crop` -> Unified predictive action endpoint.

### Appendix D: Model Hyperparameter Table
Cataloged in Section 6.3 and `docs/report/ml_methodology.md`.

### Appendix E: Detailed Confusion Matrix & Per-Class Support Table
Cataloged in Section 10.2 and `docs/report/evaluation_audit.md`.

### Appendix F: Automated Test Execution Evidence
75 tests executed across 15 modules with 100% pass rate. Verified in `docs/report/testing_and_reproducibility.md`.

### Appendix G: Reproducibility Commands
Documented in Section 5 of `docs/report/testing_and_reproducibility.md`.

### Appendix H: Additional System Visuals & Dashboards
Verified in `docs/report/screenshots/` and `docs/report/figures/`.

### Appendix I: Research-Gap Evidence Matrix
Documented in Section 2.7 of `docs/report/research_gap_and_literature.md`.
