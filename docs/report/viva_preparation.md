# Academic Viva Voce & Technical Defense Preparation Guide

**Course Assessment:** INT373 — Agile Driven Development and Project Management  
**Project:** GeoAgri AI (Crop Intelligence V5)  
**Deliverable:** Comprehensive Viva Voce Defense Question Bank with Defensible Evidence-Backed Answers (`docs/report/viva_preparation.md`)

---

## Section 1: Agile Driven Development & Project Management (INT373 Core)

### Q1: How does this project reflect Agile Driven Development principles rather than a traditional Waterfall approach?
**Defensible Answer:**
"GeoAgri AI was developed through iterative, test-driven sprints characterized by continuous empirical validation across five distinct model versions (V1 through V5). Rather than waiting for a monolithic system release, each sprint delivered an independently certified, backward-compatible increment:
- **Sprint V1/V2:** Established the foundational 26-feature pipeline and baseline multiclass XGBoost models.
- **Sprint V3:** Integrated agronomic knowledge rules (FAO EcoCrop) to prevent biophysical violations.
- **Sprint V4:** Certified yield regression ($R^2 = 0.954$) and spatiotemporal holdout benchmarks.
- **Sprint V5:** Formulated and tested candidate-level Learning-to-Rank and introduced the Multi-Objective Pareto Engine.
Crucially, Agile principles were demonstrated when empirical testing **falsified** our core LTR hypothesis. Instead of succumbing to sunk-cost fallacy, the team made an evidence-driven pivot: LambdaMART was rejected for production, and Model E1 (Multiclass Fusion) was certified, preventing an architectural regression."

### Q2: What major technical debt did you uncover during your system audit, and how would an Agile team manage it in the product backlog?
**Defensible Answer:**
"Our audit uncovered a critical architectural coupling defect: the monolithic bridge anti-pattern in `ui/server/predict_bridge.py`. The script retained legacy imports (`cv2`, `torch`, `YOLO`, `smp`) from an earlier archaeological drone project, even though pure tabular crop recommendation has zero dependency on deep computer vision. In environments lacking PyTorch or OpenCV, the worker subprocess silently crashed on startup.
In an Agile backlog, this technical debt is prioritized as an immediate Refactoring Spike in Sprint Planning. The mitigation strategy decouples `agri_inference.py` completely from vision shims, wraps optional vision imports in deferred `try...except` blocks, and updates `requirements.txt` to reflect true minimal dependencies."

---

## Section 2: Machine Learning Methodology & The LTR Benchmark

### Q3: Why did you hypothesize that LambdaMART would outperform Multiclass Softmax, and why was that hypothesis empirically falsified?
**Defensible Answer:**
"The hypothesis was that crop recommendation is fundamentally a ranking problem where farmers evaluate slates of viable crops, not a single-label classification problem. LambdaMART optimizes non-smooth listwise metrics (NDCG) directly via lambda gradients.
However, LambdaMART underperformed (NDCG@5 = 0.3580 vs Multiclass NDCG@5 = 0.3899; Top-1 = 16.85% vs 18.86%). The mathematical reason is twofold:
1. **Fixed Small Candidate Set ($K=16$):** Learning-to-Rank algorithms are designed for web search queries where candidate document pools are variable, massive ($10^2 - 10^4$), and graded ($0 - 4$). Here, every query evaluates exactly the same 16 discrete crops with binary historical ground-truth (1 observed, 15 unobserved).
2. **Softmax Partition Function:** The multiclass cross-entropy loss normalizes probabilities across all 16 mutually competing crops simultaneously in the softmax denominator, generating smoother, better-calibrated gradients than pairwise lambda gradient approximations."

### Q4: Your Top-1 classification accuracy is only 18.86%. Isn't that an unacceptably weak machine learning model?
**Defensible Answer:**
"In agricultural decision support, Top-1 accuracy is an inappropriate and misleading metric. Unlike digit recognition (MNIST), where an image of a '3' can never be a '5', agricultural land is biophysically capable of supporting multiple crops simultaneously. A fertile alluvial plain in Punjab during Rabi season can successfully support Wheat, Mustard, Chickpea, or Barley.
Evaluating the system as a single-label classifier falsely penalizes the model if it recommends Mustard when historical records happened to report Wheat. Therefore, agricultural decision support is evaluated on candidate slate viability:
- **Top-3 Accuracy = 42.31%**
- **Top-5 Accuracy = 58.24%** (significantly higher than random guessing, which is $5/16 = 31.25\%$).
The goal of GeoAgri AI is exploratory candidate screening, presenting farmers with a viable slate filtered by physiological feasibility and multi-objective trade-offs."

---

## Section 3: Failure Modes, Bias & The "MAPE Paradox"

### Q5: In your confusion matrix, Urad was predicted 308 times out of 546 test queries (56.4%). What caused this 'Urad Attractor' error?
**Defensible Answer:**
"The 'Urad Attractor' is a classic empirical class-prior bias. In the training dataset, Urad (Black Gram) was grown across a wide variety of states, moderate rainfall bands, and neutral pH soils. The unconstrained tree ensemble learned that whenever environmental indicators were near the central mean, predicting Urad maximized cross-entropy reward with minimal penalty. This led to false-positive dilution across specialized crops like Cotton and Soybean.
This exact failure mode proved why pure machine learning is insufficient for agriculture and justified our **Knowledge Fusion Layer (0.85 ML + 0.15 KB)**. The knowledge layer injects physiological specificity, penalizing generic pulses and boosting specialized cash crops when their specific soil and seasonal conditions are satisfied."

### Q6: Your yield regressor reports an $R^2$ of 0.954, but a Mean Absolute Percentage Error (MAPE) of 76.73%. How can both be true?
**Defensible Answer:**
"This is the well-documented 'MAPE Paradox' caused by low-yield denominator inflation. 
In Indian agriculture, low-yielding dryland pulses and oilseeds (Moong, Sesamum, Urad) yield only **0.25 to 0.60 tonnes per hectare**. If the true yield is $0.30 \, \text{t/ha}$ and the model predicts $0.65 \, \text{t/ha}$, the absolute error is only **0.35 t/ha**—an agronomically modest deviation. However, because $0.30$ is in the denominator:
$$\text{Percentage Error} = \frac{|0.30 - 0.65|}{0.30} \times 100\% = 116.7\%$$
Because pulses constitute over 45% of our test records, these small denominators artificially explode aggregate MAPE to 76.73%. Conversely, the **Median Absolute Error is 0.3645 t/ha**, proving that for over half of all predictions, the error is within one-third of a tonne per hectare. For academic reporting, MedAE and normalized RMSE are statistically and agronomically superior."

---

## Section 4: Generalization, Spatiotemporal Auditing & Data Leakage

### Q7: Why did you evaluate spatiotemporal holdouts rather than standard $k$-fold cross-validation?
**Defensible Answer:**
"As proved by Meyer & Pebesma (2022) in *Methods in Ecology and Evolution*, standard randomized cross-validation splits in geospatial machine learning suffer from severe spatial data leakage. Agricultural plots located in the same district share identical soil profiles, weather series, and elevation gradients. If samples from the same district are randomly assigned to both train and test folds, the model merely memorizes localized coordinates rather than learning transferable biophysical rules.
We implemented three non-random validation benchmarks:
1. **Leave-State-Out Spatial CV (Top-5 = 56.83%):** Completely excludes entire states to test cross-border transfer.
2. **Forward Temporal Holdout (Top-5 = 58.97%):** Trains on $\le 2017$, tests strictly on future years $2019–2020$.
3. **Spatiotemporal Joint Holdout (Top-5 = 43.20%):** Excludes the Southern region entirely in future years.
While metrics on these holdouts are lower than random splits, they represent true real-world operational generalization."

---

## Section 5: Multi-Objective Pareto Decision Intelligence

### Q8: What is the Multi-Objective Pareto Engine, and why is it superior to scalar weighting?
**Defensible Answer:**
"Traditional agricultural systems collapse all criteria into a single weighted sum ($\text{Score} = w_1 S + w_2 Y - w_3 W$). This scalar collapse forces arbitrary trade-offs: a farmer is given a single recommendation without knowing if the crop achieved its rank because of high yield or low water consumption.
GeoAgri AI V5 treats five dimensions as independent, orthogonal objectives: $\max \text{Suitability}$, $\max \text{Yield}$, $\min \text{Water Risk}$, $\min \text{Erosion Risk}$, and $\max \text{Decision Margin}$.
Using the Pareto dominance criterion ($A \succ B$), the engine identifies the non-dominated candidate subset—crops where no alternative provides superior performance across all dimensions simultaneously. Farmers then project this frontier through customized decision profiles (e.g., Water-Constrained for drylands vs Yield-Focused for commercial irrigated farms)."

---

## Section 6: Explainability, LLM Guardrails & Academic Scope

### Q9: Why does a positive SHAP value not prove biological causality?
**Defensible Answer:**
"TreeSHAP computes the marginal contribution of a feature to the model's mathematical prediction relative to the expected background rate. It reflects empirical correlation in the training distribution, not biological cause-and-effect. For instance, if historical data shows that high rainfall correlates with Rice cultivation, SHAP assigns positive attribution to rainfall. However, SHAP cannot determine whether rainfall directly nourished the crop or whether farmers planted Rice because subsidized canal water was released during high-rainfall years. Furthermore, collinear features (such as seasonal rainfall and 90-day rainfall) suffer from attribution smearing across tree splits."

### Q10: How do you prevent the integrated LLM (Groq Llama-3.1) from hallucinating crop advice?
**Defensible Answer:**
"The LLM is completely isolated from the numerical inference pipeline. It operates strictly downstream as a natural-language synthesizer:
1. **No Predictive Authority:** The LLM cannot recommend crops, alter rankings, or calculate suitability scores.
2. **Constrained Prompt Context:** The prompt injects strictly pre-computed values (exact crop name, suitability percentage, yield range, positive drivers from TreeSHAP, and counterfactual simulation results).
3. **Negative Constraint Enforcement:** The system prompt explicitly forbids absolute language (e.g., 'guaranteed harvest' or 'best crop') and restricts length to 3–4 evidence-grounded sentences.
4. **Deterministic Fallback:** If the Groq API key is missing or network times out, the system generates a local rule-based template without breaking execution."

### Q11: What is the exact boundary between what is implemented and what is proposed in your repository?
**Defensible Answer:**
"In strict adherence to academic honesty:
- **IMPLEMENTED & TESTED:** The 26-feature agronomic preprocessing pipeline, XGBoost multiclass model, candidate ranking benchmark, crop-conditional yield regressor, FAO EcoCrop knowledge fusion layer, 5-objective Pareto decision engine, TreeSHAP explainer, FastAPI backend, and Next.js Leaflet UI.
- **PROPOSED / ROADMAP:** Satellite remote sensing ingestion (Sentinel-2 multispectral bands), CNN/Transformer canopy segmentation (DeepLabV3+), and automated volumetric irrigation scheduling. These are future milestones targeted for KPIT Sparkle 2027 and are not operational in the current codebase."
