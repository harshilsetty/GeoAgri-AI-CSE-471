# Machine Learning Methodology & Algorithmic Architecture

**Course Assessment:** INT373 — Agile Driven Development and Project Management  
**Project:** GeoAgri AI (Crop Intelligence V5)  
**Deliverable:** Machine Learning Methodology, Model Formulations, and Technical Justifications (`docs/report/ml_methodology.md`)

---

## 1. Formal Separation of Prediction Tasks

A critical methodological requirement of GeoAgri AI is the explicit disentanglement of four distinct mathematical tasks. In prior agricultural literature, these tasks are frequently conflated into a single vague "suitability score." In GeoAgri AI V5, they are mathematically separated:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   DISCRETE ML & DECISION TASKS                         │
├─────────────────────────┬──────────────────────────────────────────────┤
│ Task 1: Classification  │ Multiclass probability distribution P(c | x) │
│ Task 2: Candidate LTR   │ Listwise ranking relevance score R(c, q)     │
│ Task 3: Yield Potential │ Continuous expected yield Y(x, c) in t/ha     │
│ Task 4: Decision Fusion │ Multi-Objective Pareto Trade-Off Frontier    │
└─────────────────────────┴──────────────────────────────────────────────┘
```

1. **Multiclass Probability Generation ($P_{\text{ML}}$):** The empirical likelihood $P(y = c \mid \mathbf{x})$ that crop $c \in \{1, \dots, 16\}$ was historically observed in environmental context $\mathbf{x}$, output via a normalized softmax distribution over the 16 classes ($\sum_{c=1}^{16} P(c \mid \mathbf{x}) = 1.0$).
2. **Physiological Compatibility ($S_{\text{KB}}$):** A deterministic, biophysically bounded suitability index in $[0.0, 1.0]$ derived from non-negotiable FAO EcoCrop physiological rules (pH bounds, rainfall minimums, slope limits).
3. **Candidate-Level Ranking Relevance ($R_{\text{LTR}}$):** A continuous score learned by LambdaMART optimizing Discounted Cumulative Gain directly on candidate slates ($16$ candidates per query block).
4. **Crop-Conditional Yield Regression ($\hat{Y}$):** A continuous regression estimate $\hat{y} \in [0.0, \infty)$ of realized harvested crop mass per unit land area (tonnes per hectare, t/ha), conditioned on one-hot candidate crop identity.
5. **Multi-Objective Pareto Dominance:** Identification of the non-dominated candidate subset across five orthogonal evaluation dimensions without scalar collapse.

---

## 2. Technical Justification for XGBoost as Core Tabular Architecture

XGBoost (Extreme Gradient Boosting; Chen & Guestrin, 2016) was selected as the foundational machine learning engine across classification, ranking, and regression. This decision is grounded in both theoretical properties and empirical realities of agricultural tabular data:

### 1. Robustness to Heterogeneous and Engineered Tabular Features
The 26-dimensional agronomic feature vector contains a mixture of continuous chemical concentrations (N, P, K), unbounded geomorphic gradients (elevation, slope), bounded derived ratios (WSI, SFI), and discrete integer encodings (`season_code`, `texture_code`). Unlike Deep Neural Networks (MLPs, Transformers), which require meticulous monotonic normalizations and are prone to uncalibrated over-smoothing on small tabular datasets ($N \approx 10^4$), gradient-boosted decision trees are invariant to monotonic feature scaling and naturally partition heterogeneous feature spaces.

### 2. Built-in Regularization and Generalization Controls
Agricultural data contains inherent environmental noise and spatial autocorrelation. XGBoost incorporates second-order Taylor expansion of the loss function paired with explicit $L_1$ (Lasso, $\alpha$) and $L_2$ (Ridge, $\lambda$) leaf weight penalties:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
This objective directly penalizes tree complexity ($T$) and extreme leaf weights ($w$), significantly reducing overfitting when evaluated across geographically distinct regions.

### 3. TreeSHAP Compatibility for Real-Time Local Explanations
Because agricultural recommendations require agronomic auditability, model predictions must be accompanied by exact local feature attributions. XGBoost tree ensembles are fully compatible with TreeSHAP (Lundberg & Lee, 2017), which computes exact Shapley values in polynomial time ($\mathcal{O}(T L D^2)$) rather than the exponential time required by model-agnostic KernelSHAP.

### 4. GPU-Accelerated Histogram Building
Using the `hist` tree method accelerated via CUDA on an NVIDIA GeForce RTX 3050 Laptop GPU, candidate-level dataset training (161,456 samples) completed in 4.3 seconds—an approximate **$14.2\times$ speedup** over CPU-bound exact greedy splitting.

---

## 3. Detailed Model Formulations & Hyperparameters

### 1. Multiclass Crop Classifier (Model A & Production Core)
- **Objective Function:** `multi:softprob` (Multinomial Logistic Loss with Softmax Output).
- **Target Variable:** Integer encoded crop class $y \in \{0, 1, \dots, 15\}$.
- **Input Dimension:** 26 continuous & encoded features.
- **Hyperparameter Configuration:**
  - `n_estimators`: 120 trees
  - `max_depth`: 5
  - `learning_rate` ($\eta$): 0.05
  - `subsample`: 0.85 (row subsampling to combat spatial noise)
  - `colsample_bytree`: 0.80 (feature subsampling)
  - `tree_method`: `hist`
  - `device`: `cuda` (with automatic fallback to `cpu`)
  - `random_state`: 42

### 2. Candidate-Level LambdaMART Ranker (Model D & V5 LTR)
- **Objective Function:** `rank:ndcg` (LambdaMART optimizing Normalized Discounted Cumulative Gain).
- **Group Structure:** Grouped queries with exactly 16 candidate items per query block.
- **Input Dimension:** 43 features (26 base query features repeated + 16 candidate one-hot identifiers + 1 knowledge base score).
- **Target Relevance:** Binary relevance $y_{q, c} \in \{0, 1\}$ indicating historical observed cultivation.
- **Hyperparameter Configuration:**
  - `n_estimators`: 100 trees
  - `max_depth`: 5
  - `learning_rate`: 0.05
  - `subsample`: 0.85
  - `ndcg_exp_gain`: `false`
  - `tree_method`: `hist`
  - `device`: `cuda`

### 3. Crop-Conditional Yield Regressor (Model C & Preserved V4/V5)
- **Objective Function:** `reg:squarederror` (Mean Squared Error Loss).
- **Target Variable:** Continuous crop yield in metric tonnes per hectare ($\text{t/ha}$).
- **Input Dimension:** 42 features (26 base environmental features + 16 candidate crop one-hot flags).
- **Hyperparameter Configuration:**
  - `n_estimators`: 150 trees
  - `max_depth`: 6
  - `learning_rate`: 0.04
  - `subsample`: 0.80
  - `colsample_bytree`: 0.85
  - `tree_method`: `hist`
  - `device`: `cuda`

---

## 4. Empirical Evaluation of the Learning-to-Rank Hypothesis

### The Central Hypothesis
In designing Crop Intelligence V5, the core theoretical hypothesis was that **candidate-level Learning-to-Rank (LambdaMART optimizing NDCG) would superiorly rank candidate crops compared to standard multiclass softmax probability generation**. 

The rationale was that multiclass softmax assumes mutually exclusive categorical choices (only one crop is "correct"), whereas agricultural planning requires ranking multiple simultaneously viable crops against non-smooth listwise metrics (NDCG@5, MRR).

### The Empirical Result: Hypothesis Falsification
The table below presents the verified results from `artifacts/crop_v5/ranking_model_comparison.csv` evaluated on the held-out chronological test set (2019–2020, $N=546$ decision queries):

| Model Architecture & Formulation | Top-1 Acc | Top-3 Acc | Top-5 Acc | NDCG@3 | NDCG@5 | MRR | MAP@5 | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Multiclass Softmax (Base)** | **18.86%** | 42.12% | **59.34%** | **0.3198** | **0.3899** | **0.3705** | **0.3236** | **0.1618** |
| **Model B: Pointwise Binary Compatibility** | 16.48% | 39.56% | 55.13% | 0.2970 | 0.3608 | 0.3510 | 0.2984 | 0.0900 |
| **Model C: Pairwise LTR (`rank:pairwise`)** | 17.40% | 38.28% | 55.13% | 0.2924 | 0.3626 | 0.3528 | 0.3021 | 0.0852 |
| **Model D: Listwise LambdaMART (`rank:ndcg`)** | 16.85% | 37.73% | 55.13% | 0.2873 | 0.3580 | 0.3471 | 0.2951 | 0.1293 |
| **Model E1: Multiclass Fused (0.85 ML + 0.15 KB)** | **18.86%** | **42.31%** | 58.24% | 0.3200 | 0.3849 | 0.3689 | 0.3204 | 0.1614 |
| **Model E2: LambdaMART Fused (0.85 LTR + 0.15 KB)** | 14.47% | 37.18% | 51.65% | 0.2738 | 0.3327 | 0.3278 | 0.2725 | 0.1177 |

### Scientific Analysis of Why LambdaMART Underperformed
1. **Candidate Set Fixed Cardinality ($K=16$):** Learning-to-Rank algorithms (LambdaMART) excel in search engines where queries match variable-length candidate document pools ($10^2 - 10^4$ items) with dense relevance gradations ($0$ to $4$). In this agronomic formulation, the candidate set is fixed at exactly 16 discrete crops per query, and historical ground-truth relevance is binary (1 observed crop, 15 unobserved crops).
2. **Loss Gradient Dynamics:** For a fixed candidate set with binary relevance, the listwise lambda gradient $\lambda_{ij} = |\Delta \text{NDCG}| \cdot \frac{-1}{1 + e^{s_i - s_j}}$ produces noisy pair updates across the 15 negative items, whereas multiclass cross-entropy directly normalizes the denominator across all 16 classes simultaneously via the exact softmax partition function.
3. **Decisive Engineering Decision:** In strict adherence to scientific integrity, Model E1 (Multiclass Fusion) was maintained as the primary ranking backbone for production inference. LambdaMART was retained within the codebase as an audited benchmark, preventing regressions.

---

## 5. Knowledge Fusion Formulations Across System Layers

A critical source of potential confusion is the existence of two distinct scoring equations in different layers of the repository. This audit documents each formulation separately:

### Layer 1: Evaluation & Benchmarking Fusion (Model E1)
Implemented in `scripts/train_crop_v5.py` and `scripts/train_crop_v4.py` for comparative model evaluation:
$$\text{Score}_{\text{Eval}}(c) = 0.85 \cdot P_{\text{ML}}(c \mid \mathbf{x}) + 0.15 \cdot S_{\text{KB}}(c \mid \text{soil}, \text{weather}, \text{slope})$$
- **$P_{\text{ML}}(c \mid \mathbf{x})$:** Calibrated probability from the 26-feature XGBoost classifier.
- **$S_{\text{KB}}(c)$:** Normalized physiological compatibility score $[0.0, 1.0]$ based on FAO EcoCrop rule violations.
- **Grid Search Optimality:** Systematic grid search across fusion ratios ($1.0/0.0$, $0.9/0.1$, $0.85/0.15$, $0.8/0.2$, $0.7/0.3$) proved that $0.85 / 0.15$ achieved optimal balance, stabilizing Top-3 accuracy (42.31%) while eliminating unviable crops.

### Layer 2: Operational Runtime Domain Suitability Engine
Implemented in `src/models/agri_inference.py` for user-facing API recommendations:
$$\text{Suitability}_{\text{Runtime}}(c) = w_{\text{model}} \cdot P_{\text{ML}}(c) + w_{\text{soil}} \cdot S_{\text{soil}}(c) + w_{\text{season}} \cdot S_{\text{season}}(c) + w_{\text{climate}} \cdot S_{\text{climate}}(c) + w_{\text{erosion}} \cdot S_{\text{erosion}}(c)$$
Where domain weights are configured as:
$$w_{\text{model}} = 0.45, \quad w_{\text{soil}} = 0.20, \quad w_{\text{season}} = 0.15, \quad w_{\text{climate}} = 0.10, \quad w_{\text{erosion}} = 0.10$$
$$\sum w_i = 1.00$$
- **Component Sub-Scores:**
  - $S_{\text{soil}}(c)$: pH optimality score based on distance from crop-specific optimal pH range $[pH_{\min}, pH_{\max}]$.
  - $S_{\text{season}}(c)$: Binary indicator (1.0 if planned season is listed in crop calendar, 0.0 otherwise).
  - $S_{\text{climate}}(c)$: Seasonal rainfall compliance against crop minimum/maximum water requirements.
  - $S_{\text{erosion}}(c)$: Slope gating ($1.0 - \text{erosion\_risk}$ if slope $\le \text{max\_slope}$, else heavily penalized).

---

## 6. Multi-Objective Pareto Decision Engine

To solve the scalar objective collapse problem, GeoAgri AI V5 introduces an operational Pareto frontier engine. For every candidate crop $c$, five evaluation vectors are computed:
1. **Objective 1 ($\max f_1$):** Agronomic Suitability Score $[0.0, 1.0]$.
2. **Objective 2 ($\max f_2$):** Expected Harvested Yield $\hat{Y}(c)$ in t/ha.
3. **Objective 3 ($\min f_3$):** Water Stress Risk (normalized WSI).
4. **Objective 4 ($\min f_4$):** Soil Erosion Vulnerability (terrain slope $\times$ geomorphic erosion probability).
5. **Objective 5 ($\max f_5$):** Decision Margin (distance between candidate probability and second-best alternative).

### Pareto Dominance Criterion
Candidate crop $A$ dominates candidate crop $B$ ($A \succ B$) if and only if:
$$\forall i \in \{1, \dots, 5\}, \quad f_i(A) \ge f_i(B) \quad \text{and} \quad \exists j \in \{1, \dots, 5\}, \quad f_j(A) > f_j(B)$$
The non-dominated candidates form the **Pareto-Optimal Frontier**:
$$\mathcal{P}^* = \{ c \in \mathcal{C} \mid \nexists c' \in \mathcal{C} : c' \succ c \}$$

### Supported Farmer Decision Profiles
Farmers select customized decision profiles that adjust the scalarized projection of the Pareto frontier:
- **Profile 1: Suitability-Focused** (Default; maximizes biophysical compatibility).
- **Profile 2: Yield-Focused** (Weights commercial productivity $\hat{Y}$ heavily; suitable for irrigated, capital-intensive farms).
- **Profile 3: Water-Constrained** (Strictly minimizes WSI; prioritizes millets, pulses, and drought-hardy crops).
- **Profile 4: Risk-Averse** (Minimizes combined erosion and climate shock vulnerability).
- **Profile 5: Balanced** (Equal harmonic weighting across all five dimensions).
