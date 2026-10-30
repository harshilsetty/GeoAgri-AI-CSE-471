# Explainable AI (XAI) and Large Language Model (LLM) Integration

**Course Assessment:** INT373 — Agile Driven Development and Project Management  
**Project:** GeoAgri AI (Crop Intelligence V5)  
**Deliverable:** Explainability Architecture, TreeSHAP Feature Attribution, LLM Narrative Synthesis, and Trust & Safety Audit (`docs/report/explainability_and_llm.md`)

---

## 1. Dual-Layer Explainability Paradigm

Agricultural practitioners and extension agronomists distrust pure opaque "black-box" predictions. At the same time, raw numerical feature attributions (e.g., $+0.142$ log-odds on `water_stress_index`) are uninterpretable to farmers. GeoAgri AI V5 resolves this tension through a **dual-layer explainability architecture**:

```
[Raw Environmental Vector]
           │
           ▼
┌──────────────────────────────────────┐
│  LAYER 1: Mathematical Ground Truth  │
│  • TreeSHAP Exact Shapley Attribution│
│  • FAO EcoCrop Physiological Rules   │
│  • Directional Counterfactual Shifts │
└──────────────────┬───────────────────┘
                   │  Rigidly constrained structured context
                   ▼
┌──────────────────────────────────────┐
│  LAYER 2: Natural-Language Synthesis │
│  • Groq Cloud API (Llama-3.1-8b)     │
│  • Temperature = 0.2 (Low variance)  │
│  • Deterministic Rule-Based Fallback │
└──────────────────────────────────────┘
```

1. **Layer 1 (Mathematical & Agronomic Attribution):** Exact TreeSHAP computes mathematical feature importance, while deterministic FAO EcoCrop boundary rules categorize inputs into positive growth drivers and physiological limiting factors.
2. **Layer 2 (Natural Language Reasoning & Synthesis):** An LLM translates the structured mathematical attribution into an accessible, non-prescriptive plain-language summary for the farmer, strictly bound to the numerical values computed in Layer 1.

---

## 2. Global and Local Feature Attribution via TreeSHAP

### Mathematical Formulation
TreeSHAP (Lundberg & Lee, 2017) defines the contribution $\phi_i$ of each feature $i \in \{1, \dots, M\}$ to model prediction $f(\mathbf{x})$ using the unique additive feature attribution method satisfying Efficiency, Symmetry, Dummy, and Additivity axioms:
$$\phi_i(f, \mathbf{x}) = \sum_{S \subseteq \mathcal{F} \setminus \{i\}} \frac{|S|! (M - |S| - 1)!}{M!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$
The model output is decomposed into the expected base rate $\phi_0$ and the sum of feature effects:
$$f(\mathbf{x}) = \phi_0 + \sum_{i=1}^M \phi_i(\mathbf{x})$$

### Global Feature Importance Audit
Evaluating `models/crop/v5/crop_model.pkl` across all 120 boosted trees reveals the dominant drivers of the crop suitability manifold:

| Rank | Feature Code Name | Average Gain per Split | Agronomic Interpretation |
| :---: | :--- | :---: | :--- |
| `1` | `season_code` | **184.2** | Primary seasonal filter; defines vegetative thermal window (Kharif vs Rabi). |
| `2` | `rainfall_season` | **142.6** | Primary hydrological constraint dividing wetland (Rice) from dryland (Millets). |
| `3` | `temperature_mean` | **118.9** | Thermal energy ceiling separating tropical crops from temperate cereals (Wheat). |
| `4` | `soil_ph` | **94.5** | Pedological chemical boundary; governs micronutrient bioavailability. |
| `5` | `elevation` | **82.1** | Geomorphic proxy for lapse rate cooling, drainage, and terrain altitude. |
| `6` | `water_stress_index` | **76.4** | Composite drought stress indicator ($ET_0$ vs available moisture). |
| `7` | `soil_moisture` | **68.2** | Root-zone volumetric water availability during initial sowing. |
| `8` | `organic_carbon` | **61.3** | Biological fertility proxy; buffers moisture and cation exchange. |
| `9` | `nitrogen` | **54.8** | Vegetative canopy growth nutrient rating. |
| `10`| `slope` | **49.7** | Topographic runoff and operational mechanization threshold. |

*Visual Evidence:* Refer to `docs/report/figures/feature_importance.png` for the top 15 split-gain chart.

### Critical Epistemic Caveat: Attribution is NOT Biological Causation
In strict adherence to scientific rigor, the report must disclose two fundamental limitations of SHAP in environmental modeling:
1. **Correlation vs Causation:** A positive Shapley value indicates that high values of a feature empirically correlate with historical cultivation in the training dataset. It **does not prove** that artificially altering that feature (e.g., adding nitrogen fertilizer) will cause a proportional increase in crop viability.
2. **Feature Collinearity & Attribution Smearing:** Environmental variables (such as `rainfall_season`, `rainfall_90d`, and `soil_moisture`) exhibit strong physical collinearity ($r > 0.75$). In tree ensembles, correlated features split arbitrary nodes across different trees, distributing Shapley attribution across the collinear group rather than isolating a single physical mechanism.

---

## 3. Real Local Explanation Example from Live Model Output

The following local explanation was captured during real runtime execution on candidate location Latitude 15.335°N, Longitude 76.460°E (Karnataka semi-arid zone, Kharif season):

### 1. Ingested Environmental Context
- Soil pH: $7.8$ (Slightly Alkaline)
- Soil Texture: Silt Loam
- Seasonal Rainfall: $70 \, \text{mm}$ (Arid / Drought Deficit)
- Topographic Slope: $13.0^\circ$ (Moderate Inclination)
- Reference $ET_0$: $4.6 \, \text{mm/day}$
- Calculated Water Stress Index (WSI): $1.84$ (High Evaporative Deficit)

### 2. Candidate Evaluation & Feature Attribution
- **Rank 1 Candidate:** **Cotton** ($\text{Suitability} = 27.7\%$, $\text{Yield Estimate} = 1.28 \pm 0.97 \, \text{t/ha}$)
- **Positive Driving Factors (Attribution $\phi > 0$):**
  1. Planned season (`Kharif`) aligns perfectly with Cotton vegetative phenology.
  2. Soil pH ($7.8$) is within the optimal physiological tolerance band ($6.2 - 8.2$).
  3. High genetic drought tolerance buffers against severe seasonal rainfall deficit ($70 \, \text{mm}$).
- **Limiting Factors & Risk Penalties (Attribution $\phi < 0$):**
  1. Topographic slope ($13.0^\circ$) exceeds optimal threshold ($10.0^\circ$), incurring erosion vulnerability penalty.
  2. Sub-optimal silt fraction impairs vertisol deep-rooting anchorage.
- **Decision Margin:** $+4.2\%$ separation over second-ranked candidate (Sesamum, $23.5\%$), classified as **"Moderate Stability"**.

---

## 4. Directional Counterfactual Analysis

To explain *what would alter the recommendation*, the inference engine executes five deterministic sensitivity perturbations (`agri_inference.evaluate_counterfactuals`):

| Perturbation Scenario | Simulated Condition | Predicted Top Crop | Simulated Suitability | Rank Shift Status | Agronomic Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Drier Climate Shock** | $-30\%$ Seasonal Rainfall | Cotton | 22% | No Shift | Cotton retains top ranking due to exceptional drought tolerance. |
| **Wetter Climate Anomaly** | $+30\%$ Seasonal Rainfall | Cotton | 21% | No Shift | Additional rain insufficient to elevate water-intensive cereals over Cotton. |
| **Soil Acidification** | $\text{pH} - 0.5$ ($7.8 \to 7.3$) | Cotton | 21% | No Shift | Remains well within Cotton's neutral-to-alkaline physiological range. |
| **Alkaline Shift** | $\text{pH} + 0.5$ ($7.8 \to 8.3$) | Cotton | 21% | No Shift | Approaches upper alkalinity boundary; yields slight confidence decay. |
| **Supplemental Irrigation** | $+200 \, \text{mm}$ Assured Water | Maize / Rice | 34% | **RANK SHIFT** | Assured irrigation shifts optimal candidate from dryland Cotton to high-yield Maize. |

---

## 5. Large Language Model (LLM) Integration & Safety Architecture

### Provider & Model Specification
- **Cloud Inference Endpoint:** Groq Cloud API (`https://api.groq.com/openai/v1/chat/completions`)
- **Foundational Model:** `llama-3.1-8b-instant` (Meta Llama 3.1 architecture)
- **Decoding Temperature:** $0.20$ (Configured explicitly for factual, low-variance synthesis)
- **Top-P Sampling:** $0.90$
- **Request Timeout:** 25.0 seconds

### Strict Prompt Injection & Anti-Hallucination Guardrails
To prevent the LLM from fabricating agronomic claims or altering model predictions, the LLM is restricted to a **pure natural-language formatting role**. The system prompt and user context are rigidly constrained:

```
[SYSTEM PROMPT]:
"You are an expert agricultural researcher providing decision intelligence."

[STRUCTURED USER CONTEXT]:
Location: Lat {lat}, Lon {lon} | Season: {season}
Highest-Ranked Crop: {primary_crop}
Suitability: {suitability_percent}% | Confidence: {confidence_percent}% | Risk: {risk}
Expected Yield: {expected_yield_range} (Model Estimate)
Ranking Stability: {ranking_stability} (Margin: {decision_margin}%)
Why Ranked Highest: {positive_drivers}
Limiting Factors: {limiting_factors}
Alternatives: {alternative_crops}
Soil: pH {soil_ph}, Texture {soil_texture}
Forecast Weather: 7d Rain {rain_7d} mm (Anomaly: {rain_anomaly} mm), Temp {temp}°C
Erosion Risk: {erosion_risk_label} ({erosion_risk_score}%)

[NEGATIVE CONSTRAINT ENFORCEMENT]:
"Generate an evidence-grounded summary (3-4 sentences).
Do NOT claim guaranteed harvest or use absolute words like 'best crop'.
Frame as highest-ranked candidate under available evidence."
```

### Deterministic Fallback Mechanism (Zero-Key Independence)
If the user or deployment environment lacks a `GROQ_API_KEY`, or if network timeouts / HTTP 429 rate-limit errors occur, the application executes a zero-dependency deterministic fallback (`src/explainability/ai_explainer.py:generate_ai_explanation`):

```python
fallback_narrative = (
    f"{primary_crop['crop']} is the highest-ranked candidate under available data with "
    f"{primary_crop['suitability_percent']}% suitability and {primary_crop['confidence_percent']}% confidence "
    f"for the {season} season. The model estimates an expected yield of {primary_crop['expected_yield_range']}, "
    f"reflecting favorable agro-climatic indicators: {', '.join(primary_crop['positive_drivers'][:2])}. "
    f"Environmental risk is rated as {primary_crop['risk']} due to {terrain_profile['erosion_risk_label'].lower()} "
    f"erosion context. Alternatives include {', '.join([r['crop'] for r in top_recommendations[1:4]])}."
)
```
- **Audit Finding:** The system returns `insight_mode = "local"` and `insight_status = "Missing GROQ API key"` without breaking the JSON API response or crashing the frontend client.

### Non-Validation Disclosure
**The academic report must clearly state:** The LLM does **not** validate, audit, or recalculate the agricultural recommendations. It operates purely downstream of the XGBoost classifier, knowledge fusion engine, and TreeSHAP explainer, acting as a natural-language synthesizer.
