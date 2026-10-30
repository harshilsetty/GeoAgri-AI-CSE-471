# End-to-End Software & Systems Architecture

**Course Assessment:** INT373 — Agile Driven Development and Project Management  
**Project:** GeoAgri AI (Crop Intelligence V5)  
**Deliverable:** End-to-End System Architecture, Component Specifications, and Technical Debt Audit (`docs/report/architecture.md`)

---

## 1. System Context & High-Level Architecture

GeoAgri AI is structured as a decoupled, multi-tier agricultural intelligence application comprising an interactive GIS Next.js frontend, a dual-mode API serving layer (FastAPI direct backend + Next.js Node.js worker bridge), and an asynchronous geospatial ML inference engine.

```mermaid
graph TD
    User["Farmer / Agricultural Researcher"] -->|HTTPS (Browser)| UI["Next.js 14 Web Application (Port 3000)"]
    
    subgraph Frontend_Layer ["Presentation Tier (Next.js / React)"]
        UI --> GIS["Leaflet GIS MapPicker (Dynamic SSR: False)"]
        UI --> Panel["CropIntelligencePanel & HeroRiskCard"]
        UI --> Profiles["Decision Profile Selector (5 Profiles)"]
    end

    subgraph API_Bridge_Tier ["Serving & Dispatch Tier"]
        UI -->|POST /api/predict| NextRoute["Next.js Route Handler (route.ts)"]
        NextRoute -->|JSON over Stdin/Stdout IPC| PyWorker["Python Worker Subprocess (predict_worker.py)"]
        DirectClient["API Client / Automated Tests"] -->|HTTP REST (Port 8000)| FastAPI["FastAPI Backend (backend_api.py)"]
    end

    subgraph Core_ML_Engine ["Inference & Intelligence Tier (src/)"]
        PyWorker --> AgriInference["Crop Decision Engine (src/models/agri_inference.py)"]
        FastAPI --> AgriInference
        AgriInference --> Preprocessing["26-Feature Preprocessing (agri_features.py)"]
        AgriInference --> GeoIngestion["Geospatial Ingestion (SoilGrids, Open-Meteo, Open-Elevation)"]
        AgriInference --> Models["Model Registry (Crop V5, Yield, Erosion)"]
        AgriInference --> KB["FAO EcoCrop Knowledge Base (JSON Rules)"]
        AgriInference --> Pareto["Multi-Objective Pareto Engine"]
        AgriInference --> XAI["TreeSHAP & Counterfactual Explainer"]
    end

    subgraph External_Services ["External Cloud Services"]
        GeoIngestion --> SoilGridsAPI["ISRIC SoilGrids REST API"]
        GeoIngestion --> MeteoAPI["Open-Meteo Weather API"]
        GeoIngestion --> ElevAPI["Open-Elevation SRTM DEM"]
        XAI --> GroqAPI["Groq Cloud API (Llama-3.1-8b)"]
    end
```

---

## 2. Component Deconstruction & Technical Stack

The table below catalogs every technology in the actual implementation, its architectural responsibility, and verified integration status:

| Technology / Component | Version | Subsystem / Role | Architectural Justification | Evidence Path in Codebase |
| :--- | :---: | :--- | :--- | :--- |
| **Next.js** | 14.2.3 | Frontend Application Framework | Server-Side Rendering (SSR) for static landing layouts; client-side React hydration for interactive GIS Leaflet maps and stateful decision profiling. | `ui/package.json`<br>`ui/app/page.tsx` |
| **React & TypeScript** | 18.x / 5.x | UI Component Library & Type Safety | Strongly-typed props and state management across complex agricultural recommendation payloads. | `ui/tsconfig.json`<br>`ui/components/` |
| **Tailwind CSS** | 3.4.1 | UI Styling & Design System | Dark-slate responsive dashboard interface (`#0B0F19`) with accessible typography, micro-animations, and status badges. | `ui/tailwind.config.ts`<br>`ui/app/globals.css` |
| **Leaflet & React-Leaflet** | 1.9.4 | Geospatial Map Interaction | Interactive open-source coordinate picking, spatial marker placement, and agro-climatic zone visualization. | `ui/components/MapPicker.tsx` |
| **FastAPI** | 0.142.2 | Production Python REST Service | High-performance asynchronous OpenAPI 3.0 compliant endpoints (`/api/agriculture/recommend`). | `src/api/backend_api.py`<br>`backend_api.py` |
| **Pydantic** | 2.13.5 | Request/Response Validation | Schema validation and type enforcement for inbound latitude, longitude, and season parameters. | `requirements.txt`<br>`src/api/backend_api.py` |
| **Python** | 3.13.3 | Core ML & Preprocessing Runtime | Unified environment for feature engineering, gradient boosting, and scientific computing. | `pyproject.toml` |
| **XGBoost** | 3.4.1 | Tabular ML (Classification, Ranking, Yield)| Gradient-boosted decision trees with GPU histogram acceleration (`tree_method='hist'`, `device='cuda'`). | `models/crop/v5/`<br>`src/models/agri_inference.py` |
| **Scikit-Learn** | 1.9.1 | Preprocessing & Metrics Evaluation | Categorical label encoding, spatial cross-validation splitting, Top-$K$ accuracy, and NDCG scoring. | `scripts/train_crop_v5.py` |
| **TreeSHAP** | 0.52.0 | Exact Local Feature Attribution | Polynomial-time Shapley feature attribution for tabular decision trees. | `src/models/agri_inference.py` |
| **Groq Cloud API** | Cloud | Natural Language Narrative Synthesis | Ultra-low latency LLM inference executing Meta Llama-3.1-8b-instant for conversational recommendations. | `src/explainability/ai_explainer.py` |
| **Pytest** | 9.1.1 | Automated Test Suite | Regression testing across model registry, constraint boundaries, and data pipelines. | `tests/`<br>`pyproject.toml` |

---

## 3. Detailed Subsystem Workflows

### 1. Geospatial Data Acquisition Pipeline
When a user selects coordinates $L = (\text{lat}, \text{lon})$:
1. `src/geospatial/fetch_soil.py`: Queries ISRIC SoilGrids for sand, silt, clay, pH, nitrogen, and organic carbon at 0–30 cm depths. If the remote service times out (> 8.0s) or fails, the module executes a fallback to the nearest historical district centroid in `data/agriculture/v1.0/raw/lat_lon_india_district.csv`.
2. `src/geospatial/fetch_weather.py`: Queries Open-Meteo for 10-year historical climatology and 7-day numerical weather predictions (temperature, precipitation series, relative humidity, $ET_0$).
3. `src/geospatial/fetch_terrain.py`: Derives elevation and cardinal gradient slope (degrees) using Open-Elevation SRTM grids.
4. `terrain_model/erosion_model.pkl`: Predicts geomorphic erosion risk probability from slope, vegetation proxy, elevation, rainfall, and soil class.

### 2. Feature Engineering & Invariant Ingestion
`src/preprocessing/agri_features.py` ingests raw environmental dictionaries and calculates the 26-dimensional model vector:
- Computes **Soil Fertility Index (SFI)** using ICAR benchmarks.
- Computes **Water Stress Index (WSI)** and **Seasonal Rainfall Deviation ($\Delta R$)**.
- Encodes USDA soil texture class and planned cropping season into discrete categorical codes.

### 3. Inference, Knowledge Gating, and Multi-Objective Pareto Filtering
1. `models/crop/v5/crop_model.pkl` outputs a 16-class probability distribution $P_{\text{ML}}$.
2. `src/models/agri_inference.py` loads `CROP_ECOLOGICAL_RULES` and scores biophysical compatibility $S_{\text{KB}}$ for each crop.
3. Hybrid suitability is calculated ($0.85 \cdot P_{\text{ML}} + 0.15 \cdot S_{\text{KB}}$).
4. `models/crop/v5/yield_model.pkl` predicts expected harvested yield $\hat{Y}(c)$ in t/ha.
5. The Pareto engine evaluates non-dominated candidates across the 5-objective frontier (Suitability, Yield, Water, Erosion, Margin) and maps results to the active farmer decision profile.
6. TreeSHAP extracts positive growth drivers and physiological limiting factors.
7. Downstream Groq Llama-3.1 API formats the structured decision payload into a concise 3-sentence advisory narrative.

---

## 4. Architectural Root Cause of the "Model Not Able to Predict" Bug

During Phase 1 auditing, a severe architectural coupling defect was identified that directly explains why the user encountered prediction failures:

```
[Next.js App / Client]
         │  POST /api/predict
         ▼
[ui/app/api/predict/route.ts]
         │  spawn("python", ["server/predict_worker.py"])
         ▼
[ui/server/predict_worker.py]
         │  line 5: from predict_bridge import ...
         ▼
[ui/server/predict_bridge.py]
         │  line 8:  import cv2
         │  line 11: import torch
         │  line 12: from ultralytics import YOLO
         │  line 13: import segmentation_models_pytorch as smp
         ▼
   CRASH ON STARTUP! ──> [ModuleNotFoundError: No module named 'cv2' / 'torch']
```

### Technical Post-Mortem & Agile Project Management Analysis
1. **The Monolithic Bridge Anti-Pattern:**  
   `predict_bridge.py` was originally authored for a previous computer-vision archaeological site mapping project (`Archaeological-Site-Mapping-AI-V4`). When the repository migrated to GeoAgri AI, agricultural recommendation functions (`recommend_crops`, `agri_inference`) were added into `predict_bridge.py` without decoupling legacy computer-vision dependencies (`cv2`, `torch`, `YOLO`, `smp`).
2. **Incomplete Dependency Specification (`requirements.txt`):**  
   The project's `requirements.txt` specified core tabular dependencies (`xgboost`, `pandas`, `scikit-learn`, `fastapi`), but completely omitted heavy deep-learning frameworks (`torch`, `ultralytics`, `opencv-python`).
3. **Silent Worker Process Failure:**  
   In `ui/app/api/predict/route.ts`, when `predict_worker.py` is spawned, Python immediately aborts with `ModuleNotFoundError` during module import. Because the worker process died before emitting the `{"type": "ready"}` handshake, all incoming prediction requests hung until hitting the 240-second timeout or returning a generic 500 error.
4. **Agile Refactoring Recommendation:**  
   To eliminate this technical debt permanently without altering core algorithms, legacy vision imports in `predict_bridge.py` must either be isolated in a dedicated vision package or wrapped in safe, deferred `try...except ImportError` blocks. Pure tabular agricultural inference (`agri_inference.py`) has zero physical dependency on `cv2` or `torch`.

---

## 5. Actual vs Planned Capability Boundary

The diagram below explicitly delineates implemented operational modules from planned roadmap features:

```mermaid
graph LR
    subgraph Operational_Core ["IMPLEMENTED & TESTED (Operational V5)"]
        A1["26-Feature Preprocessing Pipeline"]
        A2["XGBoost 16-Crop Multiclass Engine"]
        A3["XGBoost Yield Potential Regressor"]
        A4["FAO EcoCrop Knowledge Fusion Layer"]
        A5["5-Objective Pareto Decision Engine"]
        A6["TreeSHAP Local Feature Attribution"]
        A7["Groq Llama-3.1 Synthesis & Fallback"]
        A8["FastAPI REST Endpoints & Next.js UI"]
    end

    subgraph Proposed_Roadmap ["PLANNED / ROADMAP (Not Implemented)"]
        B1["Sentinel-2 Multispectral Imagery Ingestion"]
        B2["DeepLabV3+ Field Canopy Segmentation"]
        B3["Automated Volumetric Irrigation Scheduling"]
        B4["UAV Drone Orthomosaic Aerial Pipeline"]
        B5["Farmer Mobile App & Field Validation"]
    end

    Operational_Core -.->|Future Extensions| Proposed_Roadmap
```
