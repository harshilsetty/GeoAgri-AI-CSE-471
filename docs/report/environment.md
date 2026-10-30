# Execution Environment & Dependency Specification

**Course Assessment:** INT373 — Agile Driven Development and Project Management  
**Project:** GeoAgri AI (Crop Intelligence V5)  
**Deliverable:** Hardware, Operating System, Runtime, and Dependency Environment Specification (`docs/report/environment.md`)

---

## 1. Hardware Architecture & Acceleration Environment

The experimental benchmarks, training pipelines, and evaluation runs were executed on a certified local workstation environment:

| Hardware Component | Verified Workstation Specification | Technical Operational Role |
| :--- | :--- | :--- |
| **Host Machine / Architecture** | Windows 11 Enterprise x64 (Build 26100) | Primary development and assessment operating system. |
| **Central Processing Unit (CPU)**| 13th Gen Intel(R) Core(TM) i5-13450HX (16 logical cores) | Feature engineering, TreeSHAP CPU calculations, API serving. |
| **Physical Memory (RAM)** | 16.0 GB DDR5 High-Speed Memory | In-memory candidate ranking dataset construction (161,456 rows). |
| **Dedicated Graphics Processing Unit (GPU)**| **NVIDIA GeForce RTX 3050 6GB Laptop GPU** | Accelerated gradient boosting tree induction. |
| **Dedicated Video Memory (VRAM)**| 6,144 MiB (6.0 GB GDDR6) | GPU histogram memory pool (`tree_method='hist'`). |
| **NVIDIA Graphics Driver** | Version 581.86 | Hardware abstraction interface. |
| **CUDA Runtime Version** | CUDA 13.0 | Accelerated linear algebra and tree split calculation. |
| **Acceleration Benchmark Ratio**| **14.2x Speedup** over CPU greedy splitting | Verified in `artifacts/crop_v5/gpu_verification.txt`. |

---

## 2. Software Runtime Environment

| Runtime / Tool | Verified Version | Installation Path / Execution Scope |
| :--- | :---: | :--- |
| **Python** | `3.13.3` | `.venv\Scripts\python.exe` (Isolated project environment) |
| **Node.js** | `v20.12.2` | System PATH (Next.js server-side runtime & SSR engine) |
| **npm** | `10.5.0` | Node Package Manager for UI dependencies |
| **Git** | `2.45.0` | Version control & branch management |
| **PowerShell** | `7.4.x / 5.1` | Primary administrative and execution shell |

---

## 3. Python Package Environment (Verified Virtual Environment)

The packages installed in `.venv` (verified via `pip list`) are cataloged below:

| Package Identifier | Installed Version | Architectural Role / Dependency Justification |
| :--- | :---: | :--- |
| **`xgboost`** | `3.4.1` | Extreme Gradient Boosting for classification, ranking, and regression. |
| **`numpy`** | `2.5.3` | High-performance multidimensional array computing and linear algebra. |
| **`pandas`** | `3.0.6` | Tabular dataset manipulation, CSV ingestion, and time-series masks. |
| **`scikit-learn`** | `1.9.1` | Metric evaluation (Top-$K$, NDCG, F1), LabelEncoder, cross-validation. |
| **`scipy`** | `1.18.1` | Statistical distributions, Kolmogorov-Smirnov test, Population Stability Index. |
| **`shap`** | `0.52.0` | Game-theoretic TreeSHAP exact local feature attribution. |
| **`fastapi`** | `0.142.2` | Production asynchronous REST API backend. |
| **`uvicorn`** | `0.54.0` | High-performance ASGI web server for FastAPI. |
| **`pydantic`** | `2.13.5` | Inbound request data parsing and response schema validation. |
| **`requests`** | `2.34.2` | Synchronous HTTP client for SoilGrids, Open-Meteo, Open-Elevation, and Groq. |
| **`joblib`** | `1.6.0` | High-efficiency disk serialization for trained tree models and label encoders. |
| **`matplotlib`** | `3.11.2` | Publication-grade scientific plotting (confusion matrices, bar charts). |
| **`seaborn`** | `0.13.2` | Statistical heatmaps and matrix visualization styling. |
| **`python-docx`** | `1.2.0` | Automated academic DOCX report generation. |
| **`reportlab`** | `5.1.0` | Automated PDF document compilation and rendering. |
| **`opencv-python-headless`**| `5.0.0.93` | Headless computer vision library resolving legacy bridge imports. |
| **`pytest`** | `9.1.1` | Automated regression test execution framework. |
| **`pytest-asyncio`** | `1.4.0` | Asynchronous test execution runner for FastAPI endpoints. |
| **`httpx`** | `0.28.1` | HTTP test client library for FastAPI endpoint testing. |

---

## 4. Frontend Node.js Dependencies (`ui/package.json`)

The UI dashboard dependencies installed in `ui/node_modules` are cataloged below:

| NPM Package | Target Version | UI Role / Responsibility |
| :--- | :---: | :--- |
| **`next`** | `14.2.3` | React full-stack framework with App Router architecture. |
| **`react`** | `^18` | Declarative UI rendering library. |
| **`react-dom`** | `^18` | DOM rendering interface for React. |
| **`typescript`** | `^5` | Static type checking across frontend components. |
| **`tailwindcss`** | `^3.4.1` | Utility-first CSS framework adhering to dark-mode design system. |
| **`leaflet`** | `^1.9.4` | Open-source mobile-friendly interactive maps. |
| **`react-leaflet`** | `^4.2.1` | React bindings for Leaflet map elements. |
| **`framer-motion`** | `^11.1.7` | Production-ready motion animation engine for interactive transitions. |
| **`lucide-react`** | `^0.378.0` | High-contrast accessible iconography. |
| **`html2canvas`** | `^1.4.1` | Client-side visual dashboard snapshot rasterization. |
| **`jspdf`** | `^2.5.1` | Client-side PDF export generation for farmer advisory reports. |

---

## 5. Environment Variables & Security Audit

The operational configuration of the system is managed through environment variables:

| Environment Variable | Configured Value / Example | Security Classification | Default Behavior If Absent |
| :--- | :--- | :---: | :--- |
| **`GROQ_API_KEY`** | `gsk_...` (User-provided secret) | **SENSITIVE SECRET** | Gracefully falls back to local deterministic rule narrative (`insight_mode = "local"`). |
| **`PYTHON_EXECUTABLE`** | `.venv\Scripts\python.exe` | System Path Config | Defaults to `python` found on system PATH. |
| **`PREDICT_BACKEND_URL`** | `http://localhost:8000` | Network Configuration | Empty string (routes prediction to local Python worker subprocess). |
| **`GEO_AI_CROP_MODEL_VERSION`**| `v5` (or `v4`, `v3`, `v2`, `v1`) | Model Registry Config | Defaults to `v2` baseline if unconfigured. |

### Security & Non-Disclosure Audit
1. **Zero Secret Leakage:** No API keys, Groq authorization bearer tokens, or cloud credentials are hardcoded within source files or git history.
2. **Git Exclusions (`.gitignore`):** Virtual environments (`.venv/`, `env/`), API key environment files (`.env`, `.env.local`), compiled bytecode (`__pycache__/`, `*.pyc`), and Next.js build directories (`ui/.next/`) are strictly excluded from version control.
