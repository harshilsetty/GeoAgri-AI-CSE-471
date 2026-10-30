# AGP 12 — Production Safety, Isolation Verification & Release Readiness

## 1. Executive Summary

This report assesses the production readiness of GeoAgri-AI's Computer Vision and Archaeology pipeline following AGP 12 experimentation. It provides a formal audit of production model registry isolation, inspects application integration endpoints, evaluates candidate models against production deployment gates, and outlines the required verification protocol for future release cycles.

---

## 2. Production Registry & Isolation Verification

### 2.1 Active Configuration Audit
An exhaustive inspection of [`config/models_registry.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/config/models_registry.yaml) confirms:
* **Active Production Vision Model:** Pinned to `yolov8s_archaeology_prod`.
* **Production Checkpoint Integrity:** The production artifact resides at `models/yolov8s_archaeology.pt` and has not been modified, replaced, or deleted.
* **Experimental Namespace Isolation:** All YOLO11 OBB runs are cataloged strictly under the experimental namespace:
  - `geoagri_yolo11n_obb_v1`: Baseline unweighted model
  - `geoagri_yolo11n_obb_v2_refit`: EXP-B (geometry refit)
  - `geoagri_yolo11n_obb_v3_domain_aug`: EXP-C (domain-shift augmentation)
  - `geoagri_yolo11n_obb_v4_class_weighted`: EXP-D (class-weighted OBB)
  - `geoagri_yolo11s_obb_v5_class_weighted`: EXP-E (capacity scaling candidate)

### 2.2 Integration Server Safety
The integration bridge [`ui/server/predict_bridge.py`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/ui/server/predict_bridge.py) queries `models_registry.yaml` dynamically via `active_models.production_vision`. Because this reference remains untouched, live application inference continues serving production traffic through validated production weights without exposure to unstable experimental runs.

---

## 3. Candidate Promotion Decision Gates

To qualify for consideration as a replacement for `production_vision`, any candidate model must satisfy five non-negotiable gates:

| Gate | Criterion | EXP-D Performance | EXP-E Projection | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1: Minimum Val mAP@50** | $\ge 0.4000$ | **0.4150** | Under training | **PASS (EXP-D)** |
| **Gate 2: Structures Val Recall** | $\ge 0.5000$ | **0.7711** | Under training | **PASS (EXP-D)** |
| **Gate 3: Multi-Site Generalization** | Test mAP@50 $\ge 0.2500$ across unseen sites | **0.1289** (Failed) | Quarantined / Untested | **BLOCKED (All Models)** |
| **Gate 4: Inference Latency** | $\le 35.0$ ms per tile on RTX 3050 GPU | **11.1 ms** (PASS) | Projected $\approx 18-22$ ms | **PASS** |
| **Gate 5: External Benchmark** | Validation on 120-scene independent benchmark | Not yet collected | Not yet collected | **PENDING DATASET** |

### Release Assessment: NO PRODUCTION PROMOTION (HOLD)
* While EXP-D achieved a major technical milestone by breaking the zero-recall barrier on minority structures ($0.7711$ on val, $0.0716$ on test vs $0.0000$ prior), its cross-site generalization on held-out scenes ($0.1289$ mAP@50) remains well below the production threshold ($0.2500$).
* The root cause of generalization degradation is **severe geographic and geomorphological confounding in the dataset** (Hampi granite vs Khajuraho sandstone vs Nalanda brick), which cannot be solved by model capacity or loss functions alone.
* Therefore, **no candidate model will be promoted to production**. `yolov8s_archaeology_prod` remains the active production vision model.

---

## 4. Next Deployment Roadmap

1. **Maintain Experimental Branching:** Catalog EXP-E in `config/models_registry.yaml` under `experimental_models` upon completion.
2. **Execute Field Data Collection:** Collect and annotate the 120+ scene independent dataset as specified in `docs/agp12_data_collection_protocol.md`.
3. **Train on Balanced Multi-Site Distribution:** Retrain YOLO11 OBB using cross-site stratified sampling.
4. **Independent Benchmark Gate:** Run candidate evaluation exclusively against the new held-out Multi-Site Archaeological Benchmark (AGB-120).

