# AGP 11 — Production Safety and Release Readiness Report

## 1. Executive Summary

AGP 11 conducted a comprehensive root-cause error diagnosis and designed, verified, and executed **EXP-D (`geoagri_yolo11n_obb_v4_class_weighted`)** using bounded, damped inverse square-root class weighting.
* In validation, EXP-D established an unprecedented recall benchmark of **$77.11\%$ on `structures`** and **$44.08\%$ overall recall**.
* In held-out test evaluation, EXP-D successfully broke the historical zero-recall barrier on unseen Nalanda structures, achieving **$7.16\%$ recall** and **$0.0417$ mAP@50** (up from $0.0000$ in all previous baseline, EXP-B, and EXP-C runs).
* In strict adherence to production gating rules, **EXP-D remains categorized as experimental and is NOT promoted to production**. The active production model remains securely locked at **`yolov8s_archaeology_prod`**.

---

## 2. Model Registry Isolation Audit

The model registry [`config/models_registry.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/config/models_registry.yaml) was audited to confirm active vs. experimental slot boundaries:

```yaml
active_models:
  production_vision: "yolov8s_archaeology_prod"
  candidate_vision: "geoagri_yolo11n_obb_v1"
  production_erosion: "terrain_model/erosion_model.pkl"
```

### Registered Experimental Candidates:
1. `geoagri_yolo11n_obb_v1`: Baseline OBB model (40.0% val mAP50, 14.1% test mAP50, 0.0% test struct recall).
2. `geoagri_yolo11n_obb_v2_refit` (EXP-B): Geometric boundary refit (43.1% val mAP50, 60.3% precision, 63.9% val struct recall).
3. `geoagri_yolo11n_obb_v3_aug` (EXP-C): Domain augmentation model (45.0% val mAP50, 72.2% val struct recall, 11.4% test mAP50).
4. `geoagri_yolo11n_obb_v4_class_weighted` (EXP-D): Damped class-weighted model (41.5% val mAP50, **77.1% val struct recall**, **44.1% overall recall**, **7.16% test struct recall**).

All experimental checkpoints reside under `runs/obb/` without overwriting production weights.

---

## 3. Automated Test Suite Verification

A total of 15 targeted integration, weighting, hardening, and geometric isolation tests were executed via pytest:
* `tests/test_agp11_class_weighting.py` (4 passed): Verified unweighted baseline parity, `(1, 1, 5)` weight tensor broadcast, finite gradients with zero minority instances, and clean lifecycle hook attachment.
* `tests/test_yolo_obb_hardening.py` (8 passed): Verified production registry resolution, model loading, CPU fallback, API output contracts, zero-detection handling, invalid input rejection, and 8-point OBB coordinate serialization.
* `tests/test_agp09_geometry_and_isolation.py` (3 passed): Verified 100% boundary coordinate validity, polygon convexity, and production slot isolation.
* **Status:** **15 passed, 0 failed in 42.38 seconds.**

---

## 4. Gating Decision Matrix

| Operational Criterion | Required Threshold | Measured Outcome | Gate Status |
| :--- | :--- | :--- | :---: |
| **Production Model Intact** | Zero modification to active model | `yolov8s_archaeology_prod` verified active | **PASSED** |
| **Loss Weighting Safety** | Finite gradients, 0 NaNs | Verified in test_3 and across all 28 epochs | **PASSED** |
| **Structures Generalization** | Non-zero recall on held-out test split | Recall rose from 0.0000 to **0.0716** | **MILESTONE ACHIEVED** |
| **Overall Test mAP@50** | Target $> 0.25$ for production promotion | Test mAP is **0.1289** | **HELD BACK** |
| **CPU Fallback Functional** | Inference executes on CPU | Verified via `test_8_cpu_inference_fallback` | **PASSED** |
| **Promotion Decision** | Explicit sign-off required | **Do NOT promote EXP-D to production** | **ENFORCED** |

