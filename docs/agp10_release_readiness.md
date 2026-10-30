# AGP 10 — Production Safety and Release Readiness Report

## 1. Executive Summary

In AGP 10, two controlled experiments (**EXP-B** and **EXP-C**) were executed and evaluated on the NVIDIA GeForce RTX 3050 Laptop GPU. 
* While EXP-C demonstrated state-of-the-art validation performance ($0.4497$ mAP@50, $0.2638$ mAP@50–95, and $72.2\%$ structures recall), its held-out test split performance dropped to **$0.1138$ mAP@50** due to extreme architectural and material domain shift across unseen sites.
* In strict adherence to production gating rules, **EXP-C was NOT promoted to production**.
* The production model remains quarantined and locked at **`yolov8s_archaeology_prod`**.

---

## 2. Model Registry Audit

The central registry [`config/models_registry.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/config/models_registry.yaml) was audited to verify production slot isolation:

```yaml
active_models:
  production_vision: "yolov8s_archaeology_prod"
  candidate_vision: "geoagri_yolo11n_obb_v1"
  production_erosion: "terrain_model/erosion_model.pkl"
```

### Registry Status Summary:
* **Active Production Vision Model:** `yolov8s_archaeology_prod` (Unchanged, axis-aligned detection baseline).
* **Experimental Candidates:**
  * `geoagri_yolo11n_obb_v1`: Initial OBB baseline ($0.4002$ val mAP50, $0.1412$ test mAP50).
  * `geoagri_yolo11n_obb_v2_refit` (EXP-B): Geometric refitting baseline ($0.4306$ val mAP50, $0.2567$ val mAP50–95).
  * `geoagri_yolo11n_obb_v3_aug` (EXP-C): Domain augmentation model ($0.4497$ val mAP50, $0.1138$ test mAP50).
* All experimental checkpoints reside safely under `runs/obb/` without overwriting production assets.

---

## 3. Regression Test Verification

The complete regression test suite was executed in `.venv`:

### Suite 1: Full Repository Suite (`pytest`)
* **Total Tests Executed:** 93
* **Tests Passed:** **93 passed (100%)**
* **Failures / Errors:** **0**
* Execution Time: 656.07 seconds (10 min 56s)

### Suite 2: CV Hardening & Bridge Integrity (`tests/test_yolo_obb_hardening.py`)
All 8 end-to-end integration and safety tests passed:
1. `test_1_model_registry_resolution`: PASSED (Production & candidate models resolve correctly from registry).
2. `test_2_production_model_loading_and_fallback`: PASSED (Production baseline loads on GPU with CPU fallback).
3. `test_3_candidate_obb_model_loading`: PASSED (Candidate OBB models load with valid weight checkpoints).
4. `test_4_valid_image_inference_api_contract`: PASSED (Inference bridge returns valid structured schema).
5. `test_5_zero_detection_blank_image`: PASSED (Graceful handling of zero-detection images without NoneType errors).
6. `test_6_invalid_input_handling`: PASSED (Rejects non-image and corrupt inputs).
7. `test_7_obb_output_structure`: PASSED (Preserves 8-coordinate OBB quadrilateral geometry).
8. `test_8_cpu_inference_fallback`: PASSED (Inference executes reliably when `device='cpu'`).

---

## 4. Operational Gating Matrix

| Gating Criteria | Status | Evidence / Notes |
| :--- | :---: | :--- |
| **Production Model Intact** | **VERIFIED** | `yolov8s_archaeology_prod` active; checkpoint unmodified. |
| **Experimental Isolation** | **VERIFIED** | EXP-B and EXP-C isolated in separate subdirectories under `runs/obb/`. |
| **API & Bridge Compatibility** | **VERIFIED** | `predict_bridge.py` supports both axis-aligned and oriented bounding box formats. |
| **CPU Fallback Functional** | **VERIFIED** | Validated via `test_8_cpu_inference_fallback`. |
| **Automated Tests Passing** | **VERIFIED** | 93 / 93 tests passed. |
| **Production Promotion Decision**| **HELD BACK** | EXP-C test mAP ($0.1138$) does not meet promotion threshold. |

---

## 5. Deployment Recommendation

* **Do not promote EXP-C to production.**
* Retain `yolov8s_archaeology_prod` for all user-facing terrain erosion risk workflows.
* Prioritize AGP 11 on cross-site data expansion and multi-scale feature adaptation to solve the underlying geological and material domain shift.

