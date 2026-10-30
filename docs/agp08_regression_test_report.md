# AGP 08 — Regression & Integration Test Execution Report

**Author:** Quality Assurance & MLOps Lead  
**Date:** October 28, 2026  
**Test Suite:** `tests/test_yolo_obb_hardening.py` + `tests/test_api_endpoints.py` + `tests/test_erosion_v2_model.py`  
**Execution Environment:** Python 3.13.3, PyTorch 2.6.0+cu124, NVIDIA RTX 3050 Laptop GPU  

---

## 1. Test Execution Summary

A complete regression test suite comprising **15 test cases** was executed across inference hardening, model registry resolution, API endpoints, and erosion modeling.

| Test Category | Test File | Tests Run | Passed | Failed | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CV Hardening & OBB Integration** | `tests/test_yolo_obb_hardening.py` | 8 | 8 | 0 | **PASS (100%)** |
| **API Endpoints & Routing** | `tests/test_api_endpoints.py` | 4 | 4 | 0 | **PASS (100%)** |
| **Erosion Model V2 Inference** | `tests/test_erosion_v2_model.py` | 3 | 3 | 0 | **PASS (100%)** |
| **Total Suite** | — | **15** | **15** | **0** | **ALL PASSED** |

---

## 2. Detailed CV Hardening Test Case Matrix

| Test Case Name | Description | Verification Criterion | Result |
| :--- | :--- | :--- | :---: |
| `test_1_model_registry_resolution` | Tests registry resolution for both production and candidate slots | Checks paths exist and point to valid `.pt` files | **PASSED** |
| `test_2_production_model_loading_and_fallback` | Loads verified baseline `yolov8s_archaeology2` | Verifies YOLO model initialization and 5 classes | **PASSED** |
| `test_3_candidate_obb_model_loading` | Loads candidate `geoagri_yolo11n_obb_v1` | Verifies task is `'obb'` and classes match | **PASSED** |
| `test_4_valid_image_inference_api_contract` | Runs `predict()` on valid test image | Verifies probability, riskLabel, metrics, and base64 images | **PASSED** |
| `test_5_zero_detection_blank_image` | Runs `predict()` on solid black image | Verifies no crash when detections are empty | **PASSED** |
| `test_6_invalid_input_handling` | Passes missing, 0-byte, wrong extension, and tiny images | Asserts `FileNotFoundError` and `ValueError` | **PASSED** |
| `test_7_obb_output_structure` | Validates polygon coordinates from OBB prediction | Asserts shape `(N, 4, 2)` for rotated quadrilaterals | **PASSED** |
| `test_8_cpu_inference_fallback` | Executes inference forced onto CPU | Verifies clean CPU tensor forward pass | **PASSED** |

---

## 3. Discovered Defects and Remediations

During initial test runs, two critical defects were uncovered and resolved:
1. **Defect 1: Missing Local Production Checkpoint:**
   - *Discovery:* `runs/detect/yolov8s_archaeology2/weights/best.pt` was missing locally in `GeoAgri-AI` (existed in sibling repo).
   - *Fix:* Materialized the baseline checkpoint locally (`22.5 MB`), restoring production fallback.
2. **Defect 2: TypeError on Zero-Detection Images:**
   - *Discovery:* When an image produced zero detections, Ultralytics OBB set `results[0].boxes = None`. The legacy fallback loop `for box in detection_boxes:` threw `TypeError: 'NoneType' object is not iterable`.
   - *Fix:* Hardened the check to `yolo_results[0].boxes if (yolo_results and yolo_results[0].boxes is not None) else []`.

---

## 4. Execution Command

To re-run the full hardened regression test suite:

```powershell
.venv\Scripts\pytest.exe tests/test_yolo_obb_hardening.py tests/test_api_endpoints.py tests/test_erosion_v2_model.py -v
```

