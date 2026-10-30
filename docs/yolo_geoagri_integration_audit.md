# GeoAgri-AI: Computer Vision Pipeline Integration Audit

**Document ID:** `docs/yolo_geoagri_integration_audit.md`  
**Auditor:** Senior MLOps Engineer & Full-Stack System Architect  
**Codebase Traced:** `GeoAgri-AI` (`src/api/backend_api.py`, `ui/server/predict_bridge.py`, `ui/app/page.tsx`)  
**Date:** October 28, 2026  
**Status:** Traced, Verified & Operational

---

## 1. End-to-End Pipeline Execution Trace

The actual computer vision data flow was verified directly from the source code:

```
[User Browser: ui/app/page.tsx]
       |
       | Uploads Image + (lat, lon, confidence, class toggles)
       v
[Next.js API Gateway: ui/app/api/predict/route.ts]
       |
       | Proxies multipart/form-data to http://127.0.0.1:8000/api/predict
       v
[FastAPI Backend: src/api/backend_api.py]
       |
       | Writes uploaded image to temporary file, parses query params
       v
[Inference Bridge: ui/server/predict_bridge.py: predict()]
       |
       |-- 1. YOLO Inference:
       |      YOLO_MODEL.predict(image, conf=0.25, imgsz=640)
       |      Extracts detection_boxes = yolo_results[0].boxes
       |      Draws cv2.rectangle bounding boxes with CLASS_COLORS:
       |        - Class 0 (boulders):   Yellow (0, 255, 255)
       |        - Class 1 (others):     Gray (200, 200, 200)
       |        - Class 2 (ruins):      Blue (255, 0, 0)
       |        - Class 3 (structures): Red (0, 0, 255)
       |        - Class 4 (vegetation): Green (0, 255, 0)
       |
       |-- 2. Surface Ratio Extraction:
       |      Computes: boulders_ratio, vegetation_ratio, ruins_ratio, structures_ratio
       |
       |-- 3. Geographic Feature Fusion:
       |      Calls geo_features.extract_full_features(lat, lon, ...)
       |      Extracts: slope, rainfall, soil, elevation
       |
       |-- 4. Erosion Model Inference:
       |      build_erosion_features(EROSION_MODEL, slope, vegetation_ratio, ...)
       |      Calculates P(erosion risk) via XGBoost Classifier
       |      Maps P to "LOW" (<0.30), "MODERATE" (0.30-0.70), "HIGH" (>=0.70)
       |
       |-- 5. TreeSHAP Attribution & LLM Insight:
       |      Computes Shapley values for the 8 features
       |      Calls generate_groq_explanation_with_status()
       v
[FastAPI Response Validation: backend_api.py: _validated_probability]
       |
       | Verifies 0.0 <= probability <= 1.0
       v
[User Browser Dashboard: ui/app/page.tsx]
       - Renders Visual Analysis Grid (Detection, Segmentation, Heatmap)
       - Renders Hero Risk Card ("Prediction ready: 10% LOW risk")
       - Displays SHAP feature importance bars and Agronomic Advisory
```

---

## 2. Model Loader & Fallback Inspection

In [`ui/server/predict_bridge.py`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/ui/server/predict_bridge.py#L111-L144):

```python
def _load_models():
    # 1. Erosion model resolution
    erosion_model_path = _resolve_erosion_model_path()
    erosion = joblib.load(erosion_model_path)

    # 2. YOLO checkpoint resolution
    yolo_path = os.path.join(ROOT, "runs", "detect", "yolov8s_archaeology2", "weights", "best.pt")
    yolo = YOLO(yolo_path) if os.path.exists(yolo_path) and YOLO is not None else None

    # 3. Defensive device and segmentation loading
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    ...
    return yolo, seg, erosion, device, erosion_model_path
```

* **Defensive Robustness:** If `torch` or `YOLO` is absent, `YOLO_MODEL` resolves cleanly to `None` without crashing the application.
* **Output Contract:** The bridge expects the YOLO object to support `.predict(image)` returning results with a `.boxes` property containing `.xyxy`, `.cls`, and `.conf`.

---

## 3. Drop-in Replacement Safety Guidelines

When the retrained model weights are approved for deployment:
1. **Model Format:** Must be compatible with Ultralytics `YOLO(...)`.
2. **Class Names:** Must maintain the 5 classes `{0: 'boulders', 1: 'others', 2: 'ruins', 3: 'structures', 4: 'vegetation'}`.
3. **Target Destination:** Save new weights to `runs/detect/yolo11n_archaeology_v1/weights/best.pt` or `artifacts/yolo_vision_v2/weights/best.pt`.
4. **Configuration Switch:** Update `yolo_path` in `predict_bridge.py` to point to the new weights. The frontend and backend API interfaces require zero changes.

