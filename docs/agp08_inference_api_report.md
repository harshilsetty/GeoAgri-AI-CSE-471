# AGP 08 — Inference API Specification & Hardening Report

**Author:** Senior MLOps Engineer  
**Date:** October 28, 2026  
**Module:** `ui/server/predict_bridge.py`  
**API Endpoints:** `/api/predict` (FastAPI backend & Next.js proxy)  

---

## 1. API Contract & Schema Preservation

The interface between the frontend dashboard and the deep learning inference bridge was preserved without breaking schema changes. All existing consumer fields remain present and typed identically.

### 1.1 Request Signature (`predict()`)
```python
predict(
    image_path: str | Path,
    lat: float,
    lon: float,
    api_key: Optional[str] = None,
    confidence: float = 0.25,
    class_visibility: Optional[Dict[str, bool]] = None,
    use_ai_insight: bool = False,
    include_shap: bool = False,
    fast_mode: bool = False,
) -> Dict[str, Any]
```

### 1.2 Response Schema
```json
{
  "probability": 0.428,
  "riskLabel": "MODERATE",
  "explanation": "...",
  "insightMode": "default",
  "insightStatus": null,
  "metrics": {
    "vegetation": 0.354,
    "boulders": 0.128,
    "ruins": 0.082,
    "structures": 0.015,
    "slope": 14.2,
    "rainfall": 78.4,
    "elevation": 412.0,
    "soil": "loam",
    "lat": 15.335,
    "lon": 76.46
  },
  "shap": [],
  "images": {
    "original": "data:image/jpeg;base64,...",
    "detection": "data:image/jpeg;base64,...",
    "segmentation": "data:image/jpeg;base64,...",
    "combined": "data:image/jpeg;base64,...",
    "heatmap": "data:image/jpeg;base64,..."
  }
}
```

---

## 2. Hardened Error Matrix

| Input Condition | Previous Behavior | Hardened Behavior | Error Code / Exception |
| :--- | :--- | :--- | :--- |
| **Non-existent file** | Uncaught OpenCV error | File existence check | `FileNotFoundError` |
| **0-byte empty file** | OpenCV decode failure | File size verification | `ValueError("Image file is empty (0 bytes)")` |
| **Unsupported format (`.txt`, `.pdf`)** | Silent decode crash | Whitelist extension check | `ValueError("Unsupported image format...")` |
| **Tiny image ($<16\times 16$)** | CNN convolution underflow | Dimension floor verification | `ValueError("Image dimensions too small...")` |
| **Zero detections** | `TypeError: 'NoneType' object is not iterable` | Empty list fallback | Graceful return with 0.0 ratios |
| **Missing model checkpoint** | `None` model with crash | Safe fallback to production model | Logged warning with fallback |
| **Non-CUDA environment** | CUDA runtime crash | Automatic CPU fallback | Graceful CPU tensor execution |

---

## 3. Configurable Model Slot Control

To ensure production isolation, model slot resolution is governed by the `GEOAGRI_VISION_SLOT` environment variable:
* `GEOAGRI_VISION_SLOT="production_vision"` (default): Resolves to the verified YOLOv8s detection model.
* `GEOAGRI_VISION_SLOT="candidate_vision"`: Resolves to the candidate YOLO11n-OBB model.
* Configured through [`config/models_registry.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/config/models_registry.yaml) and resolved via [`src/models/registry.py`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/src/models/registry.py).

