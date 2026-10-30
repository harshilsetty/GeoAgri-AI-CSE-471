import argparse
import base64
import json
import os
import sys
import traceback

import cv2
import joblib
import numpy as np
try:
    import torch
except (ImportError, OSError):
    torch = None

try:
    from ultralytics import YOLO
except (ImportError, OSError):
    YOLO = None

try:
    import segmentation_models_pytorch as smp
except (ImportError, OSError):
    smp = None

try:
    import shap
except (ImportError, OSError):
    shap = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_SRC_DIR = os.path.join(ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Ensure project root modules (e.g., terrain_model) are importable when this script is
# executed from Next.js API routes.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geo_features
from terrain_model.ai_explainer_groq import generate_groq_explanation_with_status


_SHAP_EXPLAINER = None

# These ranges/priors are aligned with terrain_model/erosion_dataset.csv statistics.
_FEATURE_BOUNDS = {
    "slope": (0.0293265124731489, 44.88257120604852),
    "vegetation": (0.00059500556905, 0.9993911513357628),
    "elevation": (300.2677118169703, 499.645616832133),
    "rainfall": (0.0554435395069718, 199.66261561730929),
    "soil": (1.0, 3.0),
    "boulders": (0.0010080407991445, 0.4999929489708697),
    "ruins": (0.0000646995176798, 0.2999434502549515),
    "structures": (0.0005125947674578, 0.1992778292892062),
}

_FEATURE_PRIORS = {
    "vegetation": 0.5845022182441334,
    "boulders": 0.23949178379829325,
    "ruins": 0.15243554215591154,
    "structures": 0.09720548137443649,
}


def _clip_feature(name, value):
    lo, hi = _FEATURE_BOUNDS[name]
    return float(max(lo, min(hi, float(value))))


def _validate_probability(probability, context="prediction"):
    prob = float(probability)
    if not np.isfinite(prob):
        raise ValueError(f"Non-finite probability returned by model during {context}: {probability}")
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f"Out-of-range probability returned by model during {context}: {prob}")
    return prob


def _resolve_segmentation_checkpoint():
    candidates = [
        os.path.join(ROOT, "models", "deeplab_model.pth"),
        os.path.join(ROOT, "runs", "segmentation", "deeplab_model_best.pth"),
        os.path.join(ROOT, "deeplab_model.pth"),
    ]

    existing = [path for path in candidates if os.path.exists(path)]
    if not existing:
        raise FileNotFoundError("No segmentation checkpoint found.")

    return max(existing, key=os.path.getmtime)


def _resolve_erosion_model_path():
    candidates = [
        os.path.join(ROOT, "terrain_model", "erosion_model.pkl"),
        os.path.join(ROOT, "models", "erosion_model.pkl"),
        os.path.join(ROOT, "erosion_model.pkl"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("No erosion model checkpoint found.")


def _load_models():
    erosion = None
    erosion_model_path = None
    try:
        erosion_model_path = _resolve_erosion_model_path()
        erosion = joblib.load(erosion_model_path)
    except Exception:
        erosion = None
        erosion_model_path = None

    # Safe Model Registry Integration (Phase 6)
    # Production model is retained by default; candidate model requires explicit GEOAGRI_VISION_SLOT
    model_slot = os.environ.get("GEOAGRI_VISION_SLOT", "production_vision").strip()
    try:
        from src.models.registry import resolve_model_path
        yolo_path = resolve_model_path(model_slot)
    except Exception:
        yolo_path = None

    if not yolo_path or not os.path.exists(yolo_path):
        # Fallback candidates in strict order: production baseline first, then candidate
        candidates = [
            os.path.join(ROOT, "runs", "detect", "yolov8s_archaeology2", "weights", "best.pt"),
            os.path.join(ROOT, "runs", "obb", "geoagri_yolo11n_obb_v1", "weights", "best.pt"),
        ]
        yolo_path = next((p for p in candidates if os.path.exists(p)), None)

    yolo = YOLO(yolo_path) if (yolo_path and os.path.exists(yolo_path) and YOLO is not None) else None

    seg = None
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    try:
        chk = _resolve_segmentation_checkpoint()
        if os.path.exists(chk) and smp is not None and torch is not None:
            seg = smp.DeepLabV3Plus(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=6,
            )
            seg.load_state_dict(torch.load(chk, map_location="cpu"))
            seg.eval()
            seg = seg.to(device)
    except Exception:
        seg = None

    return yolo, seg, erosion, device, erosion_model_path


YOLO_MODEL, SEG_MODEL, EROSION_MODEL, DEVICE, EROSION_MODEL_PATH = _load_models()

CLASS_COLORS = {
    "boulders": (0, 255, 255),
    "others": (200, 200, 200),
    "ruins": (255, 0, 0),
    "structures": (0, 0, 255),
    "vegetation": (0, 255, 0),
}

SEG_COLORS = {
    0: [0, 0, 0],
    1: [0, 255, 255],
    2: [200, 200, 200],
    3: [255, 0, 0],
    4: [0, 0, 255],
    5: [0, 255, 0],
}

CLASS_ID_TO_NAME = {
    1: "boulders",
    2: "others",
    3: "ruins",
    4: "structures",
    5: "vegetation",
}


def _image_to_data_url(image_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("Failed to encode output image")
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_erosion_features(
    erosion_model,
    slope,
    vegetation_ratio,
    elevation,
    boulders_ratio,
    ruins_ratio,
    structures_ratio,
    rainfall=None,
    soil_value=None,
):
    expected_features = getattr(erosion_model, "n_features_in_", None)

    slope = _clip_feature("slope", slope)
    vegetation_ratio = _clip_feature("vegetation", vegetation_ratio)
    elevation = _clip_feature("elevation", elevation)
    rainfall = _clip_feature("rainfall", rainfall if rainfall is not None else 0.0)
    soil_value = _clip_feature("soil", soil_value if soil_value is not None else 2)
    boulders_ratio = _clip_feature("boulders", boulders_ratio)
    ruins_ratio = _clip_feature("ruins", ruins_ratio)
    structures_ratio = _clip_feature("structures", structures_ratio)

    if expected_features == 3:
        return np.array([[slope, vegetation_ratio, elevation]], dtype=np.float32), 3

    if expected_features == 4:
        return np.array([[slope, vegetation_ratio, ruins_ratio, elevation]], dtype=np.float32), 4

    if expected_features == 5:
        return np.array([[slope, vegetation_ratio, elevation, rainfall, soil_value]], dtype=np.float32), 5

    if expected_features == 6:
        return np.array(
            [[slope, vegetation_ratio, elevation, boulders_ratio, ruins_ratio, structures_ratio]],
            dtype=np.float32,
        ), 6

    if expected_features == 8:
        return np.array(
            [[slope, vegetation_ratio, elevation, rainfall, soil_value, boulders_ratio, ruins_ratio, structures_ratio]],
            dtype=np.float32,
        ), 8

    return np.array([[slope, vegetation_ratio, elevation]], dtype=np.float32), 3


def _resolve_feature_names(erosion_model, features, feature_mode):
    feature_names = list(getattr(erosion_model, "feature_names_in_", []))
    if len(feature_names) != features.shape[1]:
        feature_names_by_mode = {
            3: ["slope", "vegetation", "elevation"],
            4: ["slope", "vegetation", "ruins", "elevation"],
            5: ["slope", "vegetation", "elevation", "rainfall", "soil"],
            6: ["slope", "vegetation", "elevation", "boulders", "ruins", "structures"],
            8: ["slope", "vegetation", "elevation", "rainfall", "soil", "boulders", "ruins", "structures"],
        }
        feature_names = feature_names_by_mode.get(feature_mode, ["f" + str(i) for i in range(features.shape[1])])

    alias = {
        "vegetation_ratio": "vegetation",
        "boulders_ratio": "boulders",
        "ruins_ratio": "ruins",
        "structures_ratio": "structures",
        "soil_value": "soil",
    }
    return [alias.get(name, name) for name in feature_names]


def _predict_probability(erosion_model, features):
    if hasattr(erosion_model, "predict_proba"):
        raw_prob = float(erosion_model.predict_proba(features)[0][1])
    else:
        raw_prob = float(erosion_model.predict(features)[0])

    return _validate_probability(max(0.0, min(1.0, raw_prob)), context="predict_proba")


def get_model_debug_info(sample_lat=22.5726, sample_lon=88.3639):
    sample_lat = float(sample_lat)
    sample_lon = float(sample_lon)

    vegetation_ratio = 0.35
    boulders_ratio = 0.0
    ruins_ratio = 0.0
    structures_ratio = 0.0

    # Extract real geographic features using the new geo_features module
    features_dict = geo_features.extract_full_features(
        sample_lat, sample_lon,
        vegetation_ratio=vegetation_ratio,
        boulders_ratio=boulders_ratio,
        ruins_ratio=ruins_ratio,
        structures_ratio=structures_ratio
    )
    slope = float(features_dict["slope"])
    rainfall = float(features_dict["rainfall"])
    soil_value = int(features_dict["soil"])
    elevation = float(features_dict["elevation"])

    features, feature_mode = build_erosion_features(
        EROSION_MODEL,
        slope,
        vegetation_ratio,
        elevation,
        boulders_ratio,
        ruins_ratio,
        structures_ratio,
        rainfall,
        soil_value,
    )
    probability = _predict_probability(EROSION_MODEL, features)

    return {
        "erosionModelPath": EROSION_MODEL_PATH,
        "erosionModelType": f"{type(EROSION_MODEL).__module__}.{type(EROSION_MODEL).__name__}",
        "expectedFeatures": int(getattr(EROSION_MODEL, "n_features_in_", features.shape[1])),
        "featureMode": int(feature_mode),
        "featureNames": _resolve_feature_names(EROSION_MODEL, features, feature_mode),
        "sample": {
            "lat": sample_lat,
            "lon": sample_lon,
            "probability": probability,
            "features": features[0].tolist(),
        },
    }


def _compute_top_shap(erosion_model, features, feature_mode):
    global _SHAP_EXPLAINER

    feature_names = _resolve_feature_names(erosion_model, features, feature_mode)

    if _SHAP_EXPLAINER is None:
        _SHAP_EXPLAINER = shap.Explainer(erosion_model)

    shap_values = _SHAP_EXPLAINER(features)
    shap_array = np.array(getattr(shap_values, "values", shap_values))

    if shap_array.ndim == 3:
        class_index = 1 if shap_array.shape[2] > 1 else 0
        shap_vals = shap_array[0, :, class_index]
    elif shap_array.ndim == 2:
        shap_vals = shap_array[0]
    else:
        shap_vals = shap_array.reshape(-1)

    if shap_vals.shape[0] > features.shape[1]:
        shap_vals = shap_vals[: features.shape[1]]
    elif shap_vals.shape[0] < features.shape[1]:
        shap_vals = np.pad(shap_vals, (0, features.shape[1] - shap_vals.shape[0]), constant_values=0.0)

    top_indices = np.argsort(np.abs(shap_vals))[::-1][:5]
    return [{"feature": feature_names[i], "value": float(shap_vals[i])} for i in top_indices]


def _compute_local_sensitivity_fallback(erosion_model, features, feature_mode):
    feature_names = _resolve_feature_names(erosion_model, features, feature_mode)
    baseline = features.astype(np.float32)
    base_row = baseline[0].copy()

    contributions = []
    for index, name in enumerate(feature_names):
        step = max(0.01, abs(float(base_row[index])) * 0.05)

        up = base_row.copy()
        down = base_row.copy()
        up[index] += step
        down[index] -= step

        up_prob = _predict_probability(erosion_model, np.expand_dims(up, axis=0))
        down_prob = _predict_probability(erosion_model, np.expand_dims(down, axis=0))

        # Signed local contribution using symmetric finite difference.
        signed_effect = (up_prob - down_prob) / 2.0
        contributions.append({"feature": name, "value": float(signed_effect)})

    contributions.sort(key=lambda item: abs(item["value"]), reverse=True)
    return contributions[:5]


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_default_explanation(probability, slope, vegetation_ratio, rainfall):
    if probability >= 0.7:
        return (
            "High erosion risk is driven by terrain instability indicators. "
            f"Slope ({slope:.2f}) and rainfall ({rainfall:.2f}) suggest strong runoff pressure, "
            "while vegetation cover appears insufficient for full stabilization."
        )
    if probability >= 0.3:
        return (
            "Moderate erosion risk detected. Mixed indicators suggest partial stability with some vulnerable zones. "
            f"Vegetation ratio is {vegetation_ratio:.2f}, so targeted reinforcement is recommended."
        )
    return (
        "Low erosion risk detected. Current terrain indicators suggest generally stable conditions with lower immediate risk."
    )


def _soil_to_value(soil):
    if isinstance(soil, (int, np.integer)):
        return int(soil)

    soil_text = str(soil or "").strip().lower()
    if soil_text == "sandy":
        return 1
    if soil_text == "clay":
        return 3
    return 2


def generate_insights(
    metrics,
    probability,
    api_key=None,
    include_ai_insight=True,
    include_shap=True,
):
    slope = float(metrics.get("slope", 0.0))
    vegetation_ratio = float(metrics.get("vegetation", 0.0))
    rainfall = float(metrics.get("rainfall", 0.0))
    elevation = float(metrics.get("elevation", 0.0))
    boulders_ratio = float(metrics.get("boulders", 0.0))
    ruins_ratio = float(metrics.get("ruins", 0.0))
    structures_ratio = float(metrics.get("structures", 0.0))
    soil_raw = metrics.get("soil", "loam")
    soil_value = _soil_to_value(soil_raw)
    soil_map = {1: "sandy", 2: "loam", 3: "clay"}
    soil_type = soil_map.get(soil_value, "loam")

    features, feature_mode = build_erosion_features(
        EROSION_MODEL,
        slope,
        vegetation_ratio,
        elevation,
        boulders_ratio,
        ruins_ratio,
        structures_ratio,
        rainfall,
        soil_value,
    )

    ai_features = {
        "slope": slope,
        "vegetation": vegetation_ratio,
        "rainfall": rainfall,
        "soil": soil_type,
        "boulders": boulders_ratio,
        "ruins": ruins_ratio,
        "structures": structures_ratio,
    }

    insight_status = None
    if include_ai_insight:
        ai_explanation, insight_mode, insight_status = generate_groq_explanation_with_status(
            ai_features,
            probability,
            api_key=api_key or None,
        )
    else:
        ai_explanation = _build_default_explanation(probability, slope, vegetation_ratio, rainfall)
        insight_mode = "default"

    top_shap = []
    if include_shap:
        try:
            top_shap = _compute_top_shap(EROSION_MODEL, features, feature_mode)
        except Exception:
            top_shap = _compute_local_sensitivity_fallback(EROSION_MODEL, features, feature_mode)

    return {
        "explanation": ai_explanation,
        "insightMode": insight_mode,
        "insightStatus": insight_status,
        "shap": top_shap,
    }


def _to_float_or_default(value, default):
    try:
        parsed = float(value)
        if np.isfinite(parsed):
            return parsed
    except Exception:
        pass
    return float(default)


def predict_point(
    lat,
    lon,
    api_key=None,
    use_ai_insight=False,
    include_shap=True,
    vegetation_ratio=None,
    boulders_ratio=None,
    ruins_ratio=None,
    structures_ratio=None,
):
    # Point mode has no segmentation mask; use dataset priors unless caller supplies values.
    vegetation_ratio = _to_float_or_default(vegetation_ratio, _FEATURE_PRIORS["vegetation"])
    boulders_ratio = _to_float_or_default(boulders_ratio, _FEATURE_PRIORS["boulders"])
    ruins_ratio = _to_float_or_default(ruins_ratio, _FEATURE_PRIORS["ruins"])
    structures_ratio = _to_float_or_default(structures_ratio, _FEATURE_PRIORS["structures"])

    # Extract real geographic features using the new geo_features module
    features_dict = geo_features.extract_full_features(
        lat, lon,
        vegetation_ratio=vegetation_ratio,
        boulders_ratio=boulders_ratio,
        ruins_ratio=ruins_ratio,
        structures_ratio=structures_ratio
    )
    slope = float(features_dict["slope"])
    rainfall = float(features_dict["rainfall"])
    soil_value = int(features_dict["soil"])
    elevation = float(features_dict["elevation"])

    features, feature_mode = build_erosion_features(
        EROSION_MODEL,
        slope,
        vegetation_ratio,
        elevation,
        boulders_ratio,
        ruins_ratio,
        structures_ratio,
        rainfall,
        soil_value,
    )

    probability = _predict_probability(EROSION_MODEL, features)
    risk_label = "LOW" if probability < 0.3 else "MODERATE" if probability < 0.7 else "HIGH"

    soil_map = {1: "sandy", 2: "loam", 3: "clay"}
    soil_type = soil_map.get(soil_value, "loam")

    ai_features = {
        "slope": slope,
        "vegetation": vegetation_ratio,
        "rainfall": rainfall,
        "soil": soil_type,
        "boulders": boulders_ratio,
        "ruins": ruins_ratio,
        "structures": structures_ratio,
    }

    insight_status = None
    if use_ai_insight:
        explanation, insight_mode, insight_status = generate_groq_explanation_with_status(
            ai_features,
            probability,
            api_key=api_key or None,
        )
    else:
        explanation = _build_default_explanation(probability, slope, vegetation_ratio, rainfall)
        insight_mode = "default"

    top_shap = []
    if include_shap:
        try:
            top_shap = _compute_top_shap(EROSION_MODEL, features, feature_mode)
        except Exception:
            top_shap = _compute_local_sensitivity_fallback(EROSION_MODEL, features, feature_mode)

    return {
        "probability": probability,
        "riskLabel": risk_label,
        "explanation": explanation,
        "insightMode": insight_mode,
        "insightStatus": insight_status,
        "metrics": {
            "vegetation": vegetation_ratio,
            "slope": slope,
            "rainfall": rainfall,
            "elevation": elevation,
            "soil": soil_type,
            "boulders": boulders_ratio,
            "ruins": ruins_ratio,
            "structures": structures_ratio,
            "lat": lat,
            "lon": lon,
        },
        "shap": top_shap,
    }


def predict(
    image_path,
    lat,
    lon,
    api_key=None,
    confidence=0.25,
    class_visibility=None,
    use_ai_insight=False,
    include_shap=False,
    fast_mode=False,
):
    if class_visibility is None:
        class_visibility = {
            "vegetation": True,
            "ruins": True,
            "structures": True,
            "boulders": True,
            "others": True,
        }

    conf_value = float(max(0.05, min(0.95, confidence)))

    if not isinstance(image_path, (str, bytes, os.PathLike)):
        raise TypeError(f"Invalid image_path type: {type(image_path).__name__}. Expected str or PathLike.")

    if not os.path.exists(str(image_path)):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    if os.path.getsize(str(image_path)) == 0:
        raise ValueError(f"Image file is empty (0 bytes): {image_path}")

    ext = os.path.splitext(str(image_path))[1].lower()
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    if ext not in valid_exts:
        raise ValueError(f"Unsupported image format '{ext}'. Supported formats: {', '.join(sorted(valid_exts))}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError(f"Failed to decode or parse image file: {image_path}")

    h, w = image_bgr.shape[:2]
    if h < 16 or w < 16:
        raise ValueError(f"Image dimensions too small ({w}x{h}). Minimum required: 16x16 pixels.")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detection_img = image.copy()
    yolo_imgsz = 512 if fast_mode else 640
    yolo_results = (
        YOLO_MODEL.predict(image, conf=conf_value, imgsz=yolo_imgsz, verbose=False)
        if YOLO_MODEL is not None
        else []
    )
    has_obb = (
        yolo_results
        and hasattr(yolo_results[0], "obb")
        and yolo_results[0].obb is not None
        and len(yolo_results[0].obb) > 0
    )
    if has_obb:
        obb = yolo_results[0].obb
        for i in range(len(obb)):
            cls = int(obb.cls[i])
            conf = float(obb.conf[i])
            label = YOLO_MODEL.names.get(cls, str(cls))
            if not class_visibility.get(label, True):
                continue
            color = CLASS_COLORS.get(label, (255, 255, 255))
            pts = obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
            cv2.polylines(detection_img, [pts], isClosed=True, color=color, thickness=2)
            x1, y1 = pts[0]
            cv2.putText(
                detection_img,
                f"{label} {conf:.2f}",
                (int(x1), max(16, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
    else:
        detection_boxes = yolo_results[0].boxes if (yolo_results and yolo_results[0].boxes is not None) else []
        for box in detection_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = YOLO_MODEL.names[cls]
            if not class_visibility.get(label, True):
                continue
            color = CLASS_COLORS.get(label, (255, 255, 255))
            cv2.rectangle(detection_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                detection_img,
                f"{label} {conf:.2f}",
                (x1, max(16, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    if torch is None or SEG_MODEL is None:
        mask = None
    else:
        resized = cv2.resize(image, (512, 512))
        tensor = torch.tensor(resized.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = SEG_MODEL(tensor)
            mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))

    if mask is not None:
        seg_vis = image.copy()
        for class_id, color in SEG_COLORS.items():
            if class_id == 0:
                continue
            class_name = CLASS_ID_TO_NAME.get(class_id)
            if class_name and class_visibility.get(class_name, True):
                seg_vis[mask == class_id] = color
        seg_overlay = cv2.addWeighted(image, 0.85, seg_vis, 0.15, 0)
    else:
        # Honest OBB/detection visualization with filled polygon footprint
        obb_fill_overlay = image.copy()
        has_detection_any = False
        if has_obb:
            obb = yolo_results[0].obb
            for i in range(len(obb)):
                cls = int(obb.cls[i])
                conf = float(obb.conf[i])
                label = YOLO_MODEL.names.get(cls, str(cls))
                if not class_visibility.get(label, True):
                    continue
                color = CLASS_COLORS.get(label, (255, 255, 255))
                pts = obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                cv2.fillPoly(obb_fill_overlay, [pts], color=color)
                cv2.polylines(obb_fill_overlay, [pts], isClosed=True, color=color, thickness=2)
                x1, y1 = pts[0]
                cv2.putText(
                    obb_fill_overlay,
                    f"{label} {conf:.2f}",
                    (int(x1), max(16, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )
                has_detection_any = True
        elif detection_boxes:
            for box in detection_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = YOLO_MODEL.names[cls]
                if not class_visibility.get(label, True):
                    continue
                color = CLASS_COLORS.get(label, (255, 255, 255))
                cv2.rectangle(obb_fill_overlay, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(obb_fill_overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    obb_fill_overlay,
                    f"{label} {conf:.2f}",
                    (x1, max(16, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )
                has_detection_any = True

        if has_detection_any:
            seg_overlay = cv2.addWeighted(image, 0.65, obb_fill_overlay, 0.35, 0)
        else:
            seg_overlay = image.copy()

    # Heatmap generation: Feature density heatmap from genuine detections
    heat_overlay = image.copy()
    has_heat_features = False
    density_map = np.zeros((h, w), dtype=np.float32)

    if has_obb:
        obb = yolo_results[0].obb
        for i in range(len(obb)):
            cls = int(obb.cls[i])
            label = YOLO_MODEL.names.get(cls, str(cls))
            if not class_visibility.get(label, True):
                continue
            pts = obb.xyxyxyxy[i].cpu().numpy().astype(np.float32)
            cx, cy = np.mean(pts, axis=0)
            area = float(cv2.contourArea(pts.astype(np.int32)))
            radius = int(max(24, min(120, np.sqrt(max(1.0, area)) * 0.9)))
            weight = 255.0 * float(obb.conf[i])
            cv2.circle(density_map, (int(cx), int(cy)), radius, weight, -1)
            has_heat_features = True
    elif detection_boxes:
        for box in detection_boxes:
            cls = int(box.cls[0])
            label = YOLO_MODEL.names[cls]
            if not class_visibility.get(label, True):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = int(max(24, min(120, np.sqrt(max(1.0, (x2 - x1) * (y2 - y1))) * 0.9)))
            weight = 255.0 * float(box.conf[0])
            cv2.circle(density_map, (cx, cy), radius, weight, -1)
            has_heat_features = True

    if has_heat_features:
        blurred_density = cv2.GaussianBlur(density_map, (0, 0), sigmaX=35, sigmaY=35)
        norm_density = cv2.normalize(blurred_density, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_heatmap = cv2.applyColorMap(norm_density, cv2.COLORMAP_JET)
        heat_overlay = cv2.addWeighted(image, 0.58, color_heatmap, 0.42, 0)

    combined = seg_overlay.copy()
    if has_obb:
        obb = yolo_results[0].obb
        for i in range(len(obb)):
            cls = int(obb.cls[i])
            conf = float(obb.conf[i])
            label = YOLO_MODEL.names.get(cls, str(cls))
            if not class_visibility.get(label, True):
                continue
            color = CLASS_COLORS.get(label, (255, 255, 255))
            pts = obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
            cv2.polylines(combined, [pts], isClosed=True, color=color, thickness=2)
            x1, y1 = pts[0]
            cv2.putText(
                combined,
                f"{label} {conf:.2f}",
                (int(x1), max(16, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
    else:
        for box in detection_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = YOLO_MODEL.names[cls]
            if not class_visibility.get(label, True):
                continue
            color = CLASS_COLORS.get(label, (255, 255, 255))
            cv2.rectangle(combined, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                combined,
                f"{label} {conf:.2f}",
                (x1, max(16, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    if mask is not None and np.sum(mask) > 0:
        total_pixels = mask.size
        boulders_ratio = float(np.sum(mask == 1) / total_pixels)
        vegetation_ratio = float(np.sum(mask == 5) / total_pixels)
        ruins_ratio = float(np.sum(mask == 3) / total_pixels)
        structures_ratio = float(np.sum(mask == 4) / total_pixels)
    else:
        total_pixels = float(h * w)
        class_areas = {"boulders": 0.0, "vegetation": 0.0, "ruins": 0.0, "structures": 0.0}
        if has_obb:
            obb = yolo_results[0].obb
            for i in range(len(obb)):
                cls = int(obb.cls[i])
                label = YOLO_MODEL.names.get(cls, str(cls))
                if label in class_areas:
                    pts = obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                    class_areas[label] += float(cv2.contourArea(pts))
        elif detection_boxes:
            for box in detection_boxes:
                cls = int(box.cls[0])
                label = YOLO_MODEL.names[cls]
                if label in class_areas:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_areas[label] += float((x2 - x1) * (y2 - y1))

        boulders_ratio = float(min(1.0, class_areas["boulders"] / total_pixels))
        vegetation_ratio = (
            float(min(1.0, class_areas["vegetation"] / total_pixels))
            if class_areas["vegetation"] > 0
            else _FEATURE_PRIORS["vegetation"]
        )
        ruins_ratio = float(min(1.0, class_areas["ruins"] / total_pixels))
        structures_ratio = float(min(1.0, class_areas["structures"] / total_pixels))

    # Extract real geographic features using the new geo_features module
    features_dict = geo_features.extract_full_features(
        lat, lon,
        vegetation_ratio=vegetation_ratio,
        boulders_ratio=boulders_ratio,
        ruins_ratio=ruins_ratio,
        structures_ratio=structures_ratio
    )
    slope = float(features_dict["slope"])
    rainfall = float(features_dict["rainfall"])
    soil = int(features_dict["soil"])
    elevation = float(features_dict["elevation"])

    features, feature_mode = build_erosion_features(
        EROSION_MODEL,
        slope,
        vegetation_ratio,
        elevation,
        boulders_ratio,
        ruins_ratio,
        structures_ratio,
        rainfall,
        soil,
    )

    probability = _predict_probability(EROSION_MODEL, features)

    soil_map = {1: "sandy", 2: "loam", 3: "clay"}
    soil_type = soil_map.get(soil, "loam")

    ai_features = {
        "slope": slope,
        "vegetation": vegetation_ratio,
        "rainfall": rainfall,
        "soil": soil_type,
        "boulders": boulders_ratio,
        "ruins": ruins_ratio,
        "structures": structures_ratio,
    }
    insight_status = None
    if use_ai_insight:
        ai_explanation, insight_mode, insight_status = generate_groq_explanation_with_status(
            ai_features,
            probability,
            api_key=api_key or None,
        )
    else:
        ai_explanation = _build_default_explanation(probability, slope, vegetation_ratio, rainfall)
        insight_mode = "default"

    top_shap = []
    if include_shap:
        try:
            top_shap = _compute_top_shap(EROSION_MODEL, features, feature_mode)
        except Exception:
            # SHAP can fail for some model/checkpoint combinations; provide a signed local
            # sensitivity fallback so the UI still receives meaningful feature bars.
            top_shap = _compute_local_sensitivity_fallback(EROSION_MODEL, features, feature_mode)

    return {
        "probability": probability,
        "riskLabel": "LOW" if probability < 0.3 else "MODERATE" if probability < 0.7 else "HIGH",
        "explanation": ai_explanation,
        "insightMode": insight_mode,
        "insightStatus": insight_status,
        "metrics": {
            "vegetation": vegetation_ratio,
            "slope": slope,
            "rainfall": rainfall,
            "elevation": elevation,
            "soil": soil_type,
            "boulders": boulders_ratio,
            "ruins": ruins_ratio,
            "structures": structures_ratio,
            "lat": lat,
            "lon": lon,
        },
        "shap": top_shap,
        "images": {
            "original": _image_to_data_url(image),
            "detection": _image_to_data_url(detection_img),
            "segmentation": _image_to_data_url(seg_overlay),
            "combined": _image_to_data_url(combined),
            "heatmap": _image_to_data_url(heat_overlay),
        },
        "segmentationInfo": {
            "available": SEG_MODEL is not None,
            "type": "OBB Detection" if SEG_MODEL is None else "Semantic Segmentation",
            "notice": "Pixel-level segmentation is currently unavailable. The active computer vision model supports oriented object detection." if SEG_MODEL is None else "Pixel-level segmentation active.",
        },
        "heatmapInfo": {
            "available": has_heat_features,
            "type": "Feature Density",
            "notice": "Spatial feature density heatmap generated from detected archaeological & terrain features." if has_heat_features else "Spatial feature density unavailable: No terrain features detected in this tile.",
        },
    }


def recommend_crops(
    lat: float,
    lon: float,
    season: str = "Kharif",
    vegetation_ratio: float = 0.35,
    api_key: str = "",
    top_k: int = 5,
):
    import agri_inference

    return agri_inference.generate_crop_recommendation(
        lat=lat,
        lon=lon,
        season=season,
        vegetation_ratio=vegetation_ratio,
        api_key=api_key,
        top_k=top_k,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--lat", required=True, type=float)
    parser.add_argument("--lon", required=True, type=float)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--confidence", default="0.25")
    parser.add_argument("--show-vegetation", default="true")
    parser.add_argument("--show-ruins", default="true")
    parser.add_argument("--show-structures", default="true")
    parser.add_argument("--show-boulders", default="true")
    parser.add_argument("--show-others", default="true")
    parser.add_argument("--use-ai-insight", default="false")
    parser.add_argument("--include-shap", default="false")
    parser.add_argument("--fast-mode", default="false")
    args = parser.parse_args()

    try:
        payload = predict(
            args.image,
            args.lat,
            args.lon,
            args.api_key,
            confidence=float(args.confidence),
            class_visibility={
                "vegetation": _to_bool(args.show_vegetation),
                "ruins": _to_bool(args.show_ruins),
                "structures": _to_bool(args.show_structures),
                "boulders": _to_bool(args.show_boulders),
                "others": _to_bool(args.show_others),
            },
            use_ai_insight=_to_bool(args.use_ai_insight),
            include_shap=_to_bool(args.include_shap),
            fast_mode=_to_bool(args.fast_mode),
        )
        print(json.dumps({"ok": True, "data": payload}))
    except Exception as exc:
        error = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=6),
        }
        print(json.dumps(error))


if __name__ == "__main__":
    main()
