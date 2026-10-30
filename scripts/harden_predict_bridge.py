"""Harden ui/server/predict_bridge.py with robust error validation, safe model registry resolution, and production isolation.
"""

from pathlib import Path

BRIDGE_PATH = Path("ui/server/predict_bridge.py")

def main():
    text = BRIDGE_PATH.read_text(encoding="utf-8")

    # 1. Update _load_models to use registry and prevent silent replacement
    old_loader_block = """    candidates = [
        os.path.join(ROOT, "runs", "obb", "geoagri_yolo11n_obb_v1", "weights", "best.pt"),
        os.path.join(ROOT, "runs", "obb", "geoagri_obb_smoke_test", "weights", "best.pt"),
        os.path.join(ROOT, "runs", "detect", "yolov8s_archaeology2", "weights", "best.pt"),
    ]
    yolo_path = next((p for p in candidates if os.path.exists(p)), None)
    yolo = YOLO(yolo_path) if (yolo_path and os.path.exists(yolo_path) and YOLO is not None) else None"""

    new_loader_block = """    # Safe Model Registry Integration (Phase 6)
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

    yolo = YOLO(yolo_path) if (yolo_path and os.path.exists(yolo_path) and YOLO is not None) else None"""

    if old_loader_block in text:
        text = text.replace(old_loader_block, new_loader_block, 1)
        print("[*] Successfully hardened model loading block in predict_bridge.py")

    # 2. Add input validation guards to predict()
    old_predict_start = """    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError("Unable to read image")"""

    new_predict_start = """    if not isinstance(image_path, (str, bytes, os.PathLike)):
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
        raise ValueError(f"Image dimensions too small ({w}x{h}). Minimum required: 16x16 pixels.")"""

    if old_predict_start in text:
        text = text.replace(old_predict_start, new_predict_start, 1)
        print("[*] Successfully hardened image input validation in predict_bridge.py")

    BRIDGE_PATH.write_text(text, encoding="utf-8")
    print("[SUCCESS] predict_bridge.py hardening applied!")

if __name__ == "__main__":
    main()

