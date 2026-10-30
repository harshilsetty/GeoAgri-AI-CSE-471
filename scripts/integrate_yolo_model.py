"""Integrate YOLO-OBB Model with fallback to existing detection model in ui/server/predict_bridge.py.
"""

from pathlib import Path

BRIDGE_PATH = Path("ui/server/predict_bridge.py")

def main():
    text = BRIDGE_PATH.read_text(encoding="utf-8")

    # 1. Update YOLO model loading to look for YOLO11n-OBB first, then smoke-test, then legacy yolov8s
    old_load = """    yolo_path = os.path.join(ROOT, "runs", "detect", "yolov8s_archaeology2", "weights", "best.pt")
    yolo = YOLO(yolo_path) if os.path.exists(yolo_path) and YOLO is not None else None"""

    new_load = """    candidates = [
        os.path.join(ROOT, "runs", "obb", "geoagri_yolo11n_obb_v1", "weights", "best.pt"),
        os.path.join(ROOT, "runs", "obb", "geoagri_obb_smoke_test", "weights", "best.pt"),
        os.path.join(ROOT, "runs", "detect", "yolov8s_archaeology2", "weights", "best.pt"),
    ]
    yolo_path = next((p for p in candidates if os.path.exists(p)), None)
    yolo = YOLO(yolo_path) if (yolo_path and os.path.exists(yolo_path) and YOLO is not None) else None"""

    if old_load in text:
        text = text.replace(old_load, new_load, 1)
        print("[*] Replaced YOLO loader in predict_bridge.py")
    elif "geoagri_yolo11n_obb_v1" in text:
        print("[*] YOLO loader already updated in predict_bridge.py")
    else:
        print("[!] Warning: Could not locate old_load pattern in predict_bridge.py")

    # 2. Add OBB polygon drawing support to predict()
    old_draw = """    detection_boxes = yolo_results[0].boxes if yolo_results else []
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
        )"""

    new_draw = """    has_obb = (
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
        detection_boxes = yolo_results[0].boxes if yolo_results else []
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
            )"""

    if old_draw in text:
        text = text.replace(old_draw, new_draw, 1)
        print("[*] Updated detection_img drawing with OBB support")
    elif "has_obb" in text:
        print("[*] detection_img already has OBB support")
    else:
        print("[!] Warning: Could not locate old_draw pattern in predict_bridge.py")

    # 3. Add OBB polygon drawing support to combined image
    old_combined = """    combined = seg_overlay.copy()
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
        )"""

    new_combined = """    combined = seg_overlay.copy()
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
            )"""

    if old_combined in text:
        text = text.replace(old_combined, new_combined, 1)
        print("[*] Updated combined image drawing with OBB support")
    elif "if has_obb:" in text:
        print("[*] combined image already has OBB support")
    else:
        print("[!] Warning: Could not locate old_combined pattern in predict_bridge.py")

    BRIDGE_PATH.write_text(text, encoding="utf-8")
    print("[SUCCESS] ui/server/predict_bridge.py integrated successfully!")

if __name__ == "__main__":
    main()

