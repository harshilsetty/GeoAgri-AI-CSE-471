"""Generate visual diagnostic predictions on held-out test split.

Overlays predicted OBB polygons (and compares with ground truth) on test images.
Categorizes cases: correct detections, missed objects, false positives, small objects, structures.
Saves rendered images and a comprehensive contact sheet.
"""

import cv2
import json
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO

random.seed(42)

WEIGHTS_PATH = Path("runs/obb/geoagri_yolo11n_obb_v1/weights/best.pt")
TEST_IMG_DIR = Path("datasets/geoagri_obb_v2/images/test")
TEST_LBL_DIR = Path("datasets/geoagri_obb_v2/labels/test")
OUTPUT_DIR = Path("artifacts/yolo_vision_v2/evaluation/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "boulders",
    1: "others",
    2: "ruins",
    3: "structures",
    4: "vegetation"
}

PRED_COLORS = {
    0: (0, 255, 255),    # boulders: cyan
    1: (200, 200, 200),  # others: light gray
    2: (0, 140, 255),    # ruins: orange
    3: (0, 0, 255),      # structures: bright red
    4: (0, 255, 0)       # vegetation: bright green
}

GT_COLOR = (255, 255, 255)  # white dashed/thin outline for ground truth

def draw_obb(img, pts, color, label=None, thickness=2, is_gt=False):
    pts = pts.astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    if label:
        x, y = pts[0]
        prefix = "GT:" if is_gt else "PRED:"
        txt = f"{prefix} {label}"
        cv2.putText(img, txt, (int(x), max(14, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

def main():
    print(f"[*] Loading model from {WEIGHTS_PATH}...")
    model = YOLO(str(WEIGHTS_PATH))

    test_imgs = sorted(list(TEST_IMG_DIR.glob("*.jpg")))
    print(f"[*] Total test images: {len(test_imgs)}")

    # Find images containing 'structures' ground truth
    struct_imgs = []
    other_imgs = []
    for p in test_imgs:
        lbl_p = TEST_LBL_DIR / f"{p.stem}.txt"
        has_struct = False
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("3 "):
                        has_struct = True
                        break
        if has_struct:
            struct_imgs.append(p)
        else:
            other_imgs.append(p)

    print(f"[*] Found {len(struct_imgs)} test images with 'structures' class")

    # Sample 12 images: all struct_imgs (up to 8) + diverse scenes from other_imgs
    selected = struct_imgs[:8]
    needed = 12 - len(selected)
    selected.extend(random.sample(other_imgs, min(needed, len(other_imgs))))

    diagnostic_records = []

    for idx, img_p in enumerate(selected):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        vis = img_bgr.copy()

        # 1. Overlay ground truth (thin white outline)
        lbl_p = TEST_LBL_DIR / f"{img_p.stem}.txt"
        gt_count = 0
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    coords = [float(c) for c in parts[1:]]
                    pts = np.array([
                        [coords[0] * w, coords[1] * h],
                        [coords[2] * w, coords[3] * h],
                        [coords[4] * w, coords[5] * h],
                        [coords[6] * w, coords[7] * h]
                    ], dtype=np.int32)
                    c_name = CLASS_NAMES.get(cls_id, str(cls_id))
                    draw_obb(vis, pts, (220, 220, 220), label=None, thickness=1, is_gt=True)
                    gt_count += 1

        # 2. Run model prediction (conf=0.20)
        results = model.predict(img_bgr, conf=0.20, imgsz=640, verbose=False)
        pred_count = 0
        has_obb = (results and hasattr(results[0], "obb") and results[0].obb is not None)
        
        preds_summary = []
        if has_obb:
            obb = results[0].obb
            pred_count = len(obb)
            for i in range(len(obb)):
                cls_id = int(obb.cls[i])
                conf = float(obb.conf[i])
                c_name = CLASS_NAMES.get(cls_id, str(cls_id))
                pts = obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                color = PRED_COLORS.get(cls_id, (255, 255, 255))
                draw_obb(vis, pts, color, label=f"{c_name} {conf:.2f}", thickness=2)
                preds_summary.append({"class": c_name, "conf": round(conf, 3)})

        # Header banner
        banner_text = f"Test Sample {idx+1}: {img_p.name[:20]} | GT: {gt_count} | Preds: {pred_count}"
        cv2.rectangle(vis, (0, 0), (w, 24), (20, 20, 20), -1)
        cv2.putText(vis, banner_text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        out_path = OUTPUT_DIR / f"test_pred_{idx+1}_{img_p.stem}.jpg"
        cv2.imwrite(str(out_path), vis)
        diagnostic_records.append({
            "idx": idx + 1,
            "filename": img_p.name,
            "path": str(out_path),
            "gt_count": gt_count,
            "pred_count": pred_count,
            "preds": preds_summary
        })

    # Create a 3x4 contact sheet (12 images)
    print("[*] Creating 3x4 diagnostic contact sheet...")
    thumb_w, thumb_h = 320, 320
    rows, cols = 3, 4
    contact_img = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for i, rec in enumerate(diagnostic_records[:rows * cols]):
        r = i // cols
        c = i % cols
        img = cv2.imread(rec["path"])
        if img is not None:
            thumb = cv2.resize(img, (thumb_w, thumb_h))
            contact_img[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb

    contact_path = OUTPUT_DIR / "prediction_contact_sheet.jpg"
    cv2.imwrite(str(contact_path), contact_img)
    print(f"[*] Contact sheet written to {contact_path}")

    # Write summary manifest
    manifest_path = OUTPUT_DIR / "visual_predictions_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(diagnostic_records, f, indent=2)
    print(f"[*] Manifest written to {manifest_path}")

if __name__ == "__main__":
    main()

