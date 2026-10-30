"""Diagnostic Analysis of Nalanda Structures in Held-Out Test Set.

Evaluates ground-truth structures vs predictions from EXP-C (and EXP-B).
Computes exact Polygon IoU, angle deviations, center displacements,
and generates visual overlays.
"""

import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def polygon_iou(poly1_pts, poly2_pts):
    try:
        p1 = poly1_pts.astype(np.float32)
        p2 = poly2_pts.astype(np.float32)
        area1 = cv2.contourArea(p1)
        area2 = cv2.contourArea(p2)
        inter, inter_pts = cv2.intersectConvexConvex(p1, p2)
        if inter_pts is None or len(inter_pts) < 3:
            inter_area = 0.0
        else:
            inter_area = cv2.contourArea(inter_pts)
        union = area1 + area2 - inter_area
        if union <= 1e-7:
            return 0.0
        return float(inter_area / union)
    except Exception:
        return 0.0

def get_oriented_box_properties(pts):
    # pts shape (4, 2) in pixels
    rect = cv2.minAreaRect(pts.astype(np.float32))
    center = rect[0]
    (w, h) = rect[1]
    angle = rect[2]
    return center, w, h, angle

def main():
    v3_root = Path("datasets/geoagri_obb_v3")
    nalanda_imgs = sorted(list((v3_root / "images" / "test").glob("nalanda*.jpg")))
    print(f"Total Nalanda test images found: {len(nalanda_imgs)}")

    model_c = YOLO("runs/obb/geoagri_yolo11n_obb_v3_aug/weights/best.pt")

    out_dir = Path("artifacts/agp11/nalanda_diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)

    names = {0: "boulders", 1: "others", 2: "ruins", 3: "structures", 4: "vegetation"}

    analysis_records = []
    overlay_count = 0

    for img_path in nalanda_imgs:
        lbl_path = v3_root / "labels" / "test" / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        # Load GT
        gt_boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                pts = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2) * 640.0
                gt_boxes.append({"cls": cid, "pts": pts})

        gt_structs = [g for g in gt_boxes if g["cls"] == 3]
        if not gt_structs:
            continue

        # Predict
        res = model_c.predict(str(img_path), conf=0.01, device="0", verbose=False)[0]
        preds = []
        if res.obb is not None:
            for cid, conf, xyxyxyxy in zip(res.obb.cls.cpu().numpy(), res.obb.conf.cpu().numpy(), res.obb.xyxyxyxy.cpu().numpy()):
                preds.append({"cls": int(cid), "conf": float(conf), "pts": xyxyxyxy})

        # Match GT structures against predictions
        img = cv2.imread(str(img_path))
        overlay = img.copy()

        for g_idx, gt in enumerate(gt_structs):
            gt_pts = gt["pts"]
            gt_center, gt_w, gt_h, gt_angle = get_oriented_box_properties(gt_pts)

            best_iou = 0.0
            best_match = None

            for p in preds:
                iou = polygon_iou(gt_pts, p["pts"])
                if iou > best_iou:
                    best_iou = iou
                    best_match = p

            record = {
                "image": img_path.name,
                "gt_idx": g_idx,
                "best_iou": best_iou,
                "matched": best_iou >= 0.50,
                "gt_angle": gt_angle,
                "gt_dim": [float(gt_w), float(gt_h)],
            }

            if best_match:
                p_center, p_w, p_h, p_angle = get_oriented_box_properties(best_match["pts"])
                center_dist = float(np.linalg.norm(np.array(gt_center) - np.array(p_center)))
                angle_diff = abs(gt_angle - p_angle)
                if angle_diff > 90:
                    angle_diff = abs(180 - angle_diff)
                
                record.update({
                    "pred_cls": names.get(best_match["cls"]),
                    "pred_conf": best_match["conf"],
                    "center_disp_px": center_dist,
                    "angle_diff_deg": angle_diff,
                    "pred_angle": p_angle,
                    "pred_dim": [float(p_w), float(p_h)],
                })
            else:
                record.update({
                    "pred_cls": None,
                    "pred_conf": 0.0,
                    "center_disp_px": None,
                    "angle_diff_deg": None
                })

            analysis_records.append(record)

            # Draw GT on overlay in Green
            cv2.polylines(overlay, [gt_pts.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(overlay, f"GT_struct", (int(gt_pts[0][0]), int(gt_pts[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw predictions on overlay: Red for false alarms/misses, Cyan for partial matches
        for p in preds:
            if p["conf"] >= 0.15:
                color = (255, 0, 0) if p["cls"] == 3 else (0, 165, 255) # Blue for struct, Orange for ruins
                cv2.polylines(overlay, [p["pts"].astype(np.int32)], isClosed=True, color=color, thickness=2)
                lbl_text = f"{names.get(p['cls'])} {p['conf']:.2f}"
                cv2.putText(overlay, lbl_text, (int(p["pts"][0][0]), int(p["pts"][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if overlay_count < 5:
            vis_file = out_dir / f"overlay_{img_path.stem}.jpg"
            cv2.imwrite(str(vis_file), overlay)
            overlay_count += 1

    # Save records
    with open(out_dir / "nalanda_struct_analysis.json", "w") as f:
        json.dump(analysis_records, f, indent=2)

    print(f"Audited {len(analysis_records)} ground-truth structures in Nalanda test set.")
    ious = [r["best_iou"] for r in analysis_records]
    print(f"Max IoU achieved: {max(ious):.4f}")
    print(f"Mean IoU achieved: {np.mean(ious):.4f}")
    matches_50 = sum(1 for r in analysis_records if r["best_iou"] >= 0.50)
    matches_30 = sum(1 for r in analysis_records if r["best_iou"] >= 0.30)
    matches_20 = sum(1 for r in analysis_records if r["best_iou"] >= 0.20)
    print(f"Matches at IoU >= 0.50: {matches_50} / {len(analysis_records)}")
    print(f"Matches at IoU >= 0.30: {matches_30} / {len(analysis_records)}")
    print(f"Matches at IoU >= 0.20: {matches_20} / {len(analysis_records)}")

    # Class confusion of best matches
    pred_classes = [r.get("pred_cls") for r in analysis_records if r.get("pred_cls")]
    from collections import Counter
    print("Predicted class distribution for best IoU matches:", Counter(pred_classes))

if __name__ == "__main__":
    main()
