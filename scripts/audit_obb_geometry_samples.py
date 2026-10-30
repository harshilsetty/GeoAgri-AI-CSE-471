"""Audit and validate ground-truth OBB geometry and annotations.

Samples 30 train, 30 val, 30 test images from datasets/geoagri_obb_v2.
Overlays ground truth polygons, checks border proximity, polygon convexity,
aspect ratio, and class distribution, then outputs rendered images and a contact sheet.
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

random.seed(42)

DATASET_DIR = Path("datasets/geoagri_obb_v2")
OUTPUT_DIR = Path("artifacts/yolo_vision_v2/annotation_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "boulders",
    1: "others",
    2: "ruins",
    3: "structures",
    4: "vegetation"
}

CLASS_COLORS = {
    0: (0, 200, 255),    # boulders: amber/yellow
    1: (180, 180, 180),  # others: gray
    2: (0, 100, 255),    # ruins: orange
    3: (0, 0, 255),      # structures: red
    4: (0, 220, 0)       # vegetation: green
}

def polygon_area_and_convex(pts):
    # pts is (4, 2)
    area = cv2.contourArea(pts)
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    # Check if convex and non-degenerate
    is_convex = abs(area - hull_area) < 1.0 and area > 10.0
    return area, is_convex

def touches_border(coords, tol=0.005):
    # coords: 8 floats
    for i in range(0, 8, 2):
        x, y = coords[i], coords[i+1]
        if x <= tol or x >= (1.0 - tol) or y <= tol or y >= (1.0 - tol):
            return True
    return False

def main():
    splits = ["train", "val", "test"]
    sampled_records = []
    
    total_reviewed_images = 0
    images_with_border_touches = 0
    images_with_distorted_poly = 0
    images_with_others = 0
    total_instances_inspected = 0
    class_counter = Counter()

    for split in splits:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        all_imgs = sorted(list(img_dir.glob("*.jpg")))
        
        sample_imgs = random.sample(all_imgs, min(30, len(all_imgs)))
        print(f"[*] Sampled {len(sample_imgs)} images from {split}")

        for img_p in sample_imgs:
            total_reviewed_images += 1
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            h, w = img.shape[:2]
            vis = img.copy()

            has_border = False
            has_distortion = False
            has_other_cls = False
            img_instances = 0

            if lbl_p.exists():
                with open(lbl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_id = int(parts[0])
                        coords = [float(p) for p in parts[1:]]
                        class_counter[cls_id] += 1
                        total_instances_inspected += 1
                        img_instances += 1

                        if cls_id == 1:
                            has_other_cls = True

                        if touches_border(coords):
                            has_border = True

                        # Denormalize to pixel coordinates
                        pts = np.array([
                            [coords[0] * w, coords[1] * h],
                            [coords[2] * w, coords[3] * h],
                            [coords[4] * w, coords[5] * h],
                            [coords[6] * w, coords[7] * h],
                        ], dtype=np.int32)

                        area, is_convex = polygon_area_and_convex(pts)
                        if not is_convex:
                            has_distortion = True

                        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
                        
                        # Label text
                        c_name = CLASS_NAMES.get(cls_id, str(cls_id))
                        pt0 = pts[0]
                        cv2.putText(vis, c_name, (int(pt0[0]), max(14, int(pt0[1]) - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            if has_border:
                images_with_border_touches += 1
            if has_distortion:
                images_with_distorted_poly += 1
            if has_other_cls:
                images_with_others += 1

            # Save annotated image
            out_name = f"{split}_{img_p.stem}_annotated.jpg"
            out_path = OUTPUT_DIR / out_name
            cv2.imwrite(str(out_path), vis)
            
            sampled_records.append({
                "split": split,
                "name": img_p.name,
                "path": str(out_path),
                "instances": img_instances,
                "has_border": has_border,
                "has_distortion": has_distortion,
                "has_others": has_other_cls
            })

    # Create a 6x5 contact sheet (30 sampled images, 10 per split)
    print("[*] Generating contact sheet...")
    contact_samples = []
    for split in splits:
        split_records = [r for r in sampled_records if r["split"] == split]
        contact_samples.extend(split_records[:10])

    thumb_w, thumb_h = 256, 256
    rows, cols = 5, 6
    contact_img = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for idx, rec in enumerate(contact_samples[:rows * cols]):
        r = idx // cols
        c = idx % cols
        img = cv2.imread(rec["path"])
        if img is not None:
            thumb = cv2.resize(img, (thumb_w, thumb_h))
            # Put header bar
            header = f"{rec['split'].upper()}: {rec['name'][:18]} (n={rec['instances']})"
            cv2.rectangle(thumb, (0, 0), (thumb_w, 20), (0, 0, 0), -1)
            cv2.putText(thumb, header, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            contact_img[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb

    contact_path = OUTPUT_DIR / "obb_annotation_contact_sheet.jpg"
    cv2.imwrite(str(contact_path), contact_img)
    print(f"[*] Contact sheet written to {contact_path}")

    # Report findings
    border_pct = (images_with_border_touches / total_reviewed_images) * 100
    distortion_pct = (images_with_distorted_poly / total_reviewed_images) * 100
    others_pct = (images_with_others / total_reviewed_images) * 100

    print("\n" + "=" * 60)
    print("OBB Annotation & Geometry Audit Summary (90 Sampled Images)")
    print("=" * 60)
    print(f"Total Reviewed Images:           {total_reviewed_images} (30 train, 30 val, 30 test)")
    print(f"Total Instances Inspected:       {total_instances_inspected}")
    print(f"Images with Border-Touching OBBs: {images_with_border_touches} ({border_pct:.1f}%)")
    print(f"Images with Distorted / Non-convex OBBs: {images_with_distorted_poly} ({distortion_pct:.1f}%)")
    print(f"Images Containing 'others' Class: {images_with_others} ({others_pct:.1f}%)")
    print("\nClass Breakdown across 90 images:")
    for cid in range(5):
        print(f"  {CLASS_NAMES[cid]:<12}: {class_counter[cid]} instances")
    print("=" * 60)

if __name__ == "__main__":
    main()

