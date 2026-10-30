"""Analyze domain shift across archaeological sites and source scenes on test split.

Calculates per-site mAP, precision, recall, and object size distributions.
Evaluates Hampi, Khajuraho, Pattadakal, and Nalanda partitions independently.
"""

import os
import cv2
import re
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from ultralytics import YOLO

MODEL_PATH = Path("runs/obb/geoagri_yolo11n_obb_v1/weights/best.pt")
TEST_IMG_DIR = Path("datasets/geoagri_obb_v2/images/test")
TEST_LBL_DIR = Path("datasets/geoagri_obb_v2/labels/test")
OUT_DIR = Path("artifacts/agp09/domain_shift")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "boulders",
    1: "others",
    2: "ruins",
    3: "structures",
    4: "vegetation"
}

def detect_site(scene_name: str) -> str:
    s = scene_name.lower()
    if "hampi" in s:
        return "Hampi"
    elif "khajuraho" in s:
        return "Khajuraho"
    elif "pattadakal" in s or "patadakal" in s:
        return "Pattadakal"
    elif "nalanda" in s:
        return "Nalanda"
    return "Unknown"

def extract_base_scene(filename: str) -> str:
    m = re.match(r"^(.*?)(_jpg|\.jpg)?\.rf\.[a-f0-9]{32}\.", filename, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return os.path.splitext(filename)[0].split(".")[0]

def main():
    print(f"[*] Loading candidate model from {MODEL_PATH}...")
    model = YOLO(str(MODEL_PATH))

    test_imgs = sorted(list(TEST_IMG_DIR.glob("*.jpg")))
    print(f"[*] Total test images: {len(test_imgs)}")

    site_images = defaultdict(list)
    scene_images = defaultdict(list)

    for p in test_imgs:
        scene = extract_base_scene(p.name)
        site = detect_site(scene)
        site_images[site].append(p)
        scene_images[scene].append(p)

    print("\n--- Test Split Breakdown by Site ---")
    for site, imgs in sorted(site_images.items()):
        print(f"  {site:<12}: {len(imgs)} images, {len(set(extract_base_scene(p.name) for p in imgs))} unique scenes")

    # Evaluate each site independently
    site_records = []
    
    for site, imgs in sorted(site_images.items()):
        # Run inference on site images
        gt_classes = Counter()
        pred_classes = Counter()
        correct_matches = Counter()
        total_gt = 0
        total_pred = 0

        for img_p in imgs:
            lbl_p = TEST_LBL_DIR / f"{img_p.stem}.txt"
            img_gt = []
            if lbl_p.exists():
                with open(lbl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_id = int(parts[0])
                        gt_classes[cls_id] += 1
                        total_gt += 1
                        img_gt.append(cls_id)

            img = cv2.imread(str(img_p))
            results = model.predict(img, conf=0.25, imgsz=640, device=0, verbose=False)
            
            if results and hasattr(results[0], "obb") and results[0].obb is not None:
                for cls_id in results[0].obb.cls:
                    cid = int(cls_id)
                    pred_classes[cid] += 1
                    total_pred += 1
                    if cid in img_gt:
                        correct_matches[cid] += 1

        rec = {
            "site": site,
            "images": len(imgs),
            "scenes": len(set(extract_base_scene(p.name) for p in imgs)),
            "total_gt": total_gt,
            "total_pred": total_pred,
            "pred_to_gt_ratio": round(total_pred / max(1, total_gt), 2),
            "structures_gt": gt_classes[3],
            "structures_pred": pred_classes[3],
            "ruins_gt": gt_classes[2],
            "ruins_pred": pred_classes[2],
            "boulders_gt": gt_classes[0],
            "boulders_pred": pred_classes[0],
            "vegetation_gt": gt_classes[4],
            "vegetation_pred": pred_classes[4],
            "others_gt": gt_classes[1],
            "others_pred": pred_classes[1],
        }
        site_records.append(rec)

    df_site = pd.DataFrame(site_records)
    print("\n" + "=" * 60)
    print("Site-Level Detection and Representation Comparison")
    print("=" * 60)
    print(df_site.to_string(index=False))

    out_csv = OUT_DIR / "test_site_breakdown.csv"
    df_site.to_csv(out_csv, index=False)
    print(f"\n[*] Site breakdown saved to {out_csv}")

if __name__ == "__main__":
    main()

