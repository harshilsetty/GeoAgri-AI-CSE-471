"""Audit geometric effects of coordinate clamping on OBB annotations.

Compares original unclipped annotations in Archaeological-Site-Mapping-AI-V2
with clamped annotations in datasets/geoagri_obb_v2.
Measures changes in: area, edge lengths, convexity, orientation.
Generates suspicious_annotations.csv and overlay visual comparisons.
"""

import os
import cv2
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

SRC_DIR = Path(r"..\Archaeological-Site-Mapping-AI-V2\Archaeological-Site-Mapping-AI-V2\dataset")
CLAMPED_DIR = Path("datasets/geoagri_obb_v2")
OUT_DIR = Path("artifacts/agp09/annotation_review")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "boulders",
    1: "others",
    2: "ruins",
    3: "structures",
    4: "vegetation"
}

def poly_area(pts):
    # pts: (4, 2)
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0

def edge_lengths(pts):
    lengths = []
    for i in range(4):
        j = (i + 1) % 4
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        lengths.append(math.hypot(dx, dy))
    return lengths

def is_convex_poly(pts):
    # pts: (4, 2)
    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    signs = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        cp = cross_product(p1, p2, p3)
        if abs(cp) > 1e-7:
            signs.append(cp > 0)
    return len(set(signs)) <= 1

def main():
    print("[*] Indexing source original annotations...")
    src_labels = {}
    src_images = {}
    for sub in ["train", "valid", "test"]:
        lbl_dir = SRC_DIR / sub / "labels"
        img_dir = SRC_DIR / sub / "images"
        if not lbl_dir.exists():
            continue
        for lp in lbl_dir.glob("*.txt"):
            src_labels[lp.name] = lp
        for ip in img_dir.glob("*.*"):
            src_images[ip.name] = ip

    print(f"[*] Found {len(src_labels)} source label files.")

    modified_instances = []
    suspicious_cases = []
    total_clamped_instances = 0
    total_instances = 0

    splits = ["train", "val", "test"]

    for split in splits:
        clamped_lbl_dir = CLAMPED_DIR / "labels" / split
        clamped_img_dir = CLAMPED_DIR / "images" / split

        for c_lbl_p in sorted(list(clamped_lbl_dir.glob("*.txt"))):
            orig_lbl_p = src_labels.get(c_lbl_p.name)
            if not orig_lbl_p or not orig_lbl_p.exists():
                continue

            orig_lines = [l.strip() for l in open(orig_lbl_p, "r", encoding="utf-8") if l.strip()]
            clamped_lines = [l.strip() for l in open(c_lbl_p, "r", encoding="utf-8") if l.strip()]

            # Match lines by class & order
            min_len = min(len(orig_lines), len(clamped_lines))
            total_instances += min_len

            img_p = clamped_img_dir / f"{c_lbl_p.stem}.jpg"

            for idx in range(min_len):
                orig_parts = orig_lines[idx].split()
                clamp_parts = clamped_lines[idx].split()

                cls_id = int(orig_parts[0])
                orig_coords = [float(x) for x in orig_parts[1:9]]
                clamp_coords = [float(x) for x in clamp_parts[1:9]]

                # Did this instance have out-of-bounds coordinates?
                had_oob = any(c < 0.0 or c > 1.0 for c in orig_coords)
                if not had_oob:
                    continue

                total_clamped_instances += 1

                orig_pts = np.array(orig_coords).reshape((4, 2))
                clamp_pts = np.array(clamp_coords).reshape((4, 2))

                orig_area = poly_area(orig_pts)
                clamp_area = poly_area(clamp_pts)
                area_diff = abs(clamp_area - orig_area)
                area_pct_change = (area_diff / max(1e-7, orig_area)) * 100.0

                orig_edges = edge_lengths(orig_pts)
                clamp_edges = edge_lengths(clamp_pts)
                max_edge_change = max(abs(c - o) / max(1e-7, o) for o, c in zip(orig_edges, clamp_edges)) * 100.0

                orig_convex = is_convex_poly(orig_pts)
                clamp_convex = is_convex_poly(clamp_pts)
                lost_convexity = orig_convex and not clamp_convex

                is_suspicious = (
                    area_pct_change > 15.0
                    or max_edge_change > 25.0
                    or lost_convexity
                    or clamp_area < 1e-5
                )

                rec = {
                    "split": split,
                    "filename": c_lbl_p.name.replace(".txt", ".jpg"),
                    "instance_idx": idx,
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                    "orig_area": round(orig_area, 6),
                    "clamp_area": round(clamp_area, 6),
                    "area_pct_change": round(area_pct_change, 2),
                    "max_edge_pct_change": round(max_edge_change, 2),
                    "orig_convex": orig_convex,
                    "clamp_convex": clamp_convex,
                    "lost_convexity": lost_convexity,
                    "is_suspicious": is_suspicious,
                    "min_orig_coord": min(orig_coords),
                    "max_orig_coord": max(orig_coords)
                }

                modified_instances.append(rec)
                if is_suspicious:
                    suspicious_cases.append(rec)

    print(f"\n[*] Total audited instances: {total_instances}")
    print(f"[*] Total instances with clamped coordinates: {total_clamped_instances} ({total_clamped_instances/total_instances*100:.2f}%)")
    print(f"[*] Total suspicious cases flagged: {len(suspicious_cases)} ({len(suspicious_cases)/total_clamped_instances*100:.2f}% of clamped)")

    df_mod = pd.DataFrame(modified_instances)
    df_suspicious = pd.DataFrame(suspicious_cases)

    csv_path = OUT_DIR / "suspicious_annotations.csv"
    df_suspicious.to_csv(csv_path, index=False)
    print(f"[*] Suspicious annotations saved to {csv_path}")

    # Generate visual overlay comparisons for top 12 suspicious cases
    print("[*] Generating overlay comparison images...")
    top_suspicious = df_suspicious.sort_values(by="area_pct_change", ascending=False).head(12)

    for i, (_, row) in enumerate(top_suspicious.iterrows()):
        img_p = CLAMPED_DIR / "images" / row["split"] / row["filename"]
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Re-read coords
        orig_lbl_p = src_labels.get(row["filename"].replace(".jpg", ".txt"))
        clamp_lbl_p = CLAMPED_DIR / "labels" / row["split"] / row["filename"].replace(".jpg", ".txt")

        orig_coords = [float(x) for x in open(orig_lbl_p).readlines()[row["instance_idx"]].split()[1:9]]
        clamp_coords = [float(x) for x in open(clamp_lbl_p).readlines()[row["instance_idx"]].split()[1:9]]

        vis = img.copy()
        pts_orig = (np.array(orig_coords).reshape((4, 2)) * [w, h]).astype(np.int32)
        pts_clamp = (np.array(clamp_coords).reshape((4, 2)) * [w, h]).astype(np.int32)

        # Draw original in red, clamped in green
        cv2.polylines(vis, [pts_orig], isClosed=True, color=(0, 0, 255), thickness=2)   # Red = original unclamped
        cv2.polylines(vis, [pts_clamp], isClosed=True, color=(0, 255, 0), thickness=2) # Green = clamped

        header = f"{row['class_name'].upper()} | Area Change: {row['area_pct_change']}% | Edge Change: {row['max_edge_pct_change']}%"
        cv2.rectangle(vis, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(vis, header, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        out_img_p = OUT_DIR / f"overlay_suspicious_{i+1}_{row['split']}_{row['filename']}"
        cv2.imwrite(str(out_img_p), vis)

    print("[SUCCESS] Geometric audit and review generation complete!")

if __name__ == "__main__":
    main()

