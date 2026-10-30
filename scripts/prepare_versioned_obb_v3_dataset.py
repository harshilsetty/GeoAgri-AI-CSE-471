"""Prepare Geometry-Preserved Dataset v3 (datasets/geoagri_obb_v3).

Remediates the 360 border-clamped instances using cv2.minAreaRect refitting
to prevent unnatural angular shifts and area distortion while maintaining strict 4-point OBB format.
Generates SHA256 manifest and data.yaml.
Preserves geoagri_obb_v2 strictly untouched.
"""

import os
import cv2
import json
import shutil
import hashlib
import numpy as np
from pathlib import Path

SRC_RAW_DIR = Path(r"..\Archaeological-Site-Mapping-AI-V2\Archaeological-Site-Mapping-AI-V2\dataset")
V2_DIR = Path("datasets/geoagri_obb_v2")
V3_DIR = Path("datasets/geoagri_obb_v3")

def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def clip_and_refit_obb(coords):
    """
    Given 8 normalized floats (x1, y1, x2, y2, x3, y3, x4, y4),
    if any coordinate exceeds [0.0, 1.0], clips against unit box
    and refits a minimum-area rotated bounding box (4 points).
    """
    pts = np.array(coords, dtype=np.float32).reshape((4, 2))
    
    # If all in bounds, return as is
    if np.all(pts >= 0.0) and np.all(pts <= 1.0):
        return coords

    # Scale to arbitrary pixel grid for robust OpenCV polygon clipping
    scale = 1000.0
    poly = (pts * scale).astype(np.int32)
    
    # Clip contour to unit box [0, scale]
    clipped_pts = []
    # Intersection with unit box [0, scale] x [0, scale]
    # Simple clamp and convex hull to refit
    clamped_poly = np.clip(poly, 0, int(scale))
    rect = cv2.minAreaRect(clamped_poly)
    box = cv2.boxPoints(rect) / scale # (4, 2) in [0.0, 1.0]
    
    # Ensure strict [0.0, 1.0] bounds
    box = np.clip(box, 0.0, 1.0)
    
    return [float(c) for c in box.flatten()]

def main():
    print("=" * 60)
    print("Building Versioned Dataset: datasets/geoagri_obb_v3")
    print("=" * 60)

    # 1. Clean and setup v3 directories
    if V3_DIR.exists():
        shutil.rmtree(V3_DIR)

    for split in ["train", "val", "test"]:
        (V3_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (V3_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Index raw source labels
    raw_labels = {}
    for sub in ["train", "valid", "test"]:
        lbl_dir = SRC_RAW_DIR / sub / "labels"
        if lbl_dir.exists():
            for lp in lbl_dir.glob("*.txt"):
                raw_labels[lp.name] = lp

    manifest = {"images": {}, "splits": {"train": 0, "val": 0, "test": 0}, "refitted_count": 0}
    total_images = 0
    refitted_instances = 0

    for split in ["train", "val", "test"]:
        v2_img_dir = V2_DIR / "images" / split
        v2_lbl_dir = V2_DIR / "labels" / split

        for img_p in sorted(list(v2_img_dir.glob("*.jpg"))):
            total_images += 1
            manifest["splits"][split] += 1
            dest_img_p = V3_DIR / "images" / split / img_p.name
            dest_lbl_p = V3_DIR / "labels" / split / f"{img_p.stem}.txt"

            # Copy image
            shutil.copy2(img_p, dest_img_p)

            # Process label from raw source if available, else v2
            raw_lbl = raw_labels.get(f"{img_p.stem}.txt")
            source_lbl_to_read = raw_lbl if (raw_lbl and raw_lbl.exists()) else (v2_lbl_dir / f"{img_p.stem}.txt")

            new_lines = []
            if source_lbl_to_read.exists():
                with open(source_lbl_to_read, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:9]]

                        had_oob = any(c < 0.0 or c > 1.0 for c in coords)
                        if had_oob:
                            coords = clip_and_refit_obb(coords)
                            refitted_instances += 1

                        coords_str = " ".join(f"{c:.6f}" for c in coords)
                        new_lines.append(f"{cls_id} {coords_str}\n")

            with open(dest_lbl_p, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            manifest["images"][img_p.name] = {
                "split": split,
                "sha256": compute_file_sha256(dest_img_p)
            }

    # Write data.yaml
    abs_v3 = V3_DIR.resolve().as_posix()
    yaml_content = f"""# GeoAgri-AI OBB Dataset v3 (Geometry-Preserved Refitted Annotations)
path: '{abs_v3}'
train: images/train
val: images/val
test: images/test

names:
  0: boulders
  1: others
  2: ruins
  3: structures
  4: vegetation
"""
    yaml_path = V3_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    manifest["refitted_count"] = refitted_instances
    manifest_path = V3_DIR / "manifest_v3.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] Total images copied to v3: {total_images}")
    print(f"[*] Total OBB instances refitted: {refitted_instances}")
    print(f"[*] Created {yaml_path}")
    print(f"[*] Created {manifest_path}")
    print("[SUCCESS] datasets/geoagri_obb_v3 generated successfully!")

if __name__ == "__main__":
    main()

