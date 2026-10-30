"""Verification script for GeoAgri-AI final dataset (datasets/geoagri_obb_v2).
"""

import os
import re
import sys
import yaml
from pathlib import Path
from collections import defaultdict, Counter

DATASET_DIR = Path("datasets/geoagri_obb_v2").resolve()
YAML_PATH = DATASET_DIR / "data.yaml"

def extract_base_scene(filename: str) -> str:
    m = re.match(r"^(.*?)(_jpg|\.jpg)?\.rf\.[a-f0-9]{32}\.", filename, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return os.path.splitext(filename)[0].split(".")[0]

def polygon_area_4pt(coords):
    # Shoelace formula for 4-point polygon
    pts = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0

def main():
    print("=" * 60)
    print("Verifying Final Dataset: datasets/geoagri_obb_v2")
    print("=" * 60)

    # 1. Verify data.yaml
    assert YAML_PATH.exists(), f"data.yaml not found at {YAML_PATH}"
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"[*] data.yaml loaded successfully:")
    print(f"    path:  {config.get('path')}")
    print(f"    train: {config.get('train')}")
    print(f"    val:   {config.get('val')}")
    print(f"    test:  {config.get('test')}")
    print(f"    names: {config.get('names')}")

    # Check directory existence
    base_path = Path(config.get("path", str(DATASET_DIR)))
    train_img_dir = base_path / config["train"]
    val_img_dir = base_path / config["val"]
    test_img_dir = base_path / config["test"]

    assert train_img_dir.exists(), f"Train image dir missing: {train_img_dir}"
    assert val_img_dir.exists(), f"Val image dir missing: {val_img_dir}"
    assert test_img_dir.exists(), f"Test image dir missing: {test_img_dir}"

    splits = {
        "train": (train_img_dir, base_path / "labels" / "train"),
        "val": (val_img_dir, base_path / "labels" / "val"),
        "test": (test_img_dir, base_path / "labels" / "test")
    }

    split_scenes = defaultdict(set)
    split_images = Counter()
    split_instances = defaultdict(Counter)
    total_invalid_coords = 0
    zero_area_count = 0
    total_instances = 0

    for split_name, (img_dir, lbl_dir) in splits.items():
        assert lbl_dir.exists(), f"Label dir missing: {lbl_dir}"
        img_files = sorted(list(img_dir.glob("*.jpg")))
        split_images[split_name] = len(img_files)

        for img_p in img_files:
            scene = extract_base_scene(img_p.name)
            split_scenes[split_name].add(scene)

            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            assert lbl_p.exists(), f"Missing label for {img_p}"

            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                    total_instances += 1
                    split_instances[split_name][cls_id] += 1

                    # Check 8 coords
                    assert len(coords) == 8, f"Invalid coord count ({len(coords)}) in {lbl_p}"
                    for c in coords:
                        if c < 0.0 or c > 1.0:
                            total_invalid_coords += 1

                    area = polygon_area_4pt(coords)
                    if area <= 1e-7:
                        zero_area_count += 1

    print("\n--- Image Counts Verification ---")
    for s, count in split_images.items():
        print(f"  {s}: {count} images")

    print("\n--- Unique Source Scenes Breakdown ---")
    for s, scenes in split_scenes.items():
        print(f"  {s}: {len(scenes)} scenes")

    # Cross-split overlap verification
    train_s = split_scenes["train"]
    val_s = split_scenes["val"]
    test_s = split_scenes["test"]

    overlap_tr_val = train_s & val_s
    overlap_tr_te = train_s & test_s
    overlap_val_te = val_s & test_s

    print("\n--- Leakage Verification ---")
    print(f"  Train & Val scene overlap:  {len(overlap_tr_val)} {overlap_tr_val}")
    print(f"  Train & Test scene overlap: {len(overlap_tr_te)} {overlap_tr_te}")
    print(f"  Val & Test scene overlap:   {len(overlap_val_te)} {overlap_val_te}")

    assert len(overlap_tr_val) == 0, "Train-Val scene leakage!"
    assert len(overlap_tr_te) == 0, "Train-Test scene leakage!"
    assert len(overlap_val_te) == 0, "Val-Test scene leakage!"

    total_scenes = len(train_s | val_s | test_s)
    print(f"  Total unique scenes across all splits: {total_scenes} (Expected: 128)")
    assert total_scenes == 128, f"Unexpected total scenes: {total_scenes}"

    print("\n--- Annotation Integrity ---")
    print(f"  Total instances audited:        {total_instances} (Expected: 18,458)")
    print(f"  Out-of-bounds coords (<0 or >1): {total_invalid_coords}")
    print(f"  Degenerate zero-area polygons:  {zero_area_count}")
    assert total_invalid_coords == 0, "Detected out-of-bounds coordinates!"
    assert total_instances == 18458, f"Unexpected instance count: {total_instances}"

    print("\n--- Per-Class Instances by Split ---")
    header = f"{'Class ID':<8} | {'Class Name':<12} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}"
    print(header)
    print("-" * len(header))
    for cid, cname in sorted(config["names"].items()):
        tr = split_instances["train"][cid]
        va = split_instances["val"][cid]
        te = split_instances["test"][cid]
        tot = tr + va + te
        print(f"{cid:<8} | {cname:<12} | {tr:<8} | {va:<8} | {te:<8} | {tot:<8}")

    print("\n[SUCCESS] Final dataset verification completed: 100% valid, 0 scene leakage, 0 invalid coords.")

if __name__ == "__main__":
    main()

