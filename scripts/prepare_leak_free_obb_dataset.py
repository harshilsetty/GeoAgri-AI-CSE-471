"""Prepare Leak-Free Group-Aware Dataset for GeoAgri-AI YOLO-OBB Training.

This script:
1. Reads all images and annotations from the source dataset (read-only).
2. Groups augmented variants by their original 128 source photographs.
3. Performs group-aware stratified splitting across all 4 sites (Hampi, Khajuraho, Pattadakal, Nalanda)
   and ensures the minority 'structures' class is represented in train, val, and test splits.
4. Sanitizes out-of-bounds coordinates (clamping values < 0.0 to 0.0 and > 1.0 to 1.0).
5. Writes the partitioned dataset to `datasets/geoagri_obb_v2` with a valid `data.yaml`.
6. Generates full auditing statistics and verifies zero cross-split scene leakage.
"""

import os
import re
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter

SOURCE_DIR = Path(r"..\Archaeological-Site-Mapping-AI-V2\Archaeological-Site-Mapping-AI-V2\dataset")
DEST_DIR = Path(r"datasets/geoagri_obb_v2")

CLASS_NAMES = {
    0: "boulders",
    1: "others",
    2: "ruins",
    3: "structures",
    4: "vegetation"
}

def extract_base_scene(filename: str) -> str:
    # Roboflow export filenames: <base_name>_jpg.rf.<hash>.jpg or <base_name>.rf.<hash>.jpg
    m = re.match(r"^(.*?)(_jpg|\.jpg)?\.rf\.[a-f0-9]{32}\.", filename, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return os.path.splitext(filename)[0].split(".")[0]

def detect_site(scene_name: str) -> str:
    name_lower = scene_name.lower()
    if "hampi" in name_lower:
        return "Hampi"
    elif "khajuraho" in name_lower:
        return "Khajuraho"
    elif "pattadakal" in name_lower or "patadakal" in name_lower:
        return "Pattadakal"
    elif "nalanda" in name_lower:
        return "Nalanda"
    return "Unknown"

def main():
    print(f"[*] Verifying source dataset path: {SOURCE_DIR.resolve()}")
    if not SOURCE_DIR.exists():
        print(f"[!] Error: Source dataset path does not exist: {SOURCE_DIR}")
        sys.exit(1)

    # 1. Discover all images and labels in source splits
    source_subdirs = ["train", "valid", "test"]
    all_items = []
    
    for split in source_subdirs:
        img_dir = SOURCE_DIR / split / "images"
        lbl_dir = SOURCE_DIR / split / "labels"
        if not img_dir.exists():
            continue
        
        for img_path in img_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            all_items.append({
                "img_path": img_path,
                "lbl_path": lbl_path if lbl_path.exists() else None,
                "filename": img_path.name,
                "source_split": split
            })

    print(f"[*] Found total images: {len(all_items)}")

    # 2. Group items by base scene
    scene_items = defaultdict(list)
    scene_classes = defaultdict(set)
    scene_class_counts = defaultdict(Counter)
    total_coord_fixes = 0

    for item in all_items:
        base_scene = extract_base_scene(item["filename"])
        item["base_scene"] = base_scene
        scene_items[base_scene].append(item)
        
        # Read label to inspect classes
        if item["lbl_path"] and item["lbl_path"].exists():
            with open(item["lbl_path"], "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    scene_classes[base_scene].add(cls_id)
                    scene_class_counts[base_scene][cls_id] += 1
                    # Check coordinates
                    coords = [float(p) for p in parts[1:]]
                    for c in coords:
                        if c < 0.0 or c > 1.0:
                            total_coord_fixes += 1

    unique_scenes = sorted(list(scene_items.keys()))
    print(f"[*] Total unique source scenes: {len(unique_scenes)}")
    print(f"[*] Total out-of-bounds coordinate occurrences across all instances: {total_coord_fixes}")

    # Inspect site distributions
    site_scenes = defaultdict(list)
    has_structures_scenes = []
    no_structures_scenes = []

    for scene in unique_scenes:
        site = detect_site(scene)
        site_scenes[site].append(scene)
        if 3 in scene_classes[scene]:
            has_structures_scenes.append(scene)
        else:
            no_structures_scenes.append(scene)

    print("\n--- Source Scene Site Breakdown ---")
    for site, scenes in sorted(site_scenes.items()):
        print(f"  {site}: {len(scenes)} scenes")
    print(f"  Scenes containing 'structures' class (3): {len(has_structures_scenes)}")
    print(f"  Scenes without 'structures' class: {len(no_structures_scenes)}")

    # 3. Stratified Group Splitting (Target: ~70% Train, ~15% Val, ~15% Test)
    # Total 128 scenes: 90 Train, 20 Val, 18 Test
    # For scenes with structures (21 scenes): 15 Train, 3 Val, 3 Test
    random.seed(42)

    train_scenes = set()
    val_scenes = set()
    test_scenes = set()

    # First, allocate scenes containing structures per site
    struct_by_site = defaultdict(list)
    for scene in has_structures_scenes:
        struct_by_site[detect_site(scene)].append(scene)

    for site, s_list in sorted(struct_by_site.items()):
        random.shuffle(s_list)
        # Allocate: at least 1 in val and test if possible
        n = len(s_list)
        if n == 1:
            train_scenes.add(s_list[0])
        elif n == 2:
            train_scenes.add(s_list[0])
            val_scenes.add(s_list[1])
        elif n >= 3:
            test_scenes.add(s_list.pop())
            val_scenes.add(s_list.pop())
            for s in s_list:
                train_scenes.add(s)

    # Now allocate remaining scenes per site
    rem_by_site = defaultdict(list)
    for scene in no_structures_scenes:
        rem_by_site[detect_site(scene)].append(scene)

    for site, s_list in sorted(rem_by_site.items()):
        random.shuffle(s_list)
        total_site = len(site_scenes[site])
        target_val = max(1, round(total_site * 0.15))
        target_test = max(1, round(total_site * 0.15))
        
        # Current counts for this site
        cur_val = sum(1 for s in val_scenes if detect_site(s) == site)
        cur_test = sum(1 for s in test_scenes if detect_site(s) == site)
        
        need_val = max(0, target_val - cur_val)
        need_test = max(0, target_test - cur_test)
        
        for _ in range(need_val):
            if s_list:
                val_scenes.add(s_list.pop())
        for _ in range(need_test):
            if s_list:
                test_scenes.add(s_list.pop())
        for s in s_list:
            train_scenes.add(s)

    # Verify counts
    print(f"\n--- Split Scene Allocations (Zero Leakage) ---")
    print(f"Train scenes: {len(train_scenes)} ({len(train_scenes)/len(unique_scenes)*100:.1f}%)")
    print(f"Val scenes:   {len(val_scenes)} ({len(val_scenes)/len(unique_scenes)*100:.1f}%)")
    print(f"Test scenes:  {len(test_scenes)} ({len(test_scenes)/len(unique_scenes)*100:.1f}%)")
    print(f"Total:        {len(train_scenes) + len(val_scenes) + len(test_scenes)}")

    # Overlap assertions
    assert len(train_scenes & val_scenes) == 0, "Train and Val scene overlap detected!"
    assert len(train_scenes & test_scenes) == 0, "Train and Test scene overlap detected!"
    assert len(val_scenes & test_scenes) == 0, "Val and Test scene overlap detected!"
    assert len(train_scenes | val_scenes | test_scenes) == len(unique_scenes), "Not all scenes allocated!"

    # 4. Prepare Destination Directory
    if DEST_DIR.exists():
        print(f"[*] Removing existing destination: {DEST_DIR}")
        shutil.rmtree(DEST_DIR)

    for split in ["train", "val", "test"]:
        (DEST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DEST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 5. Copy images and sanitize annotations
    split_map = {}
    for s in train_scenes:
        split_map[s] = "train"
    for s in val_scenes:
        split_map[s] = "val"
    for s in test_scenes:
        split_map[s] = "test"

    split_image_counts = Counter()
    split_instance_counts = defaultdict(Counter)
    clamped_points_count = 0

    for item in all_items:
        base_scene = item["base_scene"]
        target_split = split_map[base_scene]
        dest_img_path = DEST_DIR / "images" / target_split / item["filename"]
        dest_lbl_path = DEST_DIR / "labels" / target_split / f"{Path(item['filename']).stem}.txt"

        # Copy image
        shutil.copy2(item["img_path"], dest_img_path)
        split_image_counts[target_split] += 1

        # Process and write sanitized label
        if item["lbl_path"] and item["lbl_path"].exists():
            sanitized_lines = []
            with open(item["lbl_path"], "r", encoding="utf-8") as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                    
                    # Sanitize coords
                    clamped_coords = []
                    for c in coords:
                        if c < 0.0:
                            clamped_coords.append(0.0)
                            clamped_points_count += 1
                        elif c > 1.0:
                            clamped_coords.append(1.0)
                            clamped_points_count += 1
                        else:
                            clamped_coords.append(c)

                    split_instance_counts[target_split][cls_id] += 1
                    formatted_coords = " ".join(f"{c:.6f}" for c in clamped_coords)
                    sanitized_lines.append(f"{cls_id} {formatted_coords}\n")

            with open(dest_lbl_path, "w", encoding="utf-8") as f_out:
                f_out.writelines(sanitized_lines)
        else:
            # Empty label file
            dest_lbl_path.touch()

    # 6. Generate data.yaml with absolute/portable path
    abs_dest_dir = DEST_DIR.resolve().as_posix()
    yaml_content = f"""# GeoAgri-AI Leak-Free OBB Dataset Configuration
# Created via group-aware stratification across 128 unique historical scenes.
path: '{abs_dest_dir}'
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
    yaml_path = DEST_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[*] Created data.yaml at {yaml_path}")
    print(f"[*] Clamped coordinate points: {clamped_points_count}")

    print("\n--- Final Split Image Counts ---")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {split_image_counts[s]} images")

    print("\n--- Final Split Instance Counts by Class ---")
    header = f"{'Class':<15} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}"
    print(header)
    print("-" * len(header))
    for cls_id in range(5):
        c_name = CLASS_NAMES[cls_id]
        tr = split_instance_counts["train"][cls_id]
        va = split_instance_counts["val"][cls_id]
        te = split_instance_counts["test"][cls_id]
        tot = tr + va + te
        print(f"{c_name:<15} | {tr:<8} | {va:<8} | {te:<8} | {tot:<8}")

    print("\n[SUCCESS] Leak-free group-aware dataset preparation complete!")

if __name__ == "__main__":
    main()

