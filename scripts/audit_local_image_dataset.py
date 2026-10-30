"""Comprehensive Dataset Quality & Annotation Auditor for GeoAgri-AI / YOLO Instance Segmentation.

Author: Senior Computer Vision Engineer & Dataset Quality Auditor
Project: GeoAgri-AI (Archaeological & Terrain Instance Segmentation Module)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Supported image extensions
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Default class configuration from data.yaml
DEFAULT_CLASSES = {
    0: "boulders",
    1: "others",
    2: "ruins",
    3: "structures",
    4: "vegetation",
}

CLASS_COLORS = {
    0: (255, 200, 0),    # boulders (cyan/amber)
    1: (180, 180, 180),  # others (gray)
    2: (0, 100, 255),    # ruins (orange/red)
    3: (255, 50, 50),    # structures (blue)
    4: (0, 220, 0),      # vegetation (green)
}


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_data_yaml(yaml_path: str) -> Dict[int, str]:
    classes = {}
    if not os.path.exists(yaml_path):
        return DEFAULT_CLASSES
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", {})
        if isinstance(names, dict):
            for k, v in names.items():
                classes[int(k)] = str(v).strip()
        elif isinstance(names, list):
            for i, v in enumerate(names):
                classes[i] = str(v).strip()
    except Exception as exc:
        print(f"[Warning] Failed to parse {yaml_path} with pyyaml ({exc}). Using standard fallback.")
        # Simple manual regex parser for data.yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        in_names = False
        for line in lines:
            line_s = line.strip()
            if line_s.startswith("names:"):
                in_names = True
                continue
            if in_names:
                m = re.match(r"^(\d+)\s*:\s*(.+)$", line_s)
                if m:
                    classes[int(m.group(1))] = m.group(2).strip()
                elif line_s and not line_s.startswith("#"):
                    break
    return classes if classes else DEFAULT_CLASSES


def extract_base_source_id(filename: str) -> str:
    # Roboflow export filenames follow: <base_name>_jpg.rf.<hash>.jpg or <base_name>.rf.<hash>.jpg
    m = re.match(r"^(.*?)(_jpg|\.jpg)?\.rf\.[a-f0-9]{32}\.", filename, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: remove extension
    base = os.path.splitext(filename)[0]
    return base.split(".")[0]


def audit_image_file(img_path: str) -> Dict[str, Any]:
    file_size = os.path.getsize(img_path)
    sha256 = compute_sha256(img_path)

    if file_size == 0:
        return {
            "file_size": 0,
            "width": 0,
            "height": 0,
            "aspect_ratio": 0.0,
            "format": "UNKNOWN",
            "mode": "UNKNOWN",
            "is_decodable": False,
            "is_blank": True,
            "is_corrupted": True,
            "sha256": sha256,
            "error": "Zero-byte file",
        }

    try:
        with Image.open(img_path) as pil_img:
            width, height = pil_img.size
            img_format = pil_img.format or os.path.splitext(img_path)[1].upper().replace(".", "")
            img_mode = pil_img.mode
            pil_img.verify()
    except Exception as exc:
        return {
            "file_size": file_size,
            "width": 0,
            "height": 0,
            "aspect_ratio": 0.0,
            "format": "CORRUPT",
            "mode": "CORRUPT",
            "is_decodable": False,
            "is_blank": False,
            "is_corrupted": True,
            "sha256": sha256,
            "error": f"PIL decode failure: {exc}",
        }

    # Open with OpenCV to check matrix contents and blankness
    cv_img = cv2.imread(img_path)
    if cv_img is None:
        return {
            "file_size": file_size,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / max(1, height), 4),
            "format": img_format,
            "mode": img_mode,
            "is_decodable": False,
            "is_blank": False,
            "is_corrupted": True,
            "sha256": sha256,
            "error": "OpenCV imread returned None",
        }

    # Check for blank / near-blank images
    std_dev = float(np.std(cv_img))
    is_blank = std_dev < 1.0

    return {
        "file_size": file_size,
        "width": width,
        "height": height,
        "aspect_ratio": round(width / max(1, height), 4),
        "format": img_format,
        "mode": img_mode,
        "is_decodable": True,
        "is_blank": is_blank,
        "is_corrupted": False,
        "sha256": sha256,
        "std_dev": round(std_dev, 2),
        "error": None,
    }


def audit_label_file(
    lbl_path: str,
    valid_classes: Dict[int, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    instances = []
    issues = []

    if not os.path.exists(lbl_path):
        return instances, issues

    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            issues.append({
                "issue_type": "MALFORMED_LINE",
                "line_index": line_idx,
                "detail": f"Too few tokens ({len(parts)})",
                "raw_line": line[:60],
            })
            continue

        try:
            cls_id = int(parts[0])
        except ValueError:
            issues.append({
                "issue_type": "NON_INTEGER_CLASS",
                "line_index": line_idx,
                "detail": f"Class '{parts[0]}' is not an integer",
                "raw_line": line[:60],
            })
            continue

        if cls_id not in valid_classes:
            issues.append({
                "issue_type": "INVALID_CLASS_ID",
                "line_index": line_idx,
                "detail": f"Class ID {cls_id} not in data.yaml classes {list(valid_classes.keys())}",
                "raw_line": line[:60],
            })

        # Coordinates
        raw_coords = parts[1:]
        coord_count = len(raw_coords)
        if coord_count % 2 != 0:
            issues.append({
                "issue_type": "ODD_COORDINATE_COUNT",
                "line_index": line_idx,
                "detail": f"Odd number of coordinate numbers ({coord_count})",
                "raw_line": line[:60],
            })
            continue

        num_points = coord_count // 2
        try:
            coords = [float(c) for c in raw_coords]
        except ValueError:
            issues.append({
                "issue_type": "NON_NUMERIC_COORDINATE",
                "line_index": line_idx,
                "detail": "Failed to parse coordinate as float",
                "raw_line": line[:60],
            })
            continue

        # Check bounds
        out_of_bounds = [c for c in coords if c < -0.05 or c > 1.05]
        clipped_in_bounds = [c for c in coords if (c < 0.0 or c > 1.0) and (-0.05 <= c <= 1.05)]
        if out_of_bounds:
            issues.append({
                "issue_type": "COORDINATE_OUT_OF_BOUNDS",
                "line_index": line_idx,
                "detail": f"Values significantly outside [0, 1]: {out_of_bounds}",
                "raw_line": line[:60],
            })

        # Area and geometry check
        pts = np.array(coords).reshape(-1, 2)
        # Polygon area via Shoelace formula
        x = pts[:, 0]
        y = pts[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        instances.append({
            "line_index": line_idx,
            "class_id": cls_id,
            "class_name": valid_classes.get(cls_id, f"unknown_{cls_id}"),
            "num_points": num_points,
            "area_normalized": float(area),
            "is_four_point_quad": (num_points == 4),
            "coords": coords,
        })

    return instances, issues


def generate_diagnostic_contact_sheet(
    dataset_dir: str,
    output_image_path: str,
    valid_classes: Dict[int, str],
    inventory: List[Dict[str, Any]],
):
    print("Generating diagnostic contact sheet...")
    # Select representative samples: 2 from test, 2 from valid, 4 from train across classes
    sample_targets = []
    seen_classes = set()

    for item in inventory:
        if not item["instances"]:
            continue
        item_classes = {inst["class_id"] for inst in item["instances"]}
        # Prioritize unseen classes
        if not item_classes.issubset(seen_classes) or len(sample_targets) < 8:
            sample_targets.append(item)
            seen_classes.update(item_classes)
            if len(sample_targets) >= 8:
                break

    if not sample_targets:
        return

    canvas_tiles = []
    tile_size = 400

    for sample in sample_targets:
        img_path = sample["full_path"]
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Draw instance annotations
        for inst in sample["instances"]:
            cls_id = inst["class_id"]
            color = CLASS_COLORS.get(cls_id, (0, 255, 255))
            pts = np.array(inst["coords"]).reshape(-1, 2)
            pixel_pts = (pts * [w, h]).astype(np.int32)

            # Draw polygon overlay
            overlay = img.copy()
            cv2.fillPoly(overlay, [pixel_pts], color)
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
            cv2.polylines(img, [pixel_pts], isClosed=True, color=color, thickness=2)

            # Label text
            top_pt = pixel_pts[np.argmin(pixel_pts[:, 1])]
            label_text = f"{inst['class_name']} ({inst['num_points']}pts)"
            cv2.putText(
                img,
                label_text,
                (max(5, top_pt[0]), max(15, top_pt[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        # Header overlay
        header_text = f"[{sample['split'].upper()}] {sample['source_base_id'][:16]} ({len(sample['instances'])} objs)"
        cv2.rectangle(img, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(img, header_text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        resized = cv2.resize(img, (tile_size, tile_size))
        canvas_tiles.append(resized)

    # Arrange in a 2x4 grid
    while len(canvas_tiles) < 8:
        canvas_tiles.append(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))

    row1 = np.hstack(canvas_tiles[:4])
    row2 = np.hstack(canvas_tiles[4:8])
    contact_sheet = np.vstack([row1, row2])

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, contact_sheet)
    print(f"Contact sheet saved to: {output_image_path}")


def run_full_dataset_audit(dataset_dir: str, docs_dir: str):
    print("=" * 70)
    print("GEOAGRI-AI: LOCAL IMAGE DATASET QUALITY & INTEGRITY AUDIT")
    print("=" * 70)
    print(f"Target Dataset Directory: {os.path.abspath(dataset_dir)}")

    # 1. Parse data.yaml
    yaml_candidates = [
        os.path.join(dataset_dir, "..", "data.yaml"),
        os.path.join(dataset_dir, "data.yaml"),
        os.path.join(dataset_dir, "..", "config", "data.yaml"),
    ]
    data_yaml_path = None
    for y_path in yaml_candidates:
        if os.path.exists(y_path):
            data_yaml_path = os.path.abspath(y_path)
            break

    classes = parse_data_yaml(data_yaml_path) if data_yaml_path else DEFAULT_CLASSES
    print(f"Loaded {len(classes)} Classes from {data_yaml_path}:")
    for cid, cname in classes.items():
        print(f"  Class {cid}: {cname}")

    # 2. Check splits
    splits = ["train", "valid", "test"]
    split_dirs = {}
    for s in splits:
        img_d = os.path.join(dataset_dir, s, "images")
        lbl_d = os.path.join(dataset_dir, s, "labels")
        if os.path.exists(img_d):
            split_dirs[s] = {"images": img_d, "labels": lbl_d}

    if not split_dirs:
        print(f"[Error] No standard train/valid/test split directories found in {dataset_dir}")
        sys.exit(1)

    inventory = []
    all_issues = []
    sha256_to_paths = {}
    base_id_to_splits = {}

    total_images_scanned = 0
    t0 = time.time()

    for split, dirs in split_dirs.items():
        img_dir = dirs["images"]
        lbl_dir = dirs["labels"]
        img_files = sorted([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTS])

        print(f"\nScanning split [{split}]: {len(img_files)} images found...")

        for img_name in img_files:
            total_images_scanned += 1
            full_img_path = os.path.join(img_dir, img_name)
            img_meta = audit_image_file(full_img_path)

            # Check duplicate hashes
            sha = img_meta["sha256"]
            if sha in sha256_to_paths:
                sha256_to_paths[sha].append(full_img_path)
            else:
                sha256_to_paths[sha] = [full_img_path]

            # Source scene ID before .rf.
            base_id = extract_base_source_id(img_name)
            if base_id not in base_id_to_splits:
                base_id_to_splits[base_id] = set()
            base_id_to_splits[base_id].add(split)

            # Label pairing
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            full_lbl_path = os.path.join(lbl_dir, lbl_name)
            has_label_file = os.path.exists(full_lbl_path)

            instances = []
            label_issues = []
            if has_label_file:
                instances, label_issues = audit_label_file(full_lbl_path, classes)
                for issue in label_issues:
                    issue["split"] = split
                    issue["image_file"] = img_name
                    issue["label_file"] = lbl_name
                    all_issues.append(issue)
            else:
                all_issues.append({
                    "split": split,
                    "image_file": img_name,
                    "label_file": lbl_name,
                    "issue_type": "MISSING_LABEL_FILE",
                    "line_index": -1,
                    "detail": "Image has no corresponding .txt file in labels directory",
                    "raw_line": "",
                })

            inventory.append({
                "split": split,
                "image_file": img_name,
                "label_file": lbl_name if has_label_file else None,
                "source_base_id": base_id,
                "full_path": full_img_path,
                "file_size_bytes": img_meta["file_size"],
                "width": img_meta["width"],
                "height": img_meta["height"],
                "aspect_ratio": img_meta["aspect_ratio"],
                "format": img_meta["format"],
                "mode": img_meta["mode"],
                "is_decodable": img_meta["is_decodable"],
                "is_blank": img_meta["is_blank"],
                "is_corrupted": img_meta["is_corrupted"],
                "sha256": img_meta["sha256"],
                "instance_count": len(instances),
                "is_negative_image": (len(instances) == 0 and has_label_file),
                "instances": instances,
            })

    duration = time.time() - t0
    print(f"\nCompleted scan of {total_images_scanned} images in {duration:.2f} seconds.")

    # 3. Duplicate and Leakage Analysis
    exact_duplicate_groups = {k: v for k, v in sha256_to_paths.items() if len(v) > 1}
    cross_split_scenes = {k: v for k, v in base_id_to_splits.items() if len(v) > 1}

    # Train / Valid / Test scene sets
    scenes_by_split = {s: {item["source_base_id"] for item in inventory if item["split"] == s} for s in splits}
    tv_leak = scenes_by_split["train"] & scenes_by_split.get("valid", set())
    tt_leak = scenes_by_split["train"] & scenes_by_split.get("test", set())
    vt_leak = scenes_by_split.get("valid", set()) & scenes_by_split.get("test", set())
    all_three_leak = scenes_by_split["train"] & scenes_by_split.get("valid", set()) & scenes_by_split.get("test", set())

    # 4. Class Distribution Statistics
    class_stats = []
    for cid, cname in classes.items():
        entry = {"class_id": cid, "class_name": cname}
        for s in splits:
            s_items = [item for item in inventory if item["split"] == s]
            s_insts = [inst for item in s_items for inst in item["instances"] if inst["class_id"] == cid]
            s_imgs_with_c = len([item for item in s_items if any(inst["class_id"] == cid for inst in item["instances"])])
            entry[f"{s}_instances"] = len(s_insts)
            entry[f"{s}_images"] = s_imgs_with_c
            entry[f"{s}_img_pct"] = round((s_imgs_with_c / max(1, len(s_items))) * 100, 2)
        entry["total_instances"] = sum(entry[f"{s}_instances"] for s in splits)
        class_stats.append(entry)

    class_stats_df = pd.DataFrame(class_stats)

    # 5. Coordinate & Geometry Profile
    all_instances = [inst for item in inventory for inst in item["instances"]]
    point_counts = {}
    for inst in all_instances:
        n = inst["num_points"]
        point_counts[n] = point_counts.get(n, 0) + 1

    # 6. Save Machine-Readable Artifacts
    os.makedirs(docs_dir, exist_ok=True)

    # Inventory CSV
    inv_df = pd.DataFrame([
        {
            "split": item["split"],
            "image_file": item["image_file"],
            "source_base_id": item["source_base_id"],
            "file_size_bytes": item["file_size_bytes"],
            "width": item["width"],
            "height": item["height"],
            "aspect_ratio": item["aspect_ratio"],
            "format": item["format"],
            "is_decodable": item["is_decodable"],
            "is_blank": item["is_blank"],
            "is_corrupted": item["is_corrupted"],
            "instance_count": item["instance_count"],
            "is_negative_image": item["is_negative_image"],
            "sha256": item["sha256"],
        }
        for item in inventory
    ])
    inv_csv_path = os.path.join(docs_dir, "local_image_dataset_inventory.csv")
    inv_df.to_csv(inv_csv_path, index=False)
    print(f"Saved Inventory CSV: {inv_csv_path} ({len(inv_df)} rows)")

    # Annotation Issues CSV
    issues_df = pd.DataFrame(all_issues)
    issues_csv_path = os.path.join(docs_dir, "local_image_dataset_annotation_issues.csv")
    issues_df.to_csv(issues_csv_path, index=False)
    print(f"Saved Issues CSV: {issues_csv_path} ({len(issues_df)} issues logged)")

    # Class Distribution CSV
    class_csv_path = os.path.join(docs_dir, "local_image_dataset_class_distribution.csv")
    class_stats_df.to_csv(class_csv_path, index=False)
    print(f"Saved Class Distribution CSV: {class_csv_path}")

    # Contact Sheet
    contact_sheet_path = os.path.join(docs_dir, "local_image_dataset_contact_sheet.jpg")
    try:
        generate_diagnostic_contact_sheet(dataset_dir, contact_sheet_path, classes, inventory)
    except Exception as exc:
        print(f"[Warning] Failed to generate contact sheet: {exc}")

    # 7. Print Summary Metrics
    print("\n" + "=" * 70)
    print("DATASET AUDIT SUMMARY METRICS")
    print("=" * 70)
    print(f"Total Local Images: {len(inventory)}")
    for s in splits:
        s_count = len([x for x in inventory if x["split"] == s])
        s_neg = len([x for x in inventory if x["split"] == s and x["is_negative_image"]])
        print(f"  Split [{s}]: {s_count} images (Negatives: {s_neg})")

    print(f"\nUnique Physical Scenes (Pre-Augmentation): {len(base_id_to_splits)}")
    print(f"  Train Unique Scenes: {len(scenes_by_split['train'])}")
    print(f"  Valid Unique Scenes: {len(scenes_by_split.get('valid', set()))}")
    print(f"  Test Unique Scenes:  {len(scenes_by_split.get('test', set()))}")

    print("\nDATA LEAKAGE METRICS:")
    print(f"  Scenes Shared Between Train & Valid: {len(tv_leak)} ({len(tv_leak)/max(1, len(scenes_by_split.get('valid', set())))*100:.1f}% of validation scenes leak into train!)")
    print(f"  Scenes Shared Between Train & Test:  {len(tt_leak)} ({len(tt_leak)/max(1, len(scenes_by_split.get('test', set())))*100:.1f}% of test scenes leak into train!)")
    print(f"  Scenes Appearing in ALL THREE splits: {len(all_three_leak)}")

    print("\nANNOTATION GEOMETRY AUDIT:")
    print(f"  Total Labeled Instances: {len(all_instances)}")
    for n_pts, cnt in sorted(point_counts.items()):
        print(f"  Polygons with {n_pts} vertices ({n_pts*2} coords): {cnt} instances ({cnt/len(all_instances)*100:.1f}%)")

    print("\nCLASS INSTANCE TOTALS:")
    for _, row in class_stats_df.iterrows():
        print(f"  Class {row['class_id']} ({row['class_name']}): {row['total_instances']} instances (Train: {row['train_instances']}, Valid: {row['valid_instances']}, Test: {row['test_instances']})")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Audit Local Image Dataset for GeoAgri-AI / YOLO.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=r"..\Archaeological-Site-Mapping-AI-V2\Archaeological-Site-Mapping-AI-V2\dataset",
        help="Path to the dataset directory containing train/valid/test folders",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=r"docs",
        help="Path to output documentation directory",
    )
    args = parser.parse_args()

    run_full_dataset_audit(args.dataset_dir, args.docs_dir)


if __name__ == "__main__":
    main()

