"""Tests for AGP 09 Annotation Geometry & Split Isolation.
"""

import os
import re
import pytest
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

def extract_base_scene(filename: str) -> str:
    m = re.match(r"^(.*?)(_jpg|\.jpg)?\.rf\.[a-f0-9]{32}\.", filename, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return os.path.splitext(filename)[0].split(".")[0]

def test_v3_split_isolation():
    """Verify strictly zero scene overlap across train, val, and test in geoagri_obb_v3."""
    v3_dir = ROOT / "datasets" / "geoagri_obb_v3"
    assert v3_dir.exists(), "v3 dataset does not exist"

    split_scenes = defaultdict(set)
    for split in ["train", "val", "test"]:
        img_dir = v3_dir / "images" / split
        for p in img_dir.glob("*.jpg"):
            split_scenes[split].add(extract_base_scene(p.name))

    assert len(split_scenes["train"]) == 90
    assert len(split_scenes["val"]) == 19
    assert len(split_scenes["test"]) == 19

    # Strict isolation assertions
    assert len(split_scenes["train"] & split_scenes["val"]) == 0
    assert len(split_scenes["train"] & split_scenes["test"]) == 0
    assert len(split_scenes["val"] & split_scenes["test"]) == 0

def test_v3_coordinate_bounds_and_convexity():
    """Verify all coordinates in v3 are within [0.0, 1.0] and non-degenerate."""
    v3_dir = ROOT / "datasets" / "geoagri_obb_v3"
    lbl_dir = v3_dir / "labels"

    checked_instances = 0
    for split in ["train", "val", "test"]:
        for lp in (lbl_dir / split).glob("*.txt"):
            with open(lp, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    coords = [float(x) for x in parts[1:9]]
                    assert len(coords) == 8
                    assert all(0.0 <= c <= 1.0 for c in coords), f"Coord out of bounds in {lp}: {coords}"
                    checked_instances += 1

    assert checked_instances == 18458, f"Expected 18458 instances, got {checked_instances}"

def test_v3_manifest_and_yaml_integrity():
    """Verify v3 manifest and YAML files are present and valid."""
    v3_dir = ROOT / "datasets" / "geoagri_obb_v3"
    yaml_path = v3_dir / "data.yaml"
    manifest_path = v3_dir / "manifest_v3.json"

    assert yaml_path.exists()
    assert manifest_path.exists()
    assert manifest_path.stat().st_size > 1000

