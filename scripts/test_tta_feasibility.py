"""Test Test-Time Augmentation (TTA) feasibility on GeoAgri-AI YOLO-OBB.

Tests whether Ultralytics OBB architecture supports `augment=True` during evaluation,
measures inference latency, and compares mAP@50 and per-class metrics.
"""

import sys
import time
import json
from pathlib import Path
import torch

def test_tta():
    weights = Path("runs/obb/geoagri_yolo11n_obb_v4_class_weighted/weights/best.pt")
    data_yaml = Path("datasets/geoagri_obb_v3/data.yaml")
    
    if not weights.exists():
        print(f"[!] Checkpoint not found: {weights}")
        return
        
    from ultralytics import YOLO
    
    print("[*] Loading EXP-D model...")
    model = YOLO(str(weights))
    
    # 1. Baseline validation (no TTA)
    print("\n" + "=" * 50)
    print("[*] Running Baseline Validation (augment=False)...")
    t0 = time.time()
    val_base = model.val(
        data=str(data_yaml),
        split="val",
        batch=4,
        imgsz=640,
        augment=False,
        device="0",
        verbose=False
    )
    base_time = time.time() - t0
    
    metric_obj = val_base.box if hasattr(val_base, "box") else val_base.obb
    base_map50 = float(metric_obj.map50)
    base_map50_95 = float(metric_obj.map)
    base_mp = float(metric_obj.mp)
    base_mr = float(metric_obj.mr)
    base_classes = {}
    for idx, name in val_base.names.items():
        base_classes[name] = {
            "p": float(metric_obj.p[idx]),
            "r": float(metric_obj.r[idx]),
            "map50": float(metric_obj.all_ap[idx, 0]),
            "map50_95": float(metric_obj.ap[idx])
        }
    
    print(f"Baseline Time: {base_time:.2f}s | mAP50: {base_map50:.4f} | R: {base_mr:.4f} | Speed: {val_base.speed}")
    
    # 2. TTA validation (augment=True)
    print("\n" + "=" * 50)
    print("[*] Running TTA Validation (augment=True)...")
    tta_supported = True
    tta_error = None
    t0 = time.time()
    try:
        val_tta = model.val(
            data=str(data_yaml),
            split="val",
            batch=4,
            imgsz=640,
            augment=True,
            device="0",
            verbose=False
        )
        tta_time = time.time() - t0
        metric_obj_tta = val_tta.box if hasattr(val_tta, "box") else val_tta.obb
        tta_map50 = float(metric_obj_tta.map50)
        tta_map50_95 = float(metric_obj_tta.map)
        tta_mp = float(metric_obj_tta.mp)
        tta_mr = float(metric_obj_tta.mr)
        tta_classes = {}
        for idx, name in val_tta.names.items():
            tta_classes[name] = {
                "p": float(metric_obj_tta.p[idx]),
                "r": float(metric_obj_tta.r[idx]),
                "map50": float(metric_obj_tta.all_ap[idx, 0]),
                "map50_95": float(metric_obj_tta.ap[idx])
            }
        tta_speed = val_tta.speed
        print(f"TTA Time: {tta_time:.2f}s | mAP50: {tta_map50:.4f} | R: {tta_mr:.4f} | Speed: {tta_speed}")
    except Exception as exc:
        tta_supported = False
        tta_error = str(exc)
        print(f"[!] TTA Failed with error: {exc}")
        tta_time = 0
        tta_map50 = None
        tta_map50_95 = None
        tta_mp = None
        tta_mr = None
        tta_classes = {}
        tta_speed = {}
    
    results = {
        "tta_supported": tta_supported,
        "tta_error": tta_error,
        "baseline": {
            "time_s": base_time,
            "speed": val_base.speed,
            "map50": base_map50,
            "map50_95": base_map50_95,
            "precision": base_mp,
            "recall": base_mr,
            "per_class": base_classes
        },
        "tta": {
            "time_s": tta_time,
            "speed": tta_speed,
            "map50": tta_map50,
            "map50_95": tta_map50_95,
            "precision": tta_mp,
            "recall": tta_mr,
            "per_class": tta_classes
        } if tta_supported else None
    }
    
    out_path = Path("docs/agp12_tta_feasibility_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved to {out_path}")

if __name__ == "__main__":
    test_tta()
