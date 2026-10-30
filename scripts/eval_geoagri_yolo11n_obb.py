"""Independent Test Split Evaluation Script for GeoAgri-AI YOLO11n-OBB.

Evaluates the trained model strictly on the untouched test split (19 scenes, 117 images).
Produces per-class mAP50, mAP50-95, precision, recall, and speed benchmarks.
"""

import argparse
import json
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO11n-OBB on validation or test split")
    parser.add_argument("--weights", type=str, default="runs/obb/geoagri_yolo11n_obb_v1/weights/best.pt", help="Path to model weights")
    parser.add_argument("--data", type=str, default="datasets/geoagri_obb_v3/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Split to evaluate")
    parser.add_argument("--device", type=str, default="0", help="CUDA device (e.g. 0 or cpu)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"[!] Error: Model weights not found at {weights_path}")
        sys.exit(1)

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"[!] Error: Data config not found at {data_path}")
        sys.exit(1)

    print("=" * 60)
    print("GeoAgri-AI YOLO11n-OBB Evaluation")
    print(f"Model Checkpoint: {weights_path}")
    print(f"Dataset Config:   {data_path}")
    print(f"Evaluation Split: {args.split}")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_path),
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        save_json=True,
        plots=True
    )

    print("\n" + "=" * 60)
    print(f"{args.split.upper()} Evaluation Results")
    print("=" * 60)

    # Summary metrics
    map50 = float(metrics.box.map50) if hasattr(metrics, "box") else float(metrics.obb.map50)
    map50_95 = float(metrics.box.map) if hasattr(metrics, "box") else float(metrics.obb.map)
    mp = float(metrics.box.mp) if hasattr(metrics, "box") else float(metrics.obb.mp)
    mr = float(metrics.box.mr) if hasattr(metrics, "box") else float(metrics.obb.mr)

    print(f"Overall Precision (P): {mp:.4f}")
    print(f"Overall Recall (R):    {mr:.4f}")
    print(f"Overall mAP@50:        {map50:.4f}")
    print(f"Overall mAP@50-95:     {map50_95:.4f}")

    # Per-class breakdown
    names = metrics.names
    eval_dict = {
        "split": args.split,
        "weights": str(weights_path),
        "data": str(data_path),
        "overall": {
            "precision": mp,
            "recall": mr,
            "mAP50": map50,
            "mAP50_95": map50_95
        },
        "per_class": {},
        "speed_ms": metrics.speed
    }

    print("\n--- Per-Class Performance ---")
    header = f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10} | {'mAP50-95':<10}"
    print(header)
    print("-" * len(header))

    obb_metric = metrics.obb if hasattr(metrics, "obb") else metrics.box
    for i, cls_name in names.items():
        if i < len(obb_metric.p):
            p = float(obb_metric.p[i])
            r = float(obb_metric.r[i])
            ap50 = float(obb_metric.ap50[i])
            ap = float(obb_metric.ap[i])
            eval_dict["per_class"][cls_name] = {
                "precision": p,
                "recall": r,
                "mAP50": ap50,
                "mAP50_95": ap
            }
            print(f"{cls_name:<15} | {p:<10.4f} | {r:<10.4f} | {ap50:<10.4f} | {ap:<10.4f}")

    print("\n--- Latency Breakdown (ms per image) ---")
    for k, v in metrics.speed.items():
        print(f"  {k}: {v:.2f} ms")

    # Save to json if requested
    out_json = Path(args.output) if args.output else Path(f"docs/metrics_{args.split}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=2)
    print(f"\n[*] Metrics written to {out_json}")

if __name__ == "__main__":
    main()

