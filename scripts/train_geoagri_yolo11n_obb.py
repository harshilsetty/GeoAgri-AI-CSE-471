"""Local GPU Training Script for GeoAgri-AI YOLO-OBB.

Hardware target: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)
Model: YOLO11n-obb (pretrained)
Dataset: datasets/geoagri_obb_v2/data.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

def get_gpu_memory_info():
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
            max_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
            name = torch.cuda.get_device_name(0)
            return {
                "available": True,
                "name": name,
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "max_allocated_mb": max_allocated
            }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    return {"available": False}

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n-OBB on GeoAgri dataset")
    parser.add_argument("--data", type=str, default="datasets/geoagri_obb_v3/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n-obb.pt", help="Pretrained model weights or config")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (conservative for 6GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--name", type=str, default="geoagri_yolo11n_obb_v2_refit", help="Run experiment name")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index (e.g. 0)")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 2-epoch smoke test")
    # Optional augmentation overrides for domain shift experiments
    parser.add_argument("--hsv_h", type=float, default=None, help="HSV-Hue augmentation fraction")
    parser.add_argument("--hsv_s", type=float, default=None, help="HSV-Saturation augmentation fraction")
    parser.add_argument("--hsv_v", type=float, default=None, help="HSV-Value augmentation fraction")
    parser.add_argument("--degrees", type=float, default=None, help="Rotation degrees")
    parser.add_argument("--translate", type=float, default=None, help="Translation fraction")
    parser.add_argument("--flipud", type=float, default=None, help="Vertical flip probability")
    parser.add_argument("--class_weights", type=str, default=None, help="Comma-separated class weights (e.g. 0.95,1.41,1.16,3.88,0.56)")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 2
        args.name = "geoagri_obb_smoke_test"
        args.patience = 2

    import torch
    print("=" * 60)
    print("GeoAgri-AI YOLO11n-OBB Training Pipeline")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device:     {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print("=" * 60)

    from ultralytics import YOLO

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"[!] Error: {data_yaml} not found!")
        sys.exit(1)

    print(f"[*] Dataset Config: {data_yaml}")
    print(f"[*] Hyperparameters: epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, workers={args.workers}")
    print(f"[*] Target run name: {args.name}")

    initial_mem = get_gpu_memory_info()
    print(f"[*] Initial GPU Memory: {initial_mem}")

    # Load YOLO-obb model backbone
    model = YOLO(args.model)

    if args.class_weights:
        weights = torch.tensor([float(w.strip()) for w in args.class_weights.split(",")], dtype=torch.float32)
        print(f"[*] Configured class weights: {weights}")
        def attach_weights(trainer):
            print(f"[*] Attaching class weights to trainer model: {weights}")
            trainer.model.class_weights = weights.to(trainer.device)
        model.add_callback("on_pretrain_routine_end", attach_weights)

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": str(Path("runs/obb").resolve()),
        "name": args.name,
        "patience": args.patience,
        "seed": args.seed,
        "deterministic": True,
        "save": True,
        "plots": True,
        "exist_ok": True
    }

    if args.hsv_h is not None:
        train_kwargs["hsv_h"] = args.hsv_h
    if args.hsv_s is not None:
        train_kwargs["hsv_s"] = args.hsv_s
    if args.hsv_v is not None:
        train_kwargs["hsv_v"] = args.hsv_v
    if args.degrees is not None:
        train_kwargs["degrees"] = args.degrees
    if args.translate is not None:
        train_kwargs["translate"] = args.translate
    if args.flipud is not None:
        train_kwargs["flipud"] = args.flipud

    start_time = time.time()
    try:
        results = model.train(**train_kwargs)
    except Exception as exc:
        print(f"[!] Training failed with error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed_s = time.time() - start_time
    final_mem = get_gpu_memory_info()

    print("=" * 60)
    print(f"[*] Training complete in {elapsed_s / 60:.2f} minutes ({elapsed_s:.1f} seconds)")
    print(f"[*] Peak GPU Memory Allocated: {final_mem.get('max_allocated_mb', 0):.2f} MB")
    print(f"[*] Checkpoints saved to: runs/obb/{args.name}/weights/")
    print("=" * 60)

if __name__ == "__main__":
    main()

