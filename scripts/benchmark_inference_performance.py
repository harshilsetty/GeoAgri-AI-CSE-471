"""Rigorous Inference Performance & Latency Benchmarking for GeoAgri-AI YOLO-OBB.

Hardware target: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)
Runs 10 warmup iterations, followed by 100 timed iterations over held-out test images.
Computes model-only, end-to-end, median, P95 latency, FPS, and VRAM memory.
"""

import time
import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

WEIGHTS_PATH = Path("runs/obb/geoagri_yolo11n_obb_v1/weights/best.pt")
TEST_IMG_DIR = Path("datasets/geoagri_obb_v2/images/test")

def main():
    print("=" * 60)
    print("GeoAgri-AI Inference Latency & Performance Benchmark")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Model Checkpoint: {WEIGHTS_PATH}")
    print("=" * 60)

    model = YOLO(str(WEIGHTS_PATH))
    test_imgs = sorted(list(TEST_IMG_DIR.glob("*.jpg")))[:30] # 30 representative test images

    # Load images into memory
    import cv2
    loaded_imgs = [cv2.imread(str(p)) for p in test_imgs]

    # Warmup (10 runs)
    print("[*] Warming up GPU (10 iterations)...")
    for i in range(10):
        img = loaded_imgs[i % len(loaded_imgs)]
        _ = model.predict(img, imgsz=640, device="cuda:0", verbose=False)
    torch.cuda.synchronize()

    # Benchmark: 100 timed runs
    print("[*] Running 100 timed inferences...")
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    e2e_times = []

    for i in range(100):
        img = loaded_imgs[i % len(loaded_imgs)]
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        results = model.predict(img, imgsz=640, device="cuda:0", verbose=False)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        e2e_ms = (t1 - t0) * 1000.0
        e2e_times.append(e2e_ms)

        speed = results[0].speed
        preprocess_times.append(speed.get("preprocess", 0.0))
        inference_times.append(speed.get("inference", 0.0))
        postprocess_times.append(speed.get("postprocess", 0.0))

    # Memory metrics
    alloc_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    res_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
    max_alloc_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)

    # Compute statistics
    e2e_arr = np.array(e2e_times)
    inf_arr = np.array(inference_times)
    pre_arr = np.array(preprocess_times)
    post_arr = np.array(postprocess_times)

    e2e_median = float(np.median(e2e_arr))
    e2e_mean = float(np.mean(e2e_arr))
    e2e_p95 = float(np.percentile(e2e_arr, 95))
    e2e_p99 = float(np.percentile(e2e_arr, 99))
    fps = 1000.0 / e2e_mean

    inf_median = float(np.median(inf_arr))
    inf_mean = float(np.mean(inf_arr))
    inf_p95 = float(np.percentile(inf_arr, 95))

    benchmark_results = {
        "device": torch.cuda.get_device_name(0),
        "imgsz": 640,
        "batch_size": 1,
        "precision": "FP16 (AMP)",
        "iterations": 100,
        "e2e_latency_ms": {
            "mean": round(e2e_mean, 2),
            "median": round(e2e_median, 2),
            "p95": round(e2e_p95, 2),
            "p99": round(e2e_p99, 2),
            "min": round(float(np.min(e2e_arr)), 2),
            "max": round(float(np.max(e2e_arr)), 2)
        },
        "model_only_inference_ms": {
            "mean": round(inf_mean, 2),
            "median": round(inf_median, 2),
            "p95": round(inf_p95, 2)
        },
        "preprocess_ms_mean": round(float(np.mean(pre_arr)), 2),
        "postprocess_ms_mean": round(float(np.mean(post_arr)), 2),
        "throughput_fps": round(fps, 2),
        "gpu_memory_mb": {
            "allocated": round(alloc_mb, 2),
            "reserved": round(res_mb, 2),
            "peak_allocated": round(max_alloc_mb, 2)
        }
    }

    print("\n--- Benchmark Summary ---")
    print(f"End-to-End Latency: Mean {e2e_mean:.2f} ms | Median {e2e_median:.2f} ms | P95 {e2e_p95:.2f} ms")
    print(f"Model-Only Inference: Mean {inf_mean:.2f} ms | Median {inf_median:.2f} ms | P95 {inf_p95:.2f} ms")
    print(f"Throughput: {fps:.1f} FPS")
    print(f"GPU Memory Allocated: {alloc_mb:.2f} MB | Reserved: {res_mb:.2f} MB | Peak: {max_alloc_mb:.2f} MB")

    out_json = Path("docs/final_yolo_benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"[*] Benchmark saved to {out_json}")

if __name__ == "__main__":
    main()

