# GeoAgri-AI: Local YOLO Training Plan (NVIDIA RTX 3050 6GB GPU)

**Document ID:** `docs/local_yolo_training_plan.md`  
**Author:** Senior MLOps Engineer & Computer Vision Researcher  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) | Intel Core i5-12450HX  
**Date:** October 28, 2026  
**Status:** **PROPOSED PLAN (NOT YET EXECUTED)**

---

## 1. Prerequisites Before Execution

Before running the training command, three mandatory blocking prerequisites must be executed:

1. **Install PyTorch with CUDA Support in Virtual Environment:**
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   pip install ultralytics==8.4.24
   ```
2. **Execute Leak-Free Group-Aware Dataset Split:**
   - Partition the 128 unique base scenes into `data_group_split.yaml` (90 train scenes, 20 validation scenes, 18 test scenes).
3. **Clamp Out-of-Bounds Vertices:**
   - Run label sanitization clamping the 198 coordinates to $[0.0, 1.0]$.

---

## 2. Conservative Hardware-Optimized Hyperparameters (6GB VRAM)

| Parameter | Recommended Setting | Rationale for RTX 3050 (6GB) |
| :--- | :---: | :--- |
| **Model Architecture** | `yolo11n-obb.pt` *(Primary)* or `yolov8n.pt` *(Drop-in)* | Nano variant ($\approx 2.6\ \text{M}$ params) ensures low VRAM footprint. |
| **Input Resolution (`imgsz`)** | **640** | Matches native dataset dimensions ($640 \times 640$). |
| **Batch Size (`batch`)** | **4** (Conservative) or **8** | Consumes $\approx 1.8\ \text{GB}$ VRAM with batch=4, leaving $> 4\ \text{GB}$ headroom. |
| **Precision (`amp`)** | **True** (FP16 Automatic Mixed Precision) | Halves tensor memory and accelerates RTX Tensor Cores. |
| **DataLoader Workers (`workers`)** | **4** | Optimal for 8-core / 12-thread Intel i5-12450HX on Windows. |
| **Epochs (`epochs`)** | **60** | Sufficient for fine-tuning on 520 training images. |
| **Patience (`patience`)** | **12** | Early stopping triggers if validation loss plateaus for 12 epochs. |
| **Optimizer (`optimizer`)** | `auto` (AdamW) | Adaptive weight decay for small multi-class datasets. |
| **Learning Rate (`lr0`)** | **0.005** | Safe initial learning rate for pre-trained weights transfer. |
| **Weight Decay (`weight_decay`)**| **0.0005** | Prevents memorization of repetitive archaeological textures. |
| **Memory Caching (`cache`)** | **False** | Keeps RAM usage low on Windows system memory. |
| **Target Device (`device`)** | **0** (`cuda:0`) | Routes gradients to NVIDIA GeForce RTX 3050 Laptop GPU. |
| **Save Directory (`project`/`name`)** | `runs/obb/` or `runs/detect/` | Completely isolates experiments from existing production models. |

---

## 3. Exact Proposed Training Commands

### Option A: Primary Recommendation — YOLO11 Nano Oriented Bounding Box (OBB)
*(Natively matches 4-vertex rotated boxes)*

```powershell
# [NOT YET EXECUTED - AWAITING PREREQUISITE INSTALLATION]
yolo task=obb mode=train `
  model=yolo11n-obb.pt `
  data=data_group_split.yaml `
  epochs=60 `
  batch=4 `
  imgsz=640 `
  device=0 `
  workers=4 `
  optimizer=AdamW `
  lr0=0.005 `
  patience=12 `
  project=runs/obb `
  name=yolo11n_archaeology_obb_v1 `
  save=True
```

### Option B: Fallback Recommendation — YOLO11 / YOLOv8 Nano Detection
*(Matches existing predict_bridge.py DetectionModel pipeline with zero code changes)*

```powershell
# [NOT YET EXECUTED - AWAITING PREREQUISITE INSTALLATION]
yolo task=detect mode=train `
  model=yolo11n.pt `
  data=data_group_split.yaml `
  epochs=60 `
  batch=8 `
  imgsz=640 `
  device=0 `
  workers=4 `
  optimizer=AdamW `
  lr0=0.005 `
  patience=12 `
  project=runs/detect `
  name=yolo11n_archaeology_detect_v1 `
  save=True
```

---

## 4. Rollback and Safety Strategy

* **Zero Overwrite Guarantee:** The existing production model checkpoint at:
  `runs/detect/yolov8s_archaeology2/weights/best.pt`
  will remain untouched and archived.
* **Fallback Path:** If CUDA memory error (`torch.cuda.OutOfMemoryError`) ever occurs, reducing batch size to `batch=2` with `imgsz=512` guarantees complete execution within $< 1.2\ \text{GB}$ VRAM.

