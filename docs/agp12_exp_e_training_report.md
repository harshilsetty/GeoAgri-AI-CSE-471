# AGP 12B — EXP-E Training Audit & Completion Report

## 1. Executive Summary & Training Outcome

* **Experiment ID:** EXP-E (`geoagri_yolo11s_obb_v5_class_weighted`)
* **Model Architecture:** YOLO11 Small OBB (`yolo11s-obb.pt`, 9.72M parameters, 22.7 GFLOPs)
* **Dataset:** `datasets/geoagri_obb_v3/data.yaml` (geometry-refitted v3 dataset; 514 train / 88 scenes, 109 val / 19 scenes)
* **Final Training State:** **COMPLETED via Early Stopping**
* **Final Epoch:** **Epoch 20**
* **Best Epoch:** **Epoch 8** (mAP@50: 0.3797, mAP@50-95: 0.2297, fitness: 0.22965)
* **Early Stopping Confirmation:** Formally confirmed by Ultralytics trainer:
  ```text
  EarlyStopping: Training stopped early as no improvement observed in last 12 epochs. 
  Best results observed at epoch 8, best model saved as best.pt.
  ```

---

## 2. Artifact Verification & Checkpoint Integrity

All expected run artifacts were audited in [`runs/obb/geoagri_yolo11s_obb_v5_class_weighted/`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/runs/obb/geoagri_yolo11s_obb_v5_class_weighted):

| Artifact | File Path | File Size | SHA / Integrity Status |
| :--- | :--- | :--- | :--- |
| **Best Checkpoint** | `weights/best.pt` | 19.8 MB (stripped) / 97.8 MB (raw) | Verified; loads 9,700,722 params, epoch 8 |
| **Last Checkpoint** | `weights/last.pt` | 19.8 MB (stripped) / 97.8 MB (raw) | Verified; loads 9,700,722 params, epoch 20 |
| **Training Log CSV** | `results.csv` | 2,915 bytes | 20 complete epoch entries with losses & metrics |
| **Training Args** | `args.yaml` | 2,185 bytes | Confirmed batch=4, imgsz=640, patience=12, seed=42 |
| **Visual Labels** | `labels.jpg` | 166.4 KB | Verified label distribution plots |
| **Batch Plots** | `train_batch0/1/2.jpg` | ~260–350 KB | Verified augmented mosaic inputs |

### Class Weight Verification
Inspecting the initialization log confirms callback attachment:
```text
[*] Configured class weights: tensor([0.9500, 1.4100, 1.1600, 3.8800, 0.5600])
[*] Attaching class weights to trainer model: tensor([0.9500, 1.4100, 1.1600, 3.8800, 0.5600])
```
Class weighting was dynamically bound to `trainer.model.class_weights` for weighted BCE loss computation across all 20 training epochs.

---

## 3. Hardware Telemetry & Resource Consumption

* **Hardware Target:** NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB GDDR6, Compute Capability 8.6)
* **PyTorch Memory Allocated:** **1.58 GB** peak allocation during training
* **CUDA VRAM Reserved:** **1,667 MB** ($\approx 27.1\%$ of available VRAM)
* **VRAM Headroom:** **4.47 GB** ($72.9\%$ headroom)
* **Operating Temperature:** **69°C – 74°C** under continuous 100% compute saturation
* **Total Training Wall Clock Duration:** **$\approx 9.3$ minutes** (556.3 seconds)
* **Average Epoch Latency:** $\approx 24.8$ seconds per epoch

---

## 4. Convergence & Overfitting Analysis

Examining the full loss trajectory from [`results.csv`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/runs/obb/geoagri_yolo11s_obb_v5_class_weighted/results.csv):

| Epoch | Train Box Loss | Train Cls Loss | Val Box Loss | Val Cls Loss | Val Precision | Val Recall | Val mAP@50 | Val mAP@50-95 | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 2.1688 | 3.0514 | 2.0991 | 2.7304 | 0.5133 | 0.2305 | 0.0987 | 0.0387 | Warmup phase |
| **4** | 1.6368 | 1.9684 | 1.7000 | 2.1900 | 0.2903 | 0.4207 | 0.3187 | 0.1707 | Fast convergence |
| **8** | 1.4840 | 1.6337 | 1.6153 | 1.7688 | **0.4683** | 0.3721 | **0.3797** | **0.2297** | **Best Checkpoint** |
| **12** | 1.3775 | 1.4301 | 1.6491 | 1.8744 | 0.3007 | 0.4378 | 0.3478 | 0.2126 | Val loss begins plateauing |
| **16** | 1.2837 | 1.2787 | 1.6316 | 1.8104 | 0.2706 | 0.4513 | 0.3006 | 0.1848 | Evidence of overfitting |
| **20** | 1.2170 | 1.1800 | 1.6455 | 1.8689 | 0.3550 | 0.4060 | 0.2990 | 0.1760 | Early stopping triggered |

### Overfitting Diagnosis
1. **Rapid Overfitting on Small Dataset:** While training classification loss declined steadily from $3.0514$ to $1.1800$ ($-61.3\%$), validation classification loss stopped improving at Epoch 8 ($1.7688$) and began drifting upward to $1.8689$.
2. **Capacity Penalty:** With 9.72M parameters (vs 2.66M in YOLO11n), YOLO11s possessed excessive capacity relative to the 514 training images (88 scenes). The network rapidly memorized idiosyncratic backgrounds, causing generalization to degrade after just 8 epochs.

