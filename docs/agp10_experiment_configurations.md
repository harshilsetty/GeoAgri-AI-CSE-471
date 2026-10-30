# AGP 10 — Experiment Configurations Specification

## 1. Overview & Experimental Principles

The goal of AGP 10 is to isolate and quantify the drivers of performance on GeoAgri-AI's oriented bounding box (OBB) model:
1. **EXP-B (Geometry-Refitted Baseline):** Directly measures the effect of `cv2.minAreaRect` annotation refitting on `datasets/geoagri_obb_v3` by holding all hyperparameters, initialization weights, seeds, and augmentations identical to the AGP 08 baseline.
2. **EXP-C (Domain-Shift Augmentation):** Measures whether conservative colorimetric and spatial augmentations bridge the material and architectural domain shift between sites (e.g. Nalanda red brick vs. Khajuraho buff sandstone vs. Hampi granite) without corrupting OBB geometry.

All runs are executed sequentially on a single dedicated NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) to ensure strictly reproducible memory allocation and cache isolation.

---

## 2. Shared Experimental Constants

Both experiments share the following fixed environment and training parameters:
* **Framework:** Ultralytics YOLO 8.4.191 / PyTorch 2.6.0+cu124
* **Base Architecture:** YOLO11 Nano OBB (`yolo11n-obb.pt` pretrained weights)
* **Dataset Config:** [`datasets/geoagri_obb_v3/data.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/datasets/geoagri_obb_v3/data.yaml)
* **Input Resolution:** $640 \times 640$ pixels
* **Batch Size:** 4
* **Hardware Device:** `CUDA:0` (NVIDIA GeForce RTX 3050 Laptop GPU 6GB)
* **Dataloader Workers:** 2
* **Precision:** Mixed Precision (`amp=True`)
* **Maximum Epochs:** 60
* **Early Stopping Patience:** 12 epochs without validation mAP improvement
* **Seed:** 42 (`deterministic=True`)
* **Optimizer:** Auto (SGD with `lr0=0.01`, `lrf=0.01`, `momentum=0.937`, `weight_decay=0.0005`, `warmup_epochs=3.0`)
* **Loss Components:** `box=7.5`, `cls=0.5`, `dfl=1.5`, `angle=1.0`

---

## 3. Detailed Parameter Comparison

| Hyperparameter | Baseline (AGP 08 / v1) | EXP-B (Geometry Refit) | EXP-C (Domain Augmentation) | Rationale for Change |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset** | `geoagri_obb_v2` | `geoagri_obb_v3` | `geoagri_obb_v3` | Evaluates impact of refitted boundary boxes. |
| **Run Name** | `geoagri_yolo11n_obb_v1` | `geoagri_yolo11n_obb_v2_refit` | `geoagri_yolo11n_obb_v3_aug` | Clean directory isolation in `runs/obb/`. |
| **Hue (`hsv_h`)** | `0.015` | `0.015` | **`0.050`** | Bridges spectral gap between red brick and buff sandstone. |
| **Saturation (`hsv_s`)** | `0.70` | `0.70` | **`0.80`** | Enhances contrast across mineral weathering patterns. |
| **Value (`hsv_v`)** | `0.40` | `0.40` | **`0.50`** | Simulates varying sunlight, shadows, and aerial cloud cover. |
| **Rotation (`degrees`)** | `0.0` | `0.0` | **`5.0`** | Exposes angle head to minor rotational variance. |
| **Translation (`translate`)** | `0.10` | `0.10` | **`0.15`** | Improves boundary robustness for perimeter objects. |
| **Scale (`scale`)** | `0.50` | `0.50` | `0.50` | Preserved across all experiments. |
| **Horizontal Flip (`fliplr`)**| `0.50` | `0.50` | `0.50` | Preserved across all experiments. |
| **Vertical Flip (`flipud`)** | `0.0` | `0.0` | **`0.20`** | Aerial/oblique perspectives are largely orientation-invariant. |
| **Mosaic (`mosaic`)** | `1.0` | `1.0` | `1.0` | Preserved; vital for spatial context learning. |
| **Close Mosaic (`close_mosaic`)** | `10` | `10` | `10` | Final 10 epochs train on unmixed scenes. |
| **Erasing (`erasing`)** | `0.40` | `0.40` | `0.40` | Regularizes against partial occlusions. |

---

## 4. Execution Isolation & Safety Safeguards

1. **Non-Overlapping Runs:** EXP-B runs to completion, verifies artifacts (`weights/best.pt`, `results.csv`, `args.yaml`), releases CUDA memory, and logs validation performance before EXP-C begins.
2. **Quarantine of Held-Out Test Set:** Both EXP-B and EXP-C are trained on `train` and evaluated strictly on `val`. Test-set inference is deferred until Phase 8, and is run only once on the single model selected by the Phase 7 validation gates.
3. **Zero In-Place Modifications:** No dataset files or baseline checkpoints are modified in place.

