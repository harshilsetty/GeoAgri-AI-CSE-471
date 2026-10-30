# AGP 12 — EXP-E Experimental Protocol: Model Capacity Scaling (YOLO11s-OBB)

## 1. Executive Summary & Objective

The objective of **EXP-E** is to evaluate whether scaling model capacity from YOLO11 Nano OBB (2.66M parameters, 6.8 GFLOPs) to YOLO11 Small OBB (9.74M parameters, 22.8 GFLOPs) improves the representation of rare, morphologically complex archaeological classes (principally `structures` and `ruins`) across geologically distinct terrains, without compromising training stability or real-time inference latency.

This experiment builds directly upon **EXP-D** (`geoagri_yolo11n_obb_v4_class_weighted`), which established the efficacy of inverse-frequency damped class weighting ($w_{\text{structures}} = 3.88$).

---

## 2. Hypothesis & Architectural Justification

### 2.1 Scientific Hypothesis
* **Primary Hypothesis:** The limited feature extraction capacity of YOLO11 Nano (2.66M parameters) restricts the network from learning invariant structural and geometric features across divergent lithological substrates (e.g., buff sandstone in Khajuraho vs gray granite in Hampi vs red brick in Nalanda).
* **Secondary Hypothesis:** Increasing backbone and neck capacity via YOLO11s (9.74M parameters) will yield richer multi-scale spatial representations in the C3k2 and SPPF modules, allowing the OBB angle/bounding regression heads to distinguish geometric masonry outlines from surrounding natural rubble and erosion gullies.

### 2.2 Capacity & Computational Footprint Comparison
| Property | YOLO11n-OBB (EXP-D Baseline) | YOLO11s-OBB (EXP-E Candidate) | Delta / Ratio |
| :--- | :--- | :--- | :--- |
| **Model Family** | YOLO11 Nano OBB | YOLO11 Small OBB | Capacity Tier +1 |
| **Backbone/Neck Depth** | 0.50 depth multiplier | 0.50 depth multiplier | 1.0x |
| **Channel Width** | 0.25 width multiplier | 0.50 width multiplier | 2.0x channel expansion |
| **Max Channels** | 1024 | 1024 | Same ceiling |
| **Total Layers** | 196 | 196 | Same topology |
| **Total Parameters** | 2,661,427 (2.66M) | 9,744,931 (9.74M) | **+266.2% (+3.66x)** |
| **GFLOPs (at 640x640)** | 6.8 GFLOPs | 22.8 GFLOPs | **+235.3% (+3.35x)** |
| **Estimated Peak VRAM (Batch 4)** | 2.87 GB | 4.12 GB | Fits comfortably within 6.0 GB VRAM |

---

## 3. Experimental Controls & Configuration

To maintain strict scientific comparability with EXP-D, all hyperparameters, dataset manifests, and weighting schemes are held constant:

```yaml
experiment:
  id: EXP-E
  run_name: geoagri_yolo11s_obb_v5_class_weighted
  task: obb
  base_checkpoint: yolo11s-obb.pt
  dataset_yaml: datasets/geoagri_obb_v3/data.yaml
  dataset_version: v3 (geometry-refitted polygon annotations)

hyperparameters:
  epochs: 60
  batch_size: 4
  imgsz: 640
  device: "0"
  workers: 2
  patience: 12
  seed: 42
  deterministic: true
  save: true
  plots: true
  exist_ok: true

loss_weighting:
  scheme: inverse_frequency_damped (alpha=0.5)
  weights:
    archaeological_mound: 0.95
    ancient_wall: 1.41
    ruins: 1.16
    structures: 3.88
    vegetation_crop: 0.56
```

---

## 4. Hardware Safety & Pre-Flight Validation

* **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB GDDR6 VRAM, compute capability 8.6).
* **Batch Size Safety:** Batch size is constrained to 4 with 2 dataloader workers to prevent PyTorch CUDA OOM spikes during backward passes through expanded $C=512$ feature maps.
* **Pre-Flight Memory Allocation:** Baseline reserved VRAM is < 100 MB. Projected training headroom is > 1,800 MB.

---

## 5. Evaluation & Decision Criteria

Evaluation is conducted **strictly on the 109-scene validation split** (`datasets/geoagri_obb_v3/val`). The 117-scene test split remains strictly quarantined.

### Decision Matrix (EXP-E vs EXP-D):
1. **Pass Condition (Advance as primary candidate):**
   * Validation mAP@50 $\ge$ 0.4150 (EXP-D baseline: 0.4150)
   * Validation Structures Recall $\ge$ 0.7000 (EXP-D baseline: 0.7711)
   * Validation mAP@50-95 $\ge$ 0.2500
   * Inference latency $\le$ 30.0 ms/image on RTX 3050 GPU
2. **Fail Condition (Reject capacity scaling):**
   * Validation mAP@50 < 0.4000 (evidence of overfitting on small dataset)
   * Severe regression on majority classes without structures gain
   * GPU OOM or unstable gradient divergence

