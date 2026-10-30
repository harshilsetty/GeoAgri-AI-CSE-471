# AGP 12B — Experimental Model Selection & Checkpoint Governance Report

## 1. Executive Summary & Selection Decision

Following controlled empirical evaluation across the 109-scene validation partition, this report formalizes the model selection decision among **EXP-C** (domain augmentation), **EXP-D** (damped class weighting), and **EXP-E** (capacity scaling).

* **Preferred Experimental Checkpoint Selected:** **EXP-D (`geoagri_yolo11n_obb_v4_class_weighted/weights/best.pt`)**
* **Candidate Checkpoint Rejected:** **EXP-E (`geoagri_yolo11s_obb_v5_class_weighted/weights/best.pt`)**
* **Production Status:** **UNCHANGED**. Active production vision remains strictly pinned to [`yolov8s_archaeology_prod`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/config/models_registry.yaml).

---

## 2. Multi-Criteria Decision Matrix

| Criterion | Evaluation Standard | EXP-C (Augmented) | EXP-D (Class-Weighted) | EXP-E (Capacity-Scaled) | Selection Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Overall Val mAP@50** | Higher is better | **0.4497** | 0.4150 | 0.3797 | EXP-E performs worst ($-15.6\%$) |
| **Overall Val mAP@50-95**| Higher is better | **0.2638** | 0.2514 | 0.2297 | EXP-E performs worst ($-12.9\%$) |
| **Structures Recall** | Critical domain goal | 0.7222 | **0.7711** | 0.5556 | **EXP-D wins decisively** ($+21.5\%$) |
| **Overall Recall** | Detection completeness | 0.4140 | **0.4408** | 0.3717 | **EXP-D wins** |
| **Parameter Efficiency** | Lower footprint is better| **2.66M** | **2.66M** | 9.70M | EXP-E adds $+265\%$ params for negative gain |
| **Compute Footprint** | Lower GFLOPs is better | **6.8 GFLOPs** | **6.8 GFLOPs** | 22.4 GFLOPs | EXP-E requires $3.29\times$ more compute |
| **Inference Latency** | Edge deployability | **10.1 ms** | **10.2 ms** | 13.8 ms | Nano is $36\%$ faster forward pass |

---

## 3. Rationale for Selecting EXP-D over EXP-E

1. **Failure of Capacity Scaling Hypothesis:**
   * EXP-E tested the hypothesis that a larger parameter space (9.70M) would enable YOLO11 to learn invariant structural representations.
   * Empirically, EXP-E regressed on `structures` recall ($0.5556$ vs $0.7711$ in EXP-D) and achieved lower overall mAP ($0.3797$ vs $0.4150$).
   * Early stopping occurred at Epoch 20 with the optimal checkpoint frozen at Epoch 8, demonstrating that the larger capacity model rapidly overfitted the 514 training images.
2. **Minority Class Priority:**
   * In archaeological remote sensing, recovering rare ancient masonry structures is the central technical objective. EXP-D's damped class weighting scheme ($w_{\text{structures}} = 3.88$) remains the only technique that successfully recovered structures recall ($0.7711$) while preserving respectable precision ($0.4097$).
3. **Deployment Efficiency:**
   * EXP-D operates with a 2.66M parameter footprint and 10.2 ms inference time, providing superior throughput on edge devices and client workstations.

---

## 4. Production Isolation & Governance Confirmation

1. **Zero Production Promotion:**
   * In strict accordance with AGP release governance, **neither EXP-D nor EXP-E is promoted to production**.
   * While EXP-D is superior experimentally, its performance on held-out unseen sites ($0.1289$ mAP@50) does not meet the production readiness bar ($0.2500$).
2. **Registry Integrity:**
   * [`config/models_registry.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/config/models_registry.yaml) maintains:
     ```yaml
     active_models:
       production_vision: "yolov8s_archaeology_prod"
     ```
   * Both `geoagri_yolo11n_obb_v4_class_weighted` and `geoagri_yolo11s_obb_v5_class_weighted` are archived exclusively in the experimental catalog.

