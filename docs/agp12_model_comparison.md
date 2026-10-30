# AGP 12B — Controlled Model Comparison: EXP-C vs EXP-D vs EXP-E

## 1. Executive Summary & Experimental Context

This controlled comparative study evaluates three candidate models developed across AGP 10, AGP 11, and AGP 12. All three models were trained and validated under **strictly compatible experimental conditions**:
* **Dataset:** `datasets/geoagri_obb_v3/data.yaml` (geometry-refitted polygon annotations)
* **Validation Split:** 109 images originating from **19 unique source scenes** across 4 archaeological sites
* **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB GDDR6 VRAM)
* **Software:** PyTorch 2.6.0+cu124, Ultralytics 8.4.191, CUDA 12.4
* **Image Size & Batch:** 640x640 resolution, batch size 4
* **Quarantine Boundary:** Held-out test split remained strictly untouched.

---

## 2. Global Metric Comparison (Validation Set)

| Metric | EXP-C (YOLO11n Domain-Aug) | EXP-D (YOLO11n Class-Weighted) | EXP-E (YOLO11s Class-Weighted) | Optimal Configuration |
| :--- | :---: | :---: | :---: | :---: |
| **Model Architecture** | YOLO11n-OBB | YOLO11n-OBB | **YOLO11s-OBB** | — |
| **Parameter Count** | 2,661,427 (2.66M) | 2,661,427 (2.66M) | **9,700,722 (9.70M)** | Nano (3.65x smaller) |
| **Computation (GFLOPs)**| 6.8 GFLOPs | 6.8 GFLOPs | **22.4 GFLOPs** | Nano (3.29x lighter) |
| **Overall Precision (P)**| **0.5722** | 0.3720 | 0.4680 | **EXP-C** (+22.3% over EXP-E) |
| **Overall Recall (R)** | 0.4140 | **0.4408** | 0.3717 | **EXP-D** (+18.6% over EXP-E) |
| **Overall mAP@50** | **0.4497** | 0.4150 | 0.3797 | **EXP-C** (+18.4% over EXP-E) |
| **Overall mAP@50-95** | **0.2638** | 0.2514 | 0.2297 | **EXP-C** (+14.8% over EXP-E) |
| **Inference Latency** | **10.11 ms** | **10.21 ms** | **13.84 ms** | Nano (+35.6% faster forward pass) |
| **Total Pipeline Latency**| 55.36 ms | 59.53 ms | **35.92 ms** | EXP-E (Faster NMS convergence) |

---

## 3. Per-Class Granular AP & Recall Comparison

| Target Class | Metric | EXP-C (Augmented) | EXP-D (Class-Weighted) | EXP-E (Capacity-Scaled) | Winner |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`structures`** | Precision | **0.7781** | 0.4097 | 0.5868 | EXP-C |
| (36 instances) | Recall | 0.7222 | **0.7711** | 0.5556 | **EXP-D** |
| | mAP@50 | **0.7775** | 0.7006 | 0.5404 | **EXP-C** |
| | mAP@50-95 | **0.5268** | 0.4791 | 0.3515 | **EXP-C** |
| **`ruins`** | Precision | **0.5823** | 0.2370 | 0.3907 | EXP-C |
| (335 instances) | Recall | 0.1164 | **0.1642** | 0.1224 | **EXP-D** |
| | mAP@50 | **0.2034** | 0.1443 | 0.1665 | **EXP-C** |
| | mAP@50-95 | **0.1327** | 0.0824 | 0.0863 | **EXP-C** |
| **`boulders`** | Precision | 0.3293 | 0.2937 | **0.4828** | **EXP-E** |
| (582 instances) | Recall | 0.2973 | **0.3282** | 0.2509 | EXP-D |
| | mAP@50 | 0.2487 | 0.2132 | **0.2761** | **EXP-E** |
| | mAP@50-95 | 0.1179 | 0.0961 | **0.1290** | **EXP-E** |
| **`vegetation`** | Precision | 0.3230 | **0.4402** | 0.4267 | EXP-D |
| (1,455 instances) | Recall | 0.1402 | 0.1759 | **0.2241** | **EXP-E** |
| | mAP@50 | 0.1687 | 0.2429 | **0.2679** | **EXP-E** |
| | mAP@50-95 | 0.0748 | 0.1150 | **0.1272** | **EXP-E** |
| **`others`** | Precision | **0.8484** | 0.4794 | 0.4530 | EXP-C |
| (34 instances) | Recall | **0.7941** | 0.7647 | 0.7059 | **EXP-C** |
| | mAP@50 | **0.8504** | 0.7741 | 0.6475 | **EXP-C** |
| | mAP@50-95 | 0.4666 | **0.4845** | 0.4546 | EXP-D |

---

## 4. Evaluation of the 5 Experimental Questions

### 1. Does EXP-E provide better overall detection accuracy?
**NO.** EXP-E achieved an overall mAP@50 of **0.3797**, which is **$-8.5\%$ lower than EXP-D (0.4150)** and **$-15.6\%$ lower than EXP-C (0.4497)**. Overall recall also regressed to $0.3717$ (vs $0.4408$ in EXP-D).

### 2. Does EXP-E provide better structures recall?
**NO.** EXP-E structures recall collapsed to **$0.5556$**, representing a severe **$-21.5\%$ absolute drop** compared to EXP-D ($0.7711$). EXP-D remains the definitive champion for minority structure recovery.

### 3. Does EXP-E provide better boulders and vegetation performance?
**YES.** EXP-E achieved the highest scores recorded on the two texture-dense, sample-rich majority classes:
* `boulders`: mAP@50 reached **0.2761** (vs 0.2132 in EXP-D and 0.2487 in EXP-C).
* `vegetation`: mAP@50 reached **0.2679** (vs 0.2429 in EXP-D and 0.1687 in EXP-C); recall reached **0.2241** (vs 0.1759 in EXP-D).
* *Explanation:* With 582 boulder instances and 1,455 vegetation instances, these classes provide enough training volume for the 9.7M parameter feature extractor to learn nuanced natural texture representations without immediate overfitting.

### 4. Does EXP-E provide acceptable inference latency?
**YES.** At **13.84 ms forward pass** and **35.92 ms end-to-end pipeline latency** on the RTX 3050 Laptop GPU, EXP-E comfortably operates within the $< 50$ ms budget required for interactive tile scanning.

### 5. Does EXP-E provide meaningful improvement relative to increased model complexity?
**NO.** Scaling parameters from 2.66M to 9.70M ($+265\%$) and computation from 6.8 to 22.4 GFLOPs ($+229\%$) resulted in net regressions in overall mAP and severe degradation in minority structural recall. On a dataset with only 514 training images across 88 scenes, model capacity scaling is strictly counterproductive due to overfitting.

