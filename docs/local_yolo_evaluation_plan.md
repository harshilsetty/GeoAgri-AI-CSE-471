# GeoAgri-AI: Computer Vision Evaluation & Validation Plan

**Document ID:** `docs/local_yolo_evaluation_plan.md`  
**Author:** Senior Machine Learning Research Engineer & CV Evaluator  
**Date:** October 28, 2026  
**Status:** Validated Plan

---

## 1. Evaluation Methodology on Independent Scenes

To prevent the cross-split contamination documented in Phase 3, the final candidate model must be validated strictly on the **Holdout Test Split** consisting of **18 independent source scenes** ($\approx 105$ images) that the model has never encountered during training or validation.

---

## 2. Quantitative Evaluation Metrics

### A. Detection & OBB Metrics
1. **mAP@0.50 (Mean Average Precision at IoU 0.50):**
   - Measures general localization accuracy. Target: $> 0.75$ across all classes.
2. **mAP@0.50:0.95:**
   - Measures boundary precision under strict IoU thresholds (from 0.50 to 0.95 in 0.05 steps). Target: $> 0.45$.
3. **Per-Class Precision, Recall, and F1-Score:**
   - Must be reported separately for each class:
     - `boulders` (Class 0)
     - `others` (Class 1)
     - `ruins` (Class 2)
     - `structures` (Class 3)
     - `vegetation` (Class 4)
4. **Normalized Confusion Matrix:**
   - To inspect specific cross-class confusion (e.g. false-positive confusion between `ruins` and `structures`, or `boulders` and rocky pathways in `others`).

---

## 3. Operational Performance & Edge Benchmarks (RTX 3050 Laptop GPU)

| Metric | Target Specification | Measurement Method |
| :--- | :---: | :--- |
| **Mean Inference Latency** | $< 8.0\ \text{ms}$ per image | Timed over 200 warmup-buffered test runs via PyTorch CUDA events |
| **P95 Latency** | $< 12.0\ \text{ms}$ per image | 95th percentile latency distribution under 640x640 input |
| **Throughput** | $> 80\ \text{FPS}$ | Batch inference throughput |
| **Runtime VRAM Consumption** | $< 1.5\ \text{GB}$ | Monitored via `torch.cuda.max_memory_allocated()` |
| **CPU Fallback Latency** | $< 35.0\ \text{ms}$ per image | Timed on Intel Core i5-12450HX CPU execution |

---

## 4. Qualitative Failure-Mode Analysis

Post-training inspection must visually audit:
1. **Shadow Occlusion:** Performance on ruins partially shaded by temple gopurams or trees.
2. **Dense Boulder Clusters:** Check whether individual boulders in Hampi are separated or lumped into single monolithic boxes.
3. **Vegetation Texture Confusion:** Verify whether agricultural fields or grassy temple lawns trigger false detections.
4. **Boundary Clipping:** Verify that predicted boxes respect image edges and do not generate negative coordinate artifacts.

---

## 5. Production Acceptance Criteria (Model Promotion Gate)

The new model candidate will only be approved to replace `runs/detect/yolov8s_archaeology2/weights/best.pt` if it satisfies all of the following criteria:

* [ ] Evaluated on the leak-free holdout test split (18 unseen scenes).
* [ ] Overall $\text{mAP@0.50} \ge 0.70$.
* [ ] Minority class `structures` achieves $\text{Recall} \ge 0.50$ (identifies at least half of the structures).
* [ ] `boulders` and `ruins` achieve $\text{mAP@0.50} \ge 0.75$.
* [ ] Inference latency on RTX 3050 is $\le 10\ \text{ms}$ ($> 100\ \text{FPS}$).
* [ ] Model outputs drop-in compatible bounding boxes (`boxes.xyxy`) without throwing errors in [`ui/server/predict_bridge.py`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/ui/server/predict_bridge.py).

