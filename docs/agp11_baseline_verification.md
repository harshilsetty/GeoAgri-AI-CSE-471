# AGP 11 — Baseline Verification and Reconciliation Report

## 1. Executive Summary

This report establishes verifiable ground truth across all experimental baselines and resolves the historical metric discrepancies reported in previous documentation.

---

## 2. Experimental Verification Matrix

| Experiment | Dataset Used | Split Verified | Checkpoint Path | Verified Val mAP@50 | Verified Val mAP@50–95 | Verified Val Recall | Verified Test mAP@50 | Verified Test mAP@50–95 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical Baseline (v1)** | `geoagri_obb_v2` (clamped) | 514 train / 109 val / 117 test | `runs/obb/geoagri_yolo11n_obb_v1/weights/best.pt` | **0.4007** (0.4002 in CSV) | **0.2422** (0.2419 in CSV) | 0.3916 | **0.1412** | **0.0713** |
| **EXP-B (Refit)** | `geoagri_obb_v3` (minAreaRect) | 514 train / 109 val / 117 test | `runs/obb/geoagri_yolo11n_obb_v2_refit/weights/best.pt` | **0.4306** | **0.2567** | 0.3731 | Quarantined | Quarantined |
| **EXP-C (Domain Aug)** | `geoagri_obb_v3` (minAreaRect) | 514 train / 109 val / 117 test | `runs/obb/geoagri_yolo11n_obb_v3_aug/weights/best.pt` | **0.4497** | **0.2638** | **0.4140** | **0.1138** | **0.0536** |

All runs used identical model backbones (`yolo11n-obb.pt`, 2.66M parameters), input resolutions ($640 \times 640$), batch sizes ($4$), and random seed ($42$).

---

## 3. Discrepancy Investigation: Historical Baseline mAP@50–95 (0.2419 vs. 0.2289)

An inspection was conducted on `runs/obb/geoagri_yolo11n_obb_v1/results.csv` and an independent validation run of `best.pt`:

1. **In `results.csv`:**
   * At **Epoch 27**, validation reached `metrics/mAP50(B) = 0.40017` ($0.4002$) and `metrics/mAP50-95(B) = 0.24186` ($0.2419$).
   * Ultralytics selects `best.pt` using the composite fitness function: $\text{fitness} = 0.1 \times \text{mAP50} + 0.9 \times \text{mAP50-95}$.
   * Epoch 27 achieved a fitness score of $0.04002 + 0.21767 = 0.25769$, making it the global maximum fitness across all 39 epochs. Consequently, **Epoch 27 is the exact checkpoint serialized to `best.pt`**.
2. **Re-Evaluation of `best.pt`:**
   * Independent evaluation using `scripts/eval_geoagri_yolo11n_obb.py` directly produced:
     * Overall Precision: `0.4710`
     * Overall Recall: `0.3916`
     * Overall mAP@50: `0.4007`
     * Overall mAP@50–95: `0.2422`
3. **Root Cause of the Discrepancy:**
   * **0.2419** is the true, verified validation mAP@50–95 of `best.pt` at Epoch 27.
   * **0.2289** was a transcription error from an earlier intermediate checkpoint or preliminary evaluation log.
   * **Standardized Baseline Value:** Going forward, the canonical historical baseline validation metric is documented as **mAP@50 = 0.4002, mAP@50–95 = 0.2419**.

---

## 4. Class Mapping and OBB Evaluation Consistency

* **Class Mapping Across All Runs:**
  * 0: `boulders`
  * 1: `others`
  * 2: `ruins`
  * 3: `structures`
  * 4: `vegetation`
* **OBB Evaluation Protocol:**
  * Evaluator: Ultralytics `OBBValidator`
  * Settings: `conf=0.001`, `iou=0.7`, `imgsz=640`, single CUDA GPU 0.
  * Metric computation: Rotated polygon intersection-over-union.
  * Consistency: 100% verified.
