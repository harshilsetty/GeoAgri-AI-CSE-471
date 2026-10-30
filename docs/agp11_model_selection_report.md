# AGP 11 — Model Selection Report (Validation-Gate Decision)

## 1. Selection Methodology

In accordance with strict scientific evaluation protocol, candidate models are compared and selected **exclusively on validation split metrics** (109 images across 19 independent source scenes). The held-out test split remained quarantined during all training runs and parameter design.

---

## 2. Validation Decision Gate Evaluation

| Decision Gate | Gate Requirement | Baseline (v1) | EXP-B (Refit) | EXP-C (Aug) | EXP-D (Weighted) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Gate 1: Val mAP@50** | Target $\ge 0.4000$ | 0.4002 | 0.4306 | **0.4497** | 0.4150 | **Passed** |
| **Gate 2: Val mAP@50–95**| Higher than baseline ($>0.2419$) | 0.2419 | 0.2567 | **0.2638** | **0.2514** | **Passed** |
| **Gate 3: Structures Recall** | Target $> 0.0500$ ($>5\%$) | 0.5830 | 0.6389 | 0.7222 | **0.7711** | **Passed (EXP-D Top)** |
| **Gate 4: Minority Sensitivity**| Significant gain in rare classes | Low | Precision gain | Balanced | **Highest recall across all 5 classes** | **Passed** |
| **Gate 5: Training Stability** | Finite loss, 0 NaNs | Stable | Stable | Stable | **Stable (0 NaNs, Peak VRAM 3.27 GB)** | **Passed** |
| **Gate 6: Hardware Latency** | $<50$ ms GPU inference | 6.5 ms | 9.8 ms | 10.1 ms | **10.2 ms** | **Passed** |

---

## 3. Comparative Synthesis & Candidate Selection

* **EXP-C** remains the strongest candidate for **overall balanced detection accuracy and precision** ($0.4497$ mAP@50, $0.5722$ precision).
* **EXP-D** successfully established the benchmark for **maximum minority-class sensitivity and hazard recall** ($77.11\%$ structures recall, $44.08\%$ overall recall, $0.4150$ mAP@50). In archaeological field monitoring and terrain erosion risk prevention, maximizing recall on unexcavated foundations and perimeter boulders is of paramount operational priority.

### Selection Outcome:
**EXP-D (`runs/obb/geoagri_yolo11n_obb_v4_class_weighted/weights/best.pt`) is selected for the single, quarantined evaluation on the held-out test split** to directly measure whether class weighting overcomes the Nalanda semantic misclassification discovered in Phase 2.

