# AGP 10 — Model Selection Report (Validation-Gate Decision)

## 1. Selection Methodology

In strict compliance with unbiased ML evaluation methodology, model selection is conducted **exclusively on the validation split** (109 images across 19 independent source scenes). The held-out test split was kept quarantined throughout training, tuning, and candidate comparison.

---

## 2. Decision Gate Evaluation

| Decision Gate | Gate Requirement | Baseline (v1) | EXP-B (Refit) | EXP-C (Domain Aug) | Gate Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Gate 1: Val mAP@50** | Target $\ge 0.4000$ | 0.4002 | 0.4306 | **0.4497** | **Passed** (EXP-C is top) |
| **Gate 2: Val mAP@50–95**| Higher than baseline | 0.2289 | 0.2567 | **0.2638** | **Passed** (+3.49% over baseline) |
| **Gate 3: Structures Recall** | Target $> 0.0500$ ($>5\%$) | 0.5830 | 0.6389 | **0.7222** | **Passed** (EXP-C is top) |
| **Gate 4: Multi-Class Stability** | No unexplained regression | N/A | Minor boulder recall dip | **Highest recall across boulders, others, and structures** | **Passed** |
| **Gate 5: Defect-Free Outputs** | 0 NaNs, valid OBB format | Valid | Valid | Valid | **Passed** |
| **Gate 6: Target Hardware Latency** | $<50$ ms GPU inference | 6.5 ms | 9.8 ms | 10.1 ms | **Passed** |

---

## 3. Comparative Synthesis

* **EXP-B vs. Baseline:** Demonstrates that geometry-refitted annotations directly improve precision by $+13.67\%$ and overall mAP@50 by $+3.04\%$. Boundary rect refitting removes geometric ambiguity at image edges.
* **EXP-C vs. EXP-B:** The domain-shift augmentations (HSV color variations, mild rotational perturbation, and vertical reflection) significantly improved feature extraction on minority and complex classes:
  * `structures` recall improved by $+8.33$ percentage points (reaching $72.2\%$).
  * `boulders` recall improved by $+8.94$ percentage points (reaching $29.7\%$).
  * `ruins` mAP@50 improved by $+4.98$ percentage points (reaching $20.3\%$).

---

## 4. Final Selection Decision

**Selected Candidate:** **EXP-C**
* **Run Name:** `geoagri_yolo11n_obb_v3_aug`
* **Checkpoint Path:** [`runs/obb/geoagri_yolo11n_obb_v3_aug/weights/best.pt`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/runs/obb/geoagri_yolo11n_obb_v3_aug/weights/best.pt)
* **Epoch:** 20 (Early stopped at 32)
* **Status:** Approved for one-time evaluation on the held-out test split under Phase 8.

