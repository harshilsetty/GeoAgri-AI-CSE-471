# AGP 12 — Evaluation Integrity, Split Quarantine & Benchmark Governance Report

## 1. Executive Summary

Evaluation integrity is the foundational prerequisite for scientific validity and real-world deployment safety in GeoAgri-AI. This report audits the historical exposure of dataset partitions, establishes explicit quarantine boundaries for the held-out test split, and documents the statistical methodology required to validate archaeological object detection without data leakage or premature benchmark saturation.

---

## 2. Dataset Partitioning & Historical Exposure Audit

The GeoAgri-AI dataset contains 740 high-resolution aerial/satellite tiles across 4 primary archaeological sites (Hampi, Khajuraho, Nalanda, and Pattadakal), partitioned as follows:
* **Train Split (`datasets/geoagri_obb_v3/train`):** 514 images (88 source scenes)
* **Validation Split (`datasets/geoagri_obb_v3/val`):** 109 images (21 source scenes)
* **Held-Out Test Split (`datasets/geoagri_obb_v3/test`):** 117 images (19 source scenes)

### 2.1 Historical Exposure Audit (AGP 07 – AGP 11)
| Milestone / Experiment | Evaluated Split | Test Split Exposed? | Purpose of Test Split Access | Status & Risk Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **AGP 07 (Baseline v1)** | Val & Test | Yes (1 run) | Initial baseline benchmark calibration | Standard post-training characterization |
| **AGP 08 (Hardening)** | Val & Test | Yes (1 run) | Verification of production readiness gates | Failed gate; candidate isolated to experimental |
| **AGP 10 (EXP-B & C)** | Val & Test | Yes (2 runs) | Comparative analysis of refitting vs augmentations | Identified generalization gap across sites |
| **AGP 11 (EXP-D)** | Val & Test | Yes (1 run) | Measurement of minority class weighting impact | Confirmed structures recall recovery (0.0716) |

### 2.2 Risk of Test Split Overfitting & Benchmark Exhaustion
Across AGP 07 through AGP 11, the 117-image held-out test split (originating from only 19 source scenes, dominated by Nalanda) has been evaluated multiple times. 

While the test set was never directly used for gradient backpropagation or parameter updates, **repeatedly evaluating experimental candidates against the same 19 scenes creates an indirect informational feedback loop**. If model architectures or hyperparameters are filtered or selected based on held-out test performance:
1. The test set ceases to be an independent, unbiased estimator of real-world generalization.
2. The model risks overfitting to the specific idiosyncratic lighting, sensor angles, and geomorphology of those 19 specific scenes.
3. Statistical significance deteriorates because 19 scenes are insufficient to span India's diverse archaeological geography.

---

## 3. Strict AGP 12 Quarantine Policy

In accordance with MLOps best practices and computer vision research integrity:

1. **Quarantine Mandate:**
   * During AGP 12 model selection, **no candidate (including EXP-E) shall be selected, tuned, or filtered using the test split**.
   * Model comparison and hyperparameter decisions are conducted **strictly on the 109-scene validation split (`datasets/geoagri_obb_v3/val`)**.
2. **Access Isolation:**
   * Script execution targets `--split val`.
   * Test split metrics are only retrieved from historical records for baseline comparison.
3. **No Direct Overwriting:**
   * The production registry (`config/models_registry.yaml`) remains pinned to `yolov8s_archaeology_prod`.
   * Neither EXP-D nor EXP-E shall be promoted to production without passing independent multi-site validation on unexposed benchmarks.

---

## 4. Benchmark Scaling & Protocol Recommendation

To establish true statistical generalization for GeoAgri-AI, the project must move beyond the current 128-scene dataset. 

As formulated in `docs/agp12_data_collection_protocol.md`:
* GeoAgri-AI has specified a **120+ scene field collection protocol** across 6 distinct archaeological sites (Dholavira, Rakhigarhi, Sanchi, Aihole, Badami, and Lothal) across 4 states and 3 distinct geological terrain zones.
* **Quarantine-by-Design:** The new 120-scene dataset will be permanently split into an independent **Multi-Site Archaeological Generalization Benchmark (AGB-120)** that is fully decoupled from the training distribution.
* **Pre-Registration:** Generalization benchmarks will be executed only once on frozen candidate checkpoints, ensuring rigorous, publishable ML evaluation.

