# AGP 10 — Class Imbalance Analysis & Loss Function Evaluation

## 1. Quantitative Imbalance Audit

A rigorous census of class instances, containing image frequencies, and independent source-scene distributions was conducted across all splits of [`datasets/geoagri_obb_v3`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/datasets/geoagri_obb_v3).

### Complete Split Breakdown

| Split | Class | Instances | % of Split Instances | Images with Class | Image % | Scenes with Class | Scene % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** (514 imgs, 90 scenes) | `boulders` | 2,618 | 19.8% | 123 | 23.9% | 21 | 23.3% |
| | `others` | 1,189 | 9.0% | 261 | 50.8% | 43 | 47.8% |
| | `ruins` | 1,740 | 13.2% | 212 | 41.2% | 34 | 37.8% |
| | `structures` | **157** | **1.2%** | **78** | **15.2%** | **16** | **17.8%** |
| | `vegetation` | 7,513 | 56.8% | 514 | 100.0% | 90 | 100.0% |
| **Val** (109 imgs, 19 scenes) | `boulders` | 582 | 23.8% | 24 | 22.0% | 4 | 21.1% |
| | `others` | 34 | 1.4% | 22 | 20.2% | 4 | 21.1% |
| | `ruins` | 335 | 13.7% | 37 | 33.9% | 7 | 36.8% |
| | `structures` | **36** | **1.5%** | **21** | **19.3%** | **3** | **15.8%** |
| | `vegetation` | 1,455 | 59.6% | 109 | 100.0% | 19 | 100.0% |
| **Test** (117 imgs, 19 scenes) | `boulders` | 676 | 24.2% | 35 | 29.9% | 5 | 26.3% |
| | `others` | 123 | 4.4% | 31 | 26.5% | 7 | 36.8% |
| | `ruins` | 311 | 11.1% | 46 | 39.3% | 8 | 42.1% |
| | `structures` | **37** | **1.3%** | **10** | **8.5%** | **2** | **10.5%** |
| | `vegetation` | 1,652 | 59.0% | 117 | 100.0% | 19 | 100.0% |

### Critical Analytical Takeaways:
1. **The 30:1 Ratio is Specifically Between Vegetation and Structures:** In the training split, vegetation outnumbers structures by $7,513 : 157$ ($47.85 : 1$). Between boulders and structures, the ratio is $16.68 : 1$.
2. **Scene Scarcity is More Severe Than Instance Scarcity:** While 157 structural instances exist in training, they originate from only **16 distinct scenes**. In the held-out test split, 37 structural instances appear in only **2 independent scenes** (both in Nalanda). Consequently, test recall on structures is essentially evaluating the model's ability to recognize a single architectural complex never seen during training.
3. **Vegetation Ubiquity:** 100% of images and 100% of scenes contain vegetation, providing constant positive gradients for class 4 while gradients for class 3 are sparse.

---

## 2. Ultralytics OBB Loss Architecture Audit

The OBB loss implementation resides in `ultralytics/utils/loss.py` under the class `v8OBBLoss` (inheriting from `v8DetectionLoss`).

### Loss Breakdown:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}} + \lambda_{\text{angle}} \mathcal{L}_{\text{angle}}$$
Where:
* Classification loss $\mathcal{L}_{\text{cls}}$ is computed using `nn.BCEWithLogitsLoss(reduction="none")`:
  ```python
  bce_loss = self.bce(pred_scores, target_scores.to(dtype))
  if self.class_weights is not None:
      bce_loss *= self.class_weights
  loss[1] = bce_loss.sum() / target_scores_sum
  ```
* In `v8DetectionLoss.__init__`:
  ```python
  self.class_weights = getattr(model, "class_weights", None)
  if self.class_weights is not None:
      self.class_weights = self.class_weights.to(device).view(1, 1, -1)
  ```

### Key Technical Findings:
* **Native Hook Available:** Ultralytics already contains an unadvertised hook: if `model.class_weights` is assigned a 1D tensor of shape `(num_classes,)`, `v8OBBLoss` scales the classification loss per class.
* **Localization is Decoupled:** `RotatedBboxLoss` and `RotatedTaskAlignedAssigner` rely on task-aligned matching (combining prediction score $\times$ IoU). Changing classification weights affects target score weighting in the assigner, indirectly influencing which anchor points are designated as foreground for the minority class.

---

## 3. Risks of Overemphasizing Minority-Class Weights

Applying aggressive inverse-frequency class weights (e.g. $w_{\text{structures}} \approx 48.0$):
1. **Amplified False Positives:** Because `structures` share textural characteristics with `ruins` and rectilinear stone slabs, high class weights cause the network to predict `structures` with high confidence on ambiguous rubble, causing overall precision to plummet.
2. **Noise Amplification:** In an archaeological dataset where 2,701 boundary instances were clamped, weighting minority instances disproportionately amplifies label noise.
3. **Training Instability:** Sharp gradient spikes from rare instances can destabilize batch normalization layers, particularly with small batch sizes ($B=4$).

---

## 4. Controlled Proposal: EXP-D (Class-Reweighted Training)

To protect experimental comparability, **no loss modifications will be applied to EXP-B or EXP-C**. 

Should EXP-B and EXP-C demonstrate that annotation refitting and augmentation alone are insufficient to overcome minority-class suppression, EXP-D is formally specified as:
* **Hypothesis:** Soft-damped effective number of samples weighting ($w_c \propto (1 - \beta) / (1 - \beta^{N_c})$ with $\beta = 0.999$) improves structural recall by at least 10 percentage points without dropping overall precision below 0.20.
* **Weighting Formulation:** Damped inverse square root rather than raw inverse frequency:
  $$w_c = \sqrt{\frac{\max_k N_k}{N_c + \epsilon}}$$
  Normalized such that $\sum w_c = C$. For our train split:
  * `boulders`: 0.95
  * `others`: 1.41
  * `ruins`: 1.16
  * `structures`: **3.88** (bounded to prevent gradient explosions)
  * `vegetation`: 0.56
* **Prerequisite Unit Tests:** 
  1. Test that `model.class_weights` broadcasts to `(1, 1, 5)` on GPU without throwing runtime shape mismatches.
  2. Test gradient finite-check: assert no NaN/Inf gradients when a batch contains zero `structures`.
  3. Validate deterministic checkpoint saving and export compatibility.

