# AGP 08 — Official Model Release Decision & Boundary Specification

**Author:** Senior Machine Learning Engineer, MLOps Lead & Release Auditor  
**Date:** October 28, 2026  
**Candidate Model:** YOLO11n-OBB (`runs/obb/geoagri_yolo11n_obb_v1/weights/best.pt`)  
**Production Baseline:** YOLOv8s Detection (`runs/detect/yolov8s_archaeology2/weights/best.pt`)  
**Hardened Test Suite:** 15/15 Passed (100%)  

---

## 1. Official Release Classification

### **DECISION: VALIDATION CANDIDATE (DO NOT PROMOTE TO PRODUCTION)**

```
[ ] NOT EVALUABLE
[ ] EXPERIMENTAL
[X] VALIDATION CANDIDATE (HARDENED FOR STAGING ONLY)
[ ] READY FOR CONTROLLED PROMOTION
```

### Justification:
The candidate model demonstrates outstanding hardware efficiency (**14.49 ms median latency**, **63.1 FPS**, **54.23 MB VRAM**) and cleanly passes all 15 input-hardening and regression tests. However, on the leak-free, held-out test split (19 unseen historical source scenes), the model suffers from severe cross-site domain degradation:
1. **Overall Test mAP@50 is 0.1412 (14.12%)**, compared to 0.4002 on the validation set.
2. **Minority `structures` recall is 0.0000 (mAP@50 = 0.0266)**, completely failing to detect preserved temple architecture in unseen historical scenes.

Promoting this model to production would degrade detection reliability for cultural heritage structures. The model is retained strictly as a **validation candidate** for staging and research benchmarking.

---

## 2. Limits of the Computer Vision Pipeline (Phase 8 Specification)

To maintain scientific and academic integrity, the following boundaries of the GeoAgri-AI CV pipeline are formally established:

1. **Oriented Bounding Boxes $\neq$ Dense Segmentation Masks:**
   - OBB predictions provide rotated 4-vertex quadrilateral bounding boxes identifying object locations and orientations.
   - OBB predictions do **not** represent dense pixel-level boundaries. Treating OBB boxes as dense masks overestimates surface coverage by 20% to 45% (especially for porous tree canopies and irregular boulders).
2. **No Direct Bare-Soil or Erosion Risk Output:**
   - The computer vision subsystem detects archaeological terrain objects (`boulders`, `ruins`, `structures`, `vegetation`).
   - It does **not** directly output soil texture, bare-soil percentage, or erosion probability.
3. **Decoupled System Architecture:**
   - Terrain erosion risk is predicted exclusively by the dedicated **XGBoost Classifier** (`terrain_model/erosion_model.pkl`), which consumes multi-factor geospatial inputs (slope, elevation, rainfall, soil type, and terrain object ratios).
   - Crop suitability recommendations are governed by the specialized **Crop Action Engine**.
4. **Prohibition of False Segmentation Training:**
   - **DO NOT** train semantic segmentation networks (such as DeepLabV3+ or U-Net) using the current 4-point OBB labels as if they were true pixel masks. This produces artificially rectangular mask predictions and violates scientific methodology.

### Recommended Future Semantic Segmentation Workflow:
To achieve genuine pixel-level vegetation and bare-soil coverage estimation:
* Collect aerial drone orthomosaics with multispectral NDVI channels.
* Annotate dense polygonal masks for vegetation canopy vs exposed bare soil.
* Train a dedicated segmentation network (e.g. SegFormer-B0 or DeepLabV3Plus with ResNet backbone).

---

## 3. Rollback & Production Isolation Plan

* **Active Production Model:** The production slot `production_vision` in [`config/models_registry.yaml`](file:///c:/Users/HARSHIL%20SOMISETTY/HS/Education/LPU/Academics/SEM%205/CSE%20471%20DEEP%20LEARNING%20FOR%20COMPUTER%20VISION/Arch/GeoAgri-AI/config/models_registry.yaml) points strictly to `yolov8s_archaeology_prod` (`runs/detect/yolov8s_archaeology2/weights/best.pt`).
* **Isolation Guarantee:** Even though the candidate model exists on disk, `predict_bridge.py` will **never** invoke it in production unless explicitly overridden via `GEOAGRI_VISION_SLOT="candidate_vision"`.
* **Zero Rollback Risk:** Because candidate promotion was blocked, no rollback action is required. Production endpoints remain 100% stable.

---

## 4. Recommended Next AGP

### **AGP 09 — Multi-Site Architectural Augmentation & Multimodal Segmentation Pipeline**
1. **Dataset Expansion:** Collect 150+ new high-resolution photographs of standing temple architecture across Hampi, Pattadakal, and Badami to overcome the `structures` sample starvation.
2. **Multispectral / NDVI Integration:** Integrate satellite multispectral bands (Sentinel-2) for true bare-soil and vegetation ratio calculation, replacing bounding box ratio approximations.
3. **Model Fine-Tuning:** Retrain YOLO11-OBB using frozen backbone weights and focal loss reweighting on the expanded structural dataset.

