# AGP 09 — Final Research Decision & Generalization Strategy Report

**Author:** Senior Computer Vision Research Engineer & Project Auditor  
**Date:** October 28, 2026  
**Project:** GeoAgri-AI  

---

## 1. Categorized Findings & Evidence Hierarchy

| Finding / Topic | Classification | Concrete Evidence & Audit Analysis |
| :--- | :---: | :--- |
| **Geographic Class Disjointness** | **Verified** | Analysis of test partition proves 100% of test `boulders` are in Hampi, while 97.3% of test `structures` are in Nalanda. |
| **Architectural Material Domain Shift** | **Verified** | Model was trained primarily on buff sandstone (Khajuraho); test structures consist of ancient fired-brick stupas (Nalanda), confusing brick courses with `ruins`. |
| **OBB Angular Sensitivity** | **Verified** | Model generated 38 structure detections on Nalanda at conf=0.25, but angular rotation offsets dropped polygon IoU below the 0.50 threshold. |
| **Absence of Metric Parity Error** | **Verified** | Validation and test evaluations executed under identical Ultralytics `OBBValidator` settings ($conf=0.001$, $iou=0.7$, $imgsz=640$). |
| **Clamping Geometry Impact** | **Verified** | Clamping modified 2,701 instances (14.6%), but only 360 (1.95% total) experienced area changes $>15\%$; zero polygons lost convexity. |
| **Structural Sample Scarcity** | **Verified** | `structures` represents only 1.25% of all annotations (230 instances), creating severe gradient starvation during training. |
| **Model Capacity Bottleneck** | **Strongly Supported** | YOLO11n (2.65M parameters) lacks the multi-scale receptive field required to decouple porous rock texture from eroded masonry. |
| **Cross-Site Color Augmentation Benefit** | **Strongly Supported** | Expanding hue/saturation perturbation is expected to bridge sandstone-to-brick spectral gaps. |
| **Dense Pixel Mask Suitability** | **Verified** | OBB boxes cannot serve as dense segmentation masks; UAV orthomosaics and Sentinel-2 NDVI are necessary for surface analysis. |
| **Field Performance on Unseen Sites** | **Hypothesis** | Additional unseen South Indian sites (Badami, Aihole) will follow the granite vs sandstone material divide. |

---

## 2. Synthesized Engineering Strategy

1. **Annotation Recovery:** Delivered `datasets/geoagri_obb_v3` with `minAreaRect` polygon refitting, eliminating angular skew on boundary instances without altering baseline v2.
2. **Structures Class Strategy:** Reject merging `structures` into `ruins`. Retain taxonomic separation and execute targeted 150+ multi-site scene collection.
3. **Multimodal Roadmap:** Decouple object localization (YOLO11-OBB) from land-cover surface mapping (SegFormer / DeepLabV3+) and satellite macro-vegetation (Sentinel-2 NDVI).
4. **Acceptance Status:** Model remains classified as **VALIDATION CANDIDATE**. Production baseline `yolov8s_archaeology_prod` remains active.

---

## 3. Recommended Next AGP

### **AGP 10 — Controlled Multi-Site Training Experiments (EXP-B through EXP-E) & Validation Selection**
* Execute the controlled experiment matrix on `datasets/geoagri_obb_v3/val`:
  - Run **EXP-B** (refitted geometry) and **EXP-C** (multi-site augmentation).
  - Run **EXP-D** (class-frequency reweighted loss).
  - Benchmark **EXP-E** (YOLO11s-OBB capacity scaling on RTX 3050).
* Select top validation candidate before opening the held-out test split.

