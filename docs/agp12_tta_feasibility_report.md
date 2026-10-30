# AGP 12 — Test-Time Augmentation (TTA) Feasibility Report

## 1. Executive Summary & Objective

In object detection workflows, Test-Time Augmentation (TTA) is frequently proposed to boost generalization by ensembling predictions across multiple transformed versions (e.g., multiscale scaling, horizontal flipping) of each query image.

This investigation rigorously evaluated the feasibility, execution mechanics, latency implications, and accuracy impact of TTA for oriented bounding box (OBB) detection in GeoAgri-AI using Ultralytics YOLO-OBB on the NVIDIA RTX 3050 GPU.

---

## 2. Experimental Execution & Architectural Discovery

The evaluation was conducted on the 109-scene validation set (`datasets/geoagri_obb_v3/val`) using the checkpoint from EXP-D (`runs/obb/geoagri_yolo11n_obb_v4_class_weighted/weights/best.pt`).

### 2.1 Native Framework Limitation (Key Finding)
When executing `model.val(augment=True)` or `model.predict(augment=True)` with YOLO-OBB, Ultralytics issues the following framework-level warning across every prediction batch:
```text
WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.
```

### 2.2 Mechanism Analysis
* In standard horizontal bounding box (HBB) models (e.g., `yolo11n.pt`), `augment=True` invokes multiscale forward passes (scales $1.0\times, 0.83\times, 0.67\times$) and left-right flips, followed by coordinate un-flipping and standard axis-aligned Non-Maximum Suppression (NMS).
* In oriented bounding box (OBB) models, detections are parameterized as rotated boxes $(x, y, w, h, \theta)$ or 8-point polygons $(x_1, y_1, \dots, x_4, y_4)$. Transforming angles across flips (e.g., $\theta \to \pi - \theta$) and ensembling multi-scale rotated polygons requires rotated polygon intersection and rotated NMS clustering, which is not implemented in the Ultralytics TTA forward pass.
* Consequently, Ultralytics **silently falls back to single-scale unaugmented prediction**.

---

## 3. Empirical Comparative Metrics

| Configuration | augment Flag | Execution Latency | mAP@50 | mAP@50-95 | Recall | Structures Recall | Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Baseline** | `augment=False` | 38.86s total (11.1 ms/img inf) | 0.4150 | 0.2514 | 0.4408 | 0.7711 | Normal single-scale forward pass |
| **Ultralytics TTA** | `augment=True` | 34.23s total (10.4 ms/img inf) | 0.4150 | 0.2514 | 0.4408 | 0.7711 | Reverts to single-scale; no-op |

*Per-class mAP@50 values are bitwise identical across all 5 classes (`boulders`: 0.2132, `others`: 0.7741, `ruins`: 0.1443, `structures`: 0.7006, `vegetation`: 0.2429).*

---

## 4. Engineering Trade-Off & Recommendation

### 4.1 Custom Multi-Angle TTA Feasibility
To achieve true TTA for OBB, an external pipeline would need to:
1. Rotate each incoming image by $[0^\circ, 90^\circ, 180^\circ, 270^\circ]$.
2. Run independent forward passes ($4\times$ latency multiplier: $\sim 44$ ms GPU inference + OpenCV rotations).
3. Inverse-rotate the output polygon coordinates back to original image space.
4. Merge duplicate oriented predictions via rotated polygon IoU NMS (e.g., using `shapely` or `cv2.intersectConvexConvex`).

### 4.2 Latency Impact
* Baseline inference + postprocessing: **$\sim 60$ ms/image** ($\approx 16.7$ FPS).
* Estimated 4-rotation custom TTA: **$\sim 240$ ms/image** ($\approx 4.1$ FPS), a **$4.0\times$ latency penalty**.

### 4.3 Production Recommendation: REJECT TTA FOR REAL-TIME DEPLOYMENT
1. **Zero Native Benefit:** Ultralytics OBB does not support native TTA; setting `augment=True` in production or evaluation configurations provides zero gain and introduces warning noise into logs.
2. **Excessive Latency Overhead:** For the web interface (`ui/server/predict_bridge.py`), interactive tile analysis requires sub-100ms response times. A $4\times$ latency increase is unacceptable for user experience.
3. **Core Generalization Bottleneck:** As demonstrated in the cross-site data audit (`docs/agp12_cross_site_data_audit.md`), the drop in model generalization stems from extreme lithological and structural domain shifts (granite vs sandstone vs brick), not minor test-time orientation variance. TTA cannot solve out-of-distribution geomorphological features.

