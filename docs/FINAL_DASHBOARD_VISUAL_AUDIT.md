# GeoAgri-AI — Dashboard Visual Audit & Resolution Report

**Academic Course:** CSE 471 — Deep Learning for Computer Vision  
**Evaluation Scope:** Main Dashboard Segmentation & Heatmap Verification  
**Audit Date:** October 30, 2026  
**Status:** Audit Completed & Repaired  

---

## 1. Executive Summary

During final faculty demonstration rehearsal, the GeoAgri-AI main dashboard exhibited anomalies where:
1. **Segmentation Layer:** Rendered as an unsegmented duplicate of the input aerial image without visual mask boundaries, class footprints, or user notices.
2. **Heatmap Layer:** Rendered as a blank darkened tile in the visual analysis grid and was entirely absent as an interactive risk layer on the Leaflet geospatial map.

An end-to-end investigation was conducted across the frontend (Next.js 14, React Leaflet), API bridges (`predict_bridge.py`, `backend_api.py`), and computer vision models. Root causes were diagnosed, repaired with minimal, surgical changes, and verified against the academic test suite without modifying trained model weights or production registries.

---

## 2. Root Cause Analysis

### 2.1 Segmentation Failure
* **Failure Classification:** **Model Loading Failure** & **Silent Fallback Misrepresentation**
* **Detailed Finding:**
  - `_resolve_segmentation_checkpoint()` in `ui/server/predict_bridge.py` attempted to resolve `deeplab_model.pth`. No semantic segmentation checkpoint exists in the repository.
  - The repository's aerial dataset (`datasets/geoagri_obb_v3/data.yaml`) is strictly an **Oriented Bounding Box (OBB)** dataset with 4-point oriented polygons (`boulders`, `ruins`, `structures`, `vegetation`, `others`), NOT pixel-level pixel masks.
  - When `SEG_MODEL is None`, the code silently fell back to `mask = np.zeros(...)`.
  - When blended with `cv2.addWeighted(image, 0.85, seg_vis, 0.15, 0)`, the zero mask produced a 100% duplicate of the original RGB image, giving the appearance of a broken or missing segmentation layer.
* **Ethical & Academic Grounding:** Fabricating a synthetic pixel mask from bounding boxes would misrepresent model capability. Instead, the system now provides an honest **Oriented Bounding Box (OBB)** polygon visualization and explicitly informs the user that pixel-level segmentation is not active.

### 2.2 Heatmap Failure
* **Failure Classification:** **Missing Implementation** & **Coordinate Decoupling**
* **Detailed Finding:**
  - **Image Tile Heatmap:** In `predict_bridge.py`, the image heatmap overlay was hardcoded as `heatmap[mask == 5] = [0, 0, 255]` and `heatmap[mask == 3] = [255, 0, 0]`. Because `mask` was all zeros, `heatmap` was entirely black (`np.zeros_like(image)`), resulting in a darkened, useless image tile.
  - **Geospatial Map (Leaflet):** Although `/api/predict?action=point` computed valid erosion probabilities ($0.0 \le p \le 1.0$), `MapPicker.tsx` discarded the risk intensity and rendered only hardcoded green dots (`#10b981`). Furthermore, no heatmap layer or layer toggle controls existed on the map container.

---

## 3. Repaired Components & Fixes Implemented

| Component / File | Issue Repaired | Fix Description |
| :--- | :--- | :--- |
| `ui/server/predict_bridge.py` | Silent zero mask & blank heatmap | • Implemented honest OBB semi-transparent polygon footprint overlay with vibrant class borders and labels.<br>• Implemented genuine feature density heatmap using Gaussian kernels centered at detected object centroids.<br>• Derived area ratios (`boulders_ratio`, `ruins_ratio`, etc.) directly from detected object contours.<br>• Added `segmentationInfo` and `heatmapInfo` metadata. |
| `ui/components/MapPicker.tsx` | No map heatmap layer or layer controls | • Added multi-tier radial heat halo visualization dynamically styled by erosion risk ($<30\%$ Emerald, $30-69\%$ Amber, $\ge 70\%$ Crimson).<br>• Added interactive map layer toggles (`🔥 Heatmap: ON/OFF`, `📍 Markers: ON/OFF`).<br>• Added color legend and empty state banner explaining required data when unselected. |
| `ui/components/dashboard/VisualAnalysisGrid.tsx` | Unlabeled segmentation and empty state | • Added badge support (`OBB Detected`, `Feature Density`).<br>• Displayed required informational notice: *"Pixel-level segmentation is currently unavailable. The active computer vision model supports oriented object detection."*<br>• Added informative notices in fullscreen modal and empty states. |
| `ui/app/page.tsx` | Decoupled prediction & map risk | • Passed `currentPointRisk` and `riskPoints` to `MapPicker` in both Sidebar and Terrain tabs.<br>• Updated visual tile definitions with dynamic badges and honest OBB status notices. |

---

## 4. Verification Evidence & Test Results

### 4.1 Automated Test Execution

1. **Python API Endpoints Test (`tests/test_api_endpoints.py`):**
   ```
   tests/test_api_endpoints.py ....                                         [100%]
   ================== 4 passed, 2 warnings in 61.11s ===================
   ```
   - Health endpoint: `PASSED`
   - Model debug endpoint: `PASSED`
   - Crop recommendation GET: `PASSED`
   - Point prediction POST: `PASSED`

2. **YOLO OBB Hardening Suite (`tests/test_yolo_obb_hardening.py`):**
   ```
   tests/test_yolo_obb_hardening.py ........                                [100%]
   ======================== 8 passed in 107.15s ========================
   ```
   - All 8 OBB regression and safety tests: `PASSED`

3. **Frontend Production Build & Linter (`npm run build` / `npm run lint`):**
   ```
   ✓ Compiled successfully
   ✓ Generating static pages (4/4)
   Route (app)                              Size     First Load JS
   ┌ ○ /                                    237 kB          325 kB
   ├ ○ /_not-found                          876 B          88.8 kB
   └ ƒ /api/predict                         0 B                0 B
   ```
   - TypeScript compilation: `0 errors`
   - Next.js production build: `PASSED`

### 4.2 End-to-End Pipeline Telemetry

A test image inference was verified against the updated bridge:
```json
{
  "probability": 0.0022,
  "riskLabel": "LOW",
  "segmentationInfo": {
    "available": false,
    "type": "OBB Detection",
    "notice": "Pixel-level segmentation is currently unavailable. The active computer vision model supports oriented object detection."
  },
  "heatmapInfo": {
    "available": true,
    "type": "Feature Density",
    "notice": "Spatial feature density heatmap generated from detected archaeological & terrain features."
  }
}
```

---

## 5. Visual Dashboard Features Supported

| Feature | Support Level | Implementation Details |
| :--- | :--- | :--- |
| **Oriented Bounding Box (OBB)** | **Production Active** | 4-point oriented polygon overlays with class labels and confidence scores. |
| **Semantic Segmentation** | **Informative Empty State** | Disclaims lack of pixel mask and routes to OBB detection footprint. |
| **Image Feature Heatmap** | **Production Active** | Gaussian kernel density over detected structural and archaeological entities. |
| **Geospatial Map Heatmap** | **Production Active** | Multi-tier radial gradient on Leaflet map reflecting erosion probabilities. |
| **Layer Toggles** | **Production Active** | Interactive switches for Heatmap Overlay, Markers, and Area Radius. |
| **Crop Intelligence** | **Production Active** | Multi-class XGBoost recommendation with agro-ecological physiological filters. |

---

## 6. Academic Research Boundaries & Honest Limitations

1. **OBB vs. Semantic Segmentation:** The archaeological computer vision dataset is labeled with oriented bounding boxes. While bounding boxes effectively localize rotated ancient structures and ruins, fine-grained pixel semantic boundaries (e.g., exact masonry mortar boundaries) require dense pixel mask annotation.
2. **Spatial Feature Density Heatmap:** In aerial image tiles without any detected objects, no synthetic heat is fabricated. The dashboard displays an honest disclaimer explaining that no features were detected.
3. **Map Heatmap Interpolation:** The geospatial map heatmap reflects genuine sampled risk points and area scans. Unsampled regions are not artificially filled with ungrounded random numbers.
