"""Regression and Integration Test Suite for GeoAgri-AI CV Hardening (AGP 08).

Tests:
1. Model registry resolution
2. Production model loading & fallback
3. Candidate OBB model loading
4. Valid image inference & API contract compatibility
5. Zero-detection / blank image handling
6. Invalid inputs: missing files, empty files, unsupported extensions, tiny dimensions
7. OBB coordinate structure & orientation validation
8. CPU inference fallback
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.registry import resolve_model_path, get_model_entry, load_registry_config
from ui.server import predict_bridge

TEST_IMG_DIR = ROOT / "datasets" / "geoagri_obb_v2" / "images" / "test"
SAMPLE_IMG_PATH = next(TEST_IMG_DIR.glob("*.jpg")) if TEST_IMG_DIR.exists() else None

def test_1_model_registry_resolution():
    """Verify registry resolves both production baseline and candidate checkpoints."""
    prod_path = resolve_model_path("production_vision")
    cand_path = resolve_model_path("candidate_vision")
    
    assert prod_path is not None, "Production vision model path resolution failed"
    assert os.path.exists(prod_path), f"Production checkpoint does not exist: {prod_path}"
    assert cand_path is not None, "Candidate vision model path resolution failed"
    assert os.path.exists(cand_path), f"Candidate checkpoint does not exist: {cand_path}"
    assert "geoagri_yolo11n_obb_v1" in cand_path

def test_2_production_model_loading_and_fallback():
    """Verify production YOLO model loads cleanly with 5 classes."""
    prod_path = resolve_model_path("production_vision")
    from ultralytics import YOLO
    model = YOLO(prod_path)
    assert model is not None
    assert len(model.names) == 5

def test_3_candidate_obb_model_loading():
    """Verify candidate YOLO11n-OBB model loads and reports OBB task."""
    cand_path = resolve_model_path("candidate_vision")
    from ultralytics import YOLO
    model = YOLO(cand_path)
    assert model is not None
    assert model.task == "obb"
    assert len(model.names) == 5

def test_4_valid_image_inference_api_contract():
    """Verify end-to-end predict() execution preserves API response fields."""
    assert SAMPLE_IMG_PATH is not None, "No sample image found"
    
    result = predict_bridge.predict(
        image_path=str(SAMPLE_IMG_PATH),
        lat=15.3350,
        lon=76.4600,
        confidence=0.25,
        use_ai_insight=False,
        include_shap=False
    )
    
    assert "probability" in result
    assert "riskLabel" in result
    assert result["riskLabel"] in ["LOW", "MODERATE", "HIGH"]
    assert "metrics" in result
    assert "vegetation" in result["metrics"]
    assert "boulders" in result["metrics"]
    assert "ruins" in result["metrics"]
    assert "structures" in result["metrics"]
    assert "images" in result
    assert "detection" in result["images"]
    assert "combined" in result["images"]
    assert result["images"]["detection"].startswith("data:image/jpeg;base64,")

def test_5_zero_detection_blank_image():
    """Verify blank / black image produces safe zero ratios without exceptions."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.imwrite(tmp.name, blank)
        tmp_path = tmp.name

    try:
        result = predict_bridge.predict(
            image_path=tmp_path,
            lat=15.3350,
            lon=76.4600,
            confidence=0.50
        )
        assert result["probability"] >= 0.0
        assert result["riskLabel"] in ["LOW", "MODERATE", "HIGH"]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_6_invalid_input_handling():
    """Verify input validation guards catch missing, empty, and invalid inputs."""
    # 1. Non-existent file
    with pytest.raises(FileNotFoundError):
        predict_bridge.predict("non_existent_image_12345.jpg", 15.0, 76.0)

    # 2. Empty 0-byte file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with pytest.raises(ValueError, match="empty"):
            predict_bridge.predict(tmp_path, 15.0, 76.0)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 3. Unsupported extension
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"not an image")
        tmp_path = tmp.name
    try:
        with pytest.raises(ValueError, match="Unsupported image format"):
            predict_bridge.predict(tmp_path, 15.0, 76.0)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 4. Dimensions too small (< 16x16)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tiny = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.imwrite(tmp.name, tiny)
        tmp_path = tmp.name
    try:
        with pytest.raises(ValueError, match="dimensions too small"):
            predict_bridge.predict(tmp_path, 15.0, 76.0)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_7_obb_output_structure():
    """Verify OBB predictions produce 4-vertex quadrilateral coordinates."""
    cand_path = resolve_model_path("candidate_vision")
    from ultralytics import YOLO
    model = YOLO(cand_path)
    
    assert SAMPLE_IMG_PATH is not None
    img = cv2.imread(str(SAMPLE_IMG_PATH))
    results = model.predict(img, imgsz=640, device="cuda:0", verbose=False)
    
    assert len(results) > 0
    assert hasattr(results[0], "obb")
    if results[0].obb is not None and len(results[0].obb) > 0:
        pts = results[0].obb.xyxyxyxy
        assert pts.shape[1:] == (4, 2), f"Expected (N, 4, 2) coords, got {pts.shape}"

def test_8_cpu_inference_fallback():
    """Verify candidate OBB model executes successfully on CPU."""
    cand_path = resolve_model_path("candidate_vision")
    from ultralytics import YOLO
    model = YOLO(cand_path)
    
    assert SAMPLE_IMG_PATH is not None
    img = cv2.imread(str(SAMPLE_IMG_PATH))
    results = model.predict(img, imgsz=320, device="cpu", verbose=False)
    assert len(results) > 0
    assert hasattr(results[0], "obb")

