"""Test Suite for Retrained Terrain Erosion Model V2."""

import os
import joblib
import numpy as np
import pytest
import xgboost as xgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V2_PICKLE_PATH = os.path.join(ROOT, "artifacts", "terrain_erosion_v2", "models", "erosion_model_v2.pkl")
V2_JSON_PATH = os.path.join(ROOT, "artifacts", "terrain_erosion_v2", "models", "erosion_model_v2.json")
EXPECTED_FEATURES = [
    "slope",
    "vegetation",
    "elevation",
    "rainfall",
    "soil",
    "boulders",
    "ruins",
    "structures",
]


def test_v2_artifacts_exist():
    assert os.path.exists(V2_PICKLE_PATH), "v2 pickle model not found"
    assert os.path.exists(V2_JSON_PATH), "v2 json booster not found"


def test_v2_model_inference():
    model = joblib.load(V2_PICKLE_PATH)
    sample = np.array([[15.0, 0.4, 400.0, 120.0, 2, 0.1, 0.05, 0.02]], dtype=np.float32)
    prob = float(model.predict_proba(sample)[0][1])
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"


def test_v2_native_json_inference():
    booster = xgb.Booster()
    booster.load_model(V2_JSON_PATH)
    dmat = xgb.DMatrix(
        np.array([[15.0, 0.4, 400.0, 120.0, 2, 0.1, 0.05, 0.02]], dtype=np.float32),
        feature_names=EXPECTED_FEATURES,
    )
    prob = float(booster.predict(dmat)[0])
    assert 0.0 <= prob <= 1.0, f"Native probability out of range: {prob}"
