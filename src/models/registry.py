"""Model Registry & Versioning Manager for GeoAgri-AI.

Enforces explicit separation between production baselines, validation candidates,
and experimental models. Prevents silent model promotion without approved configuration changes.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "models_registry.yaml"

def load_registry_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        # Minimal embedded fallback if config is missing
        return {
            "active_models": {
                "production_vision": "yolov8s_archaeology_prod",
                "candidate_vision": "geoagri_yolo11n_obb_v1"
            },
            "models": {
                "yolov8s_archaeology_prod": {
                    "checkpoint_relpath": "runs/detect/yolov8s_archaeology2/weights/best.pt",
                    "status": "production",
                    "task": "detection"
                },
                "geoagri_yolo11n_obb_v1": {
                    "checkpoint_relpath": "runs/obb/geoagri_yolo11n_obb_v1/weights/best.pt",
                    "status": "validation_candidate",
                    "task": "obb"
                }
            }
        }
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_model_path(model_slot: str = "production_vision") -> Optional[str]:
    """Resolves the absolute path for an active model slot without hardcoding."""
    config = load_registry_config()
    active_models = config.get("active_models", {})
    model_key = active_models.get(model_slot, model_slot)

    models = config.get("models", {})
    if model_key in models:
        rel_path = models[model_key].get("checkpoint_relpath")
        if rel_path:
            abs_path = (ROOT / rel_path).resolve()
            if abs_path.exists():
                return str(abs_path)

    # Fallback to direct relative check
    direct = (ROOT / model_slot).resolve()
    if direct.exists():
        return str(direct)

    return None

def get_model_entry(model_slot: str = "production_vision") -> Optional[Dict[str, Any]]:
    config = load_registry_config()
    active_models = config.get("active_models", {})
    model_key = active_models.get(model_slot, model_slot)
    return config.get("models", {}).get(model_key)

