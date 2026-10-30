"""Terrain Erosion Prediction Model V2 - Training, GPU Benchmark, and Evaluation Pipeline.

Author: Senior Machine Learning Engineer & Geospatial MLOps Team
Project: GeoAgri-AI (Terrain Erosion Module)
Hardware Target: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, CUDA 12.9)
"""

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

try:
    import shap
except ImportError:
    shap = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(ROOT, "terrain_model", "erosion_dataset.csv")
OUTPUT_DIR = os.path.join(ROOT, "artifacts", "terrain_erosion_v2")

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
TARGET_COL = "erosion"
RANDOM_SEED = 42


def get_dataset_fingerprint(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def audit_and_load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    assert df.shape == (1000, 9), f"Unexpected dataset shape: {df.shape}"
    assert list(df.columns) == EXPECTED_FEATURES + [TARGET_COL], "Column mismatch"
    assert df.isnull().sum().sum() == 0, "Missing values found"
    assert df.duplicated().sum() == 0, "Exact duplicates found"

    X = df[EXPECTED_FEATURES].astype(np.float32)
    y = df[TARGET_COL].astype(np.int64)
    return df, X, y


def benchmark_cpu_vs_gpu(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    print("\n--- Phase 6.1: CPU vs GPU Hardware Benchmark ---")
    base_params = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
    }

    # CPU Run
    t0_cpu = time.perf_counter()
    model_cpu = xgb.XGBClassifier(**base_params, device="cpu")
    model_cpu.fit(X_train, y_train)
    cpu_train_time = time.perf_counter() - t0_cpu
    cpu_preds = model_cpu.predict_proba(X_train)[:, 1]

    # GPU Run
    gpu_available = True
    gpu_error = None
    t0_gpu = time.perf_counter()
    try:
        model_gpu = xgb.XGBClassifier(**base_params, device="cuda")
        model_gpu.fit(X_train, y_train)
        gpu_train_time = time.perf_counter() - t0_gpu
        gpu_preds = model_gpu.predict_proba(X_train)[:, 1]
    except Exception as exc:
        gpu_available = False
        gpu_error = str(exc)
        gpu_train_time = None
        gpu_preds = None

    parity_max_diff = float(np.max(np.abs(cpu_preds - gpu_preds))) if gpu_available else None

    result = {
        "cpu_train_time_sec": round(cpu_train_time, 4),
        "gpu_train_time_sec": round(gpu_train_time, 4) if gpu_train_time else None,
        "gpu_acceleration_verified": gpu_available,
        "gpu_error": gpu_error,
        "cpu_gpu_probability_max_abs_diff": parity_max_diff,
        "speedup_ratio": round(cpu_train_time / gpu_train_time, 2) if gpu_train_time else None,
    }
    print(f"CPU fit duration (500 trees): {cpu_train_time:.4f}s")
    if gpu_available:
        print(f"GPU fit duration (500 trees): {gpu_train_time:.4f}s")
        print(f"Maximum absolute probability difference: {parity_max_diff:.6e}")
    else:
        print(f"GPU fit failed: {gpu_error}")

    return result


def cross_validate_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], pd.DataFrame]:
    print("\n--- Phase 6.2: 5-Fold Stratified Cross-Validation Parameter Search ---")
    param_grid = [
        {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 200, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 250, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 250, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 200, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},  # baseline config
        {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 200, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0},
        {"max_depth": 4, "learning_rate": 0.08, "n_estimators": 150, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.2, "reg_lambda": 1.5},
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results = []

    for idx, params in enumerate(param_grid):
        fold_f1s = []
        fold_accs = []
        fold_aucs = []
        fold_losses = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_f_tr, y_f_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_f_val, y_f_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            model = xgb.XGBClassifier(
                **params,
                tree_method="hist",
                device="cuda",
                eval_metric="logloss",
                random_state=RANDOM_SEED,
            )
            model.fit(X_f_tr, y_f_tr)

            val_probs = model.predict_proba(X_f_val)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)

            fold_accs.append(accuracy_score(y_f_val, val_preds))
            fold_f1s.append(f1_score(y_f_val, val_preds))
            fold_aucs.append(roc_auc_score(y_f_val, val_probs))
            fold_losses.append(log_loss(y_f_val, val_probs))

        mean_f1 = float(np.mean(fold_f1s))
        mean_acc = float(np.mean(fold_accs))
        mean_auc = float(np.mean(fold_aucs))
        mean_loss = float(np.mean(fold_losses))

        entry = {
            "candidate_id": idx + 1,
            **params,
            "cv_mean_f1": round(mean_f1, 4),
            "cv_std_f1": round(float(np.std(fold_f1s)), 4),
            "cv_mean_accuracy": round(mean_acc, 4),
            "cv_mean_roc_auc": round(mean_auc, 4),
            "cv_mean_logloss": round(mean_loss, 4),
        }
        results.append(entry)
        print(f"Candidate {idx+1}: depth={params['max_depth']}, lr={params['learning_rate']}, trees={params['n_estimators']} -> CV F1: {mean_f1:.4f}, AUC: {mean_auc:.4f}, Loss: {mean_loss:.4f}")

    results_df = pd.DataFrame(results).sort_values(by=["cv_mean_f1", "cv_mean_roc_auc", "cv_mean_logloss"], ascending=[False, False, True])
    best_candidate = results_df.iloc[0].to_dict()
    best_params = {
        "max_depth": int(best_candidate["max_depth"]),
        "learning_rate": float(best_candidate["learning_rate"]),
        "n_estimators": int(best_candidate["n_estimators"]),
        "subsample": float(best_candidate["subsample"]),
        "colsample_bytree": float(best_candidate["colsample_bytree"]),
        "reg_alpha": float(best_candidate["reg_alpha"]),
        "reg_lambda": float(best_candidate["reg_lambda"]),
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
    }
    print(f"\nOptimal Candidate Selected: Candidate {int(best_candidate['candidate_id'])} (CV F1: {best_candidate['cv_mean_f1']:.4f})")
    return best_params, results_df


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, best_params: Dict[str, Any]) -> xgb.XGBClassifier:
    print("\n--- Phase 6.3: Refitting Final Model on Full Training Set ---")
    try:
        model = xgb.XGBClassifier(**best_params, device="cuda")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        duration = time.perf_counter() - t0
        print(f"Final model successfully fitted on NVIDIA GPU (cuda) in {duration:.4f}s")
    except Exception as exc:
        print(f"GPU fit failed ({exc}). Falling back gracefully to CPU (hist)...")
        model = xgb.XGBClassifier(**best_params, device="cpu")
        model.fit(X_train, y_train)

    return model


def evaluate_on_untouched_test_set(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    print("\n--- Phase 7: Final Untouched Test Set Evaluation ---")
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp))

    metrics = {
        "test_sample_count": len(y_test),
        "test_class_distribution": {"class_0": int((y_test == 0).sum()), "class_1": int((y_test == 1).sum())},
        "accuracy": float(round(accuracy_score(y_test, preds), 4)),
        "precision": float(round(precision_score(y_test, preds), 4)),
        "recall": float(round(recall_score(y_test, preds), 4)),
        "f1_score": float(round(f1_score(y_test, preds), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, probs), 4)),
        "pr_auc": float(round(average_precision_score(y_test, probs), 4)),
        "balanced_accuracy": float(round(balanced_accuracy_score(y_test, preds), 4)),
        "specificity": float(round(specificity, 4)),
        "brier_score": float(round(brier_score_loss(y_test, probs), 4)),
        "log_loss": float(round(log_loss(y_test, probs), 4)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    print("Test Evaluation Results:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return metrics, probs


def validate_risk_thresholds(y_test: pd.Series, probs: np.ndarray) -> Dict[str, Any]:
    print("\n--- Phase 8: Risk Thresholds & Probability Calibration Analysis ---")
    low_mask = probs < 0.30
    mod_mask = (probs >= 0.30) & (probs < 0.70)
    high_mask = probs >= 0.70

    low_erosion_rate = float(y_test[low_mask].mean()) if low_mask.sum() > 0 else 0.0
    mod_erosion_rate = float(y_test[mod_mask].mean()) if mod_mask.sum() > 0 else 0.0
    high_erosion_rate = float(y_test[high_mask].mean()) if high_mask.sum() > 0 else 0.0

    threshold_audit = {
        "canonical_spec": {"LOW": "< 0.30", "MODERATE": "0.30 <= P < 0.70", "HIGH": ">= 0.70"},
        "distribution_on_test_set": {
            "low_count": int(low_mask.sum()),
            "moderate_count": int(mod_mask.sum()),
            "high_count": int(high_mask.sum()),
        },
        "empirical_erosion_rates_per_bin": {
            "low_risk_actual_erosion_rate": round(low_erosion_rate, 4),
            "moderate_risk_actual_erosion_rate": round(mod_erosion_rate, 4),
            "high_risk_actual_erosion_rate": round(high_erosion_rate, 4),
        },
        "monotonicity_verified": low_erosion_rate < mod_erosion_rate < high_erosion_rate,
    }

    print(f"LOW Risk (<0.30): {low_mask.sum()} samples | Actual erosion rate: {low_erosion_rate*100:.1f}%")
    print(f"MODERATE Risk (0.30-0.70): {mod_mask.sum()} samples | Actual erosion rate: {mod_erosion_rate*100:.1f}%")
    print(f"HIGH Risk (>=0.70): {high_mask.sum()} samples | Actual erosion rate: {high_erosion_rate*100:.1f}%")
    print(f"Monotonic Risk Progression: {threshold_audit['monotonicity_verified']}")

    return threshold_audit


def compute_feature_importance_and_shap(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict[str, Any]:
    print("\n--- Phase 9: Explainability and SHAP Feature Importance ---")
    booster = model.get_booster()
    gain_importance = booster.get_score(importance_type="gain")
    weight_importance = booster.get_score(importance_type="weight")

    sorted_gain = sorted(
        [{"feature": k, "gain": round(float(v), 4)} for k, v in gain_importance.items()],
        key=lambda x: x["gain"],
        reverse=True,
    )

    shap_summary = []
    if shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            for feat, val in zip(EXPECTED_FEATURES, mean_abs_shap):
                shap_summary.append({"feature": feat, "mean_abs_shap": round(float(val), 4)})
            shap_summary = sorted(shap_summary, key=lambda x: x["mean_abs_shap"], reverse=True)
            print("TreeSHAP Global Feature Ranking:")
            for item in shap_summary:
                print(f"  {item['feature']}: {item['mean_abs_shap']}")
        except Exception as exc:
            print(f"SHAP calculation error: {exc}")

    return {
        "gain_importance": sorted_gain,
        "shap_importance": shap_summary,
    }


def save_artifacts_and_metadata(
    model: xgb.XGBClassifier,
    best_params: Dict[str, Any],
    benchmark_res: Dict[str, Any],
    eval_metrics: Dict[str, Any],
    threshold_audit: Dict[str, Any],
    explainability_res: Dict[str, Any],
    cv_results_df: pd.DataFrame,
    dataset_hash: str,
):
    print("\n--- Phase 10: Saving Experimental Artifacts & Schemas ---")
    # Native XGBoost artifact (eliminates pickle version incompatibility warning)
    native_model_path = os.path.join(OUTPUT_DIR, "models", "erosion_model_v2.json")
    model.save_model(native_model_path)

    # Joblib pickle artifact
    pickle_model_path = os.path.join(OUTPUT_DIR, "models", "erosion_model_v2.pkl")
    # For maximum inference portability across systems, configure cpu device on serialized pickle
    model.set_params(device="cpu")
    joblib.dump(model, pickle_model_path)

    # Save CV Search Results
    cv_results_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation", "cv_hyperparameter_search.csv"), index=False)

    # Metadata & Schema
    metadata = {
        "model_name": "terrain_erosion_v2",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "path": "terrain_model/erosion_dataset.csv",
            "sha256": dataset_hash,
            "row_count": 1000,
            "features": EXPECTED_FEATURES,
            "target": TARGET_COL,
        },
        "train_test_split": {
            "train_size": 800,
            "test_size": 200,
            "stratified": True,
            "random_seed": RANDOM_SEED,
        },
        "hardware_and_software": {
            "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)",
            "driver": "581.86",
            "python": sys.version,
            "xgboost": xgb.__version__,
            "training_device": "cuda",
            "inference_device": "cpu_compatible",
        },
        "hyperparameters": best_params,
        "hardware_benchmark": benchmark_res,
        "test_evaluation": eval_metrics,
        "threshold_validation": threshold_audit,
        "explainability": explainability_res,
    }

    metadata_path = os.path.join(OUTPUT_DIR, "models", "metadata_v2.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    eval_summary_path = os.path.join(OUTPUT_DIR, "evaluation", "evaluation_summary.json")
    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"Artifacts successfully saved to {OUTPUT_DIR}:")
    print(f"  - {native_model_path}")
    print(f"  - {pickle_model_path}")
    print(f"  - {metadata_path}")


def main():
    print("=" * 70)
    print("GEOAGRI-AI: TERRAIN EROSION PREDICTION MODEL AUDIT & RETRAINING")
    print("=" * 70)

    dataset_hash = get_dataset_fingerprint(DATASET_PATH)
    print(f"Dataset SHA-256: {dataset_hash}")

    df, X, y = audit_and_load_data()

    # Split train/test (80/20) with strict stratification and fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Train split: {len(X_train)} samples ({y_train.sum()} class 1, {(y_train==0).sum()} class 0)")
    print(f"Test split: {len(X_test)} samples ({y_test.sum()} class 1, {(y_test==0).sum()} class 0) [HELD OUT]")

    # Phase 6.1: Hardware benchmark CPU vs GPU
    bench_res = benchmark_cpu_vs_gpu(X_train, y_train)

    # Phase 6.2: Hyperparameter search via 5-Fold Stratified CV on training split
    best_params, cv_results_df = cross_validate_hyperparameters(X_train, y_train)

    # Phase 6.3: Refit final model
    model = train_final_model(X_train, y_train, best_params)

    # Phase 7: Evaluate on untouched test set
    eval_metrics, probs = evaluate_on_untouched_test_set(model, X_test, y_test)

    # Phase 8: Risk threshold validation
    thresh_audit = validate_risk_thresholds(y_test, probs)

    # Phase 9: Feature importance & SHAP
    explainability_res = compute_feature_importance_and_shap(model, X_train, X_test)

    # Phase 10: Save all artifacts & schemas
    save_artifacts_and_metadata(
        model,
        best_params,
        bench_res,
        eval_metrics,
        thresh_audit,
        explainability_res,
        cv_results_df,
        dataset_hash,
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

