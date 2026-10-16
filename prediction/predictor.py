"""
prediction/predictor.py
-----------------------
Research-grade prediction interface for Diabetes and Anemia models.
Provides cached model loading, input validation, out-of-distribution (OOD)
detection, and clinical risk stratification.

Public API:
    predict_diabetes(patient_data: dict) -> PredictionResult
    predict_anemia(patient_data: dict)   -> PredictionResult
    get_metadata(disease: str)           -> dict
    get_model(disease: str)              -> Any
    get_train_sample(disease: str)       -> pd.DataFrame
"""

from __future__ import annotations
import os
import json
import threading
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd

from core.config import MODELS_DIR, DISEASE_CONFIGS, SYSTEM_CONFIG, logger
from core.schemas import PredictionResult, ValidationWarning


# ─────────────────────────────────────────────
# Thread-safe Model and Metadata Cache
# ─────────────────────────────────────────────
_cache_lock = threading.Lock()
_model_cache: dict[str, Tuple[Any, dict[str, Any]]] = {}
_train_sample_cache: dict[str, pd.DataFrame] = {}


def _load(disease: str) -> Tuple[Any, dict[str, Any]]:
    """Load model + metadata for the given disease with thread-safe caching."""
    if disease in _model_cache:
        return _model_cache[disease]

    with _cache_lock:
        if disease in _model_cache:
            return _model_cache[disease]

        model_file = os.path.join(MODELS_DIR, f"{disease}_model.pkl")
        meta_file = os.path.join(MODELS_DIR, f"{disease}_metadata.json")

        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Model file not found: {model_file}. "
                "Execute 'python train_models.py' to generate artifacts."
            )
        if not os.path.exists(meta_file):
            raise FileNotFoundError(
                f"Metadata file not found: {meta_file}. "
                "Execute 'python train_models.py' to generate artifacts."
            )

        model = joblib.load(model_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        _model_cache[disease] = (model, metadata)
        return model, metadata


def validate_and_screen_input(
    disease: str, patient_data: dict[str, Any], metadata: dict[str, Any]
) -> list[ValidationWarning]:
    """
    Screen patient input against physiological limits and empirical training bounds.
    Flags biologically implausible or extreme Out-of-Distribution (OOD) values.
    """
    warnings: list[ValidationWarning] = []
    config = DISEASE_CONFIGS.get(disease)
    dist_stats = metadata.get("feature_statistics", {})

    for feature, value in patient_data.items():
        if value is None:
            warnings.append(ValidationWarning(
                feature=feature,
                value=value,
                message=f"Missing value detected for clinical feature '{feature}'.",
                severity="alert",
            ))
            continue

        try:
            num_val = float(value)
        except (ValueError, TypeError):
            continue

        # 1. Physiological reference interval check
        if config and feature in config.reference_ranges:
            low, high = config.reference_ranges[feature]
            if num_val < low or num_val > high:
                warnings.append(ValidationWarning(
                    feature=feature,
                    value=num_val,
                    message=(
                        f"Value {num_val} for '{feature}' lies outside typical physiological "
                        f"limits [{low}, {high}]."
                    ),
                    severity="alert",
                ))

        # 2. Empirical OOD check from training distribution (> 3 std dev)
        if feature in dist_stats:
            stats = dist_stats[feature]
            mean = stats.get("mean")
            std = stats.get("std")
            if mean is not None and std is not None and std > 0:
                z_score = abs(num_val - mean) / std
                if z_score > 3.5:
                    warnings.append(ValidationWarning(
                        feature=feature,
                        value=num_val,
                        message=(
                            f"Value {num_val} for '{feature}' is statistical outlier "
                            f"(z-score: {z_score:.1f} against training distribution)."
                        ),
                        severity="warning",
                    ))

    return warnings


def _classify_risk_tier(prob: float) -> str:
    """Classify risk into clinically calibrated tiers."""
    if prob < SYSTEM_CONFIG.low_risk_threshold:
        return "Low Risk"
    elif prob > SYSTEM_CONFIG.high_risk_threshold:
        return "High Risk"
    else:
        return "Moderate Risk"


def _predict_generic(disease: str, patient_data: dict[str, Any]) -> PredictionResult:
    """Core prediction pipeline with validation, inference, and risk stratification."""
    model, metadata = _load(disease)
    data = patient_data.copy()

    # Normalize gender representation
    if "Gender" in data:
        if isinstance(data["Gender"], str):
            data["Gender"] = 1 if data["Gender"].strip().lower() in ("male", "m") else 0
        else:
            data["Gender"] = int(data["Gender"])

    # Feature validation
    warnings = validate_and_screen_input(disease, data, metadata)

    feature_names = metadata["feature_names"]
    # Ensure all required features are present
    missing = [f for f in feature_names if f not in data]
    if missing:
        raise ValueError(f"Incomplete patient profile. Missing required features: {missing}")

    df = pd.DataFrame([data])[feature_names]

    # Model inference
    pred_idx = int(model.predict(df)[0])
    probabilities = model.predict_proba(df)[0]
    prob_pos = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])

    label_map = metadata.get("label_map", {"0": "Negative", "1": "Positive"})
    prediction_label = label_map.get(str(pred_idx), str(pred_idx))
    risk_tier = _classify_risk_tier(prob_pos)

    return PredictionResult(
        prediction=prediction_label,
        probability=round(prob_pos, 4),
        disease=disease,
        model_name=metadata.get("model_name", type(model).__name__),
        risk_tier=risk_tier,
        warnings=warnings,
    )


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def predict_diabetes(patient_data: dict[str, Any]) -> PredictionResult:
    """Predict diabetes risk for a single patient."""
    return _predict_generic("diabetes", patient_data)


def predict_anemia(patient_data: dict[str, Any]) -> PredictionResult:
    """Predict anemia status for a single patient."""
    return _predict_generic("anemia", patient_data)


def get_metadata(disease: str) -> dict[str, Any]:
    """Retrieve metadata dictionary for a disease."""
    _, metadata = _load(disease)
    return metadata


def get_model(disease: str) -> Any:
    """Retrieve raw trained estimator object."""
    model, _ = _load(disease)
    return model


def get_train_sample(disease: str) -> pd.DataFrame:
    """Retrieve background training sample for global SHAP baseline."""
    if disease in _train_sample_cache:
        return _train_sample_cache[disease]

    with _cache_lock:
        if disease in _train_sample_cache:
            return _train_sample_cache[disease]

        sample_path = os.path.join(MODELS_DIR, f"{disease}_train_sample.pkl")
        if not os.path.exists(sample_path):
            raise FileNotFoundError(
                f"Train sample not found: {sample_path}. "
                "Execute 'python train_models.py' first."
            )
        sample = joblib.load(sample_path)
        _train_sample_cache[disease] = sample
        return sample
