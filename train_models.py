"""
train_models.py
---------------
Research-grade model training, cross-validation, and holdout evaluation.

Evaluates:
  1. Logistic Regression (with StandardScaler Pipeline)
  2. Random Forest Classifier
  3. XGBoost Classifier

Computes comprehensive clinical metrics:
  - Sensitivity (Recall)
  - Specificity (True Negative Rate)
  - Positive Predictive Value (Precision / PPV)
  - Negative Predictive Value (NPV)
  - Balanced Accuracy
  - F1-Score
  - ROC-AUC & PR-AUC
  - Brier Score (probability calibration)

Persists:
  - Model weights (.pkl)
  - Background training sample for SHAP (.pkl)
  - Research-grade metadata (.json) including holdout test metrics,
    feature distribution statistics (for OOD detection), and
    precomputed global SHAP feature importances.
"""

from __future__ import annotations
import os
import json
import warnings
from typing import Any
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    balanced_accuracy_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from core.config import BASE_DIR, MODELS_DIR, logger

# ─────────────────────────────────────────────
# Cross-Validation Strategy
# ─────────────────────────────────────────────
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def _compute_clinical_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    """Compute complete suite of clinical diagnostic metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "positive_predictive_value": float(ppv),
        "negative_predictive_value": float(npv),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def _compute_feature_statistics(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute empirical training feature distribution parameters for OOD checks."""
    stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            stats[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
            }
    return stats


def _compute_global_shap(model, background_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Precompute global SHAP importance over the training background sample."""
    try:
        if isinstance(model, Pipeline):
            scaler = model.named_steps.get("scaler")
            clf = model.named_steps.get("classifier")
            bg_scaled = pd.DataFrame(scaler.transform(background_df), columns=background_df.columns)
            explainer = shap.LinearExplainer(clf, bg_scaled)
            sv = explainer.shap_values(bg_scaled)
        elif isinstance(model, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(background_df)
        else:
            sample = shap.sample(background_df, min(50, len(background_df)))
            explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], sample)
            sv = explainer.shap_values(background_df)

        if isinstance(sv, list) and len(sv) == 2:
            matrix = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            matrix = sv[:, :, 1]
        else:
            matrix = np.array(sv)

        mean_abs = np.mean(np.abs(matrix), axis=0)
        features = list(background_df.columns)
        importance = [
            {"feature": f, "mean_abs_shap": round(float(mean_abs[i]), 4)}
            for i, f in enumerate(features)
        ]
        importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        for rank, item in enumerate(importance, start=1):
            item["rank"] = rank
        return importance
    except Exception as e:
        logger.warning(f"Global SHAP precomputation warning: {e}")
        return []


def _select_train_evaluate(models: dict, X_train, y_train, X_test, y_test) -> tuple:
    """Run CV, select champion model, fit, evaluate on holdout test set."""
    cv_results = {}
    fitted_candidates = {}

    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=CV, scoring=SCORING)
        cv_results[name] = {
            "CV Accuracy": float(scores["test_accuracy"].mean()),
            "CV Precision": float(scores["test_precision"].mean()),
            "CV Recall": float(scores["test_recall"].mean()),
            "CV F1": float(scores["test_f1"].mean()),
            "CV ROC-AUC": float(scores["test_roc_auc"].mean()),
            "F1 Std": float(scores["test_f1"].std()),
            "AUC Std": float(scores["test_roc_auc"].std()),
        }
        print(
            f"  [{name:19s}] CV F1: {cv_results[name]['CV F1']:.4f} +/- {cv_results[name]['F1 Std']:.4f} | "
            f"CV AUC: {cv_results[name]['CV ROC-AUC']:.4f} +/- {cv_results[name]['AUC Std']:.4f}"
        )
        model.fit(X_train, y_train)
        fitted_candidates[name] = model

    best_name = max(
        cv_results,
        key=lambda m: (
            round(cv_results[m]["CV F1"], 4),
            round(cv_results[m]["CV ROC-AUC"], 4),
        ),
    )
    champion_model = fitted_candidates[best_name]
    print(f"\n  [Selected Champion] {best_name}")

    # Holdout Test Set Evaluation
    y_pred = champion_model.predict(X_test)
    y_prob = champion_model.predict_proba(X_test)[:, 1]
    holdout_metrics = _compute_clinical_metrics(y_test, y_pred, y_prob)

    print("  Holdout Test Set Performance:")
    print(f"    Sensitivity (Recall): {holdout_metrics['sensitivity']:.4f}")
    print(f"    Specificity:          {holdout_metrics['specificity']:.4f}")
    print(f"    PPV (Precision):      {holdout_metrics['positive_predictive_value']:.4f}")
    print(f"    NPV:                  {holdout_metrics['negative_predictive_value']:.4f}")
    print(f"    F1 Score:             {holdout_metrics['f1_score']:.4f}")
    print(f"    ROC-AUC:              {holdout_metrics['roc_auc']:.4f}")
    print(f"    PR-AUC:               {holdout_metrics['pr_auc']:.4f}")
    print(f"    Brier Score:          {holdout_metrics['brier_score']:.4f}")

    return best_name, champion_model, cv_results, holdout_metrics


# ─────────────────────────────────────────────
# DIABETES
# ─────────────────────────────────────────────
def train_diabetes():
    print("=" * 70)
    print("  TRAINING DIABETES MODEL (RESEARCH SPECIFICATION)")
    print("=" * 70)

    csv_path = os.path.join(BASE_DIR, "diabetes_dataset.csv")
    df = pd.read_csv(csv_path)

    if "PatientID" in df.columns:
        df.drop(columns=["PatientID"], inplace=True)

    if df["Gender"].dtype == object:
        df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).fillna(0).astype(int)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=42, eval_metric="logloss",
        ),
    }

    best_name, best_model, cv_results, holdout_metrics = _select_train_evaluate(
        models, X_train, y_train, X_test, y_test
    )

    # Persist model
    model_path = os.path.join(MODELS_DIR, "diabetes_model.pkl")
    joblib.dump(best_model, model_path)

    # Persist background sample
    bg_sample = X_train.iloc[:200].reset_index(drop=True)
    sample_path = os.path.join(MODELS_DIR, "diabetes_train_sample.pkl")
    joblib.dump(bg_sample, sample_path)

    # Precompute global SHAP importance
    print("  Precomputing global SHAP importance...")
    global_shap_imp = _compute_global_shap(best_model, bg_sample)

    # Feature distribution statistics
    feature_stats = _compute_feature_statistics(X_train)

    metadata = {
        "disease": "diabetes",
        "model_name": best_name,
        "feature_names": feature_names,
        "target_column": "Outcome",
        "label_map": {"0": "Non-Diabetic", "1": "Diabetic"},
        "cv_results": cv_results,
        "holdout_test_metrics": holdout_metrics,
        "feature_statistics": feature_stats,
        "global_shap_importance": global_shap_imp,
        "model_path": "models/diabetes_model.pkl",
        "train_sample_path": "models/diabetes_train_sample.pkl",
    }
    meta_path = os.path.join(MODELS_DIR, "diabetes_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [OK] Saved artifacts for diabetes to {MODELS_DIR}")
    return best_name


# ─────────────────────────────────────────────
# ANEMIA
# ─────────────────────────────────────────────
def train_anemia():
    print("\n" + "=" * 70)
    print("  TRAINING ANEMIA MODEL (RESEARCH SPECIFICATION)")
    print("=" * 70)

    csv_path = os.path.join(BASE_DIR, "anemia Dataset.csv")
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["Result"])
    y = df["Result"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=42, eval_metric="logloss",
        ),
    }

    best_name, best_model, cv_results, holdout_metrics = _select_train_evaluate(
        models, X_train, y_train, X_test, y_test
    )

    # Persist model
    model_path = os.path.join(MODELS_DIR, "anemia_model.pkl")
    joblib.dump(best_model, model_path)

    # Persist background sample
    bg_sample = X_train.iloc[:200].reset_index(drop=True)
    sample_path = os.path.join(MODELS_DIR, "anemia_train_sample.pkl")
    joblib.dump(bg_sample, sample_path)

    # Precompute global SHAP importance
    print("  Precomputing global SHAP importance...")
    global_shap_imp = _compute_global_shap(best_model, bg_sample)

    # Feature distribution statistics
    feature_stats = _compute_feature_statistics(X_train)

    metadata = {
        "disease": "anemia",
        "model_name": best_name,
        "feature_names": feature_names,
        "target_column": "Result",
        "label_map": {"0": "No Anemia", "1": "Anemia"},
        "cv_results": cv_results,
        "holdout_test_metrics": holdout_metrics,
        "feature_statistics": feature_stats,
        "global_shap_importance": global_shap_imp,
        "model_path": "models/anemia_model.pkl",
        "train_sample_path": "models/anemia_train_sample.pkl",
    }
    meta_path = os.path.join(MODELS_DIR, "anemia_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [OK] Saved artifacts for anemia to {MODELS_DIR}")
    return best_name


if __name__ == "__main__":
    d_champ = train_diabetes()
    a_champ = train_anemia()
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE (ALL ARTIFACTS PERSISTED)")
    print(f"  Diabetes Champion: {d_champ}")
    print(f"  Anemia Champion:   {a_champ}")
    print("=" * 70)
