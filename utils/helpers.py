"""
utils/helpers.py
----------------
Shared utility functions used across the CDSS pipeline.
Provides deterministic query generation, formatting, and safe accessors.
"""

from __future__ import annotations


def build_clinical_query(
    disease: str,
    prediction: str,
    top_features: list[dict],
    n: int = 4,
) -> str:
    """
    Build a clinical retrieval query from top SHAP feature contributions.
    Features sorted by descending absolute impact are formatted into a
    semantically coherent question suited for vector similarity search.
    """
    top_n = top_features[:n]
    feature_names = [f["feature"] for f in top_n]

    if not feature_names:
        return (
            f"Clinical diagnosis criteria and biomarker significance in {disease} "
            f"assessment and risk stratification."
        )

    feature_str = ", ".join(feature_names)
    query = (
        f"Clinical significance of {feature_str} "
        f"in relation to {disease} risk prediction and clinical interpretation."
    )
    return query


def format_probability_pct(prob: float) -> str:
    """Format a 0–1 probability as a percentage string (e.g. 87.3%)."""
    return f"{prob * 100:.1f}%"


def direction_label(shap_val: float) -> str:
    """Return a human-readable direction label for a SHAP value."""
    if shap_val > 0:
        return "Increases risk"
    elif shap_val < 0:
        return "Decreases risk"
    else:
        return "Neutral"


def safe_get(d: dict, key: str, default=None):
    """Safe dictionary accessor with fallback default."""
    return d.get(key, default)


def get_base_dir() -> str:
    """Return the project root directory as an absolute path string."""
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def truncate_text(text: str, max_len: int = 80, suffix: str = "...") -> str:
    """Truncate long strings for display, preserving word boundaries."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)].rstrip() + suffix
