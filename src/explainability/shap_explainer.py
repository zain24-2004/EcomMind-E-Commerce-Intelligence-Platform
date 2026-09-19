"""
PHASE 10 — SHAP Explainability
Model explanation using SHAP (SHapley Additive exPlanations).

Provides:
  - Global feature importance (mean |SHAP|)
  - Local explanations (individual predictions)
  - Waterfall plots, summary plots, force plots
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger


def get_shap_explainer(model_pipeline: Any, X_train: pd.DataFrame):
    """
    Create the appropriate SHAP explainer for a model pipeline.

    Tries TreeExplainer first (fastest, for tree-based models),
    then falls back to KernelExplainer.
    """
    import shap

    # Extract the final estimator from a sklearn Pipeline
    if hasattr(model_pipeline, "named_steps"):
        estimator = model_pipeline.named_steps.get(
            "classifier",
            model_pipeline.named_steps.get("regressor", None),
        )
        preprocessor = model_pipeline.named_steps.get("preprocessor")
    else:
        estimator = model_pipeline
        preprocessor = None

    # Transform X_train through preprocessor if it exists
    if preprocessor is not None:
        X_transformed = preprocessor.transform(X_train)
    else:
        X_transformed = X_train.values if hasattr(X_train, "values") else X_train

    try:
        explainer = shap.TreeExplainer(estimator)
        logger.info("Using SHAP TreeExplainer (fast)")
        return explainer, X_transformed
    except Exception:
        logger.info("TreeExplainer not available — falling back to KernelExplainer (slow)")
        background = shap.sample(X_transformed, min(100, len(X_transformed)))
        explainer = shap.KernelExplainer(
            estimator.predict_proba if hasattr(estimator, "predict_proba")
            else estimator.predict,
            background,
        )
        return explainer, X_transformed


def compute_shap_values(
    model_pipeline: Any,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> dict[str, Any]:
    """
    Compute SHAP values for a trained model pipeline.

    Args:
        model_pipeline: Trained sklearn Pipeline.
        X: Feature DataFrame (pre-split, for explanation).
        max_samples: Maximum rows to explain (for speed).

    Returns:
        dict with shap_values, feature_names, base_value, X_transformed.
    """
    import shap

    X_sample = X.sample(min(max_samples, len(X)), random_state=42)
    explainer, X_transformed = get_shap_explainer(model_pipeline, X_sample)

    shap_values = explainer.shap_values(X_transformed)

    # Get feature names after preprocessing
    if hasattr(model_pipeline, "named_steps") and "preprocessor" in model_pipeline.named_steps:
        preprocessor = model_pipeline.named_steps["preprocessor"]
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
    else:
        feature_names = list(X.columns)

    # For binary classification, SHAP returns list of 2 arrays; use class 1
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    base_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else float(explainer.expected_value)
    )

    logger.info(f"SHAP computed: {X_transformed.shape[0]} samples × {X_transformed.shape[1]} features")

    return {
        "shap_values": shap_values,
        "feature_names": list(feature_names),
        "base_value": float(base_value),
        "X_transformed": X_transformed,
    }


def get_global_feature_importance(shap_result: dict[str, Any]) -> pd.DataFrame:
    """
    Compute global feature importance as mean |SHAP value|.

    Returns DataFrame: [feature_name, mean_abs_shap] sorted descending.
    """
    shap_vals = np.array(shap_result["shap_values"])
    feature_names = shap_result["feature_names"]

    mean_abs = np.abs(shap_vals).mean(axis=0)

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    logger.info(f"Top 5 features: {importance['feature'].head(5).tolist()}")
    return importance


def get_local_explanation(
    shap_result: dict[str, Any],
    sample_idx: int,
) -> dict[str, Any]:
    """
    Get local SHAP explanation for a single prediction.

    Args:
        shap_result: Output from compute_shap_values().
        sample_idx: Row index in the explained sample.

    Returns:
        dict with feature contributions for that sample.
    """
    shap_vals = shap_result["shap_values"][sample_idx]
    feature_names = shap_result["feature_names"]
    base_value = shap_result["base_value"]

    contributions = sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return {
        "base_value": base_value,
        "prediction": float(base_value + sum(shap_vals)),
        "contributions": [
            {"feature": name, "shap_value": round(float(val), 6)}
            for name, val in contributions[:20]  # top 20
        ],
    }


def save_shap_summary_plot(
    shap_result: dict[str, Any],
    output_path: Path,
    plot_type: str = "bar",
) -> None:
    """Save SHAP summary plot to file."""
    try:
        import shap
        import matplotlib.pyplot as plt

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))

        shap.summary_plot(
            shap_result["shap_values"],
            features=shap_result["X_transformed"],
            feature_names=shap_result["feature_names"],
            plot_type=plot_type,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP plot saved: {output_path}")
    except Exception as e:
        logger.warning(f"Could not save SHAP plot: {e}")
