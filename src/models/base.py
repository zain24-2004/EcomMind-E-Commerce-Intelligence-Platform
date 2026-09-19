"""
PHASE 5 — Base ML Model
Shared base class for all predictive models.

Enforces:
  - Consistent train/evaluate/explain interface
  - MLflow experiment tracking
  - Model persistence
  - Leakage-safe train/test splitting
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split

from src.config import get_settings


class BaseMLModel(ABC):
    """
    Abstract base class for all platform ML models.

    Each subclass must implement:
      - build_features(): Returns (X, y, metadata)
      - get_models(): Returns dict of {name: sklearn_pipeline}
      - target_name: str
      - task_type: 'classification' or 'regression'
    """

    target_name: str = ""
    task_type: str = "classification"  # or "regression"
    observation_period_description: str = ""
    prediction_horizon_description: str = ""

    def __init__(self, processed_dir: Optional[Path] = None):
        settings = get_settings()
        self.processed_dir = processed_dir or settings.data_processed_path
        self.models_dir = settings.models_path
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.trained_models: dict[str, Any] = {}
        self.evaluation_results: dict[str, dict] = {}

    @abstractmethod
    def build_features(self) -> tuple[pd.DataFrame, pd.Series, dict]:
        """
        Build the feature matrix X and target y.

        Returns:
            X: Feature DataFrame
            y: Target Series
            metadata: dict with dataset info (n_rows, feature_names, class_distribution, etc.)
        """
        ...

    @abstractmethod
    def get_models(self, feature_cols: list[str]) -> dict[str, Any]:
        """Return dict of {model_name: sklearn_compatible_pipeline}."""
        ...

    def load_processed(self, name: str) -> pd.DataFrame:
        """Load a processed Parquet file."""
        path = self.processed_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.data.etl` first."
            )
        return pd.read_parquet(path)

    def train_and_evaluate(
        self,
        test_size: float = 0.20,
        random_state: int = 42,
        run_mlflow: bool = True,
    ) -> dict[str, dict]:
        """
        Full training and evaluation loop.

        Steps:
          1. Build features
          2. Chronological train/test split (no data leakage)
          3. Train baseline
          4. Train all candidate models
          5. Evaluate each model
          6. Select best model
          7. Log to MLflow
          8. Save best model
        """
        logger.info(f"{'='*60}")
        logger.info(f"Training: {self.__class__.__name__}")
        logger.info(f"Target   : {self.target_name}")
        logger.info(f"Task     : {self.task_type}")
        logger.info(f"{'='*60}")

        X, y, metadata = self.build_features()

        logger.info(
            f"Dataset: {len(X):,} rows × {len(X.columns)} features | "
            f"Target distribution: {y.value_counts(normalize=True).round(3).to_dict() if self.task_type == 'classification' else f'mean={y.mean():.2f}'}"
        )

        # ── Leakage-safe split ────────────────────────────────────────────
        # For time-series-like customer data: use sort_by_date if available,
        # otherwise stratified random split with fixed seed.
        if "sort_index" in metadata:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            logger.info(f"Temporal split: train={len(X_train):,} | test={len(X_test):,}")
        else:
            stratify = y if self.task_type == "classification" else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
            logger.info(f"Random split: train={len(X_train):,} | test={len(X_test):,}")

        models = self.get_models(list(X.columns))

        # ── Train all models ──────────────────────────────────────────────
        results = {}
        for name, model in models.items():
            logger.info(f"  Training: {name}")
            model.fit(X_train, y_train)
            self.trained_models[name] = model

            metrics = self._evaluate_model(model, X_test, y_test, name)
            results[name] = metrics
            self.evaluation_results[name] = metrics

            logger.info(f"  {name}: {self._format_metrics(metrics)}")

        # ── Select best model ─────────────────────────────────────────────
        best_name = self._select_best_model(results)
        best_metrics = results[best_name]
        logger.info(f"Best model: {best_name} → {self._format_metrics(best_metrics)}")

        # ── Save best model ───────────────────────────────────────────────
        save_path = self._save_model(self.trained_models[best_name], best_name)

        # ── MLflow logging ────────────────────────────────────────────────
        if run_mlflow:
            self._log_to_mlflow(results, best_name, metadata, save_path)

        return {
            "all_results": results,
            "best_model": best_name,
            "best_metrics": best_metrics,
            "metadata": metadata,
            "save_path": str(save_path),
        }

    def _evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
    ) -> dict:
        """Compute evaluation metrics based on task type."""
        metrics: dict = {"model_name": model_name}

        if self.task_type == "classification":
            y_pred = model.predict(X_test)
            metrics["f1"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
            metrics["classification_report"] = classification_report(y_test, y_pred, zero_division=0)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                n_classes = len(np.unique(y_test))
                if n_classes == 2:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba[:, 1])), 4)
                    metrics["avg_precision"] = round(float(average_precision_score(y_test, y_proba[:, 1])), 4)
                else:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")), 4)

        elif self.task_type == "regression":
            y_pred = model.predict(X_test)
            metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
            metrics["r2"] = round(float(r2_score(y_test, y_pred)), 4)

        return metrics

    def _select_best_model(self, results: dict[str, dict]) -> str:
        """Select the best model based on task-appropriate metric."""
        if self.task_type == "classification":
            metric = "roc_auc" if "roc_auc" in next(iter(results.values())) else "f1"
        else:
            metric = "r2"

        return max(results.items(), key=lambda x: x[1].get(metric, 0))[0]

    def _format_metrics(self, metrics: dict) -> str:
        """Format metrics for display."""
        if self.task_type == "classification":
            parts = []
            if "roc_auc" in metrics:
                parts.append(f"ROC-AUC={metrics['roc_auc']}")
            if "avg_precision" in metrics:
                parts.append(f"AP={metrics['avg_precision']}")
            if "f1" in metrics:
                parts.append(f"F1={metrics['f1']}")
            return " | ".join(parts)
        else:
            return f"MAE={metrics.get('mae')} | RMSE={metrics.get('rmse')} | R²={metrics.get('r2')}"

    def _save_model(self, model: Any, model_name: str) -> Path:
        """Serialize and save model to disk."""
        save_path = self.models_dir / f"{self.__class__.__name__}_{model_name}.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f, protocol=5)
        logger.info(f"Model saved: {save_path}")
        return save_path

    def load_model(self, model_name: str) -> Any:
        """Load a previously saved model."""
        path = self.models_dir / f"{self.__class__.__name__}_{model_name}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def _log_to_mlflow(
        self,
        results: dict[str, dict],
        best_name: str,
        metadata: dict,
        save_path: Path,
    ) -> None:
        """Log experiment to MLflow."""
        try:
            import mlflow
            from src.config import get_settings
            settings = get_settings()
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            experiment = mlflow.set_experiment(settings.mlflow_experiment_name)

            for model_name, metrics in results.items():
                with mlflow.start_run(
                    run_name=f"{self.__class__.__name__}_{model_name}",
                    tags={
                        "model_class": self.__class__.__name__,
                        "target": self.target_name,
                        "task_type": self.task_type,
                        "is_best": str(model_name == best_name),
                    },
                ):
                    # Log params
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("n_train_rows", metadata.get("n_train", ""))
                    mlflow.log_param("n_features", metadata.get("n_features", ""))
                    mlflow.log_param("observation_period", self.observation_period_description)
                    mlflow.log_param("prediction_horizon", self.prediction_horizon_description)

                    # Log metrics (numeric only)
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)) and k != "model_name":
                            mlflow.log_metric(k, v)

                    # Log model artifact for best
                    if model_name == best_name:
                        mlflow.log_artifact(str(save_path))

            logger.info(f"MLflow: experiment '{settings.mlflow_experiment_name}' updated")

        except Exception as exc:
            logger.warning(f"MLflow logging failed (non-fatal): {exc}")
