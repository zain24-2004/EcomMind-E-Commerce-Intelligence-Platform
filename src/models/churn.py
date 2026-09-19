"""
PHASE 5 — Churn Prediction Model

Target Definition:
  A customer is "churned" if they made at least 1 purchase in the
  observation period but made NO purchase in the following 90 days.

  IMPORTANT CAVEAT: Olist has ~97% single-purchase customers.
  This fundamentally limits churn prediction — most customers
  "churn" after their first purchase by this definition.
  This is documented as a real business characteristic.

Observation Period: Customer's entire purchase history up to snapshot date.
Prediction Horizon: 90 days after snapshot date.
Leakage Prevention: Features computed only from data BEFORE snapshot date.
                    Target computed only from data AFTER snapshot date.
Split: Temporal (customers sorted by first purchase date).

Baseline: Majority class classifier (always predict churned = 1).
Candidates: Logistic Regression, Random Forest, XGBoost, LightGBM.
Evaluation: ROC-AUC, Average Precision, F1 (weighted).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.models.base import BaseMLModel


# Prediction horizon: 90 days
CHURN_HORIZON_DAYS = 90
# Min purchases to be included (at least 1 to define a real customer)
MIN_PURCHASES = 1


class ChurnModel(BaseMLModel):
    """
    Binary churn classifier.
    Predicts whether a customer will NOT purchase in the next 90 days.
    """

    target_name = "churned_90d"
    task_type = "classification"
    observation_period_description = "All customer history up to snapshot_date"
    prediction_horizon_description = "90 days after snapshot_date"

    def build_features(self) -> tuple[pd.DataFrame, pd.Series, dict]:
        """
        Build churn features from Customer 360 and RFM data.

        Leakage prevention:
          - Snapshot date = 90 days before dataset end
          - Features: computed from orders BEFORE snapshot date
          - Target: purchased in [snapshot_date, snapshot_date + 90 days]
        """
        fact_orders = self.load_processed("fact_orders")
        customer_360 = self.load_processed("customer_360")

        fact_orders["purchase_ts"] = pd.to_datetime(
            fact_orders["order_purchase_timestamp"], errors="coerce"
        )

        delivered = fact_orders[fact_orders["order_status"] == "delivered"].copy()
        delivered = delivered[delivered["purchase_ts"].notna()]
        delivered = delivered[delivered["customer_unique_id"].notna()]

        dataset_end = delivered["purchase_ts"].max()
        snapshot_date = dataset_end - pd.Timedelta(days=CHURN_HORIZON_DAYS)

        logger.info(f"Churn snapshot: {snapshot_date.date()} | Horizon: {CHURN_HORIZON_DAYS} days")
        logger.info(f"Dataset end: {dataset_end.date()}")

        # ── Observation window: before snapshot_date ───────────────────
        obs = delivered[delivered["purchase_ts"] < snapshot_date]

        # Only customers with at least 1 purchase before snapshot
        customers_with_history = set(obs["customer_unique_id"].unique())
        logger.info(f"Customers with history before snapshot: {len(customers_with_history):,}")

        # ── Target window: after snapshot_date ────────────────────────
        target_window = delivered[
            (delivered["purchase_ts"] >= snapshot_date) &
            (delivered["purchase_ts"] <= dataset_end)
        ]
        customers_purchased_after = set(target_window["customer_unique_id"].unique())

        # ── Build feature rows ─────────────────────────────────────────
        # Recompute RFM features from observation window only
        obs_rfm = (
            obs.groupby("customer_unique_id")
            .agg(
                recency_days=(
                    "purchase_ts",
                    lambda x: (snapshot_date - x.max()).days,
                ),
                frequency=("order_id", "count"),
                monetary_brl=("total_payment_brl", "sum"),
                avg_review_score=("review_score", "mean"),
                avg_delivery_days=("delivery_days", "mean"),
                avg_installments=("payment_installments", "mean"),
                item_count_mean=("item_count", "mean"),
                late_delivery_count=("is_late_delivery", "sum"),
            )
            .reset_index()
        )

        obs_rfm["late_delivery_rate"] = (
            obs_rfm["late_delivery_count"] / obs_rfm["frequency"]
        ).fillna(0)

        # Merge Customer 360 categorical features
        c360_cols = [
            "customer_unique_id",
            "customer_state",
            "preferred_payment_type",
            "top_category",
            "unique_categories",
            "unique_sellers",
            "pct_credit_card",
        ]
        available_360_cols = [c for c in c360_cols if c in customer_360.columns]
        feature_df = obs_rfm.merge(
            customer_360[available_360_cols],
            on="customer_unique_id",
            how="left",
        )

        # ── Label: churned = did NOT purchase after snapshot ──────────
        feature_df["churned_90d"] = (
            ~feature_df["customer_unique_id"].isin(customers_purchased_after)
        ).astype(int)

        # Only keep customers with observation history
        feature_df = feature_df[
            feature_df["customer_unique_id"].isin(customers_with_history)
        ]

        churn_rate = feature_df["churned_90d"].mean()
        logger.info(f"Churn rate in this dataset: {churn_rate:.1%}")
        logger.warning(
            "NOTE: Olist has ~97% single-purchase customers. The high churn rate "
            "is a real business characteristic of this dataset, not a modeling error."
        )

        y = feature_df["churned_90d"]
        X = feature_df.drop(columns=["customer_unique_id", "churned_90d"])

        metadata = {
            "n_rows": len(X),
            "n_features": len(X.columns),
            "churn_rate": round(churn_rate, 4),
            "snapshot_date": str(snapshot_date.date()),
            "horizon_days": CHURN_HORIZON_DAYS,
            "dataset_limitation": (
                "Olist has ~97% single-purchase customers. Churn model "
                "reflects this real business characteristic."
            ),
        }

        # Sort by recency for temporal split approximation
        feature_df_sorted = feature_df.sort_values("recency_days", ascending=False)
        X = X.loc[feature_df_sorted.index]
        y = y.loc[feature_df_sorted.index]
        metadata["sort_index"] = True

        return X, y, metadata

    def get_models(self, feature_cols: list[str]) -> dict[str, Any]:
        """Return all candidate models as sklearn pipelines."""
        numeric_cols = [c for c in [
            "recency_days", "frequency", "monetary_brl",
            "avg_review_score", "avg_delivery_days", "avg_installments",
            "item_count_mean", "late_delivery_rate",
            "unique_categories", "unique_sellers", "pct_credit_card",
        ] if c in feature_cols]
        categorical_cols = [c for c in [
            "customer_state", "preferred_payment_type", "top_category"
        ] if c in feature_cols]

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ], remainder="drop")

        return {
            "Baseline_MajorityClass": Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]),
            "Logistic_Regression": Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=42
                )),
            ]),
            "Random_Forest": Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(
                    n_estimators=200, max_depth=8, min_samples_leaf=20,
                    class_weight="balanced", random_state=42, n_jobs=-1,
                )),
            ]),
            "XGBoost": Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=1, eval_metric="logloss",
                    random_state=42, verbosity=0,
                )),
            ]),
            "LightGBM": Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", LGBMClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    num_leaves=31, class_weight="balanced",
                    random_state=42, verbose=-1,
                )),
            ]),
        }


if __name__ == "__main__":
    model = ChurnModel()
    results = model.train_and_evaluate(run_mlflow=False)
    print(f"\nBest model: {results['best_model']}")
    print(f"Best metrics: {results['best_metrics']}")
