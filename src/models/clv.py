"""
PHASE 5 — Customer Lifetime Value (CLV) Prediction

Target Definition:
  Predicted 12-month spend (BRL) per customer, starting from snapshot date.

Observation Period: Customer's full history before snapshot date.
Prediction Horizon: 12 months after snapshot date.
Leakage Prevention: Target computed only from orders AFTER snapshot date.
Split: Temporal (by first purchase date).

Baseline: Mean CLV (predict mean for all customers).
Candidates: Ridge Regression, Random Forest Regressor, XGBoost Regressor, LightGBM Regressor.
Evaluation: MAE, RMSE, R².

DATASET LIMITATION:
  Olist has ~97% single-purchase customers. Most customers will have CLV = 0
  (no repeat purchase). This is a real business characteristic.
  The model will reflect this distribution honestly.
  CLV predictions should be interpreted as "expected spend if repeat purchase occurs."
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.models.base import BaseMLModel


CLV_HORIZON_DAYS = 365  # 12 months


class CLVModel(BaseMLModel):
    """
    Customer Lifetime Value regression model.
    Predicts 12-month spend for each customer.
    """

    target_name = "clv_12m_brl"
    task_type = "regression"
    observation_period_description = "Customer history before snapshot_date"
    prediction_horizon_description = "12 months (365 days) after snapshot_date"

    def build_features(self) -> tuple[pd.DataFrame, pd.Series, dict]:
        """Build CLV features and target."""
        fact_orders = self.load_processed("fact_orders")
        customer_360 = self.load_processed("customer_360")

        fact_orders["purchase_ts"] = pd.to_datetime(
            fact_orders["order_purchase_timestamp"], errors="coerce"
        )
        delivered = fact_orders[
            (fact_orders["order_status"] == "delivered") &
            fact_orders["purchase_ts"].notna() &
            fact_orders["customer_unique_id"].notna()
        ].copy()

        dataset_end = delivered["purchase_ts"].max()
        snapshot_date = dataset_end - pd.Timedelta(days=CLV_HORIZON_DAYS)

        logger.info(f"CLV snapshot: {snapshot_date.date()} | Horizon: {CLV_HORIZON_DAYS} days")

        # ── Features: history before snapshot ──────────────────────────
        obs = delivered[delivered["purchase_ts"] < snapshot_date]

        # ── Target: spend in 12 months after snapshot ──────────────────
        target_window = delivered[
            (delivered["purchase_ts"] >= snapshot_date) &
            (delivered["purchase_ts"] <= dataset_end)
        ]
        future_spend = (
            target_window.groupby("customer_unique_id")["total_payment_brl"]
            .sum()
            .reset_index()
            .rename(columns={"total_payment_brl": "clv_12m_brl"})
        )

        # ── Aggregate observation features ─────────────────────────────
        obs_agg = (
            obs.groupby("customer_unique_id")
            .agg(
                recency_days=(
                    "purchase_ts",
                    lambda x: (snapshot_date - x.max()).days,
                ),
                frequency=("order_id", "count"),
                monetary_brl=("total_payment_brl", "sum"),
                avg_order_value=("total_payment_brl", "mean"),
                max_order_value=("total_payment_brl", "max"),
                min_order_value=("total_payment_brl", "min"),
                std_order_value=("total_payment_brl", "std"),
                avg_review_score=("review_score", "mean"),
                avg_delivery_days=("delivery_days", "mean"),
                avg_installments=("payment_installments", "mean"),
                late_delivery_count=("is_late_delivery", "sum"),
                customer_age_days=(
                    "purchase_ts",
                    lambda x: (x.max() - x.min()).days,
                ),
            )
            .reset_index()
        )

        obs_agg["late_delivery_rate"] = (
            obs_agg["late_delivery_count"] / obs_agg["frequency"]
        ).fillna(0)
        obs_agg["avg_inter_purchase_days"] = (
            obs_agg["customer_age_days"] / obs_agg["frequency"]
        ).fillna(obs_agg["recency_days"])
        obs_agg["std_order_value"] = obs_agg["std_order_value"].fillna(0)

        # Merge 360 categorical features
        c360_cat = ["customer_unique_id", "customer_state", "preferred_payment_type",
                    "top_category", "unique_categories", "unique_sellers", "pct_credit_card"]
        available = [c for c in c360_cat if c in customer_360.columns]
        feature_df = obs_agg.merge(customer_360[available], on="customer_unique_id", how="left")

        # Merge CLV target (fill 0 for customers with no future spend)
        feature_df = feature_df.merge(future_spend, on="customer_unique_id", how="left")
        feature_df["clv_12m_brl"] = feature_df["clv_12m_brl"].fillna(0)

        # Only include customers with observation history
        feature_df = feature_df[feature_df["frequency"] >= 1]

        clv_mean = feature_df["clv_12m_brl"].mean()
        clv_median = feature_df["clv_12m_brl"].median()
        pct_zero = (feature_df["clv_12m_brl"] == 0).mean()
        logger.info(f"CLV distribution: mean=R${clv_mean:.2f} | median=R${clv_median:.2f} | {pct_zero:.1%} zero")
        logger.warning(
            f"NOTE: {pct_zero:.1%} of customers have CLV=0 (no repeat purchase). "
            "This reflects Olist's real single-purchase customer characteristic."
        )

        y = feature_df["clv_12m_brl"]
        X = feature_df.drop(columns=["customer_unique_id", "clv_12m_brl"])

        metadata = {
            "n_rows": len(X),
            "n_features": len(X.columns),
            "clv_mean_brl": round(clv_mean, 2),
            "clv_median_brl": round(clv_median, 2),
            "pct_zero_clv": round(pct_zero, 4),
            "snapshot_date": str(snapshot_date.date()),
            "limitation": "~97% single-purchase rate makes CLV near-zero for most customers.",
            "sort_index": True,
        }

        feature_df = feature_df.sort_values("recency_days")
        X = X.loc[feature_df.index]
        y = y.loc[feature_df.index]

        return X, y, metadata

    def get_models(self, feature_cols: list[str]) -> dict[str, Any]:
        numeric_cols = [c for c in [
            "recency_days", "frequency", "monetary_brl", "avg_order_value",
            "max_order_value", "min_order_value", "std_order_value",
            "avg_review_score", "avg_delivery_days", "avg_installments",
            "late_delivery_rate", "customer_age_days", "avg_inter_purchase_days",
            "unique_categories", "unique_sellers", "pct_credit_card",
        ] if c in feature_cols]
        categorical_cols = [c for c in ["customer_state", "preferred_payment_type", "top_category"] if c in feature_cols]

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
            "Baseline_MeanPredictor": Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", DummyRegressor(strategy="mean")),
            ]),
            "Ridge_Regression": Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", Ridge(alpha=10.0)),
            ]),
            "Random_Forest": Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(
                    n_estimators=200, max_depth=8, min_samples_leaf=20,
                    random_state=42, n_jobs=-1,
                )),
            ]),
            "XGBoost": Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", XGBRegressor(
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=0,
                )),
            ]),
            "LightGBM": Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", LGBMRegressor(
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    num_leaves=31, random_state=42, verbose=-1,
                )),
            ]),
        }


if __name__ == "__main__":
    model = CLVModel()
    results = model.train_and_evaluate(run_mlflow=False)
    print(f"\nBest model: {results['best_model']}")
    print(f"Best metrics: {results['best_metrics']}")
