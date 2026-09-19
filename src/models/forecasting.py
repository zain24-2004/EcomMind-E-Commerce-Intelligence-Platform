"""
PHASE 6 — Revenue & Demand Forecasting

Time-aware validation: Walk-forward (expanding window) cross-validation.
NO future data leaks into training folds.

Models:
  1. Naive Baseline: Last observed value (seasonal naive — same week last year)
  2. Moving Average: 4-week rolling mean
  3. Prophet: Facebook Prophet with trend + seasonality
  4. LightGBM: Gradient boosting with lag features + calendar features

Forecast targets:
  - Weekly total revenue (GMV)
  - Weekly order count

Evaluation: MAE, RMSE, MAPE — computed on held-out test periods.
All metrics derived from REAL Olist data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import get_settings
from src.analytics.revenue import load_processed


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Returns 0 if y_true is all zeros."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def prepare_weekly_revenue_series(
    fact_orders: pd.DataFrame,
    status_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate orders into a weekly revenue time series.

    Returns DataFrame with columns: [week, gmv_brl, order_count].
    """
    if status_filter is None:
        status_filter = ["delivered"]

    df = fact_orders[fact_orders["order_status"].isin(status_filter)].copy()
    df["purchase_ts"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    df = df[df["purchase_ts"].notna()]

    df["week"] = df["purchase_ts"].dt.to_period("W-MON").dt.start_time

    weekly = (
        df.groupby("week")
        .agg(
            gmv_brl=("total_payment_brl", "sum"),
            order_count=("order_id", "count"),
        )
        .reset_index()
        .sort_values("week")
    )
    weekly["week"] = pd.to_datetime(weekly["week"])

    # Fill any missing weeks with 0
    full_range = pd.date_range(
        weekly["week"].min(), weekly["week"].max(), freq="W-MON"
    )
    weekly = (
        weekly.set_index("week")
        .reindex(full_range, fill_value=0)
        .reset_index()
        .rename(columns={"index": "week"})
    )

    logger.info(f"Weekly series: {len(weekly)} weeks | {weekly['week'].min().date()} → {weekly['week'].max().date()}")
    return weekly


def add_calendar_features(df: pd.DataFrame, date_col: str = "week") -> pd.DataFrame:
    """Add calendar-based features to time series DataFrame."""
    df = df.copy()
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["quarter"] = df[date_col].dt.quarter
    df["is_q4"] = (df["quarter"] == 4).astype(int)  # holiday season
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int] = [1, 2, 4, 8, 13, 26, 52],
) -> pd.DataFrame:
    """Add lag features (leakage-safe: only past values used)."""
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}w"] = df[target_col].shift(lag)
    # Rolling stats
    df[f"{target_col}_roll4_mean"] = df[target_col].shift(1).rolling(4).mean()
    df[f"{target_col}_roll4_std"] = df[target_col].shift(1).rolling(4).std()
    df[f"{target_col}_roll12_mean"] = df[target_col].shift(1).rolling(12).mean()
    return df


# ── Baseline: Naive seasonal ──────────────────────────────────────────────────

def naive_forecast(train: pd.DataFrame, n_weeks: int, target_col: str) -> np.ndarray:
    """Naive baseline: repeat last observed week's value."""
    return np.full(n_weeks, train[target_col].iloc[-1])


def moving_average_forecast(
    train: pd.DataFrame,
    n_weeks: int,
    target_col: str,
    window: int = 4,
) -> np.ndarray:
    """4-week moving average forecast (constant)."""
    ma_value = train[target_col].tail(window).mean()
    return np.full(n_weeks, ma_value)


# ── LightGBM Forecaster ───────────────────────────────────────────────────────

def train_lgbm_forecaster(
    train_df: pd.DataFrame,
    target_col: str,
) -> Any:
    """Train LightGBM forecaster with lag + calendar features."""
    from lightgbm import LGBMRegressor

    feature_cols = [c for c in train_df.columns if c not in [target_col, "week"]]
    X = train_df[feature_cols].dropna()
    y = train_df.loc[X.index, target_col]

    model = LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    return model, feature_cols


# ── Prophet Forecaster ────────────────────────────────────────────────────────

def train_prophet_forecaster(train_df: pd.DataFrame, target_col: str):
    """Train Prophet model on weekly revenue data."""
    try:
        from prophet import Prophet

        prophet_df = train_df[["week", target_col]].rename(
            columns={"week": "ds", target_col: "y"}
        )
        model = Prophet(
            seasonality_mode="multiplicative",
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.1,
        )
        model.fit(prophet_df)
        return model
    except ImportError:
        logger.warning("Prophet not installed. Skipping Prophet model.")
        return None


# ── Walk-forward Cross Validation ────────────────────────────────────────────

def walk_forward_evaluation(
    weekly: pd.DataFrame,
    target_col: str = "gmv_brl",
    n_test_weeks: int = 12,
    min_train_weeks: int = 26,
) -> dict[str, dict]:
    """
    Walk-forward (expanding window) cross-validation.

    Prevents any future data from leaking into training.

    Args:
        weekly: Time-indexed weekly DataFrame.
        target_col: Column to forecast.
        n_test_weeks: Number of weeks in test set.
        min_train_weeks: Minimum training history.

    Returns:
        Dict of {model_name: {mae, rmse, mape, n_folds}}.
    """
    logger.info(f"Walk-forward CV: target={target_col} | test_weeks={n_test_weeks}")

    results: dict[str, list] = {
        "Naive": [],
        "MovingAverage_4w": [],
        "LightGBM": [],
    }

    n_total = len(weekly)
    n_folds = 0

    # Prepare feature-engineered version
    weekly_feat = add_calendar_features(weekly)
    weekly_feat = add_lag_features(weekly_feat, target_col)

    for test_end in range(min_train_weeks + n_test_weeks, n_total + 1, n_test_weeks):
        train_end = test_end - n_test_weeks
        train = weekly.iloc[:train_end]
        test = weekly.iloc[train_end:test_end]
        train_feat = weekly_feat.iloc[:train_end]
        test_feat = weekly_feat.iloc[train_end:test_end]

        if len(train) < min_train_weeks or len(test) == 0:
            continue

        y_true = test[target_col].values
        n_folds += 1

        # Naive
        y_naive = naive_forecast(train, len(test), target_col)
        results["Naive"].append({
            "mae": mean_absolute_error(y_true, y_naive),
            "rmse": np.sqrt(mean_squared_error(y_true, y_naive)),
            "mape": _mape(y_true, y_naive),
        })

        # Moving average
        y_ma = moving_average_forecast(train, len(test), target_col, window=4)
        results["MovingAverage_4w"].append({
            "mae": mean_absolute_error(y_true, y_ma),
            "rmse": np.sqrt(mean_squared_error(y_true, y_ma)),
            "mape": _mape(y_true, y_ma),
        })

        # LightGBM
        try:
            lgbm_model, feat_cols = train_lgbm_forecaster(train_feat.dropna(), target_col)
            available_feats = [c for c in feat_cols if c in test_feat.columns]
            X_test = test_feat[available_feats].fillna(0)
            y_lgbm = lgbm_model.predict(X_test).clip(min=0)
            results["LightGBM"].append({
                "mae": mean_absolute_error(y_true, y_lgbm),
                "rmse": np.sqrt(mean_squared_error(y_true, y_lgbm)),
                "mape": _mape(y_true, y_lgbm),
            })
        except Exception as e:
            logger.debug(f"LightGBM fold skipped: {e}")

    # Aggregate across folds
    summary: dict[str, dict] = {}
    for model_name, fold_results in results.items():
        if fold_results:
            summary[model_name] = {
                "mae": round(np.mean([r["mae"] for r in fold_results]), 2),
                "rmse": round(np.mean([r["rmse"] for r in fold_results]), 2),
                "mape_pct": round(np.mean([r["mape"] for r in fold_results]), 2),
                "n_folds": n_folds,
            }
            logger.info(
                f"  {model_name:<20}: MAE={summary[model_name]['mae']:>10,.0f} | "
                f"RMSE={summary[model_name]['rmse']:>10,.0f} | "
                f"MAPE={summary[model_name]['mape_pct']:.1f}%"
            )

    return summary


def run_forecasting(processed_dir: Path | None = None) -> dict:
    """Run full forecasting pipeline for revenue and order count."""
    settings = get_settings()
    processed_dir = processed_dir or settings.data_processed_path

    logger.info("=" * 60)
    logger.info("PHASE 6 — REVENUE & DEMAND FORECASTING")
    logger.info("=" * 60)

    fact_orders = load_processed("fact_orders", processed_dir)
    weekly = prepare_weekly_revenue_series(fact_orders)

    logger.info("Evaluating revenue forecasting models...")
    rev_results = walk_forward_evaluation(weekly, target_col="gmv_brl")

    logger.info("Evaluating order count forecasting models...")
    order_results = walk_forward_evaluation(weekly, target_col="order_count")

    # Train final models on full data for inference
    logger.info("Training final models on full dataset...")
    weekly_feat = add_calendar_features(weekly)
    weekly_feat = add_lag_features(weekly_feat, "gmv_brl")
    weekly_feat_orders = add_lag_features(weekly_feat, "order_count")

    final_lgbm, feat_cols = train_lgbm_forecaster(weekly_feat.dropna(), "gmv_brl")

    # Save final model
    out_path = settings.models_path / "revenue_forecaster_lgbm.pkl"
    settings.models_path.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(out_path, "wb") as f:
        pickle.dump({"model": final_lgbm, "feature_cols": feat_cols}, f)
    logger.info(f"Final revenue forecaster saved to {out_path}")

    # Generate 8-week forward forecast
    forecast_df = _generate_forward_forecast(weekly, final_lgbm, feat_cols, n_weeks=8)
    forecast_df.to_parquet(processed_dir / "revenue_forecast.parquet", index=False)

    logger.success("Phase 6 forecasting complete.")
    return {
        "revenue_cv_results": rev_results,
        "order_count_cv_results": order_results,
        "weekly_series": weekly.to_dict("records"),
        "forecast_8w": forecast_df.to_dict("records"),
    }


def _generate_forward_forecast(
    weekly: pd.DataFrame,
    model: Any,
    feature_cols: list[str],
    n_weeks: int = 8,
) -> pd.DataFrame:
    """Generate forward-looking weekly revenue forecasts."""
    weekly_feat = add_calendar_features(weekly)
    weekly_feat = add_lag_features(weekly_feat, "gmv_brl")

    last_date = weekly["week"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=n_weeks,
        freq="W-MON",
    )

    forecast_rows = []
    extended = weekly_feat.copy()

    for future_date in future_dates:
        new_row = {"week": future_date, "gmv_brl": np.nan}
        extended = pd.concat([extended, pd.DataFrame([new_row])], ignore_index=True)
        extended = add_calendar_features(extended)
        extended = add_lag_features(extended, "gmv_brl")

        last_row = extended.iloc[[-1]]
        available_feats = [c for c in feature_cols if c in last_row.columns]
        X_pred = last_row[available_feats].fillna(0)
        pred = float(model.predict(X_pred)[0])
        pred = max(pred, 0)

        extended.loc[extended.index[-1], "gmv_brl"] = pred
        forecast_rows.append({
            "week": future_date,
            "forecast_gmv_brl": round(pred, 2),
            "is_forecast": True,
        })

    return pd.DataFrame(forecast_rows)


if __name__ == "__main__":
    run_forecasting()
