"""
PHASE 8 — FastAPI Application
E-Commerce Intelligence Platform REST API.

Endpoints:
  GET  /health                          — Health check
  GET  /api/v1/analytics/revenue        — Revenue KPIs
  GET  /api/v1/analytics/rfm/{id}       — Customer RFM profile
  GET  /api/v1/analytics/forecast       — Revenue forecast
  GET  /api/v1/analytics/marketing      — Marketing KPIs (SYNTHETIC)
  POST /api/v1/predict/churn            — Churn probability prediction
  POST /api/v1/predict/clv              — CLV prediction
  GET  /api/v1/segments/summary         — RFM segment summary

All endpoints include proper error handling, logging, and response documentation.
"""
from __future__ import annotations

import pickle
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.config import get_settings
from src.api.schemas import (
    HealthResponse,
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    CLVPredictionRequest,
    CLVPredictionResponse,
    RevenueKPIResponse,
    RFMProfileResponse,
    ForecastResponse,
    MarketingKPIResponse,
)

# ── App State ──────────────────────────────────────────────────────────────────
_app_state: dict[str, Any] = {}

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data on startup, clean up on shutdown."""
    logger.info("Starting E-Commerce Intelligence API...")
    settings = get_settings()

    try:
        # Load processed data
        processed_dir = settings.data_processed_path
        _app_state["fact_orders"] = _safe_load_parquet(processed_dir / "fact_orders.parquet")
        _app_state["customer_rfm"] = _safe_load_parquet(processed_dir / "customer_rfm.parquet")
        _app_state["customer_360"] = _safe_load_parquet(processed_dir / "customer_360.parquet")
        _app_state["forecast"] = _safe_load_parquet(processed_dir / "revenue_forecast.parquet")

        # Load ML models
        models_dir = settings.models_path
        _app_state["churn_model"] = _safe_load_model(models_dir, "ChurnModel")
        _app_state["clv_model"] = _safe_load_model(models_dir, "CLVModel")

        # Load synthetic marketing
        synthetic_dir = settings.data_synthetic_path
        _app_state["synthetic_campaigns"] = _safe_load_parquet(
            synthetic_dir / "synthetic_campaigns.parquet"
        )

        logger.success("API startup complete.")
    except Exception as e:
        logger.error(f"Startup error: {e}")

    yield

    # Cleanup
    _app_state.clear()
    logger.info("API shutdown complete.")


def _safe_load_parquet(path: Path) -> pd.DataFrame | None:
    """Load a Parquet file, returning None if not found."""
    if path and path.exists():
        df = pd.read_parquet(path)
        logger.info(f"Loaded {path.name}: {len(df):,} rows")
        return df
    logger.warning(f"Data not found: {path}. Run the pipeline first.")
    return None


def _safe_load_model(models_dir: Path, model_class_name: str) -> Any | None:
    """Load the best saved model for a given model class."""
    if not models_dir.exists():
        return None
    model_files = list(models_dir.glob(f"{model_class_name}_*.pkl"))
    if not model_files:
        logger.warning(f"No saved model found for {model_class_name}")
        return None
    path = model_files[0]  # Take first match
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded model: {path.name}")
    return model


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="E-Commerce Intelligence Platform API",
    description=(
        "AI/ML-powered analytics and prediction API for e-commerce intelligence. "
        "Built on the Olist Brazilian E-Commerce dataset.\n\n"
        "**Data Note**: Revenue and customer analytics use real Olist data (2016-2018). "
        "Marketing analytics use clearly labeled SYNTHETIC data."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dependency ────────────────────────────────────────────────────────────────

def require_data(key: str):
    """Dependency factory that checks app state."""
    def _check():
        if _app_state.get(key) is None:
            raise HTTPException(
                status_code=503,
                detail=f"Data '{key}' not loaded. Run the ETL pipeline first.",
            )
        return _app_state[key]
    return _check


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint — returns service status."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        environment=settings.environment,
    )


@app.get("/api/v1/status", tags=["Health"])
async def system_status():
    """Returns what data and models are loaded."""
    return {
        "loaded_datasets": {k: (v is not None and len(v) > 0) for k, v in _app_state.items()},
        "version": APP_VERSION,
    }


# ── Analytics: Revenue ────────────────────────────────────────────────────────

@app.get(
    "/api/v1/analytics/revenue",
    response_model=RevenueKPIResponse,
    tags=["Analytics"],
    summary="Revenue KPIs from real Olist data",
)
async def get_revenue_kpis():
    """
    Compute revenue KPIs from real Olist data.

    Returns: Total GMV, order volume, AOV, cancellation rate, delivery metrics.
    All values are from real Olist e-commerce transactions.
    """
    fact_orders = _app_state.get("fact_orders")
    if fact_orders is None:
        raise HTTPException(503, "Data not loaded.")

    from src.analytics.revenue import (
        compute_total_gmv, compute_order_volume, compute_aov,
        compute_cancellation_metrics, compute_delivery_metrics,
    )

    delivered = fact_orders[fact_orders["order_status"] == "delivered"]
    ts_col = "order_purchase_timestamp"
    fact_orders[ts_col] = pd.to_datetime(fact_orders[ts_col], errors="coerce")

    return RevenueKPIResponse(
        total_gmv_brl=compute_total_gmv(fact_orders),
        order_volume=compute_order_volume(fact_orders),
        aov_brl=compute_aov(fact_orders),
        cancellation_rate_pct=compute_cancellation_metrics(fact_orders)["cancellation_rate_pct"],
        avg_delivery_days=compute_delivery_metrics(fact_orders)["avg_delivery_days"],
        on_time_rate_pct=compute_delivery_metrics(fact_orders).get("on_time_rate_pct"),
        data_period_start=str(fact_orders[ts_col].min().date()),
        data_period_end=str(fact_orders[ts_col].max().date()),
        limitation_note="COGS/profit not available in Olist dataset. Gross margin cannot be computed.",
    )


# ── Analytics: RFM ────────────────────────────────────────────────────────────

@app.get(
    "/api/v1/analytics/rfm/{customer_unique_id}",
    response_model=RFMProfileResponse,
    tags=["Analytics"],
    summary="RFM profile for a specific customer",
)
async def get_customer_rfm(customer_unique_id: str):
    """
    Retrieve RFM (Recency, Frequency, Monetary) profile for a customer.
    """
    rfm = _app_state.get("customer_rfm")
    if rfm is None:
        raise HTTPException(503, "RFM data not loaded. Run segmentation pipeline.")

    customer = rfm[rfm["customer_unique_id"] == customer_unique_id]
    if customer.empty:
        raise HTTPException(404, f"Customer '{customer_unique_id}' not found in RFM data.")

    row = customer.iloc[0]
    return RFMProfileResponse(
        customer_unique_id=str(row["customer_unique_id"]),
        recency_days=int(row["recency_days"]),
        frequency=int(row["frequency"]),
        monetary_brl=float(row["monetary_brl"]),
        recency_score=int(row["recency_score"]),
        frequency_score=int(row["frequency_score"]),
        monetary_score=int(row["monetary_score"]),
        rfm_score=int(row["rfm_score"]),
        rfm_segment=str(row["rfm_segment"]),
        snapshot_date=str(row["snapshot_date"]),
    )


@app.get(
    "/api/v1/segments/summary",
    tags=["Analytics"],
    summary="RFM segment distribution summary",
)
async def get_segment_summary():
    """Return count and percentage of customers in each RFM segment."""
    rfm = _app_state.get("customer_rfm")
    if rfm is None:
        raise HTTPException(503, "RFM data not loaded.")

    total = len(rfm)
    segment_counts = rfm["rfm_segment"].value_counts().reset_index()
    segment_counts.columns = ["segment", "count"]
    segment_counts["pct"] = (segment_counts["count"] / total * 100).round(2)

    return {
        "total_customers": total,
        "segments": segment_counts.to_dict("records"),
    }


# ── Analytics: Forecast ───────────────────────────────────────────────────────

@app.get(
    "/api/v1/analytics/forecast",
    response_model=ForecastResponse,
    tags=["Analytics"],
    summary="8-week revenue forecast",
)
async def get_revenue_forecast():
    """
    Return the pre-computed 8-week revenue forecast.
    Forecasts were generated using LightGBM with walk-forward CV validation.
    """
    forecast = _app_state.get("forecast")
    if forecast is None:
        raise HTTPException(503, "Forecast data not loaded. Run forecasting pipeline.")

    return ForecastResponse(
        forecast=forecast.to_dict("records"),
        model_used="LightGBM with lag + calendar features",
        n_forecast_weeks=len(forecast),
        warning="Forecasts are probabilistic estimates. Walk-forward CV was used for evaluation.",
    )


# ── Analytics: Marketing (SYNTHETIC) ──────────────────────────────────────────

@app.get(
    "/api/v1/analytics/marketing",
    response_model=MarketingKPIResponse,
    tags=["Marketing [SYNTHETIC]"],
    summary="Marketing KPIs — SYNTHETIC DATA ONLY",
)
async def get_marketing_kpis():
    """
    ⚠️  SYNTHETIC DATA — NOT REAL BUSINESS METRICS.

    The Olist dataset has no marketing data.
    These KPIs are computed from synthetic campaign data generated for demonstration.
    Attribution represents 'attributed revenue' (last-click) — NOT causal inference.
    """
    campaigns = _app_state.get("synthetic_campaigns")
    if campaigns is None:
        raise HTTPException(503, "Synthetic campaign data not loaded.")

    from src.analytics.marketing import compute_synthetic_marketing_kpis
    kpis = compute_synthetic_marketing_kpis(campaigns)

    return MarketingKPIResponse(**kpis)


# ── Predictions: Churn ────────────────────────────────────────────────────────

@app.post(
    "/api/v1/predict/churn",
    response_model=ChurnPredictionResponse,
    tags=["Predictions"],
    summary="Predict customer churn probability",
)
async def predict_churn(request: ChurnPredictionRequest):
    """
    Predict the probability that a customer will NOT purchase in the next 90 days.

    Features must be computed from historical data BEFORE the prediction date
    to avoid leakage. See model card for full specification.
    """
    model = _app_state.get("churn_model")
    if model is None:
        raise HTTPException(503, "Churn model not trained. Run `python -m src.models.churn`.")

    try:
        input_df = _churn_request_to_df(request)
        proba = model.predict_proba(input_df)[0, 1]
        pred = bool(proba > 0.5)
        confidence = "High" if abs(proba - 0.5) > 0.3 else ("Medium" if abs(proba - 0.5) > 0.15 else "Low")

        # SHAP explanations (best-effort)
        top_factors = None
        try:
            from src.explainability.shap_explainer import compute_shap_values, get_local_explanation
            shap_result = compute_shap_values(model, input_df, max_samples=1)
            local = get_local_explanation(shap_result, sample_idx=0)
            top_factors = local["contributions"][:5]
        except Exception:
            pass

        return ChurnPredictionResponse(
            customer_unique_id=request.customer_unique_id,
            churn_probability=round(float(proba), 4),
            churn_prediction=pred,
            confidence=confidence,
            top_factors=top_factors,
            model_version="1.0.0",
        )
    except Exception as e:
        logger.error(f"Churn prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.post(
    "/api/v1/predict/clv",
    response_model=CLVPredictionResponse,
    tags=["Predictions"],
    summary="Predict 12-month Customer Lifetime Value",
)
async def predict_clv(request: CLVPredictionRequest):
    """
    Predict a customer's 12-month spend (BRL).

    DATASET LIMITATION: Olist has ~97% single-purchase customers.
    CLV predictions reflect this real business characteristic.
    """
    model = _app_state.get("clv_model")
    if model is None:
        raise HTTPException(503, "CLV model not trained. Run `python -m src.models.clv`.")

    try:
        input_df = _clv_request_to_df(request)
        predicted_clv = float(model.predict(input_df)[0])
        predicted_clv = max(predicted_clv, 0)

        if predicted_clv > 1000:
            segment = "High Value"
        elif predicted_clv > 200:
            segment = "Medium Value"
        else:
            segment = "Low Value"

        return CLVPredictionResponse(
            customer_unique_id=request.customer_unique_id,
            predicted_clv_12m_brl=round(predicted_clv, 2),
            predicted_clv_segment=segment,
            model_version="1.0.0",
        )
    except Exception as e:
        logger.error(f"CLV prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


# ── Feature Builders ──────────────────────────────────────────────────────────

def _churn_request_to_df(request: ChurnPredictionRequest) -> pd.DataFrame:
    """Convert API request to model feature DataFrame."""
    return pd.DataFrame([{
        "recency_days": request.recency_days,
        "frequency": request.frequency,
        "monetary_brl": request.monetary_brl,
        "avg_review_score": request.avg_review_score or 4.0,
        "avg_delivery_days": request.avg_delivery_days or 10.0,
        "avg_installments": request.avg_installments or 1.0,
        "item_count_mean": request.item_count_mean or 1.0,
        "late_delivery_rate": request.late_delivery_rate or 0.0,
        "unique_categories": request.unique_categories or 1,
        "unique_sellers": request.unique_sellers or 1,
        "pct_credit_card": request.pct_credit_card or 0.5,
        "customer_state": request.customer_state or "SP",
        "preferred_payment_type": request.preferred_payment_type or "credit_card",
        "top_category": request.top_category or "unknown",
    }])


def _clv_request_to_df(request: CLVPredictionRequest) -> pd.DataFrame:
    """Convert CLV API request to model feature DataFrame."""
    monetary = request.monetary_brl
    frequency = request.frequency
    return pd.DataFrame([{
        "recency_days": request.recency_days,
        "frequency": frequency,
        "monetary_brl": monetary,
        "avg_order_value": request.avg_order_value or (monetary / frequency if frequency > 0 else monetary),
        "max_order_value": request.avg_order_value or monetary,
        "min_order_value": request.avg_order_value or monetary,
        "std_order_value": 0.0,
        "avg_review_score": request.avg_review_score or 4.0,
        "avg_delivery_days": 10.0,
        "avg_installments": 1.5,
        "late_delivery_rate": 0.0,
        "customer_age_days": request.customer_age_days or 30,
        "avg_inter_purchase_days": request.customer_age_days or 30,
        "unique_categories": request.unique_categories or 1,
        "unique_sellers": 1,
        "pct_credit_card": 0.5,
        "customer_state": request.customer_state or "SP",
        "preferred_payment_type": "credit_card",
        "top_category": request.top_category or "unknown",
    }])


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
