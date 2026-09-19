"""
Pydantic schemas for the E-Commerce Intelligence API.
All models include validation and documentation.
"""
from __future__ import annotations

from datetime import date
from typing import Optional, List, Any

from pydantic import BaseModel, Field, ConfigDict


# ── Common ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    environment: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ── Churn Prediction ──────────────────────────────────────────────────────────

class ChurnPredictionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "customer_unique_id": "abc123",
            "recency_days": 45,
            "frequency": 2,
            "monetary_brl": 350.0,
            "avg_review_score": 4.2,
            "avg_delivery_days": 8.5,
            "customer_state": "SP",
            "preferred_payment_type": "credit_card",
        }
    })

    customer_unique_id: str = Field(..., description="Unique customer identifier")
    recency_days: int = Field(..., ge=0, description="Days since last purchase")
    frequency: int = Field(..., ge=1, description="Total number of orders")
    monetary_brl: float = Field(..., ge=0, description="Total spend in BRL")
    avg_review_score: Optional[float] = Field(None, ge=1, le=5)
    avg_delivery_days: Optional[float] = Field(None, ge=0)
    avg_installments: Optional[float] = Field(None, ge=1)
    late_delivery_rate: Optional[float] = Field(None, ge=0, le=1)
    customer_state: Optional[str] = Field(None, max_length=2)
    preferred_payment_type: Optional[str] = None
    top_category: Optional[str] = None
    unique_categories: Optional[int] = Field(None, ge=1)
    unique_sellers: Optional[int] = Field(None, ge=1)
    pct_credit_card: Optional[float] = Field(None, ge=0, le=1)
    item_count_mean: Optional[float] = Field(None, ge=1)


class ChurnPredictionResponse(BaseModel):
    customer_unique_id: str
    churn_probability: float = Field(..., description="Probability of churning (0-1)")
    churn_prediction: bool = Field(..., description="True = predicted to churn")
    confidence: str = Field(..., description="High/Medium/Low confidence level")
    top_factors: Optional[List[dict]] = Field(None, description="Top SHAP feature contributions")
    model_version: str


# ── CLV Prediction ────────────────────────────────────────────────────────────

class CLVPredictionRequest(BaseModel):
    customer_unique_id: str
    recency_days: int = Field(..., ge=0)
    frequency: int = Field(..., ge=1)
    monetary_brl: float = Field(..., ge=0)
    avg_order_value: Optional[float] = Field(None, ge=0)
    customer_age_days: Optional[int] = Field(None, ge=0)
    avg_review_score: Optional[float] = Field(None, ge=1, le=5)
    customer_state: Optional[str] = None
    top_category: Optional[str] = None
    unique_categories: Optional[int] = Field(None, ge=1)


class CLVPredictionResponse(BaseModel):
    customer_unique_id: str
    predicted_clv_12m_brl: float = Field(..., description="Predicted 12-month spend (BRL)")
    predicted_clv_segment: str = Field(..., description="High/Medium/Low value segment")
    top_factors: Optional[List[dict]] = None
    model_version: str
    disclaimer: str = (
        "Olist dataset has ~97% single-purchase customers. "
        "CLV predictions reflect this real business characteristic."
    )


# ── Analytics ─────────────────────────────────────────────────────────────────

class RevenueKPIResponse(BaseModel):
    total_gmv_brl: float
    order_volume: int
    aov_brl: float
    cancellation_rate_pct: float
    avg_delivery_days: float
    on_time_rate_pct: Optional[float]
    data_period_start: str
    data_period_end: str
    limitation_note: str


class RFMProfileResponse(BaseModel):
    customer_unique_id: str
    recency_days: int
    frequency: int
    monetary_brl: float
    recency_score: int
    frequency_score: int
    monetary_score: int
    rfm_score: int
    rfm_segment: str
    snapshot_date: str


class ForecastResponse(BaseModel):
    forecast: List[dict]
    model_used: str
    n_forecast_weeks: int
    warning: str = "Forecasts are probabilistic estimates, not guarantees."


class MarketingKPIResponse(BaseModel):
    synthetic_label: str
    attribution_disclaimer: str
    total_spend_brl: float
    total_attributed_revenue_brl: float
    overall_roas: float
    avg_ctr: float
    avg_conversion_rate: float
    estimated_cac_brl: float
    by_channel: List[dict]
