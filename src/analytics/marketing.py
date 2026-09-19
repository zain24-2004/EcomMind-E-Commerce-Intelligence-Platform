"""
⚠️  SYNTHETIC DATA — CLEARLY LABELED
Marketing Analytics Data Generator (Phase 7 prerequisite)

Generates realistic synthetic marketing campaign data for demonstration.
This data is NOT real business performance data.
All outputs are explicitly labeled as SYNTHETIC throughout.

Why synthetic:
  The Olist dataset contains no marketing spend, campaign, or attribution data.
  This module generates plausible synthetic data based on realistic e-commerce
  marketing benchmarks to demonstrate the platform's marketing analytics capabilities.

DO NOT present synthetic metrics as actual business performance.
All attribution analysis uses "attributed revenue" terminology — NOT causal inference.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_settings

# Synthetic data seed for reproducibility
SYNTHETIC_SEED = 42
SYNTHETIC_LABEL = "[SYNTHETIC — NOT REAL DATA]"

CHANNELS = {
    "google_search": {
        "ctr_base": 0.035,      # realistic benchmark CTR
        "cpc_base_brl": 2.50,
        "conv_rate_base": 0.025,
        "avg_order_value_brl": 250.0,
    },
    "facebook_ads": {
        "ctr_base": 0.012,
        "cpc_base_brl": 1.80,
        "conv_rate_base": 0.015,
        "avg_order_value_brl": 220.0,
    },
    "email_marketing": {
        "ctr_base": 0.025,
        "cpc_base_brl": 0.30,   # cost per email sent
        "conv_rate_base": 0.030,
        "avg_order_value_brl": 280.0,
    },
    "organic_search": {
        "ctr_base": 0.045,
        "cpc_base_brl": 0.0,    # no cost
        "conv_rate_base": 0.020,
        "avg_order_value_brl": 240.0,
    },
    "display_ads": {
        "ctr_base": 0.003,
        "cpc_base_brl": 0.80,
        "conv_rate_base": 0.008,
        "avg_order_value_brl": 200.0,
    },
}

CAMPAIGN_TYPES = [
    "Brand Awareness",
    "Retargeting",
    "New Customer Acquisition",
    "Seasonal Promotion",
    "Category Focus",
    "Cart Abandonment Recovery",   # NOTE: Cart data not in Olist — synthetic scenario
]


def generate_synthetic_campaigns(
    start_date: str = "2017-01-01",
    end_date: str = "2018-08-31",
    n_campaigns: int = 120,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic marketing campaign data.

    ⚠️  ALL VALUES ARE SYNTHETIC AND NOT REAL BUSINESS METRICS.

    Args:
        start_date: Campaign period start.
        end_date: Campaign period end.
        n_campaigns: Number of synthetic campaigns to generate.
        output_dir: Directory to save parquet output.

    Returns:
        DataFrame with synthetic campaign metrics.
        Every output contains is_synthetic=True.
    """
    logger.warning("Generating SYNTHETIC marketing data — not real business performance")
    rng = np.random.default_rng(SYNTHETIC_SEED)

    channels = list(CHANNELS.keys())
    campaign_types = CAMPAIGN_TYPES

    dates = pd.date_range(start_date, end_date, freq="D")
    campaign_starts = pd.Series(rng.choice(dates, size=n_campaigns))
    campaign_durations = rng.integers(7, 45, size=n_campaigns)
    campaign_ends = [
        min(s + pd.Timedelta(days=int(d)), pd.Timestamp(end_date))
        for s, d in zip(campaign_starts, campaign_durations)
    ]

    records = []
    for i in range(n_campaigns):
        channel = rng.choice(channels)
        params = CHANNELS[channel]
        budget_brl = float(rng.choice([500, 1000, 2500, 5000, 10000, 25000]))

        # Add realistic noise
        ctr = params["ctr_base"] * rng.uniform(0.7, 1.4)
        cpc = params["cpc_base_brl"] * rng.uniform(0.8, 1.3)
        conv_rate = params["conv_rate_base"] * rng.uniform(0.6, 1.5)

        # Compute dependent metrics
        clicks = int(budget_brl / cpc) if cpc > 0 else int(budget_brl * 100)
        impressions = int(clicks / ctr) if ctr > 0 else clicks * 30
        conversions = int(clicks * conv_rate)
        aov = params["avg_order_value_brl"] * rng.uniform(0.85, 1.20)
        attributed_revenue = conversions * aov
        roas = attributed_revenue / budget_brl if budget_brl > 0 else 0.0
        actual_cpc = budget_brl / clicks if clicks > 0 else 0.0

        records.append({
            "campaign_id": f"camp_{i+1:04d}",
            "campaign_name": f"{rng.choice(campaign_types)} — {channel.replace('_', ' ').title()} #{i+1}",
            "channel": channel,
            "campaign_type": rng.choice(campaign_types),
            "start_date": campaign_starts.iloc[i].date(),
            "end_date": campaign_ends[i].date() if hasattr(campaign_ends[i], "date") else campaign_ends[i],
            "budget_brl": round(budget_brl, 2),
            "spend_brl": round(budget_brl * rng.uniform(0.85, 1.0), 2),
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "attributed_revenue_brl": round(float(attributed_revenue), 2),
            "ctr": round(float(ctr), 6),
            "cpc_brl": round(float(actual_cpc), 4),
            "roas": round(float(roas), 4),
            "conversion_rate": round(float(conv_rate), 6),
            "is_synthetic": True,
            "synthetic_label": SYNTHETIC_LABEL,
        })

    df = pd.DataFrame(records)

    # Save output
    settings = get_settings()
    out_dir = output_dir or settings.data_synthetic_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_campaigns.parquet"
    df.to_parquet(out_path, index=False)
    logger.warning(f"Synthetic campaigns saved to {out_path} — {SYNTHETIC_LABEL}")

    return df


def generate_synthetic_sessions(
    fact_orders: pd.DataFrame,
    n_sessions_multiplier: float = 8.0,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic web session data.

    ⚠️  ALL SESSION DATA IS SYNTHETIC.
    The Olist dataset has no session/traffic data.

    Simulates a realistic funnel:
      Sessions → Product Views → Add-to-Cart → Checkout → Purchase
    """
    logger.warning("Generating SYNTHETIC session data — Olist has no session data")
    rng = np.random.default_rng(SYNTHETIC_SEED + 1)

    if "order_purchase_timestamp" not in fact_orders.columns:
        raise ValueError("fact_orders must have order_purchase_timestamp")

    # Use real purchase dates to anchor synthetic session timing
    delivered = fact_orders[fact_orders["order_status"] == "delivered"].copy()
    delivered["purchase_ts"] = pd.to_datetime(delivered["order_purchase_timestamp"], errors="coerce")
    dates = delivered["purchase_ts"].dt.date.unique()

    sessions = []
    channels = list(CHANNELS.keys())
    devices = ["mobile", "desktop", "tablet"]
    device_weights = [0.60, 0.35, 0.05]

    for date in dates:
        n_sessions = int(
            len(delivered[delivered["purchase_ts"].dt.date == date]) * n_sessions_multiplier
        )
        n_sessions = max(n_sessions, 50)

        # Funnel rates (synthetic benchmarks)
        n_product_views = int(n_sessions * rng.uniform(0.70, 0.85))
        n_add_to_cart = int(n_product_views * rng.uniform(0.25, 0.40))
        n_checkout = int(n_add_to_cart * rng.uniform(0.55, 0.75))
        n_purchase = int(n_checkout * rng.uniform(0.65, 0.90))
        n_abandoned = n_add_to_cart - n_purchase

        sessions.append({
            "date": date,
            "sessions": n_sessions,
            "product_views": n_product_views,
            "add_to_cart": n_add_to_cart,
            "checkout_started": n_checkout,
            "purchases": n_purchase,
            "cart_abandonment": n_abandoned,
            "cart_abandonment_rate": round(n_abandoned / n_add_to_cart, 4) if n_add_to_cart > 0 else 0,
            "session_conversion_rate": round(n_purchase / n_sessions, 4) if n_sessions > 0 else 0,
            "is_synthetic": True,
            "synthetic_label": SYNTHETIC_LABEL,
        })

    df = pd.DataFrame(sessions)

    settings = get_settings()
    out_dir = output_dir or settings.data_synthetic_path
    out_path = out_dir / "synthetic_sessions.parquet"
    df.to_parquet(out_path, index=False)
    logger.warning(f"Synthetic sessions saved to {out_path} — {SYNTHETIC_LABEL}")

    return df


def compute_synthetic_marketing_kpis(campaigns: pd.DataFrame) -> dict:
    """
    Compute marketing KPIs from synthetic campaign data.

    ⚠️  ALL METRICS ARE SYNTHETIC — NOT REAL BUSINESS PERFORMANCE.

    Attribution note:
      "Attributed revenue" means revenue where the customer touched this campaign.
      This does NOT prove the campaign caused the purchase (no causal inference).
    """
    logger.warning(f"Computing synthetic marketing KPIs — {SYNTHETIC_LABEL}")

    paid = campaigns[campaigns["cpc_brl"] > 0].copy()

    kpis = {
        "synthetic_label": SYNTHETIC_LABEL,
        "attribution_disclaimer": (
            "All revenue figures are 'attributed revenue' based on last-click attribution. "
            "This does NOT represent causal impact of campaigns on purchases. "
            "Causal inference would require A/B testing or causal modeling."
        ),
        "total_campaigns": int(len(campaigns)),
        "total_spend_brl": round(float(campaigns["spend_brl"].sum()), 2),
        "total_attributed_revenue_brl": round(float(campaigns["attributed_revenue_brl"].sum()), 2),
        "overall_roas": round(
            float(campaigns["attributed_revenue_brl"].sum() / campaigns["spend_brl"].sum()), 4
        ) if campaigns["spend_brl"].sum() > 0 else 0,
        "avg_ctr": round(float(campaigns["ctr"].mean()), 6),
        "avg_conversion_rate": round(float(campaigns["conversion_rate"].mean()), 6),
        "total_attributed_conversions": int(campaigns["conversions"].sum()),
        "estimated_cac_brl": round(
            float(campaigns["spend_brl"].sum() / campaigns["conversions"].sum()), 2
        ) if campaigns["conversions"].sum() > 0 else 0,
        "by_channel": (
            campaigns.groupby("channel")
            .agg(
                spend_brl=("spend_brl", "sum"),
                attributed_revenue_brl=("attributed_revenue_brl", "sum"),
                conversions=("conversions", "sum"),
                avg_roas=("roas", "mean"),
                avg_ctr=("ctr", "mean"),
            )
            .reset_index()
            .to_dict("records")
        ),
    }

    logger.info(f"Synthetic ROAS: {kpis['overall_roas']:.2f} | Synthetic CAC: R$ {kpis['estimated_cac_brl']:.2f}")
    return kpis


if __name__ == "__main__":
    df = generate_synthetic_campaigns()
    print(df.head())
    print(f"\n{SYNTHETIC_LABEL}")
    print(f"Generated {len(df)} synthetic campaigns")
