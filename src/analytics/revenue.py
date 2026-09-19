"""
PHASE 3 — Revenue Analytics
Computes revenue KPIs from real Olist data.

KPIs computed:
  - Total GMV (Gross Merchandise Value) — payment_value sum
  - Order volume — count of orders
  - Average Order Value (AOV) — GMV / orders
  - Revenue by time period (daily, weekly, monthly)
  - Revenue by order status
  - Revenue by product category
  - Freight revenue

Formula definitions are explicitly documented.
LIMITATION: COGS/profit not available — gross margin cannot be computed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np
from loguru import logger

from src.config import get_settings


PeriodType = Literal["daily", "weekly", "monthly", "quarterly"]


def load_processed(name: str, processed_dir: Path | None = None) -> pd.DataFrame:
    """Load a processed Parquet table."""
    settings = get_settings()
    processed_dir = processed_dir or settings.data_processed_path
    path = processed_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. "
            "Run `python -m src.data.etl` first."
        )
    return pd.read_parquet(path)


# ── KPI: Total GMV ────────────────────────────────────────────────────────────

def compute_total_gmv(
    fact_orders: pd.DataFrame,
    status_filter: list[str] | None = None,
) -> float:
    """
    KPI: Total Gross Merchandise Value (GMV).

    Formula: SUM(total_payment_brl) for selected order statuses.
    Required fields: total_payment_brl, order_status.
    Default status_filter: ['delivered'] (excludes canceled/fraudulent).

    Returns: Total GMV in BRL.
    """
    if status_filter is None:
        status_filter = ["delivered"]
    df = fact_orders[fact_orders["order_status"].isin(status_filter)].copy()
    gmv = float(df["total_payment_brl"].sum())
    logger.info(f"Total GMV (BRL) [{', '.join(status_filter)}]: R$ {gmv:,.2f}")
    return gmv


# ── KPI: Order Volume ─────────────────────────────────────────────────────────

def compute_order_volume(
    fact_orders: pd.DataFrame,
    status_filter: list[str] | None = None,
) -> int:
    """
    KPI: Total number of orders.

    Formula: COUNT(DISTINCT order_id).
    Required fields: order_id, order_status.
    """
    if status_filter is None:
        status_filter = ["delivered"]
    df = fact_orders[fact_orders["order_status"].isin(status_filter)]
    volume = int(df["order_id"].nunique())
    logger.info(f"Order Volume [{', '.join(status_filter)}]: {volume:,}")
    return volume


# ── KPI: Average Order Value ──────────────────────────────────────────────────

def compute_aov(
    fact_orders: pd.DataFrame,
    status_filter: list[str] | None = None,
) -> float:
    """
    KPI: Average Order Value (AOV).

    Formula: SUM(total_payment_brl) / COUNT(order_id).
    Required fields: total_payment_brl, order_status.
    """
    if status_filter is None:
        status_filter = ["delivered"]
    df = fact_orders[fact_orders["order_status"].isin(status_filter)]
    gmv = df["total_payment_brl"].sum()
    count = len(df)
    aov = float(gmv / count) if count > 0 else 0.0
    logger.info(f"AOV: R$ {aov:,.2f}")
    return aov


# ── KPI: Revenue Over Time ────────────────────────────────────────────────────

def compute_revenue_over_time(
    fact_orders: pd.DataFrame,
    period: PeriodType = "monthly",
    status_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    KPI: Revenue aggregated by time period.

    Returns DataFrame with columns: [period, gmv_brl, order_count, aov_brl].
    """
    if status_filter is None:
        status_filter = ["delivered"]
    df = fact_orders[fact_orders["order_status"].isin(status_filter)].copy()
    df["purchase_date"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")

    period_map = {
        "daily": "D",
        "weekly": "W-MON",
        "monthly": "ME",
        "quarterly": "QE",
    }
    freq = period_map.get(period, "ME")

    df["period"] = df["purchase_date"].dt.to_period(
        {"D": "D", "W-MON": "W", "ME": "M", "QE": "Q"}[freq]
    )

    result = (
        df.groupby("period")
        .agg(
            gmv_brl=("total_payment_brl", "sum"),
            order_count=("order_id", "count"),
        )
        .reset_index()
    )
    result["aov_brl"] = result["gmv_brl"] / result["order_count"]
    result["period"] = result["period"].astype(str)
    logger.info(f"Revenue over time ({period}): {len(result)} periods")
    return result


# ── KPI: Revenue by Category ──────────────────────────────────────────────────

def compute_revenue_by_category(
    fact_orders: pd.DataFrame,
    fact_order_items: pd.DataFrame,
    dim_products: pd.DataFrame,
    top_n: int = 20,
    status_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    KPI: Revenue (item price) by product category.

    Formula: SUM(price_brl) per category (uses item prices, not payment_value).
    Note: payment_value includes freight; item price is the product-level revenue.
    """
    if status_filter is None:
        status_filter = ["delivered"]
    delivered_orders = set(
        fact_orders[fact_orders["order_status"].isin(status_filter)]["order_id"]
    )
    items = fact_order_items[fact_order_items["order_id"].isin(delivered_orders)].copy()

    cat_col = "product_category_en" if "product_category_en" in dim_products.columns else "product_category_name"
    items = items.merge(
        dim_products[["product_id", cat_col]].rename(columns={cat_col: "category"}),
        on="product_id",
        how="left",
    )
    items["category"] = items["category"].fillna("unknown")

    result = (
        items.groupby("category")
        .agg(
            revenue_brl=("price_brl", "sum"),
            item_count=("order_id", "count"),
            avg_price_brl=("price_brl", "mean"),
        )
        .reset_index()
        .sort_values("revenue_brl", ascending=False)
        .head(top_n)
    )
    logger.info(f"Revenue by category: top {top_n} of {items['category'].nunique()} categories")
    return result


# ── KPI: Returns & Cancellations ─────────────────────────────────────────────

def compute_cancellation_metrics(fact_orders: pd.DataFrame) -> dict:
    """
    KPI: Cancellation rate.

    LIMITATION: Olist has no explicit return data.
    'canceled' order status is used as a proxy for returns/cancellations.

    Formula:
      - Cancellation rate = COUNT(canceled) / COUNT(all orders)
      - Canceled GMV = SUM(payment_value) for canceled orders
    """
    total = len(fact_orders)
    canceled = fact_orders[fact_orders["order_status"] == "canceled"]
    n_canceled = len(canceled)
    rate = n_canceled / total if total > 0 else 0.0
    canceled_gmv = float(canceled["total_payment_brl"].sum())

    metrics = {
        "total_orders": total,
        "canceled_orders": n_canceled,
        "cancellation_rate_pct": round(rate * 100, 3),
        "canceled_gmv_brl": round(canceled_gmv, 2),
        "limitation": (
            "Olist has no explicit returns data. "
            "canceled orders are used as a proxy."
        ),
    }
    logger.info(
        f"Cancellation rate: {metrics['cancellation_rate_pct']}% "
        f"({n_canceled:,} / {total:,} orders)"
    )
    return metrics


# ── KPI: Delivery Performance ─────────────────────────────────────────────────

def compute_delivery_metrics(fact_orders: pd.DataFrame) -> dict:
    """
    KPI: Delivery time and on-time delivery rate.

    Formula:
      - Avg delivery days = MEAN(delivered_customer_date - purchase_timestamp)
      - On-time rate = COUNT(delay_days <= 0) / COUNT(delivered orders)
    """
    delivered = fact_orders[fact_orders["order_status"] == "delivered"].copy()
    delivered = delivered[delivered["delivery_days"].notna()]

    metrics = {
        "total_delivered": int(len(delivered)),
        "avg_delivery_days": round(float(delivered["delivery_days"].mean()), 2),
        "median_delivery_days": round(float(delivered["delivery_days"].median()), 2),
        "p95_delivery_days": round(float(delivered["delivery_days"].quantile(0.95)), 2),
    }

    if "delay_days" in delivered.columns:
        on_time = (delivered["delay_days"] <= 0).sum()
        metrics["on_time_rate_pct"] = round(float(on_time / len(delivered) * 100), 2)
        metrics["avg_delay_days"] = round(float(delivered["delay_days"].mean()), 2)

    logger.info(f"Avg delivery: {metrics['avg_delivery_days']} days | On-time: {metrics.get('on_time_rate_pct', 'N/A')}%")
    return metrics


# ── Full Analytics Runner ─────────────────────────────────────────────────────

def run_revenue_analytics(processed_dir: Path | None = None) -> dict:
    """Run all revenue analytics and return results dict."""
    logger.info("Running Phase 3 — Revenue Analytics...")
    fact_orders = load_processed("fact_orders", processed_dir)
    fact_items = load_processed("fact_order_items", processed_dir)
    dim_products = load_processed("dim_products", processed_dir)

    results = {
        "total_gmv_brl": compute_total_gmv(fact_orders),
        "order_volume": compute_order_volume(fact_orders),
        "aov_brl": compute_aov(fact_orders),
        "revenue_monthly": compute_revenue_over_time(fact_orders, "monthly").to_dict("records"),
        "revenue_by_category": compute_revenue_by_category(fact_orders, fact_items, dim_products).to_dict("records"),
        "cancellation_metrics": compute_cancellation_metrics(fact_orders),
        "delivery_metrics": compute_delivery_metrics(fact_orders),
        "limitation_note": "COGS/profit not available in dataset. Gross margin cannot be computed.",
    }
    logger.success("Revenue analytics complete.")
    return results


if __name__ == "__main__":
    run_revenue_analytics()
