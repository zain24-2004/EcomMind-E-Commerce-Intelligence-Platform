"""
PHASE 4 — RFM Segmentation & Customer 360
Implements RFM scoring and customer profile aggregation.

RFM Definition:
  - Recency  (R): Days since last purchase (lower = better)
  - Frequency (F): Total number of orders
  - Monetary  (M): Total spend in BRL

Scoring: Quintile-based (1-5 per dimension, 5 = best)
Segments: Champion, Loyal Customer, Potential Loyalist, Recent Customer,
          Promising, Need Attention, About to Sleep, At Risk,
          Cannot Lose Them, Hibernating, Lost
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

from src.config import get_settings
from src.analytics.revenue import load_processed


# ── RFM Segment Map ──────────────────────────────────────────────────────────
# Based on RFM scores (R_score, F_score, M_score each 1-5)
# Combined rfm_score = R_score + F_score + M_score (3-15)

def _assign_segment(r: int, f: int, m: int) -> str:
    """Assign a business segment label based on RFM quintile scores."""
    score = r + f + m

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 4:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "Recent Customers"
    elif r >= 3 and f >= 3 and m >= 3:
        return "Potential Loyalists"
    elif r >= 3 and f <= 2 and m >= 3:
        return "Promising"
    elif r <= 2 and f >= 3 and m >= 3:
        return "At Risk"
    elif r <= 2 and f >= 4 and m >= 4:
        return "Cannot Lose Them"
    elif r <= 2 and f <= 2 and m >= 3:
        return "Hibernating"
    elif score <= 6:
        return "Lost"
    else:
        return "Need Attention"


def compute_rfm(
    fact_orders: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
    status_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute RFM scores for all customers.

    Args:
        fact_orders: Processed orders fact table.
        snapshot_date: Reference date for recency computation.
                       Defaults to the last purchase date in dataset + 1 day.
        status_filter: Order statuses to include. Defaults to ['delivered'].

    Returns:
        DataFrame with columns:
          customer_unique_id, snapshot_date, recency_days, frequency,
          monetary_brl, recency_score, frequency_score, monetary_score,
          rfm_score, rfm_segment
    """
    if status_filter is None:
        status_filter = ["delivered"]

    df = fact_orders[fact_orders["order_status"].isin(status_filter)].copy()
    df["purchase_ts"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    df = df[df["purchase_ts"].notna()]
    df = df[df["customer_unique_id"].notna()]

    if snapshot_date is None:
        snapshot_date = df["purchase_ts"].max() + pd.Timedelta(days=1)

    logger.info(f"RFM snapshot date: {snapshot_date.date()}")

    # ── Aggregate R, F, M per customer ───────────────────────────────────
    rfm = (
        df.groupby("customer_unique_id")
        .agg(
            last_purchase=("purchase_ts", "max"),
            frequency=("order_id", "count"),
            monetary_brl=("total_payment_brl", "sum"),
        )
        .reset_index()
    )

    rfm["recency_days"] = (snapshot_date - rfm["last_purchase"]).dt.days
    rfm["snapshot_date"] = snapshot_date.date()

    # ── Quintile scoring (1-5) ────────────────────────────────────────────
    # Recency: lower is better → highest recency score for lowest recency
    rfm["recency_score"] = pd.qcut(
        rfm["recency_days"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop"
    ).astype(int)

    rfm["frequency_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["monetary_score"] = pd.qcut(
        rfm["monetary_brl"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["rfm_score"] = rfm["recency_score"] + rfm["frequency_score"] + rfm["monetary_score"]

    # ── Segment assignment ────────────────────────────────────────────────
    rfm["rfm_segment"] = rfm.apply(
        lambda row: _assign_segment(
            row["recency_score"], row["frequency_score"], row["monetary_score"]
        ),
        axis=1,
    )

    # ── Cleanup ───────────────────────────────────────────────────────────
    rfm = rfm.drop(columns=["last_purchase"])
    rfm["monetary_brl"] = rfm["monetary_brl"].round(2)

    segment_counts = rfm["rfm_segment"].value_counts()
    logger.info(f"RFM computed for {len(rfm):,} customers across {rfm['rfm_segment'].nunique()} segments")
    for seg, cnt in segment_counts.items():
        pct = cnt / len(rfm) * 100
        logger.info(f"  {seg:<25}: {cnt:>6,} ({pct:.1f}%)")

    return rfm


def compute_customer_360(
    fact_orders: pd.DataFrame,
    fact_order_items: pd.DataFrame,
    dim_products: pd.DataFrame,
    rfm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a Customer 360 profile — comprehensive features per customer.

    Includes: RFM metrics, behavioral patterns, category preferences,
    payment behavior, delivery experience, review behavior.

    Returns one row per unique customer.
    """
    logger.info("Building Customer 360 profiles...")

    delivered = fact_orders[fact_orders["order_status"] == "delivered"].copy()
    delivered["purchase_ts"] = pd.to_datetime(delivered["order_purchase_timestamp"], errors="coerce")

    # ── Behavioral features ───────────────────────────────────────────────
    behavioral = (
        delivered.groupby("customer_unique_id")
        .agg(
            avg_review_score=("review_score", "mean"),
            review_count=("review_score", "count"),
            avg_delivery_days=("delivery_days", "mean"),
            late_delivery_count=("is_late_delivery", "sum"),
            avg_item_count=("item_count", "mean"),
            avg_freight_brl=("freight_total_brl", "mean"),
            avg_installments=("payment_installments", "mean"),
        )
        .reset_index()
    )
    behavioral["late_delivery_rate"] = (
        behavioral["late_delivery_count"] / behavioral["review_count"]
    ).fillna(0)

    # ── Payment type preference ───────────────────────────────────────────
    payment_mode = (
        delivered.groupby("customer_unique_id")["payment_type"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={"payment_type": "preferred_payment_type"})
    )
    pct_cc = (
        delivered.groupby("customer_unique_id")
        .apply(lambda g: (g["payment_type"] == "credit_card").mean(), include_groups=False)
        .reset_index()
        .rename(columns={0: "pct_credit_card"})
    )

    # ── Category preference (top category per customer) ──────────────────
    cat_col = "product_category_en" if "product_category_en" in dim_products.columns else "product_category_name"
    items_with_cat = fact_order_items.merge(
        dim_products[["product_id", cat_col]].rename(columns={cat_col: "category"}),
        on="product_id",
        how="left",
    )
    items_with_orders = items_with_cat.merge(
        delivered[["order_id", "customer_unique_id"]],
        on="order_id",
        how="inner",
    )
    top_category = (
        items_with_orders.groupby("customer_unique_id")["category"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
        .reset_index()
        .rename(columns={"category": "top_category"})
    )
    unique_categories = (
        items_with_orders.groupby("customer_unique_id")["category"]
        .nunique()
        .reset_index()
        .rename(columns={"category": "unique_categories"})
    )
    unique_sellers = (
        fact_order_items.merge(
            delivered[["order_id", "customer_unique_id"]], on="order_id", how="inner"
        )
        .groupby("customer_unique_id")["seller_id"]
        .nunique()
        .reset_index()
        .rename(columns={"seller_id": "unique_sellers"})
    )

    # ── Merge all features ────────────────────────────────────────────────
    profile = rfm.merge(behavioral, on="customer_unique_id", how="left")
    profile = profile.merge(payment_mode, on="customer_unique_id", how="left")
    profile = profile.merge(pct_cc, on="customer_unique_id", how="left")
    profile = profile.merge(top_category, on="customer_unique_id", how="left")
    profile = profile.merge(unique_categories, on="customer_unique_id", how="left")
    profile = profile.merge(unique_sellers, on="customer_unique_id", how="left")

    # ── Fill missing values ───────────────────────────────────────────────
    numeric_fill = {
        "avg_review_score": 0,
        "review_count": 0,
        "avg_delivery_days": 0,
        "late_delivery_rate": 0,
        "avg_item_count": 1,
        "avg_freight_brl": 0,
        "avg_installments": 1,
        "pct_credit_card": 0,
        "unique_categories": 1,
        "unique_sellers": 1,
    }
    for col, fill_val in numeric_fill.items():
        if col in profile.columns:
            profile[col] = profile[col].fillna(fill_val)

    logger.info(f"Customer 360: {len(profile):,} customers, {len(profile.columns)} features")
    return profile


def run_segmentation(processed_dir: Path | None = None) -> dict:
    """Run full RFM segmentation and Customer 360 pipeline."""
    settings = get_settings()
    processed_dir = processed_dir or settings.data_processed_path

    logger.info("=" * 60)
    logger.info("PHASE 4 — RFM SEGMENTATION & CUSTOMER 360")
    logger.info("=" * 60)

    fact_orders = load_processed("fact_orders", processed_dir)
    fact_items = load_processed("fact_order_items", processed_dir)
    dim_products = load_processed("dim_products", processed_dir)

    rfm = compute_rfm(fact_orders)
    rfm.to_parquet(processed_dir / "customer_rfm.parquet", index=False)
    logger.info(f"RFM saved to {processed_dir / 'customer_rfm.parquet'}")

    customer_360 = compute_customer_360(fact_orders, fact_items, dim_products, rfm)
    customer_360.to_parquet(processed_dir / "customer_360.parquet", index=False)
    logger.info(f"Customer 360 saved to {processed_dir / 'customer_360.parquet'}")

    logger.success("Phase 4 complete.")
    return {
        "rfm": rfm,
        "customer_360": customer_360,
        "segment_summary": rfm["rfm_segment"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    run_segmentation()
