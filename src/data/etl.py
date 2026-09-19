"""
PHASE 2 — ETL Pipeline
Loads, validates, transforms, and writes the Olist dataset.

Outputs (Parquet, PostgreSQL-ready DataFrames, and optionally PostgreSQL):
  - data/processed/dim_customers.parquet
  - data/processed/dim_products.parquet
  - data/processed/dim_sellers.parquet
  - data/processed/dim_date.parquet
  - data/processed/fact_orders.parquet
  - data/processed/fact_order_items.parquet

Usage:
    python -m src.data.etl [--load-postgres]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_settings
from src.data.inspect_dataset import load_dataframe

# ── ETL Constants ─────────────────────────────────────────────────────────────
VALID_ORDER_STATUSES = {
    "delivered", "shipped", "canceled", "unavailable",
    "invoiced", "processing", "created", "approved",
}
BRL_MAX_REASONABLE = 50_000  # orders above this are flagged but kept


def run_etl(
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    load_postgres: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Execute the full ETL pipeline.

    Returns a dict of transformed DataFrames ready for analytics.
    """
    settings = get_settings()
    raw_dir = raw_dir or settings.data_raw_path
    processed_dir = processed_dir or settings.data_processed_path
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 2 — ETL PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Load raw data ────────────────────────────────────────────
    logger.info("Step 1: Loading raw CSV files...")
    raw: dict[str, pd.DataFrame] = {}
    file_keys = [
        "orders", "order_items", "payments", "reviews",
        "customers", "products", "sellers", "category_translation",
    ]
    for key in file_keys:
        try:
            raw[key] = load_dataframe(raw_dir, key)
            logger.info(f"  Loaded '{key}': {raw[key].shape}")
        except FileNotFoundError as e:
            logger.error(f"  Could not load '{key}': {e}")

    # ── Step 2: Validate ─────────────────────────────────────────────────
    logger.info("Step 2: Validating data quality...")
    _validate_raw_data(raw)

    # ── Step 3: Transform ────────────────────────────────────────────────
    logger.info("Step 3: Transforming tables...")
    transformed = _transform_all(raw)

    # ── Step 4: Write Parquet ────────────────────────────────────────────
    logger.info("Step 4: Writing Parquet files...")
    for name, df in transformed.items():
        out = processed_dir / f"{name}.parquet"
        df.to_parquet(out, index=False, compression="snappy")
        logger.info(f"  Saved {name}.parquet — {len(df):,} rows")

    # ── Step 5: Optionally load PostgreSQL ───────────────────────────────
    if load_postgres:
        logger.info("Step 5: Loading into PostgreSQL...")
        _load_to_postgres(transformed, settings)
    else:
        logger.info("Step 5: Skipping PostgreSQL load (use --load-postgres to enable)")

    logger.success("ETL pipeline complete.")
    return transformed


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_raw_data(raw: dict[str, pd.DataFrame]) -> None:
    """Run data quality checks and log warnings."""
    if "orders" in raw:
        orders = raw["orders"]
        # Check for null order_ids
        null_ids = orders["order_id"].isna().sum()
        if null_ids > 0:
            logger.warning(f"  orders: {null_ids} null order_id values")
        # Check status values
        if "order_status" in orders.columns:
            unknown_statuses = set(orders["order_status"].unique()) - VALID_ORDER_STATUSES
            if unknown_statuses:
                logger.warning(f"  orders: unexpected statuses: {unknown_statuses}")

    if "payments" in raw:
        payments = raw["payments"]
        if "payment_value" in payments.columns:
            negatives = (payments["payment_value"] < 0).sum()
            if negatives > 0:
                logger.warning(f"  payments: {negatives} negative payment_value rows")
            high_value = (payments["payment_value"] > BRL_MAX_REASONABLE).sum()
            if high_value > 0:
                logger.warning(f"  payments: {high_value} orders > R$ {BRL_MAX_REASONABLE:,} (kept)")

    if "order_items" in raw:
        items = raw["order_items"]
        if "price" in items.columns:
            zero_price = (items["price"] == 0).sum()
            if zero_price > 0:
                logger.warning(f"  order_items: {zero_price} zero-price items")

    logger.info("  Validation complete.")


# ── Transformations ───────────────────────────────────────────────────────────

def _transform_all(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply all transformations and return named DataFrames."""
    transformed: dict[str, pd.DataFrame] = {}

    # ── Dimension: Products ──────────────────────────────────────────────
    if "products" in raw and "category_translation" in raw:
        transformed["dim_products"] = _transform_products(
            raw["products"], raw["category_translation"]
        )

    # ── Dimension: Sellers ───────────────────────────────────────────────
    if "sellers" in raw:
        transformed["dim_sellers"] = _transform_sellers(raw["sellers"])

    # ── Fact: Order Items ────────────────────────────────────────────────
    if "order_items" in raw:
        transformed["fact_order_items"] = _transform_order_items(raw["order_items"])

    # ── Payment summary per order ────────────────────────────────────────
    payment_summary: Optional[pd.DataFrame] = None
    if "payments" in raw:
        payment_summary = _aggregate_payments(raw["payments"])

    # ── Review summary per order ─────────────────────────────────────────
    review_summary: Optional[pd.DataFrame] = None
    if "reviews" in raw:
        review_summary = _aggregate_reviews(raw["reviews"])

    # ── Item summary per order ────────────────────────────────────────────
    item_summary: Optional[pd.DataFrame] = None
    if "order_items" in raw:
        item_summary = _aggregate_items(raw["order_items"])

    # ── Fact: Orders ─────────────────────────────────────────────────────
    if "orders" in raw and "customers" in raw:
        transformed["fact_orders"] = _transform_orders(
            raw["orders"],
            raw["customers"],
            payment_summary,
            review_summary,
            item_summary,
        )

    # ── Dimension: Customers ─────────────────────────────────────────────
    if "customers" in raw and "fact_orders" in transformed:
        transformed["dim_customers"] = _transform_customers(
            raw["customers"], transformed["fact_orders"]
        )

    # ── Dimension: Date ──────────────────────────────────────────────────
    if "fact_orders" in transformed:
        transformed["dim_date"] = _build_date_dimension(transformed["fact_orders"])

    return transformed


def _transform_products(products: pd.DataFrame, translation: pd.DataFrame) -> pd.DataFrame:
    """Clean products and add English category names."""
    df = products.copy()
    df = df.drop_duplicates(subset=["product_id"])
    df = df.merge(translation, on="product_category_name", how="left")
    df = df.rename(columns={"product_category_name_english": "product_category_en"})
    # Fill missing numeric fields with median
    numeric_cols = [
        "product_name_lenght", "product_description_lenght",
        "product_photos_qty", "product_weight_g",
        "product_length_cm", "product_height_cm", "product_width_cm",
    ]
    # Handle Olist's typo in column names
    rename_typos = {
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    }
    df = df.rename(columns=rename_typos)
    for col in ["product_name_length", "product_description_length",
                "product_photos_qty", "product_weight_g",
                "product_length_cm", "product_height_cm", "product_width_cm"]:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    logger.info(f"  dim_products: {len(df):,} products, {df['product_category_name'].nunique()} categories")
    return df


def _transform_sellers(sellers: pd.DataFrame) -> pd.DataFrame:
    """Clean sellers dimension."""
    df = sellers.copy()
    df = df.drop_duplicates(subset=["seller_id"])
    df = df.rename(columns={
        "seller_zip_code_prefix": "seller_zip_prefix",
    })
    df["seller_state"] = df["seller_state"].str.upper().str.strip()
    df["seller_city"] = df["seller_city"].str.title().str.strip()
    logger.info(f"  dim_sellers: {len(df):,} sellers")
    return df


def _transform_order_items(items: pd.DataFrame) -> pd.DataFrame:
    """Clean order items fact table."""
    df = items.copy()
    df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")
    df["price_brl"] = df["price"].clip(lower=0)
    df["freight_value_brl"] = df["freight_value"].clip(lower=0)
    df = df.drop(columns=["price", "freight_value"], errors="ignore")
    logger.info(f"  fact_order_items: {len(df):,} items")
    return df


def _aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate payment data to order level."""
    agg = (
        payments.groupby("order_id")
        .agg(
            total_payment_brl=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_type=("payment_type", lambda x: x.mode()[0] if len(x) > 0 else None),
        )
        .reset_index()
    )
    agg["total_payment_brl"] = agg["total_payment_brl"].clip(lower=0)
    return agg


def _aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reviews to order level (take first review per order)."""
    reviews_sorted = reviews.sort_values("review_creation_date").drop_duplicates(
        subset=["order_id"], keep="first"
    )
    return reviews_sorted[["order_id", "review_score"]].copy()


def _aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item totals to order level."""
    agg = (
        items.groupby("order_id")
        .agg(
            item_count=("order_item_id", "count"),
            item_total_brl=("price", "sum"),
            freight_total_brl=("freight_value", "sum"),
        )
        .reset_index()
    )
    return agg


def _transform_orders(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    payment_summary: Optional[pd.DataFrame],
    review_summary: Optional[pd.DataFrame],
    item_summary: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build the fact_orders table with all derived fields."""
    df = orders.copy()

    # Parse timestamps
    ts_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in ts_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Add customer_unique_id
    customer_map = customers.set_index("customer_id")["customer_unique_id"]
    df["customer_unique_id"] = df["customer_id"].map(customer_map)

    # Derived time fields
    df["purchase_date"] = df["order_purchase_timestamp"].dt.date
    delivered = df["order_delivered_customer_date"]
    purchased = df["order_purchase_timestamp"]
    estimated = df["order_estimated_delivery_date"]
    approved = df["order_approved_at"]

    df["delivery_days"] = (delivered - purchased).dt.days
    df["delay_days"] = (delivered - estimated).dt.days  # negative = early
    df["approval_hours"] = (approved - purchased).dt.total_seconds() / 3600
    df["is_late_delivery"] = df["delay_days"] > 0

    # Clip unreasonable delivery days (sanity check)
    df["delivery_days"] = df["delivery_days"].clip(lower=0, upper=365)

    # Merge payment summary
    if payment_summary is not None:
        df = df.merge(payment_summary, on="order_id", how="left")
    else:
        df["total_payment_brl"] = np.nan
        df["payment_installments"] = np.nan
        df["payment_type"] = None

    # Merge review summary
    if review_summary is not None:
        df = df.merge(review_summary, on="order_id", how="left")
    else:
        df["review_score"] = np.nan

    # Merge item summary
    if item_summary is not None:
        df = df.merge(item_summary, on="order_id", how="left")
    else:
        df["item_count"] = np.nan
        df["item_total_brl"] = np.nan
        df["freight_total_brl"] = np.nan

    # Select and rename final columns
    final_cols = [
        "order_id", "customer_unique_id", "order_status", "purchase_date",
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date", "delivery_days", "delay_days",
        "approval_hours", "is_late_delivery", "total_payment_brl",
        "payment_installments", "payment_type", "review_score",
        "item_count", "item_total_brl", "freight_total_brl",
    ]
    df = df[[c for c in final_cols if c in df.columns]].copy()

    # Status: normalize
    if "order_status" in df.columns:
        df["order_status"] = df["order_status"].str.lower().str.strip()

    logger.info(
        f"  fact_orders: {len(df):,} orders | "
        f"null customer_unique_id: {df['customer_unique_id'].isna().sum()}"
    )
    return df


def _transform_customers(
    customers: pd.DataFrame,
    fact_orders: pd.DataFrame,
) -> pd.DataFrame:
    """Build dim_customers with order aggregates."""
    # Get unique customers
    unique_customers = (
        customers.drop_duplicates(subset=["customer_unique_id"])
        [["customer_unique_id", "customer_state", "customer_city", "customer_zip_code_prefix"]]
        .copy()
    )
    unique_customers = unique_customers.rename(
        columns={"customer_zip_code_prefix": "customer_zip_prefix"}
    )
    unique_customers["customer_state"] = unique_customers["customer_state"].str.upper().str.strip()
    unique_customers["customer_city"] = unique_customers["customer_city"].str.title().str.strip()

    # Aggregate order stats per customer
    order_agg = (
        fact_orders[fact_orders["order_status"] == "delivered"]
        .groupby("customer_unique_id")
        .agg(
            first_order_date=("order_purchase_timestamp", "min"),
            last_order_date=("order_purchase_timestamp", "max"),
            total_orders=("order_id", "count"),
            total_spend_brl=("total_payment_brl", "sum"),
            avg_review_score=("review_score", "mean"),
        )
        .reset_index()
    )

    df = unique_customers.merge(order_agg, on="customer_unique_id", how="left")
    df["total_orders"] = df["total_orders"].fillna(0).astype(int)
    df["total_spend_brl"] = df["total_spend_brl"].fillna(0)

    logger.info(
        f"  dim_customers: {len(df):,} unique customers | "
        f"repeat buyers: {(df['total_orders'] > 1).sum():,}"
    )
    return df


def _build_date_dimension(fact_orders: pd.DataFrame) -> pd.DataFrame:
    """Build a complete date dimension spanning the data range."""
    if "order_purchase_timestamp" not in fact_orders.columns:
        return pd.DataFrame()

    dates = pd.to_datetime(fact_orders["order_purchase_timestamp"].dropna())
    date_range = pd.date_range(
        start=dates.min().floor("D"),
        end=dates.max().ceil("D"),
        freq="D",
    )
    df = pd.DataFrame({"date_id": date_range.date})
    df["date_id"] = pd.to_datetime(df["date_id"])
    df["year"] = df["date_id"].dt.year
    df["quarter"] = df["date_id"].dt.quarter
    df["month"] = df["date_id"].dt.month
    df["week_of_year"] = df["date_id"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date_id"].dt.dayofweek
    df["day_of_month"] = df["date_id"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["month_name"] = df["date_id"].dt.strftime("%B")
    df["day_name"] = df["date_id"].dt.strftime("%A")
    logger.info(f"  dim_date: {len(df):,} dates")
    return df


def _load_to_postgres(
    transformed: dict[str, pd.DataFrame],
    settings,
) -> None:
    """Load transformed DataFrames into PostgreSQL."""
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.postgres_url)

        # Create schema first
        schema_path = Path("src/data/schema.sql")
        if schema_path.exists():
            with engine.connect() as conn:
                sql = schema_path.read_text()
                for statement in sql.split(";"):
                    stmt = statement.strip()
                    if stmt:
                        try:
                            conn.execute(text(stmt))
                        except Exception as e:
                            logger.debug(f"SQL statement warning: {e}")
                conn.commit()
            logger.info("  Schema created/verified in PostgreSQL")

        # Load tables in dependency order
        load_order = [
            "dim_products", "dim_sellers", "dim_date",
            "dim_customers", "fact_orders", "fact_order_items",
        ]
        for table_name in load_order:
            if table_name in transformed:
                df = transformed[table_name]
                df.to_sql(
                    table_name,
                    engine,
                    if_exists="replace",
                    index=False,
                    chunksize=10_000,
                    method="multi",
                )
                logger.info(f"  Loaded {table_name}: {len(df):,} rows → PostgreSQL")

    except Exception as exc:
        logger.error(f"PostgreSQL load failed: {exc}")
        logger.info("Parquet files are still available in data/processed/")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ETL pipeline")
    parser.add_argument(
        "--load-postgres",
        action="store_true",
        help="Also load data into PostgreSQL (requires POSTGRES_* env vars)",
    )
    args = parser.parse_args()
    run_etl(load_postgres=args.load_postgres)
