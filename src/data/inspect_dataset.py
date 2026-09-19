"""
PHASE 1 — Dataset Inspection
Comprehensive profiling of the Olist dataset.

Produces:
  - Console report with all dataset statistics
  - docs/data_dictionary.md (written to disk)
  - data/processed/inspection_summary.json (machine-readable)

Usage:
    python -m src.data.inspect_dataset
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np
from loguru import logger

from src.config import get_settings

# ── File registry ────────────────────────────────────────────────────────────
OLIST_FILES: dict[str, dict] = {
    "orders": {
        "file": "olist_orders_dataset.csv",
        "description": "Master order table — one row per order",
        "date_cols": [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    },
    "order_items": {
        "file": "olist_order_items_dataset.csv",
        "description": "Line items within each order — one row per item",
        "date_cols": ["shipping_limit_date"],
    },
    "payments": {
        "file": "olist_order_payments_dataset.csv",
        "description": "Payment details per order — one row per payment installment",
        "date_cols": [],
    },
    "reviews": {
        "file": "olist_order_reviews_dataset.csv",
        "description": "Customer reviews — one row per review",
        "date_cols": ["review_creation_date", "review_answer_timestamp"],
    },
    "customers": {
        "file": "olist_customers_dataset.csv",
        "description": "Customer dimension — one row per customer_id (not unique customers)",
        "date_cols": [],
    },
    "products": {
        "file": "olist_products_dataset.csv",
        "description": "Product catalogue — one row per product",
        "date_cols": [],
    },
    "sellers": {
        "file": "olist_sellers_dataset.csv",
        "description": "Seller dimension — one row per seller",
        "date_cols": [],
    },
    "geolocation": {
        "file": "olist_geolocation_dataset.csv",
        "description": "Geographic coordinates by zip code prefix",
        "date_cols": [],
    },
    "category_translation": {
        "file": "product_category_name_translation.csv",
        "description": "Portuguese → English category name mapping",
        "date_cols": [],
    },
}


# ── Feature availability matrix ──────────────────────────────────────────────
FEATURE_AVAILABILITY: list[dict] = [
    # Revenue / Orders
    {"feature": "Revenue (GMV)", "available": True, "source": "payments.payment_value", "notes": ""},
    {"feature": "Order Volume", "available": True, "source": "orders.order_id", "notes": ""},
    {"feature": "Average Order Value (AOV)", "available": True, "source": "payments.payment_value / count(orders)", "notes": ""},
    {"feature": "Freight Value", "available": True, "source": "order_items.freight_value", "notes": ""},
    # Cost / Profit
    {"feature": "Cost of Goods Sold (COGS)", "available": False, "source": "N/A", "notes": "LIMITATION: No cost data in dataset. Gross margin cannot be calculated."},
    {"feature": "Gross Profit / Margin", "available": False, "source": "N/A", "notes": "LIMITATION: Not in dataset."},
    # Customers
    {"feature": "Customer Identity (unique)", "available": True, "source": "customers.customer_unique_id", "notes": "customer_unique_id enables cross-order tracking"},
    {"feature": "Customer Location", "available": True, "source": "customers.city, state, zip_code_prefix", "notes": ""},
    {"feature": "Customer Acquisition Date", "available": True, "source": "first order_purchase_timestamp per customer", "notes": ""},
    # Products
    {"feature": "Product Category", "available": True, "source": "products.product_category_name + translation", "notes": ""},
    {"feature": "Product Dimensions/Weight", "available": True, "source": "products.*_cm, products.product_weight_g", "notes": ""},
    {"feature": "Product Price", "available": True, "source": "order_items.price", "notes": "Selling price per item"},
    # Orders
    {"feature": "Order Status", "available": True, "source": "orders.order_status", "notes": "Values: delivered, shipped, canceled, etc."},
    {"feature": "Delivery Timestamps", "available": True, "source": "orders.*_date", "notes": "Purchase, approval, carrier, delivery, estimated"},
    {"feature": "Cancellations (proxy for returns)", "available": True, "source": "orders.order_status = 'canceled'", "notes": "No explicit return data; canceled orders used as proxy"},
    {"feature": "Payment Type", "available": True, "source": "payments.payment_type", "notes": "credit_card, boleto, voucher, debit_card"},
    {"feature": "Installments", "available": True, "source": "payments.payment_installments", "notes": ""},
    # Reviews
    {"feature": "Review Score", "available": True, "source": "reviews.review_score", "notes": "1-5 stars"},
    # Marketing
    {"feature": "Marketing Spend", "available": False, "source": "N/A", "notes": "SYNTHETIC: Will generate realistic synthetic data clearly labeled"},
    {"feature": "Impressions / Clicks / CTR", "available": False, "source": "N/A", "notes": "SYNTHETIC: Clearly labeled throughout"},
    {"feature": "CAC / ROAS / CPC", "available": False, "source": "N/A", "notes": "SYNTHETIC: Clearly labeled throughout"},
    {"feature": "Attribution", "available": False, "source": "N/A", "notes": "SYNTHETIC: Rule-based attribution on synthetic data. NOT causal inference."},
    # Sessions
    {"feature": "Web Sessions / Traffic", "available": False, "source": "N/A", "notes": "LIMITATION: No session data in Olist"},
    {"feature": "Cart Abandonment (raw)", "available": False, "source": "N/A", "notes": "LIMITATION: No session/cart data. Proxy: orders with status='canceled' after approval."},
    # Sellers
    {"feature": "Seller Data", "available": True, "source": "sellers.*", "notes": "Seller location only"},
    {"feature": "Geolocation", "available": True, "source": "geolocation.lat, lng", "notes": "By zip code prefix"},
]


def load_dataframe(raw_dir: Path, key: str) -> pd.DataFrame:
    """Load a single Olist CSV file with proper date parsing."""
    meta = OLIST_FILES[key]
    filepath = raw_dir / meta["file"]
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(
        filepath,
        parse_dates=meta["date_cols"] if meta["date_cols"] else False,
        low_memory=False,
    )
    return df


def profile_dataframe(df: pd.DataFrame, name: str) -> dict[str, Any]:
    """Generate a comprehensive profile of a single DataFrame."""
    profile: dict[str, Any] = {
        "table": name,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": {},
        "duplicated_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }

    for col in df.columns:
        series = df[col]
        col_info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
            "unique_count": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_info.update(
                {
                    "min": round(float(desc["min"]), 4) if not pd.isna(desc["min"]) else None,
                    "max": round(float(desc["max"]), 4) if not pd.isna(desc["max"]) else None,
                    "mean": round(float(desc["mean"]), 4) if not pd.isna(desc["mean"]) else None,
                    "median": round(float(series.median()), 4),
                    "std": round(float(desc["std"]), 4) if not pd.isna(desc["std"]) else None,
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            valid = series.dropna()
            col_info.update(
                {
                    "min_date": str(valid.min()) if len(valid) > 0 else None,
                    "max_date": str(valid.max()) if len(valid) > 0 else None,
                }
            )
        elif pd.api.types.is_object_dtype(series):
            top_values = series.value_counts().head(5).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        profile["columns"][col] = col_info

    return profile


def inspect_dataset(raw_dir: Path | None = None) -> dict[str, Any]:
    """
    Run full Phase 1 inspection on the Olist dataset.

    Returns a summary dict and writes:
      - docs/data_dictionary.md
      - data/processed/inspection_summary.json
    """
    settings = get_settings()
    raw_dir = raw_dir or settings.data_raw_path

    logger.info("=" * 60)
    logger.info("PHASE 1 — DATASET INSPECTION")
    logger.info("=" * 60)

    profiles: dict[str, dict] = {}
    dataframes: dict[str, pd.DataFrame] = {}

    # ── Load & profile all tables ────────────────────────────────────────
    for key, meta in OLIST_FILES.items():
        filepath = raw_dir / meta["file"]
        if not filepath.exists():
            logger.warning(f"File missing: {meta['file']} — skipping")
            continue
        logger.info(f"Profiling: {meta['file']}")
        df = load_dataframe(raw_dir, key)
        dataframes[key] = df
        profiles[key] = profile_dataframe(df, key)
        p = profiles[key]
        logger.info(
            f"  → {p['n_rows']:,} rows × {p['n_cols']} cols | "
            f"{p['duplicated_rows']} duplicates | "
            f"{p['memory_mb']} MB"
        )

    # ── Aggregate stats ───────────────────────────────────────────────────
    summary = _compute_aggregate_stats(dataframes)
    summary["profiles"] = profiles
    summary["feature_availability"] = FEATURE_AVAILABILITY
    summary["inspection_timestamp"] = datetime.now().isoformat()

    # ── Write outputs ─────────────────────────────────────────────────────
    _write_json_summary(summary, settings)
    _write_data_dictionary(profiles, summary, settings)
    _print_report(profiles, summary)

    logger.success("Phase 1 inspection complete.")
    return summary


def _compute_aggregate_stats(dataframes: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Compute cross-table statistics from real data."""
    stats: dict[str, Any] = {}

    if "orders" in dataframes:
        orders = dataframes["orders"]
        ts_col = "order_purchase_timestamp"
        if ts_col in orders.columns:
            orders[ts_col] = pd.to_datetime(orders[ts_col], errors="coerce")
            valid_dates = orders[ts_col].dropna()
            stats["time_range_start"] = str(valid_dates.min())
            stats["time_range_end"] = str(valid_dates.max())
            stats["total_days"] = int((valid_dates.max() - valid_dates.min()).days)
        stats["total_orders"] = int(len(orders))
        if "order_status" in orders.columns:
            stats["order_status_distribution"] = orders["order_status"].value_counts().to_dict()

    if "customers" in dataframes:
        customers = dataframes["customers"]
        stats["total_customer_rows"] = int(len(customers))
        if "customer_unique_id" in customers.columns:
            stats["unique_customers"] = int(customers["customer_unique_id"].nunique())
        if "customer_state" in customers.columns:
            stats["customers_by_state"] = (
                customers["customer_state"].value_counts().head(10).to_dict()
            )

    if "payments" in dataframes:
        payments = dataframes["payments"]
        if "payment_value" in payments.columns:
            total_gmv = float(payments["payment_value"].sum())
            stats["total_gmv_brl"] = round(total_gmv, 2)
            stats["avg_payment_value_brl"] = round(
                float(payments["payment_value"].mean()), 2
            )
        if "payment_type" in payments.columns:
            stats["payment_type_distribution"] = (
                payments["payment_type"].value_counts().to_dict()
            )

    if "order_items" in dataframes:
        items = dataframes["order_items"]
        if "price" in items.columns:
            stats["avg_item_price_brl"] = round(float(items["price"].mean()), 2)
            stats["total_items_sold"] = int(len(items))

    if "reviews" in dataframes:
        reviews = dataframes["reviews"]
        if "review_score" in reviews.columns:
            stats["avg_review_score"] = round(float(reviews["review_score"].mean()), 3)
            stats["review_score_distribution"] = (
                reviews["review_score"].value_counts().sort_index().to_dict()
            )

    if "products" in dataframes:
        products = dataframes["products"]
        stats["total_products"] = int(len(products))
        if "product_category_name" in products.columns:
            stats["total_categories"] = int(products["product_category_name"].nunique())

    if "sellers" in dataframes:
        stats["total_sellers"] = int(len(dataframes["sellers"]))

    # Repeat purchase rate (key business metric for Olist)
    if "customers" in dataframes and "orders" in dataframes:
        orders = dataframes["orders"]
        customers = dataframes["customers"]
        if "customer_id" in orders.columns and "customer_unique_id" in customers.columns:
            merged = orders.merge(
                customers[["customer_id", "customer_unique_id"]],
                on="customer_id",
                how="left",
            )
            orders_per_customer = merged.groupby("customer_unique_id")["order_id"].count()
            repeat_customers = int((orders_per_customer > 1).sum())
            total_unique = int(len(orders_per_customer))
            stats["repeat_purchase_rate_pct"] = round(
                repeat_customers / total_unique * 100, 2
            )
            stats["single_purchase_customers"] = total_unique - repeat_customers
            stats["repeat_purchase_customers"] = repeat_customers

    return stats


def _write_json_summary(summary: dict, settings) -> None:
    """Write machine-readable inspection summary as JSON."""
    out_dir = settings.data_processed_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inspection_summary.json"

    # Make JSON-serializable
    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return str(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_default)
    logger.info(f"Inspection summary saved to {out_path}")


def _write_data_dictionary(
    profiles: dict[str, dict],
    summary: dict,
    settings,
) -> None:
    """Write the human-readable data dictionary to docs/data_dictionary.md."""
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    out_path = docs_dir / "data_dictionary.md"

    lines = [
        "# Data Dictionary — Olist Brazilian E-Commerce Dataset",
        "",
        f"> Generated: {summary.get('inspection_timestamp', 'N/A')}",
        f"> Time Range: {summary.get('time_range_start', 'N/A')} → {summary.get('time_range_end', 'N/A')}",
        f"> Total Orders: {summary.get('total_orders', 'N/A'):,}",
        f"> Unique Customers: {summary.get('unique_customers', 'N/A'):,}",
        f"> Total GMV (BRL): R$ {summary.get('total_gmv_brl', 'N/A'):,.2f}",
        "",
        "---",
        "",
        "## Feature Availability Matrix",
        "",
        "| Feature | Available | Source | Notes |",
        "|---------|-----------|--------|-------|",
    ]
    for fa in FEATURE_AVAILABILITY:
        avail = "✅" if fa["available"] else "❌"
        if "SYNTHETIC" in fa["notes"]:
            avail = "⚠️ SYNTHETIC"
        lines.append(
            f"| {fa['feature']} | {avail} | `{fa['source']}` | {fa['notes']} |"
        )

    lines += ["", "---", "", "## Tables", ""]

    for key, profile in profiles.items():
        meta = OLIST_FILES.get(key, {})
        lines += [
            f"### `{key}` — {meta.get('description', '')}",
            "",
            f"- **File**: `{meta.get('file', '')}`",
            f"- **Rows**: {profile['n_rows']:,}",
            f"- **Columns**: {profile['n_cols']}",
            f"- **Duplicate rows**: {profile['duplicated_rows']}",
            f"- **Memory**: {profile['memory_mb']} MB",
            "",
            "| Column | Type | Nulls | Null% | Unique | Notes |",
            "|--------|------|-------|-------|--------|-------|",
        ]
        for col, info in profile["columns"].items():
            extra = ""
            if "min_date" in info:
                extra = f"Range: {info['min_date']} → {info['max_date']}"
            elif "min" in info:
                extra = f"min={info['min']}, max={info['max']}, mean={info['mean']}"
            elif "top_values" in info:
                tv = list(info["top_values"].items())[:3]
                extra = ", ".join([f"{k}: {v:,}" for k, v in tv])
            lines.append(
                f"| `{col}` | {info['dtype']} | {info['null_count']:,} | "
                f"{info['null_pct']}% | {info['unique_count']:,} | {extra} |"
            )
        lines += [""]

    # Limitations section
    lines += [
        "---",
        "",
        "## ⚠️ Explicit Limitations",
        "",
        "The following features were requested but are **NOT available** in the Olist dataset:",
        "",
        "1. **COGS / Profit Margins**: No cost data exists. Gross margin metrics cannot be calculated from real data.",
        "2. **Marketing Spend / CAC / ROAS / CTR**: No marketing data in Olist. These metrics use **[SYNTHETIC]** data generated for demonstration purposes only.",
        "3. **Web Sessions / Traffic**: No session-level data. Funnel analytics and true cart abandonment cannot be measured.",
        "4. **Cart Abandonment**: No cart or session data. Proxy: orders with `status='canceled'` post-approval.",
        "5. **Low Repeat Purchase Rate**: Olist has ~97% single-purchase customers. This is a real characteristic of the dataset that impacts CLV and churn model design.",
        "6. **Attribution ≠ Causation**: All marketing attribution uses rule-based methods (last-click, linear, time-decay). These represent *attributed* revenue, NOT causal impact.",
        "",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Data dictionary written to {out_path}")


def _print_report(profiles: dict[str, dict], summary: dict) -> None:
    """Pretty-print the inspection report to console."""
    print("\n" + "=" * 70)
    print("PHASE 1 — OLIST DATASET INSPECTION REPORT")
    print("=" * 70)

    print(f"\n📅 Time Range   : {summary.get('time_range_start', 'N/A')} → {summary.get('time_range_end', 'N/A')}")
    print(f"📦 Total Orders : {summary.get('total_orders', 'N/A'):,}")
    print(f"👤 Unique Customers: {summary.get('unique_customers', 'N/A'):,}")
    print(f"💰 Total GMV (BRL) : R$ {summary.get('total_gmv_brl', 0):,.2f}")
    print(f"⭐ Avg Review Score: {summary.get('avg_review_score', 'N/A')}")
    print(f"🏪 Total Sellers : {summary.get('total_sellers', 'N/A'):,}")
    print(f"📦 Total Products: {summary.get('total_products', 'N/A'):,}")
    print(f"🔄 Repeat Purchase Rate: {summary.get('repeat_purchase_rate_pct', 'N/A')}%")

    print("\n── Tables ─────────────────────────────────────────────────────")
    print(f"{'Table':<25} {'Rows':>10} {'Cols':>6} {'Dupes':>8} {'MB':>8}")
    print("-" * 65)
    for key, p in profiles.items():
        print(
            f"{key:<25} {p['n_rows']:>10,} {p['n_cols']:>6} "
            f"{p['duplicated_rows']:>8,} {p['memory_mb']:>8.2f}"
        )

    if "order_status_distribution" in summary:
        print("\n── Order Status Distribution ──────────────────────────────────")
        for status, count in sorted(summary["order_status_distribution"].items(), key=lambda x: -x[1]):
            print(f"  {status:<30} {count:>8,}")

    if "payment_type_distribution" in summary:
        print("\n── Payment Types ───────────────────────────────────────────────")
        for ptype, count in sorted(summary["payment_type_distribution"].items(), key=lambda x: -x[1]):
            print(f"  {ptype:<30} {count:>8,}")

    print("\n── Feature Availability ────────────────────────────────────────")
    available = [f for f in FEATURE_AVAILABILITY if f["available"]]
    unavailable = [f for f in FEATURE_AVAILABILITY if not f["available"]]
    synthetic = [f for f in FEATURE_AVAILABILITY if "SYNTHETIC" in f["notes"]]

    print(f"  ✅ Available    : {len(available)} features")
    print(f"  ❌ Not Available: {len(unavailable)} features (documented limitations)")
    print(f"  ⚠️  Synthetic   : {len(synthetic)} features (clearly labeled)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    inspect_dataset()
