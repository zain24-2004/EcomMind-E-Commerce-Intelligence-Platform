"""
Master Pipeline Runner
Executes all platform phases in sequence.

Usage:
    python run_pipeline.py --phases all
    python run_pipeline.py --phases 1,2,3
    python run_pipeline.py --phases 1   (download & inspect only)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger
from src.config import get_settings


def run_phase_1(settings) -> dict:
    """Phase 1: Dataset inspection."""
    logger.info("RUNNING PHASE 1: Download & Dataset Inspection")
    from src.data.download import download_olist_dataset
    raw_dir = download_olist_dataset()

    from src.data.inspect_dataset import inspect_dataset
    summary = inspect_dataset(raw_dir)
    return summary


def run_phase_2(settings) -> dict:
    """Phase 2: ETL pipeline."""
    logger.info("RUNNING PHASE 2: ETL Pipeline")
    from src.data.etl import run_etl
    transformed = run_etl(load_postgres=False)
    return {"tables": list(transformed.keys()), "row_counts": {k: len(v) for k, v in transformed.items()}}


def run_phase_3(settings) -> dict:
    """Phase 3: Business analytics."""
    logger.info("RUNNING PHASE 3: Business Analytics")
    from src.analytics.revenue import run_revenue_analytics
    return run_revenue_analytics()


def run_phase_4(settings) -> dict:
    """Phase 4: RFM segmentation."""
    logger.info("RUNNING PHASE 4: RFM Segmentation & Customer 360")
    from src.segmentation.rfm import run_segmentation
    return run_segmentation()


def run_phase_5(settings) -> dict:
    """Phase 5: ML models."""
    logger.info("RUNNING PHASE 5: ML Model Training")
    results = {}

    logger.info("  Training Churn Model...")
    from src.models.churn import ChurnModel
    churn = ChurnModel()
    results["churn"] = churn.train_and_evaluate(run_mlflow=False)

    logger.info("  Training CLV Model...")
    from src.models.clv import CLVModel
    clv = CLVModel()
    results["clv"] = clv.train_and_evaluate(run_mlflow=False)

    return results


def run_phase_6(settings) -> dict:
    """Phase 6: Forecasting."""
    logger.info("RUNNING PHASE 6: Revenue & Demand Forecasting")
    from src.models.forecasting import run_forecasting
    return run_forecasting()


def run_phase_7(settings) -> dict:
    """Phase 7: Synthetic marketing data."""
    logger.info("RUNNING PHASE 7: Marketing Analytics (SYNTHETIC DATA)")
    from src.analytics.marketing import generate_synthetic_campaigns, generate_synthetic_sessions

    campaigns = generate_synthetic_campaigns()
    logger.warning("Generated synthetic campaigns — NOT real business data")

    fact_orders_path = settings.data_processed_path / "fact_orders.parquet"
    if fact_orders_path.exists():
        import pandas as pd
        fact_orders = pd.read_parquet(fact_orders_path)
        sessions = generate_synthetic_sessions(fact_orders)
        logger.warning("Generated synthetic sessions — NOT real business data")
    else:
        sessions = None

    return {
        "synthetic_campaigns": len(campaigns),
        "synthetic_sessions": len(sessions) if sessions is not None else 0,
        "note": "All marketing data is SYNTHETIC — not real business performance",
    }


def main():
    parser = argparse.ArgumentParser(description="Run the E-Commerce Intelligence Pipeline")
    parser.add_argument(
        "--phases",
        default="all",
        help="Comma-separated phase numbers (1-7) or 'all'. Example: --phases 1,2,3",
    )
    parser.add_argument(
        "--kaggle-username",
        help="Kaggle username (or set KAGGLE_USERNAME env var)",
    )
    parser.add_argument(
        "--kaggle-key",
        help="Kaggle API key (or set KAGGLE_KEY env var)",
    )
    args = parser.parse_args()

    import os
    if args.kaggle_username:
        os.environ["KAGGLE_USERNAME"] = args.kaggle_username
    if args.kaggle_key:
        os.environ["KAGGLE_KEY"] = args.kaggle_key

    settings = get_settings()

    phase_map = {
        1: run_phase_1,
        2: run_phase_2,
        3: run_phase_3,
        4: run_phase_4,
        5: run_phase_5,
        6: run_phase_6,
        7: run_phase_7,
    }

    if args.phases == "all":
        phases_to_run = list(phase_map.keys())
    else:
        phases_to_run = [int(p.strip()) for p in args.phases.split(",")]

    logger.info(f"Running phases: {phases_to_run}")
    logger.info("=" * 60)

    overall_start = time.time()
    results = {}

    for phase_num in phases_to_run:
        if phase_num not in phase_map:
            logger.warning(f"Unknown phase {phase_num} — skipping")
            continue

        phase_start = time.time()
        try:
            result = phase_map[phase_num](settings)
            elapsed = time.time() - phase_start
            logger.success(f"Phase {phase_num} complete in {elapsed:.1f}s")
            results[f"phase_{phase_num}"] = {"status": "success", "elapsed_s": round(elapsed, 1)}
        except Exception as exc:
            logger.error(f"Phase {phase_num} FAILED: {exc}")
            results[f"phase_{phase_num}"] = {"status": "failed", "error": str(exc)}

    total_elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.success(f"Pipeline complete in {total_elapsed:.1f}s")

    print("\n── Phase Results ───────────────────────────────────────────")
    for phase, result in results.items():
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {phase}: {result['status']} ({result.get('elapsed_s', '?')}s)")
    print()


if __name__ == "__main__":
    main()
