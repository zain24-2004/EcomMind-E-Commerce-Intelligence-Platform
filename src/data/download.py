"""
PHASE 1 — Dataset Download
Downloads the Olist Brazilian E-Commerce dataset from Kaggle.

Usage:
    python -m src.data.download

Requirements:
    - KAGGLE_USERNAME and KAGGLE_KEY set in .env
    - OR ~/.kaggle/kaggle.json exists
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

from loguru import logger

from src.config import get_settings


DATASET_SLUG = "olistbr/brazilian-ecommerce"
EXPECTED_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def _configure_kaggle_credentials(settings) -> None:
    """Set Kaggle env vars if provided in settings (avoids hardcoding)."""
    if settings.kaggle_username and settings.kaggle_key:
        os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
        os.environ["KAGGLE_KEY"] = settings.kaggle_key
        logger.debug("Kaggle credentials loaded from environment settings.")
    else:
        logger.debug("Kaggle credentials not in settings — expecting ~/.kaggle/kaggle.json")


def all_files_present(raw_dir: Path) -> bool:
    """Check if all expected Olist CSV files are already present."""
    missing = [f for f in EXPECTED_FILES if not (raw_dir / f).exists()]
    if missing:
        logger.info(f"Missing {len(missing)} files: {missing}")
    return len(missing) == 0


def download_olist_dataset(raw_dir: Path | None = None) -> Path:
    """
    Download the Olist dataset from Kaggle and extract to raw_dir.

    Returns the path to the raw data directory.
    """
    settings = get_settings()
    raw_dir = raw_dir or settings.data_raw_path
    raw_dir.mkdir(parents=True, exist_ok=True)

    if all_files_present(raw_dir):
        logger.info(f"All {len(EXPECTED_FILES)} Olist files already present in {raw_dir}. Skipping download.")
        return raw_dir

    _configure_kaggle_credentials(settings)

    try:
        import kaggle  # noqa: F401 — triggers credential check
        from kaggle.api.kaggle_api_extended import KaggleApiExtended

        api = KaggleApiExtended()
        api.authenticate()

        logger.info(f"Downloading dataset '{DATASET_SLUG}' from Kaggle...")
        api.dataset_download_files(
            DATASET_SLUG,
            path=str(raw_dir),
            unzip=False,
            quiet=False,
        )
        logger.info("Download complete.")
    except Exception as exc:
        logger.error(f"Kaggle download failed: {exc}")
        logger.info("Attempting manual download instructions...")
        _print_manual_instructions(raw_dir)
        raise

    # ── Unzip ────────────────────────────────────────────────────────────
    zip_files = list(raw_dir.glob("*.zip"))
    for zf in zip_files:
        logger.info(f"Extracting {zf.name}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(raw_dir)
        zf.unlink()  # Remove zip after extraction
        logger.info(f"Extracted and removed {zf.name}")

    # ── Verify ───────────────────────────────────────────────────────────
    if not all_files_present(raw_dir):
        missing = [f for f in EXPECTED_FILES if not (raw_dir / f).exists()]
        raise FileNotFoundError(
            f"Download succeeded but {len(missing)} expected files are missing: {missing}"
        )

    logger.success(f"All {len(EXPECTED_FILES)} Olist dataset files ready in {raw_dir}")
    return raw_dir


def _print_manual_instructions(raw_dir: Path) -> None:
    """Print manual download instructions when Kaggle API fails."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print(f"1. Go to: https://www.kaggle.com/datasets/{DATASET_SLUG}")
    print("2. Click 'Download' (ZIP)")
    print(f"3. Extract all CSV files to: {raw_dir.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    from src.logger import logger  # noqa — ensure logging is configured

    download_olist_dataset()
