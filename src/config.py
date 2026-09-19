"""
Configuration management using pydantic-settings.
All secrets loaded from environment variables — never hardcoded.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Kaggle ──────────────────────────────────────────────────────────
    kaggle_username: str = ""
    kaggle_key: str = ""

    # ── PostgreSQL ──────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ecommerce_intel"
    postgres_user: str = "ecom_user"
    postgres_password: str = ""

    # ── MLflow ──────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "ecommerce_intelligence"

    # ── Paths ────────────────────────────────────────────────────────────
    data_raw_path: Path = Path("data/raw")
    data_processed_path: Path = Path("data/processed")
    data_synthetic_path: Path = Path("data/synthetic")
    models_path: Path = Path("models/saved")

    # ── API ─────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "dev-secret-key-change-in-production"

    # ── Dashboard ────────────────────────────────────────────────────────
    dashboard_port: int = 8501

    # ── App ─────────────────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_async_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
