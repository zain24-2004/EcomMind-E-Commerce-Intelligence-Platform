"""
Test suite for the ETL pipeline.
Tests data transformations without requiring the full dataset.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from pathlib import Path


# ── Sample data fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_orders():
    return pd.DataFrame({
        "order_id": ["ord1", "ord2", "ord3", "ord4"],
        "customer_id": ["cust1", "cust2", "cust1", "cust3"],
        "order_status": ["delivered", "canceled", "delivered", "shipped"],
        "order_purchase_timestamp": pd.to_datetime([
            "2017-01-15 10:00:00",
            "2017-02-20 12:00:00",
            "2017-05-10 09:00:00",
            "2017-06-01 14:00:00",
        ]),
        "order_approved_at": pd.to_datetime([
            "2017-01-15 11:00:00", None, "2017-05-10 10:00:00", "2017-06-01 15:00:00",
        ]),
        "order_delivered_customer_date": pd.to_datetime([
            "2017-01-25 16:00:00", None, "2017-05-22 11:00:00", None,
        ]),
        "order_estimated_delivery_date": pd.to_datetime([
            "2017-01-30 00:00:00", None, "2017-05-25 00:00:00", None,
        ]),
        "order_delivered_carrier_date": [None, None, None, None],
    })


@pytest.fixture
def sample_customers():
    return pd.DataFrame({
        "customer_id": ["cust1", "cust2", "cust3"],
        "customer_unique_id": ["unique1", "unique2", "unique1"],
        "customer_state": ["sp", "rj", "sp"],
        "customer_city": ["sao paulo", "rio de janeiro", "campinas"],
        "customer_zip_code_prefix": ["01310", "20040", "13010"],
    })


@pytest.fixture
def sample_payments():
    return pd.DataFrame({
        "order_id": ["ord1", "ord1", "ord2", "ord3"],
        "payment_sequential": [1, 2, 1, 1],
        "payment_type": ["credit_card", "voucher", "boleto", "credit_card"],
        "payment_installments": [3, 1, 1, 6],
        "payment_value": [150.0, 50.0, 200.0, 300.0],
    })


@pytest.fixture
def sample_products():
    return pd.DataFrame({
        "product_id": ["prod1", "prod2"],
        "product_category_name": ["cama_mesa_banho", "informatica_acessorios"],
        "product_name_lenght": [40, 30],
        "product_description_lenght": [500, 300],
        "product_photos_qty": [3, 5],
        "product_weight_g": [500.0, 1200.0],
        "product_length_cm": [30.0, 45.0],
        "product_height_cm": [10.0, 15.0],
        "product_width_cm": [20.0, 30.0],
    })


@pytest.fixture
def sample_translation():
    return pd.DataFrame({
        "product_category_name": ["cama_mesa_banho", "informatica_acessorios"],
        "product_category_name_english": ["bed_bath_table", "computers_accessories"],
    })


# ── Tests: ETL transformations ────────────────────────────────────────────────

class TestPaymentAggregation:
    def test_aggregation_sums_correctly(self, sample_payments):
        from src.data.etl import _aggregate_payments
        result = _aggregate_payments(sample_payments)

        ord1_row = result[result["order_id"] == "ord1"].iloc[0]
        assert abs(ord1_row["total_payment_brl"] - 200.0) < 0.01

    def test_payment_type_mode(self, sample_payments):
        from src.data.etl import _aggregate_payments
        result = _aggregate_payments(sample_payments)

        # ord1 has credit_card (1x) and voucher (1x) — either is valid mode
        ord1_type = result[result["order_id"] == "ord1"]["payment_type"].iloc[0]
        assert ord1_type in ["credit_card", "voucher"]

    def test_no_negative_payments(self, sample_payments):
        from src.data.etl import _aggregate_payments
        sample_payments_with_neg = sample_payments.copy()
        sample_payments_with_neg.loc[0, "payment_value"] = -50.0
        result = _aggregate_payments(sample_payments_with_neg)
        assert (result["total_payment_brl"] >= 0).all()


class TestProductTransformation:
    def test_category_translation_merged(self, sample_products, sample_translation):
        from src.data.etl import _transform_products
        result = _transform_products(sample_products, sample_translation)
        assert "product_category_en" in result.columns
        assert "bed_bath_table" in result["product_category_en"].values

    def test_typo_columns_renamed(self, sample_products, sample_translation):
        from src.data.etl import _transform_products
        result = _transform_products(sample_products, sample_translation)
        assert "product_name_length" in result.columns
        assert "product_description_length" in result.columns

    def test_no_duplicates(self, sample_products, sample_translation):
        from src.data.etl import _transform_products
        duped = pd.concat([sample_products, sample_products])
        result = _transform_products(duped, sample_translation)
        assert result["product_id"].nunique() == result.shape[0]


class TestOrderTransformation:
    def test_delivery_days_computed(self, sample_orders, sample_customers):
        from src.data.etl import _transform_orders
        result = _transform_orders(sample_orders, sample_customers, None, None, None)
        ord1 = result[result["order_id"] == "ord1"].iloc[0]
        # 2017-01-25 - 2017-01-15 = 10 days
        assert ord1["delivery_days"] == 10

    def test_customer_unique_id_mapped(self, sample_orders, sample_customers):
        from src.data.etl import _transform_orders
        result = _transform_orders(sample_orders, sample_customers, None, None, None)
        assert "customer_unique_id" in result.columns
        assert "unique1" in result["customer_unique_id"].values

    def test_is_late_delivery_correct(self, sample_orders, sample_customers):
        from src.data.etl import _transform_orders
        result = _transform_orders(sample_orders, sample_customers, None, None, None)
        # ord1: delivered 2017-01-25, estimated 2017-01-30 → NOT late
        ord1 = result[result["order_id"] == "ord1"].iloc[0]
        assert ord1["is_late_delivery"] == False  # noqa: E712


class TestCustomerTransformation:
    def test_unique_customers_only(self, sample_customers, sample_orders):
        from src.data.etl import _transform_orders, _transform_customers
        fact_orders = _transform_orders(sample_orders, sample_customers, None, None, None)
        result = _transform_customers(sample_customers, fact_orders)
        # unique1 appears twice in customers but should be deduplicated
        assert result["customer_unique_id"].nunique() == result.shape[0]

    def test_state_normalized_uppercase(self, sample_customers, sample_orders):
        from src.data.etl import _transform_orders, _transform_customers
        fact_orders = _transform_orders(sample_orders, sample_customers, None, None, None)
        result = _transform_customers(sample_customers, fact_orders)
        assert all(result["customer_state"].str.isupper())


# ── Tests: Analytics ──────────────────────────────────────────────────────────

class TestRevenueAnalytics:
    @pytest.fixture
    def fact_orders_df(self):
        return pd.DataFrame({
            "order_id": ["o1", "o2", "o3", "o4"],
            "customer_unique_id": ["c1", "c2", "c3", "c1"],
            "order_status": ["delivered", "delivered", "canceled", "delivered"],
            "total_payment_brl": [100.0, 200.0, 50.0, 150.0],
            "order_purchase_timestamp": pd.to_datetime([
                "2017-01", "2017-02", "2017-02", "2017-03",
            ]),
            "delivery_days": [10, 8, None, 15],
            "delay_days": [-5, 2, None, -3],
            "is_late_delivery": [False, True, None, False],
            "review_score": [5, 4, None, 4],
            "payment_type": ["credit_card", "boleto", "boleto", "credit_card"],
        })

    def test_gmv_excludes_canceled(self, fact_orders_df):
        from src.analytics.revenue import compute_total_gmv
        gmv = compute_total_gmv(fact_orders_df)
        assert abs(gmv - 450.0) < 0.01  # 100 + 200 + 150

    def test_gmv_includes_all_statuses(self, fact_orders_df):
        from src.analytics.revenue import compute_total_gmv
        gmv_all = compute_total_gmv(fact_orders_df, status_filter=["delivered", "canceled"])
        assert abs(gmv_all - 500.0) < 0.01  # all 4

    def test_aov_formula(self, fact_orders_df):
        from src.analytics.revenue import compute_aov, compute_total_gmv, compute_order_volume
        gmv = compute_total_gmv(fact_orders_df)
        vol = compute_order_volume(fact_orders_df)
        aov = compute_aov(fact_orders_df)
        assert abs(aov - gmv / vol) < 0.01

    def test_cancellation_rate_formula(self, fact_orders_df):
        from src.analytics.revenue import compute_cancellation_metrics
        metrics = compute_cancellation_metrics(fact_orders_df)
        # 1 canceled out of 4 = 25%
        assert abs(metrics["cancellation_rate_pct"] - 25.0) < 0.01
        assert "limitation" in metrics  # must document the limitation


# ── Tests: RFM Segmentation ───────────────────────────────────────────────────

class TestRFMSegmentation:
    @pytest.fixture
    def fact_orders_df(self):
        return pd.DataFrame({
            "order_id": [f"o{i}" for i in range(20)],
            "customer_unique_id": [f"c{i % 5}" for i in range(20)],
            "order_status": ["delivered"] * 20,
            "order_purchase_timestamp": pd.date_range("2017-01-01", periods=20, freq="7D"),
            "total_payment_brl": np.random.default_rng(42).uniform(50, 500, 20),
            "review_score": np.random.default_rng(42).choice([3, 4, 5], 20),
            "delivery_days": np.random.default_rng(42).integers(5, 20, 20).astype(float),
            "delay_days": np.random.default_rng(42).integers(-10, 10, 20).astype(float),
            "is_late_delivery": [False] * 20,
            "payment_installments": [1.0] * 20,
            "item_count": [1.0] * 20,
            "freight_total_brl": [15.0] * 20,
            "payment_type": ["credit_card"] * 20,
        })

    def test_rfm_scores_are_1_to_5(self, fact_orders_df):
        from src.segmentation.rfm import compute_rfm
        rfm = compute_rfm(fact_orders_df)
        for score_col in ["recency_score", "frequency_score", "monetary_score"]:
            assert rfm[score_col].between(1, 5).all(), f"{score_col} out of range"

    def test_rfm_has_segment_column(self, fact_orders_df):
        from src.segmentation.rfm import compute_rfm
        rfm = compute_rfm(fact_orders_df)
        assert "rfm_segment" in rfm.columns
        assert rfm["rfm_segment"].notna().all()

    def test_rfm_one_row_per_customer(self, fact_orders_df):
        from src.segmentation.rfm import compute_rfm
        rfm = compute_rfm(fact_orders_df)
        assert rfm["customer_unique_id"].nunique() == len(rfm)


# ── Tests: API schemas ────────────────────────────────────────────────────────

class TestAPISchemas:
    def test_churn_request_validation(self):
        from src.api.schemas import ChurnPredictionRequest
        req = ChurnPredictionRequest(
            customer_unique_id="test123",
            recency_days=30,
            frequency=2,
            monetary_brl=350.0,
        )
        assert req.customer_unique_id == "test123"
        assert req.recency_days == 30

    def test_clv_request_validation(self):
        from src.api.schemas import CLVPredictionRequest
        req = CLVPredictionRequest(
            customer_unique_id="test456",
            recency_days=60,
            frequency=1,
            monetary_brl=250.0,
        )
        assert req.frequency == 1

    def test_clv_response_has_disclaimer(self):
        from src.api.schemas import CLVPredictionResponse
        resp = CLVPredictionResponse(
            customer_unique_id="test",
            predicted_clv_12m_brl=150.0,
            predicted_clv_segment="Low Value",
            model_version="1.0.0",
        )
        assert "single-purchase" in resp.disclaimer.lower()
