-- ============================================================
-- E-Commerce Intelligence Platform
-- PostgreSQL Schema (Phase 2)
-- Star schema optimized for analytics
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ────────────────────────────────────────────────────────────
-- DIMENSION TABLES
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_customers (
    customer_unique_id      VARCHAR(50)     PRIMARY KEY,
    customer_state          VARCHAR(5),
    customer_city           VARCHAR(100),
    customer_zip_prefix     VARCHAR(10),
    first_order_date        TIMESTAMP,
    last_order_date         TIMESTAMP,
    total_orders            INTEGER         DEFAULT 0,
    total_spend_brl         NUMERIC(12, 2)  DEFAULT 0,
    avg_review_score        NUMERIC(4, 2),
    created_at              TIMESTAMP       DEFAULT NOW(),
    updated_at              TIMESTAMP       DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dim_products (
    product_id              VARCHAR(50)     PRIMARY KEY,
    product_category_name   VARCHAR(100),
    product_category_en     VARCHAR(100),
    product_name_length     INTEGER,
    product_description_length INTEGER,
    product_photos_qty      INTEGER,
    product_weight_g        NUMERIC(10, 2),
    product_length_cm       NUMERIC(10, 2),
    product_height_cm       NUMERIC(10, 2),
    product_width_cm        NUMERIC(10, 2),
    created_at              TIMESTAMP       DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dim_sellers (
    seller_id               VARCHAR(50)     PRIMARY KEY,
    seller_state            VARCHAR(5),
    seller_city             VARCHAR(100),
    seller_zip_prefix       VARCHAR(10),
    created_at              TIMESTAMP       DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_id                 DATE            PRIMARY KEY,
    year                    INTEGER,
    quarter                 INTEGER,
    month                   INTEGER,
    week_of_year            INTEGER,
    day_of_week             INTEGER,        -- 0=Monday
    day_of_month            INTEGER,
    is_weekend              BOOLEAN,
    month_name              VARCHAR(20),
    day_name                VARCHAR(20)
);

-- ────────────────────────────────────────────────────────────
-- FACT TABLES
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_orders (
    order_id                    VARCHAR(50)     PRIMARY KEY,
    customer_unique_id          VARCHAR(50)     REFERENCES dim_customers(customer_unique_id),
    order_status                VARCHAR(30),
    purchase_date               DATE            REFERENCES dim_date(date_id),
    purchase_timestamp          TIMESTAMP,
    approved_at                 TIMESTAMP,
    delivered_carrier_date      TIMESTAMP,
    delivered_customer_date     TIMESTAMP,
    estimated_delivery_date     TIMESTAMP,
    -- Derived fields
    delivery_days               INTEGER,        -- delivered - purchase
    delay_days                  INTEGER,        -- delivered - estimated (negative = early)
    approval_hours              NUMERIC(8, 2),  -- approved - purchase in hours
    is_late_delivery            BOOLEAN,
    -- Payment summary
    total_payment_brl           NUMERIC(12, 2),
    payment_installments        INTEGER,
    payment_type                VARCHAR(30),
    -- Review
    review_score                INTEGER,
    -- Item counts
    item_count                  INTEGER,
    item_total_brl              NUMERIC(12, 2),
    freight_total_brl           NUMERIC(12, 2),
    -- Metadata
    created_at                  TIMESTAMP       DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fact_order_items (
    order_id                VARCHAR(50)     REFERENCES fact_orders(order_id),
    order_item_id           INTEGER,
    product_id              VARCHAR(50)     REFERENCES dim_products(product_id),
    seller_id               VARCHAR(50)     REFERENCES dim_sellers(seller_id),
    shipping_limit_date     TIMESTAMP,
    price_brl               NUMERIC(12, 2),
    freight_value_brl       NUMERIC(12, 2),
    PRIMARY KEY (order_id, order_item_id)
);

-- ────────────────────────────────────────────────────────────
-- ANALYTICS / DERIVED TABLES
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS customer_rfm (
    customer_unique_id      VARCHAR(50)     PRIMARY KEY REFERENCES dim_customers(customer_unique_id),
    snapshot_date           DATE,
    recency_days            INTEGER,        -- days since last purchase
    frequency               INTEGER,        -- total orders
    monetary_brl            NUMERIC(12, 2), -- total spend
    recency_score           INTEGER,        -- 1-5 quintile
    frequency_score         INTEGER,
    monetary_score          INTEGER,
    rfm_score               INTEGER,        -- R+F+M composite
    rfm_segment             VARCHAR(30),    -- Champion, Loyal, At Risk, etc.
    calculated_at           TIMESTAMP       DEFAULT NOW()
);

-- ML feature store (pre-computed features for models)
CREATE TABLE IF NOT EXISTS customer_features (
    customer_unique_id          VARCHAR(50)     PRIMARY KEY,
    snapshot_date               DATE,
    -- RFM
    recency_days                INTEGER,
    frequency                   INTEGER,
    monetary_brl                NUMERIC(12, 2),
    -- Behavioral
    avg_review_score            NUMERIC(4, 2),
    review_count                INTEGER,
    avg_delivery_days           NUMERIC(6, 2),
    late_delivery_rate          NUMERIC(5, 4),
    avg_item_price_brl          NUMERIC(12, 2),
    unique_categories           INTEGER,
    unique_sellers              INTEGER,
    pct_credit_card             NUMERIC(5, 4),
    avg_installments            NUMERIC(5, 2),
    -- Geo
    customer_state              VARCHAR(5),
    -- Targets (populated at training time)
    churned_90d                 BOOLEAN,
    purchased_30d               BOOLEAN,
    clv_12m_brl                 NUMERIC(12, 2),
    computed_at                 TIMESTAMP       DEFAULT NOW()
);

-- ────────────────────────────────────────────────────────────
-- SYNTHETIC MARKETING DATA
--   CLEARLY LABELED AS SYNTHETIC — NOT REAL BUSINESS DATA
-- ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS synthetic_campaigns (
    campaign_id             VARCHAR(50)     PRIMARY KEY,
    campaign_name           VARCHAR(100),
    channel                 VARCHAR(50),    -- google_ads, facebook, email, organic
    start_date              DATE,
    end_date                DATE,
    -- SYNTHETIC metrics
    impressions             INTEGER,
    clicks                  INTEGER,
    spend_brl               NUMERIC(12, 2),
    conversions             INTEGER,
    attributed_revenue_brl  NUMERIC(12, 2),
    -- SYNTHETIC derived
    ctr                     NUMERIC(8, 6),
    cpc_brl                 NUMERIC(8, 4),
    roas                    NUMERIC(8, 4),
    conversion_rate         NUMERIC(8, 6),
    is_synthetic            BOOLEAN         DEFAULT TRUE,
    created_at              TIMESTAMP       DEFAULT NOW()
);

-- ────────────────────────────────────────────────────────────
-- INDEXES
-- ────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_fact_orders_customer ON fact_orders(customer_unique_id);
CREATE INDEX IF NOT EXISTS idx_fact_orders_date ON fact_orders(purchase_date);
CREATE INDEX IF NOT EXISTS idx_fact_orders_status ON fact_orders(order_status);
CREATE INDEX IF NOT EXISTS idx_fact_items_product ON fact_order_items(product_id);
CREATE INDEX IF NOT EXISTS idx_fact_items_seller ON fact_order_items(seller_id);
CREATE INDEX IF NOT EXISTS idx_rfm_segment ON customer_rfm(rfm_segment);
CREATE INDEX IF NOT EXISTS idx_features_snapshot ON customer_features(snapshot_date);
