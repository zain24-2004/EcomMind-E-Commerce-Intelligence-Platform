"""
PHASE 9 — Streamlit Dashboard
EcomMind — Interactive Analytics Dashboard.

Pages:
  1. Executive Overview (real KPIs)
  2. Revenue Analysis (time series, categories)
  3. Customer Analytics (RFM segments, cohorts)
  4. ML Predictions (interactive churn/CLV)
  5. Forecasting (revenue forecast chart)
  6. Marketing Analytics (SYNTHETIC — clearly labeled)

Run:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings

settings = get_settings()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcomMind",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #f5f5f7 !important;
        color: #1a1a1a !important;
    }

    [data-testid="stSidebar"] {
        background: #111111 !important;
        border-radius: 0 28px 28px 0 !important;
        padding: 1.8rem 0.5rem !important;
        box-shadow: 6px 0 40px rgba(0,0,0,0.28) !important;
        min-width: 150px !important;
        max-width: 150px !important;
    }
    [data-testid="stSidebar"] > div { padding: 0 !important; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stMarkdown {
        color: #555 !important;
        font-size: 0.62rem !important;
        line-height: 1.4 !important;
    }
        /* Hide native radio circles completely */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label { display: none !important; }
    [data-testid="stSidebar"] .stRadio input { display: none !important; }
    [data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] > div:first-child { display: none !important; }
    
    /* Hide nav label */
    [data-testid="stSidebar"] .stRadio > label { display: none !important; }
    /* Nav radio items */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label { display: flex !important; flex-direction: row !important; align-items: center !important; }
    [data-testid="stSidebar"] .stRadio input { display: none !important; }
    [data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] > div:first-child { display: none !important; }
    [data-testid="stSidebar"] .stRadio label > div { font-size: 0.85rem !important; white-space: nowrap !important; }
    [data-testid="stSidebar"] .stRadio label { width: 100% !important; height: 40px !important; padding: 0 10px !important; justify-content: flex-start !important; }

    [data-testid="stSidebar"] .stRadio { padding: 0 !important; }
    [data-testid="stSidebar"] .stRadio > label { display: none !important; }
    [data-testid="stSidebar"] .stRadio > div {
        display: flex !important; flex-direction: column !important; gap: 4px !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        display: flex !important; flex-direction: column !important;
        align-items: center !important; justify-content: center !important;
        width: 78px !important; height: 66px !important;
        margin: 0 auto !important; border-radius: 16px !important;
        cursor: pointer !important;
        transition: all 0.22s cubic-bezier(.4,0,.2,1) !important;
        background: transparent !important; color: #666 !important;
        border: 1px solid transparent !important;
        gap: 3px !important;
    }
    [data-testid="stSidebar"] .stRadio label > div {
        font-size: 0.68rem !important; font-weight: 600 !important;
        color: inherit !important; letter-spacing: 0.2px !important;
        white-space: nowrap !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,107,53,0.12) !important;
        color: #FF6B35 !important; border-color: rgba(255,107,53,0.2) !important;
        transform: scale(1.05) !important;
    }
    

    .main .block-container {
        background: #f5f5f7 !important;
        padding: 2rem 2.5rem !important;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 20px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        transition: all 0.3s cubic-bezier(.4,0,.2,1);
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.5s ease both;
    }
    .metric-card::before {
        content: '';
        position: absolute; top: 0; left: 0;
        width: 4px; height: 100%;
        background: linear-gradient(180deg, #FF6B35, #FF8C60);
        border-radius: 4px 0 0 4px;
        opacity: 0; transition: opacity 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 40px rgba(0,0,0,0.13);
        border-color: rgba(255,107,53,0.25);
    }
    .metric-card:hover::before { opacity: 1; }

    .metric-icon { font-size: 1.6rem; margin-bottom: 0.5rem; display: block; }
    .metric-value {
        font-size: 1.8rem; font-weight: 800; color: #111;
        letter-spacing: -0.5px; line-height: 1; margin-bottom: 4px;
    }
    .metric-label { font-size: 0.82rem; color: #888; font-weight: 500; margin-top: 4px; }
    .metric-badge {
        display: inline-flex; align-items: center; gap: 3px;
        font-size: 0.72rem; font-weight: 600;
        padding: 2px 9px; border-radius: 20px; margin-top: 8px;
    }
    .badge-up    { background: #e8f9f0; color: #1a8a4a; }
    .badge-down  { background: #fde8e8; color: #c0392b; }
    .badge-neutral { background: #f0f0f0; color: #666; }

    .sparkline-wrapper {
        position: absolute; bottom: 12px; right: 12px; opacity: 0.3;
    }

    .limitation-box {
        background: #fff8f5;
        border: 1px solid rgba(255,107,53,0.2);
        border-left: 4px solid #FF6B35;
        border-radius: 0 12px 12px 0;
        padding: 0.9rem 1.1rem; margin: 1rem 0;
        color: #666; font-size: 0.87rem;
        animation: slideInLeft 0.4s ease;
    }

    .synthetic-banner {
        background: linear-gradient(135deg, #fff3ee, #ffe8dd);
        border: 1px solid rgba(255,107,53,0.3);
        border-radius: 14px; padding: 0.9rem 1.3rem;
        color: #c0451a; font-weight: 600;
        margin-bottom: 1.2rem; font-size: 0.9rem;
    }

    .segment-chip {
        display: inline-block; padding: 0.3rem 0.85rem;
        border-radius: 20px; font-size: 0.78rem; font-weight: 600;
        margin: 0.2rem; transition: all 0.2s ease;
    }
    .segment-chip:hover { transform: scale(1.06); }

    .stButton > button {
        background: linear-gradient(135deg, #FF6B35, #e8541d) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; padding: 0.65rem 2.2rem !important;
        font-weight: 700 !important; font-size: 0.9rem !important;
        transition: all 0.25s cubic-bezier(.4,0,.2,1) !important;
        box-shadow: 0 4px 14px rgba(255,107,53,0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(255,107,53,0.55) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    .stTabs [data-baseweb="tab-list"] {
        background: #ebebeb !important;
        border-radius: 12px !important; padding: 4px !important; gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important; padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important; color: #666 !important;
        transition: all 0.2s !important;
    }
    .stTabs [aria-selected="true"] {
        background: #FF6B35 !important; color: white !important;
        box-shadow: 0 2px 10px rgba(255,107,53,0.4) !important;
    }

    .stDataFrame { border-radius: 14px !important; overflow: hidden !important; }
    .stDataFrame th { background: #f0f0f0 !important; color: #333 !important; font-weight: 700 !important; }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-12px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    hr { border: none !important; border-top: 1px solid rgba(0,0,0,0.06) !important; margin: 1.5rem 0 !important; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: #ddd; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #FF6B35; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Helper: premium metric card ───────────────────────────────────────────────

def metric_card(icon: str, label: str, value: str, note: str = "",
                badge: str = "", badge_type: str = "neutral") -> str:
    sparkline = (
        '<svg width="70" height="30" viewBox="0 0 70 30">'
        '<polyline points="0,28 12,20 24,22 36,10 48,15 60,6 70,12"'
        ' fill="none" stroke="#FF6B35" stroke-width="2.5"'
        ' stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="70" cy="12" r="3" fill="#FF6B35"/></svg>'
    )
    badge_html = ""
    if badge:
        arrow = "up arrow" if badge_type == "up" else ("down arrow" if badge_type == "down" else "dot")
        arrow_sym = "\u2191" if badge_type == "up" else ("\u2193" if badge_type == "down" else "\u00b7")
        badge_html = f'<span class="metric-badge badge-{badge_type}">{arrow_sym} {badge}</span>'
    note_html = (
        f'<div class="metric-label" style="font-size:0.72rem;color:#bbb;margin-top:4px">{note}</div>'
        if note else ""
    )
    return (
        f'<div class="metric-card">'
        f'<span class="metric-icon">{icon}</span>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'{badge_html}{note_html}'
        f'<div class="sparkline-wrapper">{sparkline}</div>'
        f'</div>'
    )

# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Load all processed data with caching."""
    data = {}
    processed_dir = settings.data_processed_path

    files = {
        "fact_orders": "fact_orders.parquet",
        "fact_items": "fact_order_items.parquet",
        "dim_products": "dim_products.parquet",
        "dim_customers": "dim_customers.parquet",
        "customer_rfm": "customer_rfm.parquet",
        "customer_360": "customer_360.parquet",
        "forecast": "revenue_forecast.parquet",
    }

    for key, filename in files.items():
        path = processed_dir / filename
        if path.exists():
            data[key] = pd.read_parquet(path)
        else:
            data[key] = None

    # Synthetic marketing
    synthetic_path = settings.data_synthetic_path / "synthetic_campaigns.parquet"
    data["synthetic_campaigns"] = pd.read_parquet(synthetic_path) if synthetic_path.exists() else None

    return data


@st.cache_resource
def load_models():
    """Load ML models with resource caching."""
    models = {}
    models_dir = settings.models_path
    if not models_dir.exists():
        return models

    for model_cls in ["ChurnModel", "CLVModel"]:
        files = list(models_dir.glob(f"{model_cls}_*.pkl"))
        if files:
            with open(files[0], "rb") as f:
                models[model_cls] = pickle.load(f)
    return models


# ── Sidebar Navigation ────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;margin-bottom:1.5rem;'>"
            "<span style='font-size:1.5rem;font-weight:800;color:#fff;'>ECOM</span>"
            "<p style='color:#555;font-size:0.65rem;margin:4px 0 0 0;"
            "letter-spacing:1px;text-transform:uppercase;'>EcomMind</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        page = st.radio(
            "nav",
            options=[
                "Overview",
                "Revenue",
                "Customers",
                "ML Models",
                "Forecast",
                "Marketing",
                "My Data",
            ],
            label_visibility="hidden",
        )

        st.markdown(
            "<hr style='border-color:#222;margin:1.5rem 0;'/>"
            "<div style='text-align:center;'>"
            "<p style='color:#444;font-size:0.62rem;letter-spacing:0.5px;'>Olist Dataset<br/>2016\u20132018</p>"
            "<p style='color:#FF6B35;font-size:0.62rem;font-weight:700;'>Marketing: SYNTHETIC</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    return page

# ── Page: Executive Overview ──────────────────────────────────────────────────

def page_executive_overview(data: dict, models: dict):
    st.markdown("""
    <div class="main-header">
        <h1>🛒 EcomMind</h1>
        <p>Real analytics from the Olist Brazilian E-Commerce Dataset · 2016–2018</p>
    </div>
    """, unsafe_allow_html=True)

    fact_orders = data.get("fact_orders")
    if fact_orders is None:
        st.error("⚠️ Data not loaded. Run the ETL pipeline first: `python -m src.data.etl`")
        st.code("python -m src.data.download\npython -m src.data.etl")
        return

    fact_orders["purchase_ts"] = pd.to_datetime(fact_orders["order_purchase_timestamp"], errors="coerce")
    delivered = fact_orders[fact_orders["order_status"] == "delivered"]

    from src.analytics.revenue import compute_total_gmv, compute_order_volume, compute_aov, compute_cancellation_metrics, compute_delivery_metrics

    gmv = compute_total_gmv(fact_orders)
    vol = compute_order_volume(fact_orders)
    aov = compute_aov(fact_orders)
    cancel_metrics = compute_cancellation_metrics(fact_orders)
    delivery_metrics = compute_delivery_metrics(fact_orders)

    rfm = data.get("customer_rfm")
    n_customers = len(rfm) if rfm is not None else 0
    avg_score = float(fact_orders["review_score"].mean()) if "review_score" in fact_orders.columns else 0

    # ── KPI Cards ─────────────────────────────────────────────────────────
    cols = st.columns(4)
    kpis = [
        ("💰 Total GMV", f"R$ {gmv:,.0f}", "Delivered orders only"),
        ("📦 Order Volume", f"{vol:,}", "Delivered orders"),
        ("🧾 Avg Order Value", f"R$ {aov:,.2f}", "GMV / delivered orders"),
        ("👤 Unique Customers", f"{n_customers:,}", "customer_unique_id"),
    ]
    for col, (label, value, note) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label" style="font-size:0.75rem">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("&nbsp;")
    cols2 = st.columns(4)
    kpis2 = [
        ("⭐ Avg Review Score", f"{avg_score:.2f}/5", "All delivered orders"),
        ("❌ Cancellation Rate", f"{cancel_metrics['cancellation_rate_pct']:.2f}%", "Proxy for returns"),
        ("🚚 Avg Delivery Days", f"{delivery_metrics['avg_delivery_days']:.1f}", "Purchase → Customer"),
        ("✅ On-Time Rate", f"{delivery_metrics.get('on_time_rate_pct', 0):.1f}%", "Delivered ≤ estimated"),
    ]
    for col, (label, value, note) in zip(cols2, kpis2):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label" style="font-size:0.75rem">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div class="limitation-box">
    ⚠️ <strong>Documented Limitations:</strong>
    COGS/profit data not available in Olist — gross margin cannot be computed.
    Cancellation rate uses 'canceled' status as a proxy for returns (no explicit return data).
    </div>""", unsafe_allow_html=True)

    # ── Revenue Time Series ────────────────────────────────────────────────
    st.markdown("### 📈 Monthly Revenue Trend")
    from src.analytics.revenue import compute_revenue_over_time
    monthly = compute_revenue_over_time(fact_orders, "monthly")
    monthly["period"] = pd.to_datetime(monthly["period"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["period"], y=monthly["gmv_brl"],
        name="GMV (BRL)", marker_color="#e94560", opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=monthly["period"], y=monthly["aov_brl"],
        name="AOV (BRL)", yaxis="y2",
        line=dict(color="#00d4ff", width=2.5),
        mode="lines+markers",
    ))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(249,249,249,1)",
        yaxis=dict(title="GMV (BRL)"),
        yaxis2=dict(title="AOV (BRL)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Order status distribution ──────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📦 Order Status Distribution")
        status_counts = fact_orders["order_status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig_s = px.pie(
            status_counts, values="count", names="status",
            color_discrete_sequence=px.colors.qualitative.Set3,
            template="plotly_white",
        )
        fig_s.update_layout(paper_bgcolor="rgba(255,255,255,0)", height=300)
        st.plotly_chart(fig_s, use_container_width=True)

    with col_b:
        st.markdown("### 💳 Payment Types")
        if "payment_type" in fact_orders.columns:
            pay_counts = fact_orders["payment_type"].value_counts().reset_index()
            pay_counts.columns = ["type", "count"]
            fig_p = px.bar(
                pay_counts, x="type", y="count",
                color="count", color_continuous_scale="Reds",
                template="plotly_white",
            )
            fig_p.update_layout(paper_bgcolor="rgba(255,255,255,0)", height=300, showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)


# ── Page: Revenue Analysis ────────────────────────────────────────────────────

def page_revenue_analysis(data: dict):
    st.markdown("## 💰 Revenue Analysis")

    fact_orders = data.get("fact_orders")
    fact_items = data.get("fact_items")
    dim_products = data.get("dim_products")

    if fact_orders is None:
        st.error("Data not loaded."); return

    from src.analytics.revenue import compute_revenue_by_category, compute_revenue_over_time

    period = st.selectbox("Time Period", ["monthly", "weekly", "quarterly"])
    rev_df = compute_revenue_over_time(fact_orders, period)
    rev_df["period"] = pd.to_datetime(rev_df["period"], errors="coerce")

    fig = px.area(
        rev_df, x="period", y="gmv_brl",
        title=f"{period.capitalize()} GMV (BRL)",
        template="plotly_white",
        color_discrete_sequence=["#e94560"],
    )
    fig.update_layout(paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(249,249,249,1)")
    st.plotly_chart(fig, use_container_width=True)

    if fact_items is not None and dim_products is not None:
        st.markdown("### 🏆 Top Categories by Revenue")
        cat_df = compute_revenue_by_category(fact_orders, fact_items, dim_products, top_n=15)
        fig2 = px.bar(
            cat_df, x="revenue_brl", y="category", orientation="h",
            color="avg_price_brl", color_continuous_scale="Reds",
            template="plotly_white",
        )
        fig2.update_layout(paper_bgcolor="rgba(255,255,255,0)", yaxis={"autorange": "reversed"}, height=500)
        st.plotly_chart(fig2, use_container_width=True)


# ── Page: Customer Analytics ──────────────────────────────────────────────────

def page_customer_analytics(data: dict):
    st.markdown("## 👥 Customer Analytics & RFM Segmentation")

    rfm = data.get("customer_rfm")
    if rfm is None:
        st.error("RFM data not loaded. Run: `python -m src.segmentation.rfm`"); return

    # Segment distribution
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Segment Distribution")
        seg_counts = rfm["rfm_segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig_seg = px.pie(
            seg_counts, values="count", names="segment",
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_white",
        )
        fig_seg.update_layout(paper_bgcolor="rgba(255,255,255,0)", height=350)
        st.plotly_chart(fig_seg, use_container_width=True)

    with col2:
        st.markdown("### RFM Score Distribution")
        fig_rfm = px.scatter(
            rfm.sample(min(5000, len(rfm)), random_state=42),
            x="recency_days", y="monetary_brl",
            color="rfm_segment", size="frequency",
            size_max=20, opacity=0.7,
            template="plotly_white",
            title="Recency vs Monetary (sized by Frequency)",
            log_y=True,
        )
        fig_rfm.update_layout(paper_bgcolor="rgba(255,255,255,0)", height=350)
        st.plotly_chart(fig_rfm, use_container_width=True)

    st.markdown("### Segment Summary Table")
    total = len(rfm)
    seg_summary = (
        rfm.groupby("rfm_segment")
        .agg(
            customers=("customer_unique_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary_brl", "mean"),
            avg_rfm_score=("rfm_score", "mean"),
        )
        .reset_index()
        .sort_values("avg_rfm_score", ascending=False)
    )
    seg_summary["pct"] = (seg_summary["customers"] / total * 100).round(2)
    st.dataframe(
        seg_summary.style.format({
            "customers": "{:,.0f}",
            "avg_recency": "{:.0f}d",
            "avg_frequency": "{:.2f}",
            "avg_monetary": "R$ {:.2f}",
            "avg_rfm_score": "{:.1f}",
            "pct": "{:.2f}%",
        }),
        use_container_width=True,
    )

    st.markdown("""<div class="limitation-box">
    ⚠️ Olist has ~97% single-purchase customers. Most segments will have Frequency = 1.
    This is a real business characteristic of the Olist marketplace, not a data error.
    </div>""", unsafe_allow_html=True)


# ── Page: ML Predictions ──────────────────────────────────────────────────────

def page_ml_predictions(data: dict, models: dict):
    st.markdown("## 🤖 ML Predictions")

    tab1, tab2 = st.tabs(["🔴 Churn Prediction", "💎 CLV Prediction"])

    with tab1:
        st.markdown("### Churn Prediction (90-day horizon)")
        st.markdown(
            "Predict whether a customer will NOT purchase in the next 90 days. "
            "Built on real Olist customer data."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            recency = st.slider("Recency (days since last purchase)", 1, 720, 45)
            frequency = st.slider("Frequency (total orders)", 1, 30, 2)
            monetary = st.number_input("Total Spend (BRL)", 10.0, 50000.0, 350.0, step=10.0)
        with col_b:
            review_score = st.slider("Avg Review Score", 1.0, 5.0, 4.0, step=0.5)
            delivery_days = st.slider("Avg Delivery Days", 1, 60, 10)
            state = st.selectbox("Customer State", ["SP", "RJ", "MG", "RS", "PR", "BA", "Other"])

        if st.button("🔮 Predict Churn", key="churn_btn"):
            churn_model = models.get("ChurnModel")
            if churn_model is None:
                st.error("Churn model not trained. Run: `python -m src.models.churn`")
            else:
                import pandas as pd
                input_df = pd.DataFrame([{
                    "recency_days": recency, "frequency": frequency, "monetary_brl": monetary,
                    "avg_review_score": review_score, "avg_delivery_days": float(delivery_days),
                    "avg_installments": 1.5, "item_count_mean": 1.2, "late_delivery_rate": 0.1,
                    "unique_categories": 2, "unique_sellers": 2, "pct_credit_card": 0.6,
                    "customer_state": state, "preferred_payment_type": "credit_card",
                    "top_category": "unknown",
                }])
                proba = churn_model.predict_proba(input_df)[0, 1]
                pred = proba > 0.5

                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    color = "#e94560" if pred else "#00c851"
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: {color};">
                        <div class="metric-label">Churn Probability</div>
                        <div class="metric-value" style="color: {color};">{proba:.1%}</div>
                        <div class="metric-label">{'⚠️ Likely to Churn' if pred else '✅ Likely to Return'}</div>
                    </div>""", unsafe_allow_html=True)
                with col_r2:
                    # Gauge chart
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba * 100,
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#e94560" if pred else "#00c851"},
                            "steps": [
                                {"range": [0, 30], "color": "#1a3a1a"},
                                {"range": [30, 60], "color": "#3a3a1a"},
                                {"range": [60, 100], "color": "#3a1a1a"},
                            ],
                        },
                        title={"text": "Churn Risk"},
                    ))
                    fig_g.update_layout(
                        template="plotly_white",
                        paper_bgcolor="rgba(255,255,255,0)",
                        height=200,
                    )
                    st.plotly_chart(fig_g, use_container_width=True)

    with tab2:
        st.markdown("### CLV Prediction (12-month horizon)")
        st.markdown(
            "Predict a customer's expected spend in the next 12 months."
        )
        st.markdown("""<div class="limitation-box">
        ⚠️ Olist has ~97% single-purchase customers.
        Most CLV predictions will be near R$ 0 — this reflects the real dataset characteristic.
        </div>""", unsafe_allow_html=True)

        col_c, col_d = st.columns(2)
        with col_c:
            clv_recency = st.slider("Recency (days)", 1, 720, 60, key="clv_r")
            clv_freq = st.slider("Frequency (orders)", 1, 30, 1, key="clv_f")
            clv_monetary = st.number_input("Total Spend (BRL)", 10.0, 50000.0, 250.0, key="clv_m")
        with col_d:
            clv_review = st.slider("Avg Review Score", 1.0, 5.0, 4.0, key="clv_rv", step=0.5)
            clv_state = st.selectbox("State", ["SP", "RJ", "MG", "RS", "PR", "BA", "Other"], key="clv_st")

        if st.button("🔮 Predict CLV", key="clv_btn"):
            clv_model = models.get("CLVModel")
            if clv_model is None:
                st.error("CLV model not trained. Run: `python -m src.models.clv`")
            else:
                input_df = pd.DataFrame([{
                    "recency_days": clv_recency, "frequency": clv_freq, "monetary_brl": clv_monetary,
                    "avg_order_value": clv_monetary / clv_freq, "max_order_value": clv_monetary,
                    "min_order_value": clv_monetary, "std_order_value": 0.0,
                    "avg_review_score": clv_review, "avg_delivery_days": 10.0,
                    "avg_installments": 1.5, "late_delivery_rate": 0.0, "customer_age_days": 90,
                    "avg_inter_purchase_days": 90, "unique_categories": 2, "unique_sellers": 2,
                    "pct_credit_card": 0.5, "customer_state": clv_state,
                    "preferred_payment_type": "credit_card", "top_category": "unknown",
                }])
                pred_clv = max(0, float(clv_model.predict(input_df)[0]))
                segment = "High Value" if pred_clv > 1000 else ("Medium Value" if pred_clv > 200 else "Low Value")
                color = {"High Value": "#00c851", "Medium Value": "#ffbb33", "Low Value": "#e94560"}[segment]

                st.markdown(f"""
                <div class="metric-card" style="border-color: {color};">
                    <div class="metric-label">Predicted 12-Month CLV</div>
                    <div class="metric-value" style="color: {color};">R$ {pred_clv:,.2f}</div>
                    <div class="metric-label">{segment}</div>
                </div>""", unsafe_allow_html=True)


# ── Page: Forecasting ─────────────────────────────────────────────────────────

def page_forecasting(data: dict):
    st.markdown("## 📈 Revenue Forecasting")

    fact_orders = data.get("fact_orders")
    forecast_df = data.get("forecast")

    if fact_orders is None:
        st.error("Data not loaded."); return

    from src.models.forecasting import prepare_weekly_revenue_series
    weekly = prepare_weekly_revenue_series(fact_orders)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly["week"], y=weekly["gmv_brl"],
        name="Actual Weekly Revenue",
        line=dict(color="#00d4ff", width=2),
        mode="lines",
    ))

    if forecast_df is not None and not forecast_df.empty:
        forecast_df["week"] = pd.to_datetime(forecast_df["week"])
        fig.add_trace(go.Scatter(
            x=forecast_df["week"], y=forecast_df["forecast_gmv_brl"],
            name="8-Week Forecast (LightGBM)",
            line=dict(color="#e94560", width=2.5, dash="dash"),
            mode="lines+markers",
        ))

    fig.update_layout(
        title="Weekly Revenue (Actual + 8-Week Forecast)",
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(249,249,249,1)",
        yaxis_title="GMV (BRL)",
        xaxis_title="Week",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "📌 Forecast uses LightGBM with lag features and calendar features. "
        "Evaluated with walk-forward cross-validation (no future leakage). "
        "This is a statistical estimate, not a guarantee."
    )


# ── Page: Marketing Analytics ─────────────────────────────────────────────────

def page_marketing_analytics(data: dict):
    st.markdown("""
    <div class="synthetic-banner">
        ⚠️ [SYNTHETIC DATA] — All marketing metrics below are generated from
        synthetic data. The Olist dataset contains no marketing spend data.
        These figures are for demonstration purposes only and do NOT represent
        real business performance.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📣 Marketing Analytics [SYNTHETIC]")

    campaigns = data.get("synthetic_campaigns")
    if campaigns is None:
        st.error("Synthetic data not found. Run: `python -m src.analytics.marketing`"); return

    from src.analytics.marketing import compute_synthetic_marketing_kpis
    kpis = compute_synthetic_marketing_kpis(campaigns)

    # KPI cards
    cols = st.columns(4)
    metrics = [
        ("💸 Total Spend [SYN]", f"R$ {kpis['total_spend_brl']:,.0f}", "SYNTHETIC"),
        ("📊 Attributed Rev [SYN]", f"R$ {kpis['total_attributed_revenue_brl']:,.0f}", "Attributed, not causal"),
        ("🎯 ROAS [SYN]", f"{kpis['overall_roas']:.2f}x", "Attributed revenue / spend"),
        ("👤 Est. CAC [SYN]", f"R$ {kpis['estimated_cac_brl']:.2f}", "Spend / conversions"),
    ]
    for col, (label, value, note) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #ff6b00;">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color: #ff8c00;">{value}</div>
                <div class="metric-label" style="font-size:0.75rem">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("&nbsp;")

    # Channel performance
    channel_df = pd.DataFrame(kpis["by_channel"])
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(
            channel_df, x="channel", y="spend_brl",
            title="Spend by Channel [SYNTHETIC]",
            color="avg_roas", color_continuous_scale="Oranges",
            template="plotly_white",
        )
        fig1.update_layout(paper_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            channel_df, x="spend_brl", y="attributed_revenue_brl",
            size="conversions", color="channel",
            title="Spend vs Attributed Revenue [SYNTHETIC]",
            template="plotly_white",
        )
        fig2.update_layout(paper_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""<div class="limitation-box">
    ⚠️ <strong>Attribution Disclaimer:</strong>
    "Attributed revenue" means the customer interacted with this campaign before purchasing.
    This is last-click attribution — it does NOT prove the campaign CAUSED the purchase.
    Causal inference requires A/B testing or uplift modeling.
    </div>""", unsafe_allow_html=True)


# ── Page: Analyze My Data ────────────────────────────────────────────────────

def page_my_data(models: dict):
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <h1 style='font-size:1.9rem;font-weight:800;color:#111;margin:0 0 4px 0;letter-spacing:-0.5px;'>
            🔍 Analyze My Data
        </h1>
        <p style='color:#888;font-size:0.9rem;margin:0;'>
            Upload your own order/customer CSV and let the platform analyze it instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload area ───────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#fff;border:2px dashed rgba(255,107,53,0.35);border-radius:20px;
         padding:2rem;text-align:center;margin-bottom:1.5rem;animation:fadeInUp 0.5s ease;'>
        <div style='font-size:2.5rem;margin-bottom:0.5rem;'>📂</div>
        <div style='font-size:1rem;font-weight:700;color:#333;'>Upload your data file</div>
        <div style='font-size:0.82rem;color:#888;margin-top:4px;'>
            Supported: CSV · Accepts order data or customer data
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        label_visibility="hidden",
        key="my_data_uploader",
    )

    if uploaded is None:
        # Show expected format hints
        st.markdown("### 📋 Expected Column Formats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <span class='metric-icon'>📦</span>
                <div class='metric-value' style='font-size:1rem;'>Order Data</div>
                <div class='metric-label' style='margin-top:8px;font-size:0.8rem;line-height:1.8;'>
                    <code>order_id</code>, <code>customer_id</code>,<br/>
                    <code>order_date</code>, <code>order_value</code>,<br/>
                    <code>order_status</code>, <code>payment_type</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <span class='metric-icon'>👤</span>
                <div class='metric-value' style='font-size:1rem;'>Customer Data</div>
                <div class='metric-label' style='margin-top:8px;font-size:0.8rem;line-height:1.8;'>
                    <code>customer_id</code>, <code>recency_days</code>,<br/>
                    <code>frequency</code>, <code>monetary_value</code>,<br/>
                    <code>review_score</code>, <code>state</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return

    # ── Load uploaded data ────────────────────────────────────────────────
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    st.success(f"✅ Loaded **{len(df):,} rows** × **{len(df.columns)} columns**")

    # ── Auto EDA ──────────────────────────────────────────────────────────
    with st.expander("🔬 Data Preview & Summary", expanded=True):
        tab_prev, tab_stats, tab_types = st.tabs(["Preview", "Statistics", "Column Types"])

        with tab_prev:
            st.dataframe(df.head(50), use_container_width=True, height=300)

        with tab_stats:
            desc = df.describe(include="all").T.reset_index()
            desc.columns = ["Column"] + list(desc.columns[1:])
            st.dataframe(desc, use_container_width=True)

        with tab_types:
            type_df = pd.DataFrame({
                "Column": df.columns,
                "Type": [str(df[c].dtype) for c in df.columns],
                "Non-Null": [df[c].notna().sum() for c in df.columns],
                "Null %": [(df[c].isna().mean() * 100).round(1) for c in df.columns],
                "Unique": [df[c].nunique() for c in df.columns],
                "Sample": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "" for c in df.columns],
            })
            st.dataframe(type_df, use_container_width=True)

    # ── KPI snapshot ──────────────────────────────────────────────────────
    st.markdown("### 📊 Quick KPI Snapshot")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "ts", "_at"])]

    kpi_cols = st.columns(4)
    kpi_vals = [
        ("Rows",    f"{len(df):,}",              "Total records"),
        ("Columns", f"{len(df.columns)}",        "Features"),
        ("Numeric", f"{len(numeric_cols)}",      "Numeric columns"),
        ("Missing", f"{df.isna().mean().mean():.1%}", "Avg null rate"),
    ]
    for col, (lbl, val, note) in zip(kpi_cols, kpi_vals):
        with col:
            st.markdown(metric_card("📋", lbl, val, note), unsafe_allow_html=True)

    # ── Numeric distribution charts ───────────────────────────────────────
    if numeric_cols:
        st.markdown("### 📈 Numeric Column Distributions")
        sel_col = st.selectbox("Select column to visualize", numeric_cols, key="eda_col")
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(
                df, x=sel_col, nbins=40,
                title=f"{sel_col} — Distribution",
                template="plotly_white",
                color_discrete_sequence=["#FF6B35"],
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(249,249,249,1)",
                height=300,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        with c2:
            fig_box = px.box(
                df, y=sel_col,
                title=f"{sel_col} — Box Plot",
                template="plotly_white",
                color_discrete_sequence=["#FF6B35"],
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(249,249,249,1)",
                height=300,
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        st.markdown("### 🔗 Correlation Matrix")
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Pearson Correlation",
            template="plotly_white",
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            height=max(300, len(numeric_cols) * 45),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── RFM Analysis on uploaded data ────────────────────────────────────
    rfm_cols_needed = {"customer_id", "order_date", "order_value"}
    has_rfm_cols = rfm_cols_needed.issubset({c.lower().strip() for c in df.columns})

    # Normalize column names for detection
    df_lower = {c.lower().strip(): c for c in df.columns}
    alt_rfm = (
        ("recency_days" in df_lower or "recency" in df_lower) and
        ("frequency" in df_lower) and
        ("monetary" in df_lower or "monetary_value" in df_lower or "order_value" in df_lower or "monetary_brl" in df_lower)
    )

    if has_rfm_cols or alt_rfm:
        st.markdown("### 🎯 RFM Segmentation on Your Data")

        try:
            if has_rfm_cols:
                cid   = df_lower["customer_id"]
                cdate = df_lower["order_date"]
                cval  = df_lower["order_value"]
                df[cdate] = pd.to_datetime(df[cdate], errors="coerce")
                snapshot = df[cdate].max() + pd.Timedelta(days=1)
                rfm_df = (
                    df.groupby(df[cid])
                    .agg(
                        recency_days=(cdate, lambda x: (snapshot - x.max()).days),
                        frequency=(cid, "count"),
                        monetary_brl=(cval, "sum"),
                    )
                    .reset_index()
                )
            else:
                r_col = df_lower.get("recency_days") or df_lower.get("recency")
                f_col = df_lower.get("frequency")
                m_col = (df_lower.get("monetary") or df_lower.get("monetary_value")
                         or df_lower.get("order_value") or df_lower.get("monetary_brl"))
                rfm_df = df[[r_col, f_col, m_col]].copy()
                rfm_df.columns = ["recency_days", "frequency", "monetary_brl"]
                rfm_df = rfm_df.dropna()

            # Score
            rfm_df["R"] = pd.qcut(rfm_df["recency_days"], 5, labels=[5,4,3,2,1], duplicates="drop").astype(int)
            rfm_df["F"] = pd.qcut(rfm_df["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
            rfm_df["M"] = pd.qcut(rfm_df["monetary_brl"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
            rfm_df["RFM_Score"] = rfm_df["R"] + rfm_df["F"] + rfm_df["M"]

            def seg(r, f, m):
                if r >= 4 and f >= 4 and m >= 4: return "Champions"
                elif r >= 3 and f >= 4:          return "Loyal Customers"
                elif r >= 4 and f <= 2:          return "Recent Customers"
                elif r >= 3 and f >= 3 and m >= 3: return "Potential Loyalists"
                elif r <= 2 and f >= 3 and m >= 3: return "At Risk"
                elif r <= 2 and f <= 2:          return "Lost / Hibernating"
                else:                            return "Need Attention"

            rfm_df["Segment"] = rfm_df.apply(lambda row: seg(row["R"], row["F"], row["M"]), axis=1)

            seg_colors = {
                "Champions": "#FF6B35", "Loyal Customers": "#f9a825",
                "Recent Customers": "#29b6f6", "Potential Loyalists": "#66bb6a",
                "At Risk": "#ef5350", "Lost / Hibernating": "#bdbdbd",
                "Need Attention": "#ab47bc",
            }

            cr1, cr2 = st.columns([1, 2])
            with cr1:
                seg_counts = rfm_df["Segment"].value_counts().reset_index()
                seg_counts.columns = ["Segment", "Count"]
                fig_seg = px.pie(
                    seg_counts, values="Count", names="Segment",
                    color="Segment", color_discrete_map=seg_colors,
                    template="plotly_white",
                    title="Segment Distribution",
                )
                fig_seg.update_layout(paper_bgcolor="rgba(255,255,255,0)", height=320)
                st.plotly_chart(fig_seg, use_container_width=True)

            with cr2:
                fig_scatter = px.scatter(
                    rfm_df.sample(min(3000, len(rfm_df)), random_state=42),
                    x="recency_days", y="monetary_brl",
                    color="Segment", size="frequency",
                    color_discrete_map=seg_colors,
                    size_max=18, opacity=0.75,
                    template="plotly_white",
                    title="Recency vs Monetary Value (sized by Frequency)",
                    log_y=True,
                )
                fig_scatter.update_layout(paper_bgcolor="rgba(255,255,255,0)", height=320)
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.dataframe(
                rfm_df[["recency_days", "frequency", "monetary_brl", "R", "F", "M", "RFM_Score", "Segment"]]
                .sort_values("RFM_Score", ascending=False)
                .head(100)
                .style.format({"recency_days": "{:.0f}d", "monetary_brl": "R$ {:.2f}"}),
                use_container_width=True,
            )

        except Exception as e:
            st.warning(f"Could not compute RFM: {e}")

    # ── ML Scoring on uploaded data ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Score Your Customers with ML Models")
    st.markdown("""
    <div class='limitation-box'>
    Map your CSV columns to the required features below, then click <strong>Run Predictions</strong>
    to get churn probability and CLV estimates for every row.
    </div>
    """, unsafe_allow_html=True)

    churn_model = models.get("ChurnModel")
    clv_model   = models.get("CLVModel")

    if churn_model is None and clv_model is None:
        st.warning("No trained models found. Run the pipeline first.")
        return

    required = {
        "recency_days":    "Recency (days since last purchase)",
        "frequency":       "Frequency (number of orders)",
        "monetary_brl":    "Monetary value (total spend)",
        "avg_review_score": "Avg review score (1–5)",
        "avg_delivery_days": "Avg delivery days",
    }

    mapping = {}
    st.markdown("#### Column Mapping")
    col_opts = ["(not available)"] + list(df.columns)
    mcols = st.columns(len(required))
    for (feat, label), mc in zip(required.items(), mcols):
        with mc:
            mapping[feat] = st.selectbox(label, col_opts, key=f"map_{feat}")

    if st.button("🚀 Run ML Predictions", key="run_ml_btn"):
        # Build input DataFrame
        score_df = pd.DataFrame()
        missing = []
        for feat, col in mapping.items():
            if col == "(not available)":
                missing.append(feat)
                score_df[feat] = 0.0
            else:
                score_df[feat] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Fill optional columns with defaults
        defaults = {
            "avg_installments": 1.5, "item_count_mean": 1.0,
            "late_delivery_rate": 0.0, "unique_categories": 2,
            "unique_sellers": 2, "pct_credit_card": 0.5,
            "customer_state": "SP", "preferred_payment_type": "credit_card",
            "top_category": "unknown",
        }
        for col, val in defaults.items():
            if col not in score_df.columns:
                score_df[col] = val

        if missing:
            st.warning(f"Using default 0 for unmapped columns: {', '.join(missing)}")

        results = df.copy()

        if churn_model:
            try:
                churn_probas = churn_model.predict_proba(score_df)[:, 1]
                results["Churn_Probability"] = churn_probas.round(3)
                results["Churn_Risk"] = pd.cut(
                    churn_probas, bins=[0, 0.4, 0.7, 1.0],
                    labels=["Low", "Medium", "High"]
                )
            except Exception as e:
                st.warning(f"Churn model error: {e}")

        if clv_model:
            clv_extra = {
                "avg_order_value": score_df.get("monetary_brl", pd.Series([0]*len(score_df))) / score_df.get("frequency", pd.Series([1]*len(score_df))),
                "max_order_value": score_df.get("monetary_brl", pd.Series([0]*len(score_df))),
                "min_order_value": score_df.get("monetary_brl", pd.Series([0]*len(score_df))),
                "std_order_value": 0.0,
                "customer_age_days": score_df.get("recency_days", pd.Series([90]*len(score_df))),
                "avg_inter_purchase_days": 90.0,
                "late_delivery_count": 0,
            }
            clv_input = score_df.copy()
            for k, v in clv_extra.items():
                if k not in clv_input:
                    clv_input[k] = v
            try:
                clv_preds = clv_model.predict(clv_input)
                results["CLV_12m_BRL"] = np.maximum(0, clv_preds).round(2)
            except Exception as e:
                st.warning(f"CLV model error: {e}")

        # Results display
        st.success(f"✅ Scored {len(results):,} customers!")

        pred_cols = [c for c in ["Churn_Probability", "Churn_Risk", "CLV_12m_BRL"] if c in results.columns]
        if pred_cols:
            # Summary KPIs
            skpi = st.columns(len(pred_cols))
            if "Churn_Probability" in results.columns:
                with skpi[0]:
                    st.markdown(metric_card("⚠️", "Avg Churn Risk",
                        f"{results['Churn_Probability'].mean():.1%}",
                        "Across all uploaded customers"), unsafe_allow_html=True)
            if "CLV_12m_BRL" in results.columns:
                idx = 1 if "Churn_Probability" in results.columns else 0
                with skpi[idx]:
                    st.markdown(metric_card("💎", "Avg Predicted CLV",
                        f"R$ {results['CLV_12m_BRL'].mean():.2f}",
                        "12-month estimate per customer"), unsafe_allow_html=True)

            # Risk distribution
            if "Churn_Risk" in results.columns:
                risk_counts = results["Churn_Risk"].value_counts().reset_index()
                risk_counts.columns = ["Risk", "Count"]
                fig_risk = px.bar(
                    risk_counts, x="Risk", y="Count",
                    color="Risk",
                    color_discrete_map={"Low": "#4caf50", "Medium": "#ff9800", "High": "#f44336"},
                    title="Churn Risk Distribution",
                    template="plotly_white",
                )
                fig_risk.update_layout(paper_bgcolor="rgba(255,255,255,0)", height=280)
                st.plotly_chart(fig_risk, use_container_width=True)

            # Full results table
            st.markdown("#### 📋 Full Scored Results")
            show_cols = list(df.columns[:5]) + pred_cols
            st.dataframe(results[show_cols].head(200), use_container_width=True)

            # Download
            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Full Results CSV",
                data=csv_out,
                file_name="ecommerce_predictions.csv",
                mime="text/csv",
                key="download_results",
            )


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    page = sidebar()

    with st.spinner("Loading data..."):
        data = load_data()
        models = load_models()

    page_map = {
        "Overview": lambda: page_executive_overview(data, models),
        "Revenue": lambda: page_revenue_analysis(data),
        "Customers": lambda: page_customer_analytics(data),
        "ML Models": lambda: page_ml_predictions(data, models),
        "Forecast": lambda: page_forecasting(data),
        "Marketing": lambda: page_marketing_analytics(data),
        "My Data": lambda: page_my_data(models),
    }

    if page in page_map:
        page_map[page]()
    else:
        st.error(f"Page not found: {page}")


if __name__ == "__main__":
    main()
