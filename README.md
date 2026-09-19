# E-Commerce Intelligence & Growth Prediction Platform

A production-grade Machine Learning and Analytics platform built on the **Olist Brazilian E-Commerce dataset**. This project demonstrates a complete end-to-end data pipeline, advanced business analytics, Customer 360 profiling, predictive modeling (Churn, CLV), and a robust deployment architecture.

## ⚠️ Data Limitations & Honesty Constraints
This project enforces strict honesty regarding the real Olist dataset capabilities:
1. **Profit Margins**: Not calculated. There is no COGS (Cost of Goods Sold) data in Olist.
2. **CLV Distribution**: Olist has ~97% single-purchase customers. The CLV model explicitly accounts for and documents this near-zero distribution.
3. **Marketing Analytics**: Olist contains no marketing spend data. All marketing KPIs (CAC, ROAS, CTR) are generated using **SYNTHETIC** data that is clearly labeled throughout the codebase and UI.
4. **Attribution**: All marketing attribution is rule-based (e.g., last-click) and labeled as "attributed revenue." **No causal inference is claimed.**

## 🏗️ Architecture
- **Data Engineering**: Python, Pandas, PostgreSQL, Parquet
- **Machine Learning**: Scikit-Learn, LightGBM, XGBoost, Prophet
- **Explainability**: SHAP (TreeExplainer)
- **Experiment Tracking**: MLflow
- **API**: FastAPI, Pydantic
- **Dashboard**: Streamlit, Plotly
- **Infrastructure**: Docker, Docker Compose, GitHub Actions (CI/CD)

## 🚀 Quickstart

### 1. Prerequisites
- Docker and Docker Compose
- Kaggle API credentials (for downloading the dataset)

### 2. Environment Setup
```bash
cp .env.example .env
# Edit .env and add your KAGGLE_USERNAME and KAGGLE_KEY
```

### 3. Run the Pipeline (Local)
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Execute all phases (Download → ETL → Analytics → Segmentation → ML Models → Forecasting):
```bash
python run_pipeline.py --phases all
```

### 4. Run Services (Docker)
Start the PostgreSQL database, MLflow tracking server, FastAPI backend, and Streamlit dashboard:
```bash
docker-compose up -d
```

- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000

## 📂 Documentation
- [Data Dictionary](docs/data_dictionary.md): Comprehensive schema and dataset stats.
- [ML Model Cards](docs/ml_model_cards.md): Detailed model architectures, leakage prevention, and metrics.
- [API Reference](docs/api_reference.md): Endpoint documentation and schemas.

## 🧪 Testing
```bash
pytest tests/ -v
```
