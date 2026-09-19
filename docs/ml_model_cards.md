# ML Model Cards

## 1. Churn Prediction (`src/models/churn.py`)

### Model Details
- **Task**: Binary Classification
- **Target**: `churned_90d` (1 if no purchase in the next 90 days, 0 otherwise).
- **Algorithms Evaluated**: Baseline (Majority Class), Logistic Regression, Random Forest, XGBoost, LightGBM.

### Observation Period & Prediction Horizon
- **Observation Period**: Customer's entire purchase history up to the `snapshot_date`.
- **Prediction Horizon**: 90 days following the `snapshot_date`.
- **Leakage Prevention**: Features are computed *only* from orders placed strictly before the `snapshot_date`. The target is computed *only* from orders placed on or after the `snapshot_date`. A temporal split is used (training on older snapshots, testing on newer ones).

### Data Limitations
- **Olist Reality**: The dataset exhibits a ~97% single-purchase rate. By definition, a customer who buys once and never again "churns." Therefore, the churn rate is exceptionally high. The model accurately reflects this real business characteristic rather than fabricating high retention.

---

## 2. Customer Lifetime Value (`src/models/clv.py`)

### Model Details
- **Task**: Regression
- **Target**: `clv_12m_brl` (Total spend in BRL over the next 12 months).
- **Algorithms Evaluated**: Baseline (Mean Predictor), Ridge Regression, Random Forest, XGBoost, LightGBM.

### Observation Period & Prediction Horizon
- **Observation Period**: Customer's purchase history before the `snapshot_date`.
- **Prediction Horizon**: 365 days (12 months) following the `snapshot_date`.
- **Leakage Prevention**: Same strict temporal isolation as the Churn model.

### Data Limitations
- **Olist Reality**: Due to the ~97% single-purchase rate, the vast majority of target CLV values are exactly `0.0`. The model correctly learns this distribution. Predictions should be interpreted as the "expected probabilistic spend," which will naturally be low for most customers.

---

## 3. Revenue Forecasting (`src/models/forecasting.py`)

### Model Details
- **Task**: Time Series Forecasting
- **Targets**: Weekly GMV (`gmv_brl`) and Weekly Order Count (`order_count`).
- **Algorithms Evaluated**: Naive (Last Observation), 4-Week Moving Average, LightGBM (with lag and calendar features). Prophet support is included if installed.

### Evaluation Strategy
- **Walk-Forward Cross-Validation**: To prevent future data leakage, the model is evaluated using expanding window cross-validation (e.g., train on weeks 1-26, test on 27-38; train on 1-38, test on 39-50).
- **Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE).

### Explainability
- **SHAP**: Model explainability is implemented for all predictive models (`src/explainability/shap_explainer.py`). SHAP TreeExplainer is used to generate feature importance and local explanations (Waterfall/Force plots) for both Churn and CLV predictions.
