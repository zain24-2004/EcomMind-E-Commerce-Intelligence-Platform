# API Reference

The E-Commerce Intelligence API is built with FastAPI. Interactive documentation (Swagger UI) is automatically available at `http://localhost:8000/docs` when the server is running.

## Base URL
`http://localhost:8000/api/v1`

---

## Analytics Endpoints

### `GET /analytics/revenue`
Returns high-level business KPIs based on real Olist data.
- **Response**: `RevenueKPIResponse` (GMV, AOV, Order Volume, Cancellation Rate, Delivery Metrics).

### `GET /analytics/rfm/{customer_unique_id}`
Returns the RFM (Recency, Frequency, Monetary) profile and segment for a specific customer.
- **Parameters**: `customer_unique_id` (string, path parameter).
- **Response**: `RFMProfileResponse`.

### `GET /segments/summary`
Returns the distribution of customers across the 11 RFM segments.

### `GET /analytics/forecast`
Returns the pre-computed 8-week forward revenue forecast.
- **Response**: `ForecastResponse` (Array of weekly forecast data).

### `GET /analytics/marketing`
**[SYNTHETIC DATA]** Returns marketing KPIs (CAC, ROAS, CTR) based on generated synthetic data.
- **Response**: `MarketingKPIResponse`.

---

## Prediction Endpoints

### `POST /predict/churn`
Predicts the probability that a customer will churn (make no purchases) in the next 90 days.
- **Request Body**: `ChurnPredictionRequest` (Requires recency, frequency, monetary, and basic behavioral stats).
- **Response**: `ChurnPredictionResponse` (Probability, boolean prediction, confidence level, and top SHAP factors).

### `POST /predict/clv`
Predicts a customer's expected spend in BRL over the next 12 months.
- **Request Body**: `CLVPredictionRequest`.
- **Response**: `CLVPredictionResponse` (Predicted BRL value, segment tier).
