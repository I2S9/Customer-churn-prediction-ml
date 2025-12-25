# Churn Prediction API

Simple FastAPI service for churn prediction inference.

## Quick Start

### Start the API server

```bash
# From project root
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint. Returns model status.

### `POST /load-model`
Load model from specified path.

**Request:**
```json
{
  "model_path": "models/baseline.pkl"
}
```

### `POST /predict`
Predict churn probability for a single customer.

**Request:**
```json
{
  "recency_days": 5.0,
  "frequency_30d": 12.0,
  "frequency_90d": 45.0,
  "monetary_30d": 150.0,
  "monetary_90d": 450.0,
  "events_last_7d": 3.0,
  "events_last_30d": 12.0,
  "events_last_90d": 45.0,
  "customer_tenure_days": 180.0,
  "monthly_revenue": 50.0
}
```

**Response:**
```json
{
  "churn_probability": 0.65,
  "predicted_churn": true,
  "recommended_action": "intervene",
  "confidence": "medium",
  "expected_value": 55.0
}
```

### `POST /predict-batch`
Predict churn probability for multiple customers.

**Request:**
```json
[
  {
    "recency_days": 5.0,
    "frequency_30d": 12.0,
    ...
  },
  {
    "recency_days": 30.0,
    "frequency_30d": 2.0,
    ...
  }
]
```

**Response:**
```json
[
  {
    "churn_probability": 0.65,
    "predicted_churn": true,
    "recommended_action": "intervene",
    "confidence": "medium",
    "expected_value": 55.0
  },
  {
    "churn_probability": 0.25,
    "predicted_churn": false,
    "recommended_action": "no_action",
    "confidence": "high",
    "expected_value": -5.0
  }
]
```

## Example Usage

### Using curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "recency_days": 5.0,
    "frequency_30d": 12.0,
    "frequency_90d": 45.0,
    "monetary_30d": 150.0,
    "monetary_90d": 450.0,
    "events_last_7d": 3.0,
    "events_last_30d": 12.0,
    "events_last_90d": 45.0,
    "customer_tenure_days": 180.0,
    "monthly_revenue": 50.0
  }'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "recency_days": 5.0,
        "frequency_30d": 12.0,
        "frequency_90d": 45.0,
        "monetary_30d": 150.0,
        "monetary_90d": 450.0,
        "events_last_7d": 3.0,
        "events_last_30d": 12.0,
        "events_last_90d": 45.0,
        "customer_tenure_days": 180.0,
        "monthly_revenue": 50.0
    }
)

result = response.json()
print(f"Churn probability: {result['churn_probability']:.2%}")
print(f"Recommended action: {result['recommended_action']}")
```

## Recommended Actions

- **intervene**: High churn risk and positive expected value. Recommend retention intervention.
- **monitor**: High churn risk but negative expected value. Monitor closely but don't intervene yet.
- **no_action**: Low churn risk. No action needed.

## Notes

- The API automatically loads the model from `models/baseline.pkl` on startup if available.
- Optimal threshold is loaded from `reports/threshold_analysis.json` if available.
- Business parameters (retention cost, churn loss) are loaded from `reports/business_evaluation_default.json` if available.
- If model is not found at startup, use `/load-model` endpoint to load it manually.

