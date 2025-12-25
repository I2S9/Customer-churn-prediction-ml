"""FastAPI application for churn prediction inference."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.paths import MODELS_DIR, REPORTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn probability",
    version="0.1.0",
)

# Global model and threshold cache
_model = None
_optimal_threshold = 0.5
_retention_cost = 10.0
_churn_loss = 100.0


class CustomerFeatures(BaseModel):
    """Customer features for prediction."""

    # RFM features
    recency_days: float = Field(..., description="Days since last event")
    frequency_30d: float = Field(..., description="Event frequency in last 30 days")
    frequency_90d: float = Field(..., description="Event frequency in last 90 days")
    monetary_30d: float = Field(..., description="Total value in last 30 days")
    monetary_90d: float = Field(..., description="Total value in last 90 days")

    # Temporal features
    events_last_7d: float = Field(default=0.0, description="Events in last 7 days")
    events_last_30d: float = Field(default=0.0, description="Events in last 30 days")
    events_last_90d: float = Field(default=0.0, description="Events in last 90 days")

    # Customer profile
    customer_tenure_days: float = Field(..., description="Days since signup")
    monthly_revenue: Optional[float] = Field(default=None, description="Monthly revenue")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "recency_days": 5.0,
                "frequency_30d": 12.0,
                "frequency_90d": 45.0,
                "monetary_30d": 150.0,
                "monetary_90d": 450.0,
                "events_last_7d": 3.0,
                "events_last_30d": 12.0,
                "events_last_90d": 45.0,
                "customer_tenure_days": 180.0,
                "monthly_revenue": 50.0,
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response."""

    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    predicted_churn: bool = Field(..., description="Binary churn prediction")
    recommended_action: str = Field(..., description="Recommended action")
    confidence: str = Field(..., description="Confidence level")
    expected_value: float = Field(..., description="Expected value of intervention")


def load_model(model_path: Optional[str | Path] = None) -> None:
    """
    Load model from disk.

    Parameters
    ----------
    model_path : str | Path, optional
        Path to model pickle file. Defaults to models/baseline.pkl.
    """
    global _model

    if model_path is None:
        model_path = MODELS_DIR / "baseline.pkl"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        _model = pickle.load(f)

    logger.info("Model loaded successfully")

    # Try to load optimal threshold from reports
    threshold_path = REPORTS_DIR / "threshold_analysis.json"
    if threshold_path.exists():
        import json

        with open(threshold_path) as f:
            threshold_data = json.load(f)
        global _optimal_threshold
        _optimal_threshold = threshold_data.get("optimal_threshold", 0.5)
        logger.info(f"Loaded optimal threshold: {_optimal_threshold:.4f}")


def load_business_params() -> None:
    """Load business parameters from evaluation reports."""
    global _retention_cost, _churn_loss

    business_path = REPORTS_DIR / "business_evaluation_default.json"
    if business_path.exists():
        import json

        with open(business_path) as f:
            business_data = json.load(f)
        scenario = business_data.get("scenario", {})
        _retention_cost = scenario.get("retention_cost_per_customer", 10.0)
        _churn_loss = scenario.get("churn_loss_per_customer", 100.0)
        logger.info(f"Loaded business params: retention_cost={_retention_cost}, churn_loss={_churn_loss}")


@app.on_event("startup")
async def startup_event():
    """Load model and parameters on startup."""
    try:
        load_model()
        load_business_params()
    except FileNotFoundError as e:
        logger.warning(f"Model not found at startup: {e}. Use /load-model endpoint to load.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")


def prepare_features(features: CustomerFeatures) -> pd.DataFrame:
    """
    Prepare features for model prediction.

    Parameters
    ----------
    features : CustomerFeatures
        Customer features.

    Returns
    -------
    pandas.DataFrame
        Prepared features DataFrame.
    """
    # Convert to dict and create DataFrame
    feature_dict = features.dict()

    # Create DataFrame with single row
    df = pd.DataFrame([feature_dict])

    # Ensure all expected columns are present (fill missing with 0)
    expected_cols = [
        "recency_days",
        "frequency_30d",
        "frequency_90d",
        "monetary_30d",
        "monetary_90d",
        "events_last_7d",
        "events_last_30d",
        "events_last_90d",
        "customer_tenure_days",
        "monthly_revenue",
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[expected_cols]


def calculate_expected_value(probability: float) -> float:
    """
    Calculate expected value of intervention.

    Parameters
    ----------
    probability : float
        Churn probability.

    Returns
    -------
    float
        Expected value (positive = worth intervening).
    """
    # Expected value = P(churn) * churn_loss - retention_cost
    expected_value = probability * _churn_loss - _retention_cost
    return expected_value


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "0.1.0",
        "endpoints": {
            "/predict": "POST - Predict churn probability",
            "/load-model": "POST - Load model from path",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    model_loaded = _model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "optimal_threshold": _optimal_threshold,
    }


@app.post("/load-model")
async def load_model_endpoint(model_path: str):
    """
    Load model from specified path.

    Parameters
    ----------
    model_path : str
        Path to model pickle file.
    """
    try:
        load_model(model_path)
        return {"status": "success", "message": f"Model loaded from {model_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CustomerFeatures) -> PredictionResponse:
    """
    Predict churn probability for a customer.

    Parameters
    ----------
    features : CustomerFeatures
        Customer features.

    Returns
    -------
    PredictionResponse
        Prediction results with recommended action.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Use /load-model endpoint or ensure model exists at models/baseline.pkl",
        )

    try:
        # Prepare features
        X = prepare_features(features)

        # Predict probability
        probability = float(_model.predict_proba(X)[0, 1])

        # Binary prediction using optimal threshold
        predicted_churn = probability >= _optimal_threshold

        # Calculate expected value
        expected_value = calculate_expected_value(probability)

        # Determine recommended action
        if predicted_churn and expected_value > 0:
            recommended_action = "intervene"
            confidence = "high" if probability > 0.7 else "medium" if probability > 0.5 else "low"
        elif predicted_churn:
            recommended_action = "monitor"
            confidence = "low"
        else:
            recommended_action = "no_action"
            confidence = "high" if probability < 0.3 else "medium"

        return PredictionResponse(
            churn_probability=probability,
            predicted_churn=predicted_churn,
            recommended_action=recommended_action,
            confidence=confidence,
            expected_value=expected_value,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(features_list: List[CustomerFeatures]) -> List[PredictionResponse]:
    """
    Predict churn probability for multiple customers.

    Parameters
    ----------
    features_list : List[CustomerFeatures]
        List of customer features.

    Returns
    -------
    List[PredictionResponse]
        List of prediction results.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Use /load-model endpoint or ensure model exists at models/baseline.pkl",
        )

    try:
        # Prepare features for all customers
        X_list = [prepare_features(f) for f in features_list]
        X = pd.concat(X_list, ignore_index=True)

        # Predict probabilities
        probabilities = _model.predict_proba(X)[:, 1]

        # Generate responses
        responses = []
        for i, prob in enumerate(probabilities):
            predicted_churn = prob >= _optimal_threshold
            expected_value = calculate_expected_value(prob)

            if predicted_churn and expected_value > 0:
                recommended_action = "intervene"
                confidence = "high" if prob > 0.7 else "medium" if prob > 0.5 else "low"
            elif predicted_churn:
                recommended_action = "monitor"
                confidence = "low"
            else:
                recommended_action = "no_action"
                confidence = "high" if prob < 0.3 else "medium"

            responses.append(
                PredictionResponse(
                    churn_probability=float(prob),
                    predicted_churn=predicted_churn,
                    recommended_action=recommended_action,
                    confidence=confidence,
                    expected_value=expected_value,
                )
            )

        return responses

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

