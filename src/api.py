from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Define project directories
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"

# Load the trained model
model = joblib.load(MODEL_PATH)

# Get the expected feature names and their order
feature_names = list(model.feature_names_in_)

app = FastAPI(
    title="MLOps Model Monitoring API",
    description="Breast cancer classification with a trained ML model",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Feature names and numerical values for one prediction",
    )


@app.get("/")
def home():
    return {
        "message": "MLOps Prediction API is running",
        "features_required": len(feature_names),
        "docs": "/docs",
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    # Check that all expected features are supplied
    missing = [
        name for name in feature_names
        if name not in request.features
    ]

    extra = [
        name for name in request.features
        if name not in feature_names
    ]

    if missing or extra:
        raise HTTPException(
            status_code=400,
            detail={
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    # Arrange features in the same order used during training
    input_data = pd.DataFrame(
        [[request.features[name] for name in feature_names]],
        columns=feature_names,
    )

    prediction = int(model.predict(input_data)[0])

    result = (
        "malignant"
        if prediction == 0
        else "benign"
    )

    return {
        "prediction": prediction,
        "classification": result,
    }