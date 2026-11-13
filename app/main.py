from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import HeartFeatures, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "heart_model.joblib"

# Load the trained model once when the app starts
model = joblib.load(MODEL_PATH)

# Feature order is the same as the training data columns
FEATURES = list(HeartFeatures.model_fields.keys())

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the presence of heart disease from patient features.",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "model_type": type(model).__name__,
        "n_features": len(FEATURES),
        "features": FEATURES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HeartFeatures):
    row = pd.DataFrame([features.model_dump()])[FEATURES]
    prediction = int(model.predict(row)[0])
    return PredictionResponse(heart_disease=bool(prediction))