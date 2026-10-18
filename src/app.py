
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Load the trained models
regression_model = joblib.load(
    MODELS_DIR / "regression_model.joblib"
)

classification_model = joblib.load(
    MODELS_DIR / "classification_model.joblib"
)

clustering_model = joblib.load(
    MODELS_DIR / "clustering_model.joblib"
)

clustering_preprocessor = joblib.load(
    MODELS_DIR / "clustering_preprocessor.joblib"
)

# Create the API
app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description="Predict farm yield, high-risk status, and farm cluster.",
    version="1.0.0",
)

VERIFICATION_CODE = "AI-A2-7196"


class FarmInput(BaseModel):
    rainfall_mm: float = Field(ge=0)
    avg_temp_c: float
    soil_moisture_pct: float = Field(ge=0, le=100)
    fertilizer_kg: float = Field(ge=0)
    field_size_ha: float = Field(gt=0)
    season: str = Field(pattern="^[AB]$")
    cooperative_support: int = Field(ge=0, le=1)



@app.get("/")
def home():
    return {
        "service": "Musanze Harvest Intelligence Service",
        "verification_code": VERIFICATION_CODE,
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "verification_code": VERIFICATION_CODE,
    }


@app.get("/model-info")
def model_info():
    return {
        "service": "Musanze Harvest Intelligence Service",
        "verification_code": VERIFICATION_CODE,
        "regression_model": "LinearRegression",
        "classification_model": "LogisticRegression",
        "clustering_model": "KMeans",
        "selected_clusters": 2,
        "dataset_type": "Synthetic classroom dataset",
    }



@app.post("/predict")
def predict_farm(farm: FarmInput):
    # Convert the submitted farm information into a DataFrame
    farm_data = pd.DataFrame([farm.model_dump()])

    # Regression prediction
    predicted_yield = float(
        regression_model.predict(farm_data)[0]
    )

    # Classification prediction
    high_risk = int(
        classification_model.predict(farm_data)[0]
    )

    risk_probability = float(
        classification_model.predict_proba(farm_data)[0][1]
    )

    # Clustering preprocessing must match training feature order
    cluster_features = farm_data[
        [
            "rainfall_mm",
            "avg_temp_c",
            "soil_moisture_pct",
            "fertilizer_kg",
            "field_size_ha",
            "cooperative_support",
            "season",
        ]
    ].copy()

    cluster_features["season"] = (
        cluster_features["season"].map({"A": 0, "B": 1})
    )

    transformed_features = clustering_preprocessor.transform(
        cluster_features
    )

    cluster = int(
        clustering_model.predict(transformed_features)[0]
    )

    return {
        "verification_code": VERIFICATION_CODE,
        "predicted_yield_tons": round(predicted_yield, 2),
        "high_risk_prediction": high_risk,
        "high_risk_probability": round(risk_probability, 4),
        "cluster": cluster,
        "interpretation": {
            "high_risk_prediction": (
                "High risk" if high_risk == 1 else "Lower risk"
            ),
            "cluster_note": (
                "Cluster labels identify groups; they are not risk levels."
            ),
        },
    }
