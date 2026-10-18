
from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["verification_code"] == "AI-A2-7196"


def test_prediction_endpoint():
    farm = {
        "rainfall_mm": 1000,
        "avg_temp_c": 20,
        "soil_moisture_pct": 50,
        "fertilizer_kg": 120,
        "field_size_ha": 2,
        "season": "A",
        "cooperative_support": 1,
    }

    response = client.post("/predict", json=farm)

    assert response.status_code == 200

    result = response.json()
    assert result["verification_code"] == "AI-A2-7196"
    assert isinstance(result["predicted_yield_tons"], float)
    assert result["high_risk_prediction"] in [0, 1]
    assert 0 <= result["high_risk_probability"] <= 1
    assert isinstance(result["cluster"], int)


def test_invalid_season_is_rejected():
    farm = {
        "rainfall_mm": 1000,
        "avg_temp_c": 20,
        "soil_moisture_pct": 50,
        "fertilizer_kg": 120,
        "field_size_ha": 2,
        "season": "C",
        "cooperative_support": 1,
    }

    response = client.post("/predict", json=farm)

    assert response.status_code == 422
