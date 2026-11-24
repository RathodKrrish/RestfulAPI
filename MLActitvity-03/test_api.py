import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_returns_ok_and_model_loaded():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}

def test_model_info_has_features_and_classes():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert len(data["features"]) == 4
    assert len(data["classes"]) == 3

def test_valid_prediction_response_structure():
    payload = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["setosa", "versicolor", "virginica"]
    assert data["class_id"] in [0, 1, 2]
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-6

def test_missing_field_returns_validation_error():
    payload = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_out_of_range_value_returns_validation_error():
    payload = {
        "sepal_length_cm": 99,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_unknown_extra_field_is_rejected():
    payload = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
        "unexpected": "reject me"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
