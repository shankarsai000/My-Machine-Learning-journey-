"""API tests for the Breast Cancer prediction service."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLE_FEATURES = [
    17.99,
    10.38,
    122.8,
    1001.0,
    0.1184,
    0.2776,
    0.3001,
    0.1471,
    0.2419,
    0.07871,
    1.095,
    0.9053,
    8.589,
    153.4,
    0.006399,
    0.04904,
    0.05373,
    0.01587,
    0.03003,
    0.006193,
    25.38,
    17.33,
    184.6,
    2019.0,
    0.1622,
    0.6656,
    0.7119,
    0.2654,
    0.4601,
    0.1189,
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_metadata_endpoint(client):
    response = client.get("/metadata")

    assert response.status_code == 200
    assert response.json()["number_of_features"] == 30


def test_prediction_endpoint(client):
    response = client.post(
        "/predict", json={"features": SAMPLE_FEATURES}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_class"] in [0, 1]
    assert result["predicted_label"] in ["malignant", "benign"]
    assert set(result["probabilities"]) == {"malignant", "benign"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


def test_invalid_feature_count(client):
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]},
    )

    assert response.status_code == 422
