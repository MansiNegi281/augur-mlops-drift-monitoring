from fastapi.testclient import TestClient

from src.api import app, feature_names

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["features_required"] == 30


def test_valid_prediction():
    # Supply every feature expected by the model.
    features = {name: 0.0 for name in feature_names}

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]
    assert response.json()["classification"] in [
        "malignant",
        "benign",
    ]


def test_missing_feature():
    # An empty feature dictionary should be rejected.
    response = client.post(
        "/predict",
        json={"features": {}},
    )

    assert response.status_code == 400
    assert "missing_features" in response.json()["detail"]


def test_unexpected_feature():
    features = {name: 0.0 for name in feature_names}
    features["unexpected_feature"] = 1.0

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 400
    assert "unexpected_feature" in response.json()["detail"][
        "unexpected_features"
    ]