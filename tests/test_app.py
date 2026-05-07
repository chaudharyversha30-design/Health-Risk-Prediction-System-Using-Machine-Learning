from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
webapp = importlib.import_module("app.app")


@pytest.fixture()
def client(tmp_path):
    webapp.DB_PATH = tmp_path / "test_predictions.db"
    webapp.init_db()
    webapp.app.config["TESTING"] = True
    with webapp.app.test_client() as test_client:
        yield test_client


def valid_payload():
    return {
        "age": 57,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 236,
        "fbs": 0,
        "restecg": 1,
        "thalach": 174,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 2,
        "ca": 1,
        "thal": 3,
    }


def test_predict_route_success(client):
    response = client.post("/predict", data=valid_payload())
    assert response.status_code == 200
    assert b"Risk Assessment" in response.data


def test_api_predict_missing_field_returns_400(client):
    payload = valid_payload()
    payload.pop("age")
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400


def test_validation_rejects_negative_values():
    with pytest.raises(ValueError):
        webapp.validate_features({**valid_payload(), "trestbps": -1})


def test_prediction_history_logging(client):
    response = client.post("/api/predict", json=valid_payload())
    assert response.status_code == 200
    history_page = client.get("/history")
    assert history_page.status_code == 200
    assert b"At Risk" in history_page.data or b"Not At Risk" in history_page.data
