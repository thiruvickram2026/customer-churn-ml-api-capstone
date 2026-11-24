from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_valid_payload():
    payload = {
        "tenure_months": 2.0,
        "support_tickets": 5,
        "monthly_spend_inr": 1200.0,
        "last_login_days": 28,
        "plan_type": "Basic",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert set(data.keys()) == {
        "churn_prediction",
        "churn_probability",
        "no_churn_probability",
        "risk_level",
    }
    assert data["churn_prediction"] in (0, 1)
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert 0.0 <= data["no_churn_probability"] <= 1.0
    assert data["risk_level"] in ("Low", "High")


def test_predict_invalid_plan_type():
    payload = {
        "tenure_months": 6.0,
        "support_tickets": 1,
        "monthly_spend_inr": 2500.0,
        "last_login_days": 3,
        "plan_type": "Enterprise",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
