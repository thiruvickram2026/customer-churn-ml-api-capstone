import os
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "churn_model.joblib")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time ML inference service for predicting customer churn probability.",
    version="1.0.0",
)


class ChurnPredictionRequest(BaseModel):
    tenure_months: float = Field(..., ge=0, description="Customer tenure in months")
    support_tickets: int = Field(..., ge=0, description="Number of support tickets raised")
    monthly_spend_inr: float = Field(..., ge=0, description="Monthly spend in INR")
    last_login_days: int = Field(..., ge=0, description="Days since last login")
    plan_type: Literal["Basic", "Standard", "Pro"] = Field(
        ..., description="Subscription plan type"
    )


class ChurnPredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    no_churn_probability: float
    risk_level: str


def load_artifact():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(payload: ChurnPredictionRequest):
    try:
        artifact = load_artifact()
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]

        row = {
            "tenure_months": payload.tenure_months,
            "support_tickets": payload.support_tickets,
            "monthly_spend_inr": payload.monthly_spend_inr,
            "last_login_days": payload.last_login_days,
            "plan_type_Basic": 1 if payload.plan_type == "Basic" else 0,
            "plan_type_Pro": 1 if payload.plan_type == "Pro" else 0,
            "plan_type_Standard": 1 if payload.plan_type == "Standard" else 0,
        }

        X_input = pd.DataFrame([row])[feature_columns]
        probs = model.predict_proba(X_input)[0]
        no_churn_prob = float(probs[0])
        churn_prob = float(probs[1])
        pred = int(churn_prob >= 0.5)
        risk_level = "High" if churn_prob >= 0.5 else "Low"

        return ChurnPredictionResponse(
            churn_prediction=pred,
            churn_probability=round(churn_prob, 4),
            no_churn_probability=round(no_churn_prob, 4),
            risk_level=risk_level,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
