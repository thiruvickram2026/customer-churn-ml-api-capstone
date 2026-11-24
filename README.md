# Customer Churn Prediction - Real-Time ML Inference API

## Overview
Real-time ML inference REST API built with FastAPI, containerized with Docker, and tested with pytest.

## Project Structure
- app/main.py: FastAPI app with /health and /predict endpoints.
- churn_model.joblib: Saved Decision Tree model and feature column order.
- customer_churn_processed.csv: Processed dataset (12 rows, 8 columns).
- Dockerfile: Container setup with pinned dependencies on port 8000.
- requirements.txt: Pinned Python package versions.
- tests/test_api.py: Unit tests for /health, /predict, and 422 errors.

## Run Tests, API, and Docker
- Run tests: PYTHONPATH=. pytest -v
- Run API: uvicorn app.main:app --host 0.0.0.0 --port 8000
- Build Docker image: docker build -t customer-churn-api .
- Run Docker container: docker run -p 8000:8000 customer-churn-api
