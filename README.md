# End-to-End MLOps Pipeline for Data Drift Detection and Model Monitoring

## 1. Project Overview

This project demonstrates an end-to-end Machine Learning Operations (MLOps) workflow for training a machine learning model, tracking experiments, monitoring data drift, serving predictions through an API, testing endpoints, and packaging the application with Docker.

The project is inspired by research on realistic drift detection in production machine learning systems. It is a student implementation and does not claim to reproduce the original research paper.

## 2. Objectives

- Train and evaluate a machine learning classification model.
- Track model parameters and evaluation metrics using MLflow.
- Compare reference data with current data using Evidently.
- Simulate data drift by shifting a feature's values.
- Serve predictions through a FastAPI REST API.
- Validate API behavior with automated tests.
- Containerize the prediction service using Docker.

## 3. Technologies Used

- Python
- pandas and NumPy
- scikit-learn
- MLflow
- Evidently
- FastAPI
- pytest
- Docker

## 4. Dataset and Model

The project uses the Breast Cancer Wisconsin dataset available through scikit-learn.

A Random Forest classifier is trained using an 80/20 train-test split with stratification and a fixed random seed.

The recorded evaluation results are:

| Metric | Result |
|---|---:|
| Accuracy | 0.9561 |
| F1-score | 0.9655 |

These results describe this particular experiment and are not a guarantee of performance on other data.

## 5. System Architecture

1. **Data and training:** Load the dataset, split it into training and testing sets, and train the Random Forest classifier.
2. **Experiment tracking:** Log model parameters and evaluation metrics with MLflow.
3. **Reference data:** Save reference data for subsequent monitoring.
4. **Drift monitoring:** Use Evidently to compare reference data with current data.
5. **Drift simulation:** Shift the `mean radius` feature by adding 100 to its values and generate a second monitoring report.
6. **API serving:** Load the saved model and expose prediction endpoints using FastAPI.
7. **Testing:** Use pytest and FastAPI's TestClient to check the home endpoint, valid predictions, missing features, and unexpected features.
8. **Containerization:** Package the API, model, and runtime dependencies into a Docker image.

## 6. Project Structure

```text
augur-mlops-project/
├── src/
│   ├── train.py
│   ├── monitor.py
│   └── api.py
├── models/
│   ├── model.joblib
│   └── reference_data.joblib
├── reports/
│   ├── baseline_report.html
│   └── drift_report.html
├── tests/
│   └── test_api.py
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
├── README.md
└── .gitignore
