End-to-End MLOps Pipeline for Data Drift Detection and Model Monitoring

An end-to-end MLOps project demonstrating machine learning model training, experiment tracking, data drift detection, REST API deployment, automated testing, and Docker containerization.

GitHub Repository: augur-mlops-drift-monitoring

⸻

Table of Contents

* Project Overview
* Objectives
* Technologies Used
* Dataset and Model
* System Architecture
* Project Structure
* Installation and Setup
* Running the Project
* Data Drift Monitoring
* Model Evaluation Results
* API Testing
* Docker Deployment
* Screenshots
* Limitations
* Future Scope
* Conclusion

Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) workflow for training, evaluating, monitoring, and deploying a machine learning model.

The project explores how data drift monitoring can help identify changes in incoming data distributions. It uses Evidently to compare reference data with current data and demonstrates feature-level drift detection through an intentionally modified dataset.

The application also integrates MLflow for experiment tracking, FastAPI for model serving, pytest for automated testing, and Docker for containerized deployment.

This is an educational implementation inspired by production MLOps practices. It does not claim to reproduce a specific research paper.

Objectives

* Train and evaluate a machine learning classification model.
* Track model parameters and evaluation metrics using MLflow.
* Monitor data distributions using Evidently.
* Simulate data drift by deliberately shifting a feature.
* Generate HTML reports for baseline and simulated drift experiments.
* Serve predictions through a FastAPI REST API.
* Validate API endpoints and input handling with automated tests.
* Containerize the prediction service using Docker.

Technologies Used

Category	Technologies
Programming Language	Python
Data Processing	pandas, NumPy
Machine Learning	scikit-learn
Experiment Tracking	MLflow
Data Drift Monitoring	Evidently
API Development	FastAPI
Automated Testing	pytest
Model Serialization	Joblib
Containerization	Docker
Version Control	Git, GitHub

Dataset and Model

The project uses the Breast Cancer Wisconsin dataset available through scikit-learn.

A Random Forest classifier is trained using an 80/20 train-test split with stratification and a fixed random seed.

The trained model is serialized and saved for use by the prediction API.

Model Evaluation Metrics

Metric	Result
Accuracy	0.9561
F1-score	0.9655

These values represent the recorded training and evaluation experiment. They are not a guarantee of performance on future or production data.

System Architecture

The project follows this workflow:

1. Data Loading: Load the Breast Cancer Wisconsin dataset.
2. Model Training: Split the dataset into training and testing sets and train a Random Forest classifier.
3. Experiment Tracking: Log model parameters and evaluation metrics using MLflow.
4. Reference Data Storage: Save reference data for subsequent monitoring.
5. Data Drift Monitoring: Compare reference data with a sample of current data using Evidently.
6. Drift Simulation: Add 100 to every value of the mean radius feature to simulate a distribution shift.
7. Report Generation: Generate baseline and simulated drift HTML reports.
8. API Development: Serve predictions through FastAPI.
9. Automated Testing: Test API availability, valid predictions, and input validation.
10. Containerization: Package the prediction service and its dependencies into a Docker image.

Project Structure

augur-mlops-drift-monitoring/
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
├── screenshots/
│   ├── baseline_drift.png
│   ├── simulated_drift.png
│   ├── feature_drift_detection.png
│   ├── local_api_prediction.png
│   └── docker_api_prediction.png
├── tests/
│   └── test_api.py
├── mlruns/
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
├── README.md
└── .gitignore

Note: The screenshot filenames above are suggested names. Make sure they match the files you actually save in your screenshots/ folder. The venv/ directory is excluded from the repository.

Installation and Setup

Prerequisites

* Python 3.9 or a compatible Python version supported by the project dependencies.
* Git.
* Docker, if you want to run the containerized application.

1. Clone the Repository

git clone https://github.com/MansiNegi281/augur-mlops-drift-monitoring.git
cd augur-mlops-drift-monitoring

2. Create a Virtual Environment

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

Use the dependency versions specified in the project requirements files.

Running the Project

1. Train the Model

python src/train.py

This script trains the Random Forest classifier, evaluates its performance, logs the experiment, and saves the trained model and reference data.

2. Generate Data Drift Reports

python src/monitor.py

The script generates two reports:

* reports/baseline_report.html
* reports/drift_report.html

On macOS, open the reports using:

open reports/baseline_report.html
open reports/drift_report.html

3. Run the FastAPI Application

uvicorn src.api:app --reload

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Use the documentation to inspect the available endpoints and submit prediction requests.

4. Run Automated Tests

python -m pytest -v

The tests validate the API endpoints and input handling.

Data Drift Monitoring

Data drift occurs when the distribution of incoming data differs from the distribution of the reference data used during model development.

This project uses Evidently to compare reference data with current data and generate HTML monitoring reports.

Experiment 1: Baseline Monitoring

The reference dataset was compared with a sample of 200 rows without intentionally modifying its feature values.

The generated report showed:

Measurement	Result
Reference dataset size	455 rows
Current dataset size	200 rows
Monitored features	30
Drifted features	0
Share of drifted features	0%
Overall dataset drift	Not detected

Experiment 2: Simulated Feature Drift

To test the monitoring system, the mean radius feature was deliberately modified by adding 100 to each value in the current dataset.

The resulting feature means were:

Measurement	Result
Original mean	14.0978
Modified mean	114.0978
Mean increase	100.0000
Total monitored features	30
Drifted features	1
Share of drifted features	3.33%
mean radius status	Detected
Overall dataset drift	Not detected

The feature-level report identified mean radius as drifted.

The overall dataset-level status remained Not Detected because the configured threshold was 0.5 (50%), while only 1 of the 30 features was flagged.

This demonstrates the distinction between detecting drift in an individual feature and triggering an overall dataset-level drift decision.

The shift was artificially introduced to validate the monitoring workflow. It does not demonstrate that real-world production data has drifted.

Model Evaluation Results

The recorded model evaluation metrics were:

Metric	Score
Accuracy	95.61%
F1-score	96.55%

Accuracy measures the proportion of predictions that were correct. The F1-score combines precision and recall into a single metric.

These scores describe the recorded experiment and should be interpreted in the context of its dataset, split, and evaluation configuration.

API Testing

Automated tests were executed using pytest.

python -m pytest -v

Test Results

Test Case	Result
Home endpoint	Passed
Valid prediction	Passed
Missing feature validation	Passed
Unexpected feature validation	Passed

Summary: 4 tests passed, 0 failed.

Docker Deployment

The FastAPI application was packaged into the Docker image:

augur-mlops-api:latest

A container named augur-mlops-container was run with the following port mapping:

8001:8000

This maps port 8001 on the host machine to port 8000 inside the container.

Run the Container

docker run -d --name augur-mlops-container -p 8001:8000 augur-mlops-api:latest

Access the Containerized API

Open:

http://127.0.0.1:8001/docs

Use the interactive documentation to test the prediction endpoint.

If the container is already running, do not start another container with the same name.

Screenshots

The screenshots/ directory contains implementation evidence, including:

1. Baseline data drift report.
2. Simulated data drift report.
3. Feature-level detection for mean radius.
4. Successful local API prediction.
5. Successful Docker-hosted API prediction.

The screenshots demonstrate the monitoring results and API deployment workflow.

Limitations

* The drift experiment uses an artificial feature shift rather than naturally occurring production drift.
* The overall dataset-level drift decision depends on the configured threshold.
* Model evaluation results do not guarantee performance on unseen production data.
* The project demonstrates a monitoring workflow but does not implement continuous production monitoring or automatic model retraining.
* The API and monitoring workflow have been tested locally and through Docker, but this does not establish production-scale reliability.

Future Scope

Potential improvements include:

* Scheduled data drift monitoring.
* Automated alerts when important features drift.
* Model versioning and experiment comparison.
* Monitoring prediction quality as ground-truth labels become available.
* CI/CD pipelines for automated testing and deployment.
* Automated model retraining and validation.
* Deployment to a cloud environment.
* Additional monitoring experiments involving different drift patterns.

Conclusion

This project demonstrates an end-to-end MLOps workflow combining model training, experiment tracking, data drift monitoring, API-based predictions, automated testing, and Docker containerization.

The simulated drift experiment successfully identified the deliberately modified mean radius feature, and all four automated API tests passed.

The implementation provides a foundation for further exploration of production-oriented MLOps practices, including continuous monitoring, alerting, model versioning, and automated retraining.