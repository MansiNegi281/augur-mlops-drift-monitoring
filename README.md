End-to-End MLOps Pipeline for Data Drift Detection and Model Monitoring

1. Project Overview

This project demonstrates an end-to-end Machine Learning Operations (MLOps) workflow for training a machine learning model, tracking experiments, monitoring data drift, serving predictions through an API, testing endpoints, and packaging the application with Docker.

The project demonstrates how monitoring can help identify changes in incoming data distributions that may affect machine learning systems. It is a student implementation inspired by research on production machine learning monitoring and does not claim to reproduce a specific research paper.

2. Objectives

* Train and evaluate a machine learning classification model.
* Track model parameters and evaluation metrics using MLflow.
* Compare reference data with current data using Evidently.
* Simulate data drift by deliberately shifting a feature’s values.
* Serve predictions through a FastAPI REST API.
* Validate API behavior with automated tests.
* Containerize the prediction service using Docker.

3. Technologies Used

* Language: Python
* Data processing: pandas, NumPy
* Machine learning: scikit-learn
* Experiment tracking: MLflow
* Data drift monitoring: Evidently
* API development: FastAPI
* Testing: pytest
* Containerization: Docker

4. Dataset and Model

The project uses the Breast Cancer Wisconsin dataset available through scikit-learn.

A Random Forest classifier is trained using an 80/20 train-test split with stratification and a fixed random seed.

Model Evaluation Results

Metric	Result
Accuracy	0.9561
F1-score	0.9655

These metrics represent the recorded experiment and are not a guarantee of performance on unseen production data.

5. System Architecture

The project follows this workflow:

1. Data loading: Load the Breast Cancer Wisconsin dataset.
2. Model training: Split the data into training and testing sets and train a Random Forest classifier.
3. Experiment tracking: Record model parameters and evaluation metrics using MLflow.
4. Reference data storage: Save the reference dataset for monitoring.
5. Drift monitoring: Compare reference data with a sample of current data using Evidently.
6. Drift simulation: Add 100 to every value of the mean radius feature and generate a second report.
7. API serving: Load the trained model and expose prediction endpoints using FastAPI.
8. Automated testing: Test API availability, valid predictions, missing features, and unexpected features.
9. Containerization: Package the API, model, and runtime dependencies into a Docker image.

6. Project Structure

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
├── mlruns/
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
├── README.md
└── .gitignore

7. Installation and Setup

Prerequisites

* Python 3.9 or a compatible Python version supported by the project dependencies.
* Docker, if you want to run the containerized API.

Set up the Python environment

Clone the repository and enter the project directory:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd augur-mlops-project

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install the dependencies:

pip install -r requirements.txt

Use the project’s pinned dependency versions if they are specified in requirements.txt.

8. Running the Project

Train the model

python src/train.py

This trains the classifier, logs the experiment, and saves the model and reference data.

Generate drift reports

python src/monitor.py

The script generates two HTML reports:

* reports/baseline_report.html
* reports/drift_report.html

Open the reports on macOS:

open reports/baseline_report.html
open reports/drift_report.html

Run the FastAPI application

uvicorn src.api:app --reload

Open the interactive API documentation at:

http://127.0.0.1:8000/docs

The API provides a home endpoint and a prediction endpoint. Use the interactive documentation to inspect the request schema and test predictions.

9. Data Drift Monitoring Results

Two monitoring experiments were performed.

Experiment 1: Baseline Monitoring

The reference dataset was compared with a sample of 200 rows from the same dataset, without intentionally modifying its feature values.

The generated baseline report showed:

* 30 monitored columns.
* 0 drifted columns.
* 0% share of drifted columns.
* Overall dataset drift not detected.

Experiment 2: Simulated Feature Drift

The mean radius feature was deliberately modified by adding 100 to every value in the current dataset.

Measurement	Result
Original mean of mean radius	14.0978
Modified mean of mean radius	114.0978
Mean increase	100.0000
Total monitored columns	30
Drifted columns	1
Share of drifted columns	3.33%

The feature-level report marked mean radius as Detected.

However, the overall dataset drift status remained Not Detected, because the report’s configured dataset-level threshold was 0.5 (50%), while only 1 out of 30 columns was flagged.

This experiment demonstrates why feature-level drift results and an overall dataset-level drift decision should be interpreted separately.

The experiment uses an artificial shift to validate monitoring behavior; it does not establish that real-world data has drifted.

10. API Testing Results

Automated tests were run using:

python -m pytest -v

All four tests passed:

Test	Result
Home endpoint	Passed
Valid prediction	Passed
Missing feature validation	Passed
Unexpected feature validation	Passed

Total: 4 passed, 0 failed.

11. Docker Deployment

The API was packaged into the Docker image augur-mlops-api:latest.

A container named augur-mlops-container was started with port mapping 8001:8000.

The containerized API documentation is available at:

http://127.0.0.1:8001/docs

Use the interactive documentation to submit a valid prediction request.

12. Screenshots and Implementation Evidence

Include screenshots demonstrating:

1. Baseline drift report.
2. Simulated drift report.
3. Feature-level detection for mean radius.
4. Successful local API prediction.
5. Successful Docker-hosted API prediction.

Store the screenshots in a project folder named screenshots/ and include them in the final submission or repository if required.

13. Limitations

* The drift experiment uses an artificially shifted feature rather than naturally occurring production drift.
* The overall dataset drift decision depends on the configured detection threshold.
* Model evaluation results are based on a single recorded experiment and do not guarantee future performance.
* The project demonstrates a monitoring workflow but does not establish continuous production monitoring or automatic model retraining.

14. Conclusion

This project demonstrates a practical MLOps workflow combining model training, experiment tracking, data drift monitoring, API-based prediction, automated testing, and Docker containerization.

The simulated drift experiment successfully identified the deliberately modified mean radius feature. The project also passed all four automated API tests and demonstrated local and containerized API deployment.

The implementation provides a foundation for exploring more advanced MLOps practices, including scheduled monitoring, alerting, model versioning, and automated retraining.