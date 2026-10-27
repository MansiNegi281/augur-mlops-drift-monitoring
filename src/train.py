from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# Define project directories
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model():
    # Step 1: Load the dataset
    dataset = load_breast_cancer(as_frame=True)

    # Step 2: Separate features and target
    X = dataset.data
    y = dataset.target

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Step 4: Train the Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Step 5: Evaluate the model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Step 6: Track the experiment using MLflow
    mlflow.set_experiment("Augur-Drift-Monitoring")

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

    # Step 7: Save the trained model and reference data
    joblib.dump(model, MODEL_DIR / "model.joblib")
    joblib.dump(X_train, MODEL_DIR / "reference_data.joblib")

    print("\nModel training completed!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Model saved at: {MODEL_DIR / 'model.joblib'}")
    print(f"Reference data saved at: {MODEL_DIR / 'reference_data.joblib'}")


if __name__ == "__main__":
    train_model()