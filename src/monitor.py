from pathlib import Path

import joblib
from evidently import Report
from evidently.presets import DataDriftPreset


# Define project directories
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# Create reports directory if it does not exist
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def generate_drift_report(reference_data, current_data, report_name):
    """Compare reference and current data and save an HTML report."""

    report = Report(
        metrics=[DataDriftPreset()]
    )

    snapshot = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    report_path = REPORT_DIR / report_name
    snapshot.save_html(str(report_path))

    print(f"Report saved: {report_path}")


if __name__ == "__main__":

    # Load reference training data
    reference_data = joblib.load(
        MODEL_DIR / "reference_data.joblib"
    )

    print("Reference dataset shape:", reference_data.shape)

    # Create a sample representing incoming production data
    current_data = reference_data.sample(
        n=min(200, len(reference_data)),
        random_state=42,
    ).copy()

    print("Current dataset shape:", current_data.shape)

    # Experiment 1: Generate baseline report without intentional drift
    print("\nGenerating baseline report...")

    generate_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        report_name="baseline_report.html",
    )

    # Experiment 2: Introduce artificial data drift
    print("\nGenerating simulated drift report...")

    drifted_data = current_data.copy()

    # Explicitly select the feature to modify
    feature = "mean radius"

    if feature not in drifted_data.columns:
        raise ValueError(
            f"Feature '{feature}' was not found in the dataset. "
            f"Available columns: {list(drifted_data.columns)}"
        )

    # Record the original mean
    original_mean = drifted_data[feature].mean()

    # Shift the feature values by 100
    drifted_data[feature] = drifted_data[feature] + 100

    # Record the shifted mean
    shifted_mean = drifted_data[feature].mean()

    print(f"Feature modified: {feature}")
    print(f"Original mean: {original_mean:.4f}")
    print(f"Shifted mean: {shifted_mean:.4f}")
    print(f"Mean increase: {shifted_mean - original_mean:.4f}")

    # Generate drift report
    generate_drift_report(
        reference_data=reference_data,
        current_data=drifted_data,
        report_name="drift_report.html",
    )

    print("\nMonitoring experiments completed successfully!")
    print("Baseline report: reports/baseline_report.html")
    print("Drift report: reports/drift_report.html")