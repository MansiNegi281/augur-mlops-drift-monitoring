from pathlib import Path

import joblib
from evidently import Report
from evidently.presets import DataDriftPreset


# Define project directories
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

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
    # Load the reference training data
    reference_data = joblib.load(
        MODEL_DIR / "reference_data.joblib"
    )

    # Create a sample representing incoming data
    current_data = reference_data.sample(
        n=min(200, len(reference_data)),
        random_state=42,
    ).copy()

    print("Generating baseline report...")

    # Experiment 1: No intentional changes
    generate_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        report_name="baseline_report.html",
    )

    print("\nGenerating simulated drift report...")

    # Experiment 2: Introduce an artificial feature shift
    drifted_data = current_data.copy()
    feature = drifted_data.columns[0]

    drifted_data[feature] = drifted_data[feature] + 100

    print(f"Artificially shifted feature: {feature}")

    generate_drift_report(
        reference_data=reference_data,
        current_data=drifted_data,
        report_name="drift_report.html",
    )

    print("\nBoth monitoring experiments completed!")