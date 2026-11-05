
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_detectors(detection_results):
    if "ground_truth_jerk" not in detection_results.columns:
        raise ValueError(
            "Missing ground_truth_jerk. "
            "Add ground-truth labels to simulator.py first."
        )

    actual = (
        detection_results["ground_truth_jerk"]
        .fillna(False)
        .astype(bool)
    )

    detector_columns = {
        "Rule-Based": "rule_detected",
        "Machine Learning": "ml_detected",
    }

    rows = []

    for detector_name, column in detector_columns.items():
        predicted = (
            detection_results[column]
            .fillna(False)
            .astype(bool)
        )

        tn, fp, fn, tp = confusion_matrix(
            actual,
            predicted,
            labels=[False, True],
        ).ravel()

        rows.append({
            "Detector": detector_name,
            "True Positives": int(tp),
            "False Positives": int(fp),
            "False Negatives": int(fn),
            "True Negatives": int(tn),
            "Precision": precision_score(
                actual, predicted, zero_division=0
            ),
            "Recall": recall_score(
                actual, predicted, zero_division=0
            ),
            "F1-score": f1_score(
                actual, predicted, zero_division=0
            ),
            "Accuracy": accuracy_score(actual, predicted),
        })

    return pd.DataFrame(rows)

