
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DEFAULT_JERK_THRESHOLD = 2.5


def calculate_jerk(samples):
    """Calculate jerk from acceleration changes over elapsed time."""
    result = samples.copy()

    if result.empty:
        result["calculated_jerk_mps3"] = pd.Series(
            index=result.index, dtype=float
        )
        return result

    required_columns = {
        "vehicle_id",
        "timestamp",
        "acceleration_mps2",
    }
    missing = required_columns - set(result.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    result["timestamp"] = pd.to_datetime(
        result["timestamp"], errors="coerce"
    )

    result = result.sort_values(
        ["vehicle_id", "timestamp"]
    ).copy()

    time_diff = (
        result.groupby("vehicle_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    acceleration_diff = (
        result.groupby("vehicle_id")["acceleration_mps2"]
        .diff()
    )

    valid_time = time_diff.where(time_diff > 0)

    result["calculated_jerk_mps3"] = (
        acceleration_diff / valid_time
    )

    result["calculated_jerk_mps3"] = (
        result["calculated_jerk_mps3"]
        .replace([np.inf, -np.inf], np.nan)
    )

    return result


def detect_rule_based(samples, threshold=DEFAULT_JERK_THRESHOLD):
    """Flag samples whose calculated jerk exceeds the threshold."""
    result = calculate_jerk(samples)

    result["rule_detected"] = (
        result["calculated_jerk_mps3"].abs() >= threshold
    ).fillna(False).astype(bool)

    return result


def detect_ml_anomalies(samples, contamination=0.05):
    """Flag statistical anomalies; these are not confirmed road defects."""
    result = calculate_jerk(samples)
    result["ml_detected"] = False

    if len(result) < 10:
        return result

    feature_columns = [
        "acceleration_mps2",
        "calculated_jerk_mps3",
    ]

    features = (
        result[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )

    model.fit(features)
    result["ml_detected"] = model.predict(features) == -1

    return result


def compare_detectors(samples):
    """Return both detector results without merging their detections."""
    rule_results = detect_rule_based(samples)
    ml_results = detect_ml_anomalies(samples)

    result = rule_results.copy()

    # Align results by row index rather than assuming identical row order.
    result["ml_detected"] = (
        ml_results["ml_detected"]
        .reindex(result.index)
        .fillna(False)
        .astype(bool)
    )

    return result
