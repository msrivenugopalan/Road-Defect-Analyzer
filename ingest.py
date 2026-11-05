"""
Real sensor data ingestion.

Lets the SAME detector / hotspot / email pipeline run on data recorded by
a real phone (any "sensor logger" app that exports CSV) instead of the
simulator. The user maps their CSV columns to the fields the detector
needs; units are converted; gravity is removed by the detector itself.

Assumption: the phone is mounted roughly flat, so the mapped axis is
approximately vertical. Arbitrary phone orientation is NOT handled here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from geo import haversine_m

G = 9.80665

ACCEL_UNITS = {"m/s²": 1.0, "g": G, "mg": G / 1000.0}
SPEED_UNITS = {"m/s": 1.0, "km/h": 1 / 3.6, "mph": 0.44704, "knots": 0.514444}

_HINTS = {
    "timestamp": ["timestamp", "time", "datetime", "date", "epoch", "seconds"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng", "long"],
    "vertical_accel": [
        "accelerometerz", "acc_z", "accel_z", "az", "z", "vertical",
        "acceleration z", "linear_acceleration_z",
    ],
    "speed": ["speed", "velocity", "gps_speed"],
    "vehicle_id": ["vehicle", "device", "vehicle_id", "device_id"],
}


def suggest_mapping(columns) -> dict:
    """Guess which CSV column is which, from common naming conventions."""
    lowered = {c: str(c).lower().strip() for c in columns}
    mapping = {}

    for field, hints in _HINTS.items():
        for hint in hints:  # earlier hints are more specific
            exact = [c for c, l in lowered.items() if l == hint]
            partial = [c for c, l in lowered.items() if hint in l]
            match = exact or partial

            if match:
                mapping[field] = match[0]
                break

    return mapping


def _parse_time(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().mean() > 0.9:
        median = numeric.median()
        # Heuristic: epoch nanoseconds / milliseconds / seconds / relative.
        if median > 1e17:
            return pd.to_datetime(numeric, unit="ns", errors="coerce")
        if median > 1e11:
            return pd.to_datetime(numeric, unit="ms", errors="coerce")
        if median > 1e8:
            return pd.to_datetime(numeric, unit="s", errors="coerce")
        # Relative seconds since start of recording.
        base = pd.Timestamp.now().normalize()
        return base + pd.to_timedelta(numeric, unit="s")

    return pd.to_datetime(series, errors="coerce")


def standardise_sensor_dataframe(
    raw: pd.DataFrame,
    mapping: dict,
    vehicle_id: str = "REAL-001",
    accel_unit: str = "m/s²",
    speed_unit: str = "m/s",
) -> pd.DataFrame:
    """Convert a user CSV into the columns the detector expects."""
    for field in ("timestamp", "latitude", "longitude", "vertical_accel"):
        if not mapping.get(field) or mapping[field] not in raw.columns:
            raise ValueError(f"Please map a column for '{field}'.")

    if accel_unit not in ACCEL_UNITS or speed_unit not in SPEED_UNITS:
        raise ValueError("Unknown unit.")

    speed_column = mapping.get("speed")
    speed = (
        pd.to_numeric(raw[speed_column], errors="coerce")
        * SPEED_UNITS[speed_unit]
        if speed_column and speed_column in raw.columns
        else np.nan
    )

    frame = pd.DataFrame({
        "timestamp": _parse_time(raw[mapping["timestamp"]]),
        "latitude": pd.to_numeric(raw[mapping["latitude"]], errors="coerce"),
        "longitude": pd.to_numeric(raw[mapping["longitude"]], errors="coerce"),
        "sensor_vertical_acceleration_mps2": (
            pd.to_numeric(raw[mapping["vertical_accel"]], errors="coerce")
            * ACCEL_UNITS[accel_unit]
        ),
        "speed_mps": speed,  # carried through cleaning with its own row
    })

    column = mapping.get("vehicle_id")
    frame["vehicle_id"] = (
        raw[column].astype(str) if column and column in raw.columns
        else vehicle_id
    )

    required = [
        "timestamp", "latitude", "longitude",
        "sensor_vertical_acceleration_mps2",
    ]
    frame = frame.dropna(subset=required).query(
        "latitude >= -90 and latitude <= 90 "
        "and longitude >= -180 and longitude <= 180"
    )
    frame = frame.sort_values(["vehicle_id", "timestamp"]).reset_index(
        drop=True
    )

    if frame.empty:
        raise ValueError("No valid rows after cleaning the file.")

    # Derive speed from GPS when it was not provided.
    if frame["speed_mps"].isna().all():
        speeds = np.zeros(len(frame))

        for _, group in frame.groupby("vehicle_id"):
            index = group.index.to_numpy()
            seconds = group["timestamp"].diff().dt.total_seconds().to_numpy()
            metres = np.r_[
                0.0,
                haversine_m(
                    group["latitude"].to_numpy()[:-1],
                    group["longitude"].to_numpy()[:-1],
                    group["latitude"].to_numpy()[1:],
                    group["longitude"].to_numpy()[1:],
                ),
            ]
            with np.errstate(divide="ignore", invalid="ignore"):
                speeds[index] = np.where(seconds > 0, metres / seconds, 0.0)

        frame["speed_mps"] = speeds

    return frame
