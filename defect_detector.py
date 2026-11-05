"""
Road-defect detector based on VERTICAL acceleration.

Why vertical? A pothole or a rough surface shakes the car up and down.
Speeding up or braking (what the old jerk detector measured) is a
different thing and is caused by the driver, not the road.

Two defect kinds are recognised:

pothole     a short, sharp spike that is both
            (a) above an absolute level, and
            (b) several times larger than the local road texture.
            Condition (b) stops every bump on an already rough road from
            being called a pothole.

rough_road  sustained vibration, measured as the rolling MEDIAN of the
            absolute vertical acceleration. The median ignores single
            spikes, so a lone pothole does not make a road "rough".

The detector outputs ONE event per vehicle per pass over a defect (not one
per sample), which is what the hotspot engine needs to count vehicles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DetectionConfig

EVENT_COLUMNS = [
    "vehicle_id",
    "kind",
    "timestamp",
    "start_time",
    "end_time",
    "latitude",
    "longitude",
    "peak_mps2",
    "rms_mps2",
    "severity",
    "speed_mps",
    "duration_s",
    "n_samples",
    "distance_along_route_m",
    "detector",
]

REQUIRED_COLUMNS = {
    "vehicle_id",
    "timestamp",
    "latitude",
    "longitude",
    "sensor_vertical_acceleration_mps2",
}


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _odd_window(window_s: float, dt: float) -> int:
    """Convert a window in seconds to an odd number of samples (>= 3)."""
    samples = max(3, int(round(window_s / max(dt, 1e-3))))
    return samples if samples % 2 == 1 else samples + 1


def _group_runs(mask: np.ndarray, seconds: np.ndarray, merge_gap_s: float):
    """
    Split the True positions of `mask` into runs. Two flagged samples
    belong to the same run if they are at most `merge_gap_s` apart.
    """
    indices = np.flatnonzero(mask)

    if indices.size == 0:
        return []

    breaks = np.flatnonzero(np.diff(seconds[indices]) > merge_gap_s) + 1

    return np.split(indices, breaks)


def _prepare(samples: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(samples.columns)

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    frame = samples.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(
        subset=[
            "timestamp",
            "latitude",
            "longitude",
            "sensor_vertical_acceleration_mps2",
        ]
    )

    return frame.sort_values(["vehicle_id", "timestamp"]).reset_index(
        drop=True
    )


def _severity_for_peak(peak: float, cfg: DetectionConfig) -> str:
    if peak >= cfg.high_peak_mps2:
        return "high"
    if peak >= cfg.medium_peak_mps2:
        return "medium"
    return "low"


def _severity_for_sigma(sigma: float, cfg: DetectionConfig) -> str:
    if sigma >= cfg.rough_high_sigma_mps2:
        return "high"
    if sigma >= cfg.rough_medium_sigma_mps2:
        return "medium"
    return "low"


def _make_event(group, vehicle, kind, value, severity, detector, peak_at=None):
    """Build one event row from the sample indices of a pass."""
    rows = vehicle.iloc[group]
    anchor = vehicle.iloc[peak_at] if peak_at is not None else None

    if anchor is not None:
        latitude = float(anchor["latitude"])
        longitude = float(anchor["longitude"])
        timestamp = anchor["timestamp"]
        speed = anchor.get("speed_mps", np.nan)
        distance = anchor.get("distance_along_route_m", np.nan)
    else:
        # A stretch of road: describe it by its middle.
        latitude = float(rows["latitude"].median())
        longitude = float(rows["longitude"].median())
        middle = rows.iloc[len(rows) // 2]
        timestamp = middle["timestamp"]
        speed = rows["speed_mps"].mean() if "speed_mps" in rows else np.nan
        distance = (
            middle.get("distance_along_route_m", np.nan)
        )

    start, end = rows["timestamp"].iloc[0], rows["timestamp"].iloc[-1]

    return {
        "vehicle_id": rows["vehicle_id"].iloc[0],
        "kind": kind,
        "timestamp": timestamp,
        "start_time": start,
        "end_time": end,
        "latitude": latitude,
        "longitude": longitude,
        "peak_mps2": float(value) if kind == "pothole" else np.nan,
        "rms_mps2": float(value) if kind == "rough_road" else np.nan,
        "severity": severity,
        "speed_mps": float(speed) if pd.notna(speed) else np.nan,
        "duration_s": float((end - start).total_seconds()),
        "n_samples": int(len(rows)),
        "distance_along_route_m": (
            float(distance) if pd.notna(distance) else np.nan
        ),
        "detector": detector,
    }


def analyse_samples(samples: pd.DataFrame, cfg: DetectionConfig | None = None):
    """
    Run the detector.

    Returns (annotated_samples, events):
      annotated_samples  the input rows plus the intermediate signals
                         (vertical_dynamic_mps2, local_noise_mps2, flags)
                         so the dashboard can show WHY something was flagged.
      events             one row per vehicle per detected defect pass.
    """
    cfg = cfg or DetectionConfig()
    frame = _prepare(samples)

    if frame.empty:
        annotated = frame.copy()
        for column in (
            "vertical_dynamic_mps2",
            "local_noise_mps2",
            "local_roughness_mps2",
        ):
            annotated[column] = pd.Series(dtype=float)
        annotated["pothole_flag"] = pd.Series(dtype=bool)
        annotated["rough_flag"] = pd.Series(dtype=bool)
        return annotated, _empty_events()

    dynamic = np.zeros(len(frame))
    noise = np.zeros(len(frame))
    roughness = np.zeros(len(frame))
    pothole_flag = np.zeros(len(frame), dtype=bool)
    rough_flag = np.zeros(len(frame), dtype=bool)

    events = []

    for _, vehicle in frame.groupby("vehicle_id", sort=False):
        positions = frame.index.get_indexer(vehicle.index)
        vehicle = vehicle.reset_index(drop=True)

        seconds = (
            vehicle["timestamp"] - vehicle["timestamp"].iloc[0]
        ).dt.total_seconds().to_numpy()

        if len(vehicle) < 3:
            continue

        dt = float(np.median(np.diff(seconds))) or 0.2
        window = _odd_window(cfg.baseline_window_s, dt)

        raw = vehicle["sensor_vertical_acceleration_mps2"].astype(float)

        # 1. Remove gravity / slow drift: subtract the rolling median.
        baseline = raw.rolling(window, center=True, min_periods=1).median()
        dyn = (raw - baseline).to_numpy()
        absolute = pd.Series(np.abs(dyn))

        # 2. Short-window median of |dyn|: used to find sustained roughness.
        rolling_median_abs = absolute.rolling(
            _odd_window(cfg.rough_window_s, dt), center=True, min_periods=1
        ).median()

        # 3. Local road texture over a LONGER window. 1.4826 * median(|x|)
        #    estimates sigma and is not fooled by a single spike; the long
        #    window stops a pothole inflating its own noise estimate.
        noise_window = _odd_window(cfg.noise_window_s, dt)
        local_sigma = np.maximum(
            1.4826
            * absolute.rolling(
                noise_window, center=True, min_periods=1
            ).median().to_numpy(),
            cfg.min_noise_floor_mps2,
        )

        # 4. Potholes: big AND prominent against local texture.
        pothole_level = np.maximum(
            cfg.pothole_peak_threshold_mps2,
            cfg.pothole_prominence_k * local_sigma,
        )
        pothole_mask = absolute.to_numpy() >= pothole_level

        # 5. Rough road: sustained vibration (rolling median of |dyn|).
        rough_mask = (
            rolling_median_abs.to_numpy()
            >= cfg.rough_median_threshold_mps2
        )

        dynamic[positions] = dyn
        noise[positions] = local_sigma
        roughness[positions] = rolling_median_abs.to_numpy()
        pothole_flag[positions] = pothole_mask
        rough_flag[positions] = rough_mask

        # Rough stretches first, so potholes inside them can be suppressed.
        rough_spans = []

        for group in _group_runs(rough_mask, seconds, cfg.merge_gap_s):
            duration = seconds[group[-1]] - seconds[group[0]]

            if duration < cfg.min_rough_duration_s:
                continue

            sigma = float(np.median(local_sigma[group]))
            rough_spans.append((seconds[group[0]], seconds[group[-1]]))

            events.append(
                _make_event(
                    group, vehicle, "rough_road", sigma,
                    _severity_for_sigma(sigma, cfg),
                    "vertical_accel",
                )
            )

        # A bump inside (or at the ragged edge of) a rough stretch cannot be
        # told apart from the stretch itself, so it is not reported as a
        # separate pothole. The stretch is still reported.
        pad = cfg.baseline_window_s

        for group in _group_runs(pothole_mask, seconds, cfg.merge_gap_s):
            peak_at = int(group[np.argmax(absolute.to_numpy()[group])])
            peak_time = seconds[peak_at]

            if any(lo - pad <= peak_time <= hi + pad for lo, hi in rough_spans):
                continue

            peak = float(absolute.iloc[peak_at])

            events.append(
                _make_event(
                    group, vehicle, "pothole", peak,
                    _severity_for_peak(peak, cfg),
                    "vertical_accel", peak_at=peak_at,
                )
            )

    annotated = frame.copy()
    annotated["vertical_dynamic_mps2"] = dynamic
    annotated["local_noise_mps2"] = noise
    annotated["local_roughness_mps2"] = roughness
    annotated["pothole_flag"] = pothole_flag
    annotated["rough_flag"] = rough_flag

    if not events:
        return annotated, _empty_events()

    result = (
        pd.DataFrame(events, columns=EVENT_COLUMNS)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return annotated, result


def detect_defects(samples: pd.DataFrame, cfg: DetectionConfig | None = None):
    """Convenience wrapper that returns only the event table."""
    return analyse_samples(samples, cfg)[1]


def events_from_flags(
    samples: pd.DataFrame,
    flag_column: str,
    value_column: str,
    detector_name: str,
    merge_gap_s: float = 1.0,
) -> pd.DataFrame:
    """
    Turn a per-sample boolean flag (as produced by the legacy jerk rule or
    IsolationForest detectors) into one event per pass, so that every
    detector can be compared fairly on the same footing.
    """
    frame = samples.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "latitude", "longitude"])
    frame = frame.sort_values(["vehicle_id", "timestamp"])

    events = []

    for _, vehicle in frame.groupby("vehicle_id", sort=False):
        vehicle = vehicle.reset_index(drop=True)
        seconds = (
            vehicle["timestamp"] - vehicle["timestamp"].iloc[0]
        ).dt.total_seconds().to_numpy()
        mask = vehicle[flag_column].fillna(False).astype(bool).to_numpy()
        values = vehicle[value_column].abs().fillna(0).to_numpy()

        for group in _group_runs(mask, seconds, merge_gap_s):
            peak_at = int(group[np.argmax(values[group])])

            events.append(
                _make_event(
                    group, vehicle, "pothole", values[peak_at],
                    "unknown", detector_name, peak_at=peak_at,
                )
            )

    if not events:
        return _empty_events()

    return pd.DataFrame(events, columns=EVENT_COLUMNS)
