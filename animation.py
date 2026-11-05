"""
Vehicle animation helpers (pure functions, no Streamlit).

Frames are chosen along the FOCUS vehicle's route distance rather than at
equal time steps. That lets the animation run quickly on open road and
automatically switch to slow motion around a defect, so you can actually
see the car brake, cross it, and speed back up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulator import approach_distance_m


def build_tracks(samples: pd.DataFrame) -> dict:
    """Per-vehicle NumPy arrays for fast lookup while animating."""
    frame = samples.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    t0 = frame["timestamp"].min()

    lat_col = "true_latitude" if "true_latitude" in frame else "latitude"
    lon_col = "true_longitude" if "true_longitude" in frame else "longitude"

    tracks = {}

    for vehicle, group in frame.sort_values("timestamp").groupby("vehicle_id"):
        speed = group["speed_mps"].to_numpy(float)
        moving = speed[speed > 0.5]

        tracks[vehicle] = {
            "t": (group["timestamp"] - t0).dt.total_seconds().to_numpy(),
            "lat": group[lat_col].to_numpy(float),
            "lon": group[lon_col].to_numpy(float),
            "speed": speed,
            "dist": group["distance_along_route_m"].to_numpy(float),
            # The speed this driver holds on open road.
            "cruise": float(np.percentile(moving, 95)) if len(moving) else 1.0,
        }

    return tracks


def _defect_span(defect: dict):
    start = float(defect["distance_along_route_m"])
    length = float(defect.get("length_m", 0.0)) if (
        defect.get("defect_type") == "rough_road") else 0.0
    return start, start + length


def frame_times(
    track: dict,
    defects: list,
    coarse_step_m: float = 60.0,
    fine_step_m: float = 3.0,
    before_m: float = 70.0,
    after_m: float = 50.0,
    slow_motion: bool = True,
) -> np.ndarray:
    """Times (seconds) at which to render, spaced by focus-vehicle distance."""
    end = float(track["dist"].max())
    spans = [_defect_span(d) for d in (defects or [])]
    distances = [0.0]
    d = 0.0

    while d < end:
        near = slow_motion and any(
            s - before_m <= d <= e + after_m for s, e in spans)
        d += fine_step_m if near else coarse_step_m
        distances.append(min(d, end))

    unique_d, first = np.unique(track["dist"], return_index=True)
    return np.interp(np.array(distances), unique_d, track["t"][first])


def positions_at(tracks: dict, t: float) -> pd.DataFrame:
    """Where every active vehicle is at time t, coloured by how slow it is."""
    rows = []

    for vehicle, tr in tracks.items():
        if t < tr["t"][0] or t > tr["t"][-1]:
            continue  # not started yet, or already arrived

        i = max(0, int(np.searchsorted(tr["t"], t, side="right")) - 1)
        speed = float(tr["speed"][i])
        ratio = float(np.clip(speed / tr["cruise"], 0.0, 1.0))

        rows.append({
            "vehicle_id": vehicle,
            "latitude": float(tr["lat"][i]),
            "longitude": float(tr["lon"][i]),
            "distance_m": float(tr["dist"][i]),
            "speed_mps": speed,
            "speed_kmh": speed * 3.6,
            "cruise_mps": tr["cruise"],
            "ratio": ratio,
            # slow = red, full speed = green
            "color": [int(255 * (1 - ratio)), int(200 * ratio), 60, 255],
            "label": f"{vehicle}: {speed * 3.6:.0f} km/h",
        })

    return pd.DataFrame(rows)


def focus_status(distance_m, speed_mps, cruise_mps, defects):
    """
    Plain-English state of the followed vehicle and distance to the next
    defect ahead (None if there is none).
    """
    approach = approach_distance_m(cruise_mps)
    status, next_ahead = "Cruising at constant speed", None

    for defect in defects or []:
        start, end = _defect_span(defect)
        name = str(defect["defect_type"]).replace("_", " ")

        if distance_m < start - 3.0:
            ahead = start - distance_m
            next_ahead = ahead if next_ahead is None else min(next_ahead, ahead)

        if start - 3.0 <= distance_m <= end + 3.0:
            return f"Crossing {name}", next_ahead

        if start - approach <= distance_m < start - 3.0:
            status = f"Slowing down for {name}"
        elif (
            end + 3.0 < distance_m <= end + 120.0
            and speed_mps < 0.97 * cruise_mps
            and not status.startswith("Slowing")
        ):
            status = "Speeding back up to cruise speed"

    return status, next_ahead
