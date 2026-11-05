"""
Hotspot engine: turns individual defect events into reportable locations.

THE RULE (from the project brief)
    A location is reported when at least `min_vehicles` DISTINCT vehicles
    have each detected a defect there.

Three details make that rule work on real data:

1. "Same coordinate" means "within a radius". GPS is never exact (phones
   are typically off by 3-10 m), so events are grouped within `radius_m`.
2. DISTINCT vehicles. One car driving over the same pothole ten times
   counts once; ten events from one car are not corroboration.
3. Old events expire (`window_days`), so a pothole repaired last year does
   not keep a location "reportable" forever.

The tracker is INCREMENTAL: events are fed in one at a time, in time order,
exactly as they would arrive from a live fleet. The moment the threshold is
crossed, `add_event` returns a trigger. The batch helper `find_hotspots`
simply replays events through the same tracker, so batch results and live
behaviour can never disagree.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from geo import haversine_m

SEVERITY_WEIGHT = {"unknown": 1, "low": 1, "medium": 2, "high": 3}
SEVERITY_ORDER = ["unknown", "low", "medium", "high"]

HOTSPOT_COLUMNS = [
    "hotspot_id",
    "latitude",
    "longitude",
    "event_count",
    "distinct_vehicle_count",
    "vehicle_ids",
    "first_seen",
    "last_seen",
    "defect_type",
    "max_severity",
    "peak_mps2",
    "spread_m",
    "elongated",
    "priority_score",
    "report_eligible",
]


class HotspotTracker:
    def __init__(self, radius_m=10.0, min_vehicles=5, window_days=30):
        if radius_m <= 0:
            raise ValueError("radius_m must be positive.")
        if min_vehicles < 1:
            raise ValueError("min_vehicles must be at least 1.")

        self.radius_m = float(radius_m)
        self.min_vehicles = int(min_vehicles)
        self.window = (
            pd.Timedelta(days=window_days) if window_days else None
        )
        self._clusters = []  # each: {"lat_sum","lon_sum","events":[...]}
        self._latest = None

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _vehicle_of(event):
        # vehicle_key is unique across runs; fall back to vehicle_id.
        key = event.get("vehicle_key")

        if key is None or (isinstance(key, float) and np.isnan(key)):
            key = event["vehicle_id"]

        return str(key)

    def _centroid(self, cluster):
        count = len(cluster["events"])
        return cluster["lat_sum"] / count, cluster["lon_sum"] / count

    def _distinct_vehicles(self, cluster, now):
        events = cluster["events"]

        if self.window is not None and now is not None:
            events = [e for e in events if e["timestamp"] >= now - self.window]

        return {self._vehicle_of(e) for e in events}

    def _merge_if_close(self, index):
        """If two clusters' centres drifted within the radius, merge them."""
        base = self._clusters[index]
        lat, lon = self._centroid(base)

        for other_index in range(len(self._clusters) - 1, -1, -1):
            if other_index == index:
                continue

            other = self._clusters[other_index]
            other_lat, other_lon = self._centroid(other)

            if haversine_m(lat, lon, other_lat, other_lon) <= self.radius_m:
                base["events"].extend(other["events"])
                base["lat_sum"] += other["lat_sum"]
                base["lon_sum"] += other["lon_sum"]
                del self._clusters[other_index]

                if other_index < index:
                    index -= 1

                base = self._clusters[index]
                lat, lon = self._centroid(base)

        return index

    # -- public API ------------------------------------------------------

    def add_event(self, event):
        """
        Add one event (a dict or Series with vehicle_id, latitude, longitude,
        timestamp, kind, severity, peak_mps2...).

        Returns a summary dict for the hotspot IF this event is the one that
        pushed it across the threshold; otherwise None.
        """
        
        event = dict(event)
        event["timestamp"] = pd.to_datetime(
            event["timestamp"], utc=True
        )

        lat, lon = float(event["latitude"]), float(event["longitude"])

        if not (np.isfinite(lat) and np.isfinite(lon)):
            return None

        
        latest_timestamp = (
            pd.to_datetime(self._latest, utc=True)
            if self._latest is not None
            else None
        )

        if (
            latest_timestamp is None
            or event["timestamp"] > latest_timestamp
        ):
            self._latest = event["timestamp"]


        target = None

        if self._clusters:
            centres = np.array(
                [self._centroid(c) for c in self._clusters]
            )
            distances = haversine_m(lat, lon, centres[:, 0], centres[:, 1])
            nearest = int(np.argmin(distances))

            if distances[nearest] <= self.radius_m:
                target = nearest

        if target is None:
            self._clusters.append(
                {"lat_sum": 0.0, "lon_sum": 0.0, "events": []}
            )
            target = len(self._clusters) - 1

        cluster = self._clusters[target]
        before = len(self._distinct_vehicles(cluster, event["timestamp"]))

        cluster["events"].append(event)
        cluster["lat_sum"] += lat
        cluster["lon_sum"] += lon

        target = self._merge_if_close(target)
        cluster = self._clusters[target]
        after = len(self._distinct_vehicles(cluster, event["timestamp"]))

        if before < self.min_vehicles <= after:
            summary = self._summarise(cluster, event["timestamp"], 0)
            summary["hotspot_id"] = None  # ranks only exist in hotspots()
            return summary

        return None

    def _summarise(self, cluster, now, rank):
        events = cluster["events"]

        if self.window is not None and now is not None:
            events = [e for e in events if e["timestamp"] >= now - self.window]

        if not events:
            return None

        lats = np.array([e["latitude"] for e in events], dtype=float)
        lons = np.array([e["longitude"] for e in events], dtype=float)

        # The median is robust to the odd badly-positioned GPS fix.
        centre_lat, centre_lon = float(np.median(lats)), float(np.median(lons))
        spread = float(
            np.max(haversine_m(centre_lat, centre_lon, lats, lons))
        )

        vehicles = sorted({self._vehicle_of(e) for e in events})
        kinds = Counter(e.get("kind", "unknown") for e in events)
        severities = [e.get("severity", "unknown") for e in events]
        max_severity = max(
            severities,
            key=lambda s: SEVERITY_ORDER.index(s)
            if s in SEVERITY_ORDER else 0,
        )
        peaks = [
            e.get("peak_mps2")
            for e in events
            if e.get("peak_mps2") is not None and pd.notna(e.get("peak_mps2"))
        ]

        return {
            "hotspot_id": f"HS-{rank + 1:03d}",
            "latitude": centre_lat,
            "longitude": centre_lon,
            "event_count": len(events),
            "distinct_vehicle_count": len(vehicles),
            "vehicle_ids": ", ".join(vehicles),
            "first_seen": min(e["timestamp"] for e in events),
            "last_seen": max(e["timestamp"] for e in events),
            "defect_type": kinds.most_common(1)[0][0],
            "max_severity": max_severity,
            "peak_mps2": float(max(peaks)) if peaks else np.nan,
            "spread_m": spread,
            # A very wide cluster usually means a long stretch of bad road
            # chained together rather than one spot.
            "elongated": bool(spread > 3 * self.radius_m),
            "priority_score": len(vehicles)
            * SEVERITY_WEIGHT.get(max_severity, 1),
            "report_eligible": len(vehicles) >= self.min_vehicles,
        }

    def hotspots(self, now=None):
        """All locations seen so far, most corroborated first."""
        
        now = (
            pd.to_datetime(now, utc=True)
            if now is not None
            else self._latest
        )


        rows = [
            summary
            for summary in (
                self._summarise(cluster, now, 0)
                for cluster in self._clusters
            )
            if summary is not None
        ]

        if not rows:
            return pd.DataFrame(columns=HOTSPOT_COLUMNS)

        frame = pd.DataFrame(rows).sort_values(
            ["report_eligible", "priority_score", "distinct_vehicle_count"],
            ascending=False,
        ).reset_index(drop=True)

        frame["hotspot_id"] = [f"HS-{i + 1:03d}" for i in range(len(frame))]

        return frame[HOTSPOT_COLUMNS]


def find_hotspots(
    events: pd.DataFrame,
    radius_m=10.0,
    min_vehicles=5,
    window_days=30,
):
    """
    Batch helper: replay an event table through the tracker in time order.

    Returns (hotspots_table, triggers) where `triggers` lists the moment
    each hotspot first crossed the threshold.
    """
    tracker = HotspotTracker(radius_m, min_vehicles, window_days)
    triggers = []

    if events is None or events.empty:
        return tracker.hotspots(), triggers

    
    ordered = events.copy()
    ordered["timestamp"] = pd.to_datetime(
        ordered["timestamp"], utc=True, errors="coerce"
    )
    ordered = ordered.dropna(subset=["timestamp"])
    ordered = ordered.sort_values("timestamp")


    for event in ordered.to_dict("records"):
        trigger = tracker.add_event(event)

        if trigger is not None:
            trigger["triggered_at"] = pd.Timestamp(event["timestamp"])
            trigger["triggered_by"] = tracker._vehicle_of(event)
            triggers.append(trigger)

    return tracker.hotspots(), triggers
