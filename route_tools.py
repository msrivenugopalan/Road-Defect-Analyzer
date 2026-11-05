"""
Route geometry helpers for click-to-place defects.

A route is just a list of coordinates. RouteIndex adds the one thing the
UI needs: a way to go between "metres along the route" and a map position,
so a click on the road can become a defect at `distance_along_route_m`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from geo import haversine_m


class RouteIndex:
    def __init__(self, route: dict):
        coordinates = route["coordinates"]

        if len(coordinates) < 2:
            raise ValueError("A route needs at least two points.")

        self.lat = np.array([c["latitude"] for c in coordinates], float)
        self.lon = np.array([c["longitude"] for c in coordinates], float)

        segments = haversine_m(
            self.lat[:-1], self.lon[:-1], self.lat[1:], self.lon[1:]
        )
        self.cumulative = np.concatenate([[0.0], np.cumsum(segments)])
        self.total_m = float(self.cumulative[-1])

        if self.total_m <= 0:
            raise ValueError("The route has zero length.")

    def position_at(self, distance_m):
        """(latitude, longitude) at a distance along the route. Vectorised."""
        d = np.clip(np.asarray(distance_m, float), 0.0, self.total_m)
        return (
            np.interp(d, self.cumulative, self.lat),
            np.interp(d, self.cumulative, self.lon),
        )

    def points(self, step_m: float = 20.0) -> pd.DataFrame:
        """Evenly spaced points on the road: the targets a click can hit."""
        distances = np.unique(
            np.append(np.arange(0.0, self.total_m, step_m), self.total_m)
        )
        lat, lon = self.position_at(distances)

        return pd.DataFrame({
            "distance_m": distances,
            "latitude": lat,
            "longitude": lon,
            "label": [f"{d:,.0f} m along route" for d in distances],
        })

    def segment(self, start_m: float, end_m: float) -> list:
        """[[lon, lat], ...] for the part of the route between two distances."""
        start_m = float(np.clip(start_m, 0.0, self.total_m))
        end_m = float(np.clip(end_m, start_m, self.total_m))

        inside = (self.cumulative > start_m) & (self.cumulative < end_m)
        distances = np.concatenate(
            [[start_m], self.cumulative[inside], [end_m]]
        )
        lat, lon = self.position_at(distances)

        return [[float(b), float(a)] for a, b in zip(lat, lon)]
