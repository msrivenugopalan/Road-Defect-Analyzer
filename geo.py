"""Small geographic helpers shared by the whole project."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in metres. Works on scalars or NumPy arrays
    (broadcasting), so one call can compare a point with many points.
    """
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(value, dtype=float))
        for value in (lat1, lon1, lat2, lon2)
    )

    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )

    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def offset_latlon(lat, lon, north_m, east_m):
    """Shift a coordinate by a number of metres north and east."""
    lat = np.asarray(lat, dtype=float)
    dlat = np.degrees(np.asarray(north_m, dtype=float) / EARTH_RADIUS_M)
    dlon = np.degrees(
        np.asarray(east_m, dtype=float)
        / (EARTH_RADIUS_M * np.cos(np.radians(lat)))
    )
    return lat + dlat, np.asarray(lon, dtype=float) + dlon


def maps_url(lat: float, lon: float) -> str:
    """Link that opens a coordinate in Google Maps."""
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"


def osm_url(lat: float, lon: float) -> str:
    """Link that opens a coordinate in OpenStreetMap."""
    return (
        f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}"
        f"#map=19/{lat:.6f}/{lon:.6f}"
    )
