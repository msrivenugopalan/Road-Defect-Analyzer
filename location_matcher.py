
import numpy as np
import pandas as pd


def distance_metres(lat1, lon1, lat2, lon2):
    """Calculate the distance between two GPS coordinates."""
    earth_radius = 6_371_000

    lat1, lon1, lat2, lon2 = np.radians(
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * np.arcsin(np.sqrt(a))


def group_events_by_location(events, radius_metres=10):
    """
    Group events within 10 metres of each group's first event.

    Each group counts distinct vehicles, not total events.
    """
    columns = [
        "location_id",
        "latitude",
        "longitude",
        "event_count",
        "distinct_vehicle_count",
        "vehicle_ids",
        "report_eligible",
    ]

    if events.empty:
        return pd.DataFrame(columns=columns)

    groups = []

    for _, event in events.iterrows():
        matched_group = None

        for group in groups:
            distance = distance_metres(
                group["anchor_latitude"],
                group["anchor_longitude"],
                event["latitude"],
                event["longitude"],
            )

            if distance <= radius_metres:
                matched_group = group
                break

        if matched_group is None:
            matched_group = {
                "anchor_latitude": float(event["latitude"]),
                "anchor_longitude": float(event["longitude"]),
                "events": [],
                "vehicles": set(),
            }
            groups.append(matched_group)

        matched_group["events"].append(event)
        matched_group["vehicles"].add(str(event["vehicle_id"]))

    results = []

    for index, group in enumerate(groups, start=1):
        group_events = pd.DataFrame(group["events"])

        results.append({
            "location_id": f"LOC-{index:03d}",
            "latitude": group_events["latitude"].mean(),
            "longitude": group_events["longitude"].mean(),
            "event_count": len(group_events),
            "distinct_vehicle_count": len(group["vehicles"]),
            "vehicle_ids": ", ".join(sorted(group["vehicles"])),
            "report_eligible": len(group["vehicles"]) >= 5,
        })

    return pd.DataFrame(results, columns=columns)
