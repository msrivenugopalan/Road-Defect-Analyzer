
import pandas as pd
from hotspot_engine import find_hotspots

# Five different vehicles report near the same location.
events = pd.DataFrame([
    {
        "vehicle_id": f"VEH-{i:03d}",
        "timestamp": f"2026-10-20 10:00:0{i}",
        "latitude": 13.001000 + i * 0.000001,
        "longitude": 80.001000 + i * 0.000001,
        "kind": "pothole",
        "severity": "high",
        "peak_mps2": 15.2,
    }
    for i in range(1, 5)
])

hotspots, triggers = find_hotspots(
    events,
    radius_m=10.0,
    min_vehicles=5,
    window_days=30,
)

print(hotspots[
    ["distinct_vehicle_count", "report_eligible"]
].to_string(index=False))

print("Eligible hotspots:", int(hotspots["report_eligible"].sum()))
print("Triggers:", len(triggers))

