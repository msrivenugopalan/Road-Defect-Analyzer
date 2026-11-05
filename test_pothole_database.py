
from simulator import simulate_vehicles
from database import (
    initialize_database,
    save_simulation_run,
    load_vehicle_samples,
)

initialize_database()

config = {
    "simulation_mode": "test",
    "vehicle_count": 1,
    "target_speed_kmh": 36.0,
    "max_acceleration_mps2": 3.0,
    "max_braking_mps2": 7.0,
    "max_jerk_mps3": 2.0,
    "timestep_seconds": 0.2,
    "driving_behavior": "normal",
}

samples, events = simulate_vehicles(
    vehicle_count=1,
    defect_config={
        "defect_id": "pothole_1",
        "defect_type": "pothole",
        "distance_along_route_m": 100.0,
        "severity": "high",
    },
)

run_id = save_simulation_run(config, samples)
loaded_samples = load_vehicle_samples(run_id)

print("Saved run ID:", run_id)
print("Generated samples:", len(samples))
print("Loaded samples:", len(loaded_samples))
print(
    loaded_samples[
        [
            "distance_along_route_m",
            "sensor_vertical_acceleration_mps2",
            "ground_truth_defect",
            "defect_id",
            "ground_truth_defect_type",
        ]
    ].loc[loaded_samples["ground_truth_defect"] == 1]
)

assert len(samples) == len(loaded_samples)
assert "sensor_vertical_acceleration_mps2" in loaded_samples.columns
assert "ground_truth_defect_type" in loaded_samples.columns
assert loaded_samples["ground_truth_defect"].sum() > 0

print("Database test passed!")
