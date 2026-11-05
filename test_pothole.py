from simulator import simulate_vehicles
from defect_detector import analyse_samples


def test_simulated_pothole_is_detected():
    samples, _ = simulate_vehicles(
        vehicle_count=1,
        defect_config={
            "defect_id": "pothole_1",
            "defect_type": "pothole",
            "distance_along_route_m": 100.0,
            "severity": "high",
        },
    )

    annotated, detected = analyse_samples(samples)

    # Verify the simulation actually generated a ground-truth encounter.
    assert samples["ground_truth_defect"].any(), (
        "The vehicle did not encounter the configured pothole."
    )

    # Verify the detector produced a pothole event.
    potholes = detected[detected["kind"] == "pothole"]
    assert not potholes.empty, (
        "The detector missed the simulated pothole."
    )

    assert potholes["severity"].eq("high").any(), (
        "The high-severity pothole was not classified as high."
    )

    # A short pothole pulse should not be classified as sustained rough road.
    assert not detected["kind"].eq("rough_road").any(), (
        "The short pothole pulse was incorrectly classified as rough road."
    )


def test_clean_simulation_has_no_defect_events():
    samples, _ = simulate_vehicles(vehicle_count=1)

    _, detected = analyse_samples(samples)

    assert not samples["ground_truth_defect"].any(), (
        "The clean simulation unexpectedly contains ground-truth defects."
    )
    assert detected.empty, (
        f"Expected no defect events on a clean simulation, "
        f"but found {len(detected)}."
    )

import numpy as np
import pandas as pd


def test_sustained_rough_road_is_detected():
    # Generate 15 seconds of sustained, moderate vertical vibration at 10 Hz.
    sample_count = 150
    elapsed = np.arange(sample_count) / 10.0

    samples = pd.DataFrame({
        "vehicle_id": "VEH-ROUGH-001",
        "timestamp": (
            pd.Timestamp("2026-10-21 10:00:00")
            + pd.to_timedelta(elapsed, unit="s")
        ),
        "latitude": 13.0827 + elapsed * 0.000001,
        "longitude": 80.2716 + elapsed * 0.000001,
        "sensor_vertical_acceleration_mps2": (
            0.9 * np.sin(2 * np.pi * 3 * elapsed)
        ),
    })

    _, detected = analyse_samples(samples)

    rough_events = detected[detected["kind"] == "rough_road"]
    assert not rough_events.empty, (
        "The detector missed sustained rough-road vibration."
    )

    pothole_events = detected[detected["kind"] == "pothole"]
    assert pothole_events.empty, (
        "Moderate sustained vibration was incorrectly classified as a pothole."
    )
