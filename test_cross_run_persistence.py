import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

import database
from config import MailConfig, HotspotConfig
from hotspot_engine import find_hotspots
from reporting import send_report


@contextmanager
def closed_connection():
    """Commit successful work and always close the SQLite connection."""
    connection = sqlite3.connect(database.DATABASE_PATH)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def make_event(vehicle_id, timestamp, latitude=13.001, longitude=80.001):
    return {
        "vehicle_id": vehicle_id,
        "timestamp": timestamp,
        "latitude": latitude,
        "longitude": longitude,
        "kind": "pothole",
        "peak_mps2": 5.5,
        "rms_mps2": 3.0,
        "severity": "high",
        "speed_mps": 10.0,
        "duration_s": 0.2,
        "n_samples": 3,
        "distance_along_route_m": 100.0,
        "detector": "vertical_accel",
    }


def make_run(vehicle_count=2):
    """Create a valid simulation run using the app's database schema."""
    config = {
        "simulation_mode": "Live Simulation",
        "vehicle_count": vehicle_count,
        "target_speed_kmh": 36.0,
        "max_acceleration_mps2": 3.0,
        "max_braking_mps2": 7.0,
        "max_jerk_mps3": 2.0,
        "timestep_seconds": 0.2,
        "driving_behavior": "automatic",
        "route_start": "Test Start",
        "route_destination": "Test Destination",
    }

    with database.get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO simulation_runs (
                simulation_mode, vehicle_count,
                route_start, route_destination,
                target_speed_kmh, max_acceleration_mps2,
                max_braking_mps2, max_jerk_mps3,
                timestep_seconds, driving_behavior, total_samples
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                config["simulation_mode"],
                config["vehicle_count"],
                config["route_start"],
                config["route_destination"],
                config["target_speed_kmh"],
                config["max_acceleration_mps2"],
                config["max_braking_mps2"],
                config["max_jerk_mps3"],
                config["timestep_seconds"],
                config["driving_behavior"],
            ),
        )
        return cursor.lastrowid


with tempfile.TemporaryDirectory(prefix="roadsense_cross_run_") as temp:
    temp_path = Path(temp)

    # Isolate this test from the real project database.
    database.DATABASE_PATH = temp_path / "test.db"
    database.get_connection = closed_connection
    database.initialize_database()

    cfg = MailConfig(
        mode="dry_run",
        host="",
        port=587,
        user="",
        password="",
        use_tls=True,
        sender="roadsense-test@example.com",
        recipients=("authority-test@example.com",),
        dedupe_radius_m=25.0,
        cooldown_days=30,
        max_emails_per_batch=10,
        outbox_dir=temp_path / "outbox",
    )

    # Run 1: four different vehicles report the same location.
    run1 = make_run(vehicle_count=4)
    run1_events = pd.DataFrame([
        make_event(f"VEH-{i:03}", "2026-10-20 10:00:00")
        for i in range(1, 5)
    ])

    database.save_road_defect_events(run1, run1_events)

    loaded1 = database.load_recent_road_defect_events(window_days=30)
    loaded1 = loaded1.rename(columns={"event_timestamp": "timestamp"})
    loaded1["vehicle_key"] = (
        loaded1["run_id"].astype(str) + ":" +
        loaded1["vehicle_id"].astype(str)
    )

    hotspots1, _ = find_hotspots(
        loaded1,
        radius_m=10,
        min_vehicles=5,
        window_days=30,
    )

    assert not hotspots1["report_eligible"].any()
    print("PASS: Run 1 has four vehicles; hotspot is not eligible.")

    # Run 2: a fifth vehicle encounters the same location.
    run2 = make_run(vehicle_count=1)
    run2_events = pd.DataFrame([
        make_event("VEH-001", "2026-10-20 10:02:00")
    ])

    database.save_road_defect_events(run2, run2_events)

    # Reload the persisted evidence, just as the app does.
    loaded2 = database.load_recent_road_defect_events(window_days=30)
    loaded2 = loaded2.rename(columns={"event_timestamp": "timestamp"})
    loaded2["vehicle_key"] = (
        loaded2["run_id"].astype(str) + ":" +
        loaded2["vehicle_id"].astype(str)
    )

    hotspots2, _ = find_hotspots(
        loaded2,
        radius_m=10,
        min_vehicles=5,
        window_days=30,
    )

    eligible = hotspots2[hotspots2["report_eligible"]]

    assert len(eligible) == 1
    hotspot = eligible.iloc[0]
    assert hotspot["distinct_vehicle_count"] == 5
    print("PASS: Persisted evidence from both runs reaches five vehicles.")

    # Confirm report creation and duplicate prevention.
    first = send_report(
        hotspot.to_dict(),
        loaded2,
        cfg,
        data_source="simulation",
        run_id=run2,
    )
    second = send_report(
        hotspot.to_dict(),
        loaded2,
        cfg,
        data_source="simulation",
        run_id=run2,
    )

    assert first["status"] == "dry_run"
    assert second["status"] == "duplicate"
    assert len(list(cfg.outbox_dir.glob("*.eml"))) == 1

    print("PASS: One dry-run report created; duplicate blocked.")
    print("PASS: Cross-run persistence test completed.")