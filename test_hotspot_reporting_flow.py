
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

import database
from config import MailConfig
from hotspot_engine import find_hotspots
from reporting import send_report


@contextmanager
def closed_connection():
    """Commit successful database work and always close the connection."""
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
        "severity": "high",
        "peak_mps2": 5.5,
        "speed_mps": 10.0,
    }


def make_hotspot(row):
    return row.to_dict()


with tempfile.TemporaryDirectory(prefix="roadsense_flow_test_") as temp:
    temp_path = Path(temp)

    # Isolate the test from the real project database and outbox.
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

    timestamp = "2026-10-20 10:00:00"

    # Four vehicles plus a repeated event from VEH-001.
    first_four = pd.DataFrame([
        make_event(f"VEH-{i:03}", timestamp)
        for i in range(1, 5)
    ] + [
        make_event("VEH-001", "2026-10-20 10:01:00")
    ])

    hotspots_before, _ = find_hotspots(
        first_four, radius_m=10, min_vehicles=5, window_days=30
    )

    assert not hotspots_before["report_eligible"].any()
    print("PASS: Four distinct vehicles are not enough.")

    # A fifth distinct vehicle confirms the same location.
    all_events = pd.concat([
        first_four,
        pd.DataFrame([
            make_event("VEH-005", "2026-10-20 10:02:00")
        ]),
    ], ignore_index=True)

    hotspots, triggers = find_hotspots(
        all_events, radius_m=10, min_vehicles=5, window_days=30
    )

    eligible = hotspots[hotspots["report_eligible"]]

    assert len(eligible) == 1
    hotspot = eligible.iloc[0]
    assert hotspot["distinct_vehicle_count"] == 5
    assert len(triggers) == 1

    print("PASS: Five distinct vehicles make one hotspot eligible.")
    print("Distinct vehicles:", hotspot["distinct_vehicle_count"])

    # First report should be created in the temporary outbox.
    first = send_report(
        make_hotspot(hotspot), all_events, cfg, data_source="simulation"
    )

    # Repeating the same report should be blocked.
    duplicate = send_report(
        make_hotspot(hotspot), all_events, cfg, data_source="simulation"
    )

    assert first["status"] == "dry_run"
    assert duplicate["status"] == "duplicate"
    assert len(list(cfg.outbox_dir.glob("*.eml"))) == 1

    print("First report:", first["status"])
    print("Repeated report:", duplicate["status"])
    print("Temporary report files:", len(list(cfg.outbox_dir.glob("*.eml"))))

    print("PASS: End-to-end hotspot reporting checks completed.")
