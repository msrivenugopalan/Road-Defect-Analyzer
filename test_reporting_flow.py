
import tempfile
from pathlib import Path

import pandas as pd

import database
from config import MailConfig
from reporting import send_report


def make_hotspot(latitude, longitude):
    return {
        "hotspot_id": f"TEST-{latitude}",
        "latitude": latitude,
        "longitude": longitude,
        "defect_type": "pothole",
        "max_severity": "high",
        "distinct_vehicle_count": 5,
        "event_count": 8,
        "first_seen": "2026-10-20 10:00:00",
        "last_seen": "2026-10-20 10:05:00",
        "spread_m": 5.0,
    }


with tempfile.TemporaryDirectory(prefix="roadsense_test_") as temp:
    temp_path = Path(temp)

    # Redirect all database.py connections to an isolated database.
    database.DATABASE_PATH = temp_path / "test.db"
    database.initialize_database()
    database.get_connection().close()

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

    events = pd.DataFrame([
        {
            "vehicle_id": f"VEH-{i:03}",
            "timestamp": "2026-10-20 10:00:00",
            "latitude": 13.001,
            "longitude": 80.001,
            "kind": "pothole",
            "severity": "high",
            "peak_mps2": 5.5,
            "speed_mps": 10.0,
        }
        for i in range(1, 6)
    ])

    first = send_report(
        make_hotspot(13.001, 80.001), events, cfg
    )
    duplicate = send_report(
        make_hotspot(13.001, 80.001), events, cfg
    )
    separate = send_report(
        make_hotspot(13.010, 80.010), events, cfg
    )

    print("First location:", first["status"])
    print("Repeated location:", duplicate["status"])
    print("Separate location:", separate["status"])
    print("Temporary .eml files:", len(list(cfg.outbox_dir.glob("*.eml"))))

    assert first["status"] == "dry_run"
    assert duplicate["status"] == "duplicate"
    assert separate["status"] == "dry_run"
    assert len(list(cfg.outbox_dir.glob("*.eml"))) == 2

    print("PASS: Reporting and duplicate prevention verified.")
