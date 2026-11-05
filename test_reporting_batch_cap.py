
import pandas as pd
from unittest.mock import patch

from config import MailConfig
from reporting import send_reports_for_hotspots


def main():
    # Temporary test configuration: limit the batch to 2.
    cfg = MailConfig(
        mode="dry_run",
        recipients=("test@example.com",),
        max_emails_per_batch=2,
    )

    # Create 4 eligible hotspots at different locations.
    hotspots = pd.DataFrame([
        {
            "hotspot_id": f"TEST_{i}",
            "latitude": 13.0 + i * 0.01,
            "longitude": 80.0 + i * 0.01,
            "defect_type": "pothole",
            "max_severity": "medium",
            "distinct_vehicle_count": 5,
            "event_count": 5,
            "first_seen": "2026-10-21 09:00:00",
            "last_seen": "2026-10-21 09:10:00",
            "spread_m": 4.0,
            "report_eligible": True,
            "priority_score": 10 - i,
        }
        for i in range(4)
    ])

    events = pd.DataFrame(columns=["latitude", "longitude"])

    # Mock send_report so no email or database record is created.
    with patch(
        "reporting.send_report",
        return_value={"status": "dry_run"},
    ) as mocked_send:
        results = send_reports_for_hotspots(
            hotspots=hotspots,
            events=events,
            cfg=cfg,
            data_source="simulation",
        )

    statuses = [result["status"] for result in results]

    print("Eligible hotspots:", len(hotspots))
    print("Configured batch cap:", cfg.max_emails_per_batch)
    print("Mocked report calls:", mocked_send.call_count)
    print("Results:", statuses)
    print("Skipped by cap:", statuses.count("skipped_cap"))

    assert mocked_send.call_count == 2
    assert statuses.count("skipped_cap") == 2
    print("PASS: Batch cap is respected.")


if __name__ == "__main__":
    main()
