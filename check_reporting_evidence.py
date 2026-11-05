
import pandas as pd
from config import MailConfig
from reporting import build_report_email

cfg = MailConfig.from_env()

hotspot = {
    "latitude": 13.0827,
    "longitude": 80.2707,
    "defect_type": "pothole",
    "max_severity": "high",
    "distinct_vehicle_count": 2,
    "event_count": 2,
    "first_seen": "2026-10-20 10:00:00",
    "last_seen": "2026-10-20 10:00:05",
    "spread_m": 5,
}

events = pd.DataFrame([
    {
        "vehicle_id": "VEH-TEST-001",
        "timestamp": "2026-10-20 10:00:00",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "kind": "pothole",
        "severity": "high",
        "peak_mps2": 6.0,
        "speed_mps": 8.0,
    },
    {
        "vehicle_id": "VEH-TEST-002",
        "timestamp": "2026-10-20 10:00:05",
        "latitude": 13.08271,
        "longitude": 80.27071,
        "kind": "pothole",
        "severity": "high",
        "peak_mps2": 5.5,
        "speed_mps": 7.5,
    },
])

message = build_report_email(
    hotspot=hotspot,
    events=events,
    cfg=cfg,
    data_source="simulation",
)

attachments = [
    part for part in message.iter_attachments()
    if part.get_filename() == "evidence_detections.csv"
]

assert len(attachments) == 1, (
    f"Expected one evidence attachment, found {len(attachments)}"
)

csv_text = attachments[0].get_content()
evidence = pd.read_csv(pd.io.common.StringIO(csv_text))

assert len(evidence) == 2, (
    f"Expected 2 evidence rows, found {len(evidence)}"
)
assert "vehicle" in evidence.columns
assert "vehicle_id" not in evidence.columns

print("PASS: Evidence CSV is attached.")
print(f"Evidence rows: {len(evidence)}")
print(f"Columns: {list(evidence.columns)}")
print("PASS: Raw vehicle IDs are not exposed.")
print("PASS: No email was sent.")
