import unittest, tempfile, pathlib, smtplib
from unittest import mock
import numpy as np, pandas as pd
from tests.helpers import fresh_db, demo_route
import database
from config import MailConfig, REALISM_TYPICAL
from simulator import simulate_vehicles
from defect_detector import detect_defects
from hotspot_engine import find_hotspots
import reporting
import ingest

ROUTE = demo_route()
POT = {"defect_type": "pothole", "distance_along_route_m": 5000, "severity": "high"}
ALWAYS_HIT = {**REALISM_TYPICAL, "defect_hit_probability": 1.0}


def pipeline(n=8, seed=11):
    s, _ = simulate_vehicles(n, seed=seed, route=ROUTE, defect_config=POT, realism=ALWAYS_HIT)
    ev = detect_defects(s)
    hs, trig = find_hotspots(ev)
    return s, ev, hs, trig


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        fresh_db()
        self.outbox = pathlib.Path(tempfile.mkdtemp())
        self.cfg = MailConfig(mode="dry_run", recipients=("roads@city.gov.in",), sender="bot@x.com", outbox_dir=self.outbox)

    def test_full_pipeline_one_real_hotspot_no_false_ones(self):
        _, ev, hs, trig = pipeline()
        self.assertEqual(len(hs), 1); self.assertTrue(hs.report_eligible[0])
        self.assertEqual(len(trig), 1)
        self.assertEqual(trig[0]["distinct_vehicle_count"], 5)   # fired at the 5th vehicle

    def test_too_few_vehicles_no_report(self):
        _, ev, hs, _ = pipeline(n=4)
        self.assertEqual(int(hs.report_eligible.sum()), 0)
        self.assertEqual(reporting.send_reports_for_hotspots(hs, ev, self.cfg), [])

    def test_dry_run_writes_eml_and_logs_and_labels_simulation(self):
        _, ev, hs, _ = pipeline()
        res = reporting.send_reports_for_hotspots(hs, ev, self.cfg)
        self.assertEqual(res[0]["status"], "dry_run")
        text = pathlib.Path(res[0]["outbox_path"]).read_bytes().decode(errors="ignore")
        self.assertIn("SIMULATED DATA", text); self.assertIn("roads@city.gov.in", text)
        self.assertNotIn("VEH-001", text)                       # anonymised
        self.assertEqual(len(database.load_reports()), 1)

    def test_duplicate_is_blocked_then_force_overrides(self):
        _, ev, hs, _ = pipeline()
        reporting.send_reports_for_hotspots(hs, ev, self.cfg)
        second = reporting.send_reports_for_hotspots(hs, ev, self.cfg)
        self.assertEqual(second[0]["status"], "duplicate")
        forced = reporting.send_reports_for_hotspots(hs, ev, self.cfg, force=True)
        self.assertEqual(forced[0]["status"], "dry_run")

    def test_dry_run_does_not_block_a_real_send(self):
        _, ev, hs, _ = pipeline()
        reporting.send_reports_for_hotspots(hs, ev, self.cfg)   # dry run first
        smtp_cfg = MailConfig(mode="smtp", host="h", recipients=("a@b.com",), sender="s@b.com", outbox_dir=self.outbox)
        with mock.patch("smtplib.SMTP") as smtp:
            res = reporting.send_reports_for_hotspots(hs, ev, smtp_cfg)
        self.assertEqual(res[0]["status"], "sent"); smtp.assert_called()

    def test_smtp_retries_then_logs_failure(self):
        _, ev, hs, _ = pipeline()
        smtp_cfg = MailConfig(mode="smtp", host="h", recipients=("a@b.com",), sender="s@b.com", outbox_dir=self.outbox)
        with mock.patch("smtplib.SMTP", side_effect=OSError("network down")) as smtp, mock.patch("time.sleep"):
            res = reporting.send_reports_for_hotspots(hs, ev, smtp_cfg)
        self.assertEqual(res[0]["status"], "failed"); self.assertEqual(smtp.call_count, 3)
        self.assertEqual(database.load_reports().status[0], "failed")
        # a failed attempt must NOT block a later retry
        with mock.patch("smtplib.SMTP"):
            again = reporting.send_reports_for_hotspots(hs, ev, smtp_cfg)
        self.assertEqual(again[0]["status"], "sent")

    def test_invalid_config_is_rejected_before_anything_happens(self):
        _, ev, hs, _ = pipeline()
        bad = MailConfig(mode="smtp", host="", recipients=("not-an-email",))
        res = reporting.send_reports_for_hotspots(hs, ev, bad)
        self.assertEqual(res[0]["status"], "invalid_config")
        self.assertEqual(len(database.load_reports()), 0)

    def test_batch_cap(self):
        hs = pd.DataFrame([{"hotspot_id": f"HS-{i}", "latitude": 13 + i * 0.01, "longitude": 80.0,
            "event_count": 6, "distinct_vehicle_count": 6, "vehicle_ids": "", "first_seen": pd.Timestamp.now(),
            "last_seen": pd.Timestamp.now(), "defect_type": "pothole", "max_severity": "high", "peak_mps2": 6.0,
            "spread_m": 3.0, "elongated": False, "priority_score": 18 - i, "report_eligible": True} for i in range(5)])
        ev = pd.DataFrame({"vehicle_id": ["a"], "latitude": [13.0], "longitude": [80.0]})
        cfg = MailConfig(mode="dry_run", recipients=("a@b.com",), outbox_dir=self.outbox, max_emails_per_batch=2)
        res = reporting.send_reports_for_hotspots(hs, ev, cfg)
        self.assertEqual([r["status"] for r in res], ["dry_run", "dry_run", "skipped_cap", "skipped_cap", "skipped_cap"])


class DatabaseTests(unittest.TestCase):
    def setUp(self): fresh_db()

    def test_events_roundtrip_and_per_run_vehicle_keys(self):
        _, ev, _, _ = pipeline(n=3)
        n1 = database.save_events(ev, run_id=None, source="real_data")
        loaded = database.load_events()
        self.assertEqual(n1, len(ev)); self.assertEqual(len(loaded), len(ev))
        self.assertTrue((loaded.vehicle_key == loaded.vehicle_id).all())
        # simulated runs get run-scoped keys so fleets from different runs don't collide
        cfg = {"simulation_mode": "t", "vehicle_count": 3, "target_speed_kmh": 36, "max_acceleration_mps2": 3,
               "max_braking_mps2": 7, "max_jerk_mps3": 2, "timestep_seconds": .2, "driving_behavior": "auto"}
        r1 = database.save_simulation_run(cfg, pd.DataFrame(), store_samples=False)
        r2 = database.save_simulation_run(cfg, pd.DataFrame(), store_samples=False)
        database.save_events(ev, run_id=r1); database.save_events(ev, run_id=r2)
        keys = database.load_events(run_id=r2).vehicle_key
        self.assertTrue(keys.str.startswith(f"run{r2}:").all())

    def test_store_samples_false_keeps_database_small(self):
        s, _ = simulate_vehicles(2, seed=1, route=ROUTE)
        cfg = {"simulation_mode": "t", "vehicle_count": 2, "target_speed_kmh": 36, "max_acceleration_mps2": 3,
               "max_braking_mps2": 7, "max_jerk_mps3": 2, "timestep_seconds": .2, "driving_behavior": "auto"}
        a = database.save_simulation_run(cfg, s, store_samples=True)
        b = database.save_simulation_run(cfg, s, store_samples=False)
        self.assertEqual(len(database.load_vehicle_samples(a)), len(s))
        self.assertEqual(len(database.load_vehicle_samples(b)), 0)

    def test_cross_run_accumulation_reaches_threshold(self):
        # 3 vehicles on one run + 3 on another = 6 distinct vehicles
        cfg = {"simulation_mode": "t", "vehicle_count": 3, "target_speed_kmh": 36, "max_acceleration_mps2": 3,
               "max_braking_mps2": 7, "max_jerk_mps3": 2, "timestep_seconds": .2, "driving_behavior": "auto"}
        for seed in (21, 22):
            s, _ = simulate_vehicles(3, seed=seed, route=ROUTE, defect_config=POT, realism=ALWAYS_HIT)
            run = database.save_simulation_run(cfg, pd.DataFrame(), store_samples=False)
            database.save_events(detect_defects(s), run_id=run)
        hs, _ = find_hotspots(database.load_events())
        self.assertEqual(hs.distinct_vehicle_count[0], 6); self.assertTrue(hs.report_eligible[0])


class IngestTests(unittest.TestCase):
    def test_real_style_csv_with_gravity_and_g_units_is_detected(self):
        s, _ = simulate_vehicles(1, seed=3, route=ROUTE, defect_config=POT, realism=ALWAYS_HIT)
        raw = pd.DataFrame({
            "Time": (s.timestamp.astype("int64") // 10**6),             # epoch ms
            "Latitude": s.latitude, "Longitude": s.longitude,
            "accelerometerZ": (s.sensor_vertical_acceleration_mps2 + 9.80665) / 9.80665,  # g, gravity included
            "Speed (km/h)": s.speed_mps * 3.6})
        mapping = ingest.suggest_mapping(raw.columns)
        self.assertEqual(mapping["vertical_accel"], "accelerometerZ")
        std = ingest.standardise_sensor_dataframe(raw, mapping, "PHONE-1", accel_unit="g", speed_unit="km/h")
        ev = detect_defects(std)
        self.assertEqual(len(ev), 1); self.assertEqual(ev.severity[0], "high")
        self.assertAlmostEqual(std.speed_mps.iloc[500], s.speed_mps.iloc[500], places=3)

    def test_missing_mapping_and_bad_rows(self):
        raw = pd.DataFrame({"t": [1, 2], "a": [1, 2]})
        with self.assertRaises(ValueError): ingest.standardise_sensor_dataframe(raw, {"timestamp": "t"})

if __name__ == "__main__":
    unittest.main()
