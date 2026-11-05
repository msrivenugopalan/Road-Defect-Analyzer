import unittest
import numpy as np, pandas as pd
from tests.helpers import demo_route
from config import REALISM_TYPICAL
from simulator import simulate_vehicles, normalise_defects
from defect_detector import analyse_samples, detect_defects
from evaluation import evaluate_detectors, ground_truth_passes

ROUTE = demo_route()
POT = {"defect_type": "pothole", "distance_along_route_m": 5000, "severity": "high"}
ROUGH = {"defect_type": "rough_road", "distance_along_route_m": 8000, "length_m": 80, "severity": "medium"}


class SimulatorTests(unittest.TestCase):
    def test_seed_reproducible_and_different_seeds_differ(self):
        a, _ = simulate_vehicles(3, seed=1, route=ROUTE, defect_config=POT, realism=REALISM_TYPICAL)
        b, _ = simulate_vehicles(3, seed=1, route=ROUTE, defect_config=POT, realism=REALISM_TYPICAL)
        c, _ = simulate_vehicles(3, seed=2, route=ROUTE, defect_config=POT, realism=REALISM_TYPICAL)
        self.assertTrue(np.allclose(a.sensor_vertical_acceleration_mps2, b.sensor_vertical_acceleration_mps2))
        self.assertFalse(np.allclose(a.sensor_vertical_acceleration_mps2[:1000], c.sensor_vertical_acceleration_mps2[:1000]))

    def test_gps_error_is_applied_and_reasonable(self):
        from geo import haversine_m
        s, _ = simulate_vehicles(3, seed=1, route=ROUTE, realism=REALISM_TYPICAL)
        err = haversine_m(s.latitude, s.longitude, s.true_latitude, s.true_longitude)
        self.assertGreater(np.mean(err), 1.0); self.assertLess(np.mean(err), 8.0)

    def test_no_pothole_runs_and_has_no_ground_truth(self):
        s, _ = simulate_vehicles(2, seed=1, route=ROUTE)
        self.assertEqual(int(s.ground_truth_defect.sum()), 0)

    def test_validation(self):
        with self.assertRaises(ValueError): normalise_defects({"defect_type": "lava", "distance_along_route_m": 1})
        with self.assertRaises(ValueError): normalise_defects({"distance_along_route_m": 1, "severity": "huge"})
        with self.assertRaises(ValueError): simulate_vehicles(0, route=ROUTE)


class DetectorTests(unittest.TestCase):
    def test_clean_road_has_no_detections(self):
        s, _ = simulate_vehicles(5, seed=4, route=ROUTE, realism=REALISM_TYPICAL)
        self.assertEqual(len(detect_defects(s)), 0)

    def test_detects_high_pothole_one_event_per_vehicle(self):
        s, _ = simulate_vehicles(6, seed=5, route=ROUTE, defect_config=POT,
                                 realism={**REALISM_TYPICAL, "defect_hit_probability": 1.0})
        ev = detect_defects(s)
        self.assertEqual(sorted(ev.vehicle_id.unique()), sorted(s.vehicle_id.unique()))
        self.assertEqual(len(ev), 6)           # ONE event per pass, not per sample
        self.assertTrue((ev.kind == "pothole").all())
        self.assertTrue((ev.severity == "high").all())

    def test_rough_road_is_one_segment_not_many_potholes(self):
        s, _ = simulate_vehicles(5, seed=6, route=ROUTE, defect_config=ROUGH,
                                 realism={**REALISM_TYPICAL, "defect_hit_probability": 1.0})
        ev = detect_defects(s)
        self.assertEqual((ev.kind == "rough_road").sum(), 5)
        self.assertEqual((ev.kind == "pothole").sum(), 0)

    def test_gravity_offset_does_not_matter(self):
        s, _ = simulate_vehicles(4, seed=7, route=ROUTE, defect_config=POT,
                                 realism={**REALISM_TYPICAL, "defect_hit_probability": 1.0})
        shifted = s.copy(); shifted["sensor_vertical_acceleration_mps2"] += 9.81
        self.assertEqual(len(detect_defects(s)), len(detect_defects(shifted)))

    def test_missing_columns_and_empty_input(self):
        with self.assertRaises(ValueError): detect_defects(pd.DataFrame({"vehicle_id": [1]}))
        _, ev = analyse_samples(pd.DataFrame(columns=[
            "vehicle_id", "timestamp", "latitude", "longitude", "sensor_vertical_acceleration_mps2"]))
        self.assertTrue(ev.empty)

    def test_evaluation_matches_ground_truth(self):
        s, _ = simulate_vehicles(8, seed=8, route=ROUTE, defect_config=[POT, ROUGH], realism=REALISM_TYPICAL)
        res = evaluate_detectors(s, {"new": detect_defects(s)}).iloc[0]
        truth = len(ground_truth_passes(s))
        self.assertEqual(res["Defect passes (truth)"], truth)
        self.assertGreaterEqual(res["Recall"], 0.85)
        self.assertGreaterEqual(res["Precision"], 0.95)

    def test_original_jerk_rule_finds_nothing_on_realistic_data(self):
        # Documents WHY the detector was replaced.
        from jerk_detector import compare_detectors
        from defect_detector import events_from_flags
        s, _ = simulate_vehicles(6, seed=9, route=ROUTE, defect_config=POT, realism=REALISM_TYPICAL)
        det = compare_detectors(s)
        self.assertEqual(int(det.rule_detected.sum()), 0)

if __name__ == "__main__":
    unittest.main()
