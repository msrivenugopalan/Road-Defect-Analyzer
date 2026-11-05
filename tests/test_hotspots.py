import unittest
from tests.helpers import make_event
import pandas as pd
from hotspot_engine import HotspotTracker, find_hotspots
from geo import offset_latlon

LAT, LON = 13.0, 80.2


def ev(i, north=0.0, east=0.0, ts="2026-10-19 10:00:00", vehicle=None):
    la, lo = offset_latlon(LAT, LON, north, east)
    return make_event(vehicle or f"V{i}", float(la), float(lo), ts)


class HotspotRuleTests(unittest.TestCase):
    def test_five_distinct_vehicles_triggers_exactly_at_five(self):
        t = HotspotTracker(15, 5)
        fired = [t.add_event(ev(i)) for i in range(5)]
        self.assertEqual([f is not None for f in fired], [False] * 4 + [True])

    def test_four_vehicles_is_not_enough(self):
        hs, trig = find_hotspots(pd.DataFrame([ev(i) for i in range(4)]))
        self.assertEqual(int(hs.report_eligible.sum()), 0)
        self.assertEqual(trig, [])

    def test_same_vehicle_many_times_counts_once(self):
        events = [ev(i, vehicle="ONLY") for i in range(20)]
        hs, _ = find_hotspots(pd.DataFrame(events))
        self.assertEqual(hs.distinct_vehicle_count[0], 1)
        self.assertEqual(hs.event_count[0], 20)
        self.assertFalse(hs.report_eligible[0])

    def test_gps_scatter_within_radius_is_one_hotspot(self):
        offsets = [(0, 0), (4, -3), (-5, 6), (7, 2), (-3, -6)]
        hs, _ = find_hotspots(pd.DataFrame([ev(i, n, e) for i, (n, e) in enumerate(offsets)]))
        self.assertEqual(len(hs), 1)
        self.assertTrue(hs.report_eligible[0])

    def test_far_apart_places_are_not_merged(self):
        events = [ev(i, north=0) for i in range(3)] + [ev(i + 10, north=500) for i in range(3)]
        hs, _ = find_hotspots(pd.DataFrame(events))
        self.assertEqual(len(hs), 2)
        self.assertEqual(int(hs.report_eligible.sum()), 0)

    def test_distinct_vehicles_split_across_two_places_do_not_add_up(self):
        events = [ev(i, north=0) for i in range(3)] + [ev(i + 10, north=100) for i in range(3)]
        hs, _ = find_hotspots(pd.DataFrame(events))
        self.assertEqual(int(hs.report_eligible.sum()), 0)

    def test_old_events_expire(self):
        events = [ev(i, ts="2026-01-01 10:00:00") for i in range(4)] + [ev(9, ts="2026-10-19 10:00:00")]
        hs, _ = find_hotspots(pd.DataFrame(events), window_days=30)
        self.assertEqual(hs.distinct_vehicle_count[0], 1)

    def test_batch_equals_streaming(self):
        events = pd.DataFrame([ev(i, i % 3, i % 2) for i in range(8)])
        hs, trig = find_hotspots(events)
        t = HotspotTracker(15, 5)
        live = [x for x in (t.add_event(e) for e in events.to_dict("records")) if x]
        self.assertEqual(len(trig), len(live))
        self.assertEqual(hs.distinct_vehicle_count[0], 8)

    def test_clusters_that_drift_together_merge(self):
        t = HotspotTracker(15, 5)
        t.add_event(ev(0, north=0))
        t.add_event(ev(1, north=16))   # 16 m away: starts a second cluster
        self.assertEqual(len(t.hotspots()), 2)
        t.add_event(ev(2, north=8))    # bridges them; centres now 12 m apart
        merged = t.hotspots()
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.distinct_vehicle_count[0], 3)

    def test_empty_input(self):
        hs, trig = find_hotspots(pd.DataFrame())
        self.assertTrue(hs.empty); self.assertEqual(trig, [])

    def test_invalid_args(self):
        with self.assertRaises(ValueError): HotspotTracker(0, 5)
        with self.assertRaises(ValueError): HotspotTracker(15, 0)

if __name__ == "__main__":
    unittest.main()
