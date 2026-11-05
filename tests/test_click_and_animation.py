import unittest
import numpy as np
from tests.helpers import demo_route
from route_tools import RouteIndex
from animation import build_tracks, frame_times, positions_at, focus_status
from simulator import simulate_vehicles, _prepare_route, approach_distance_m

ROUTE = demo_route()
INDEX = RouteIndex(ROUTE)
POT = {"defect_type": "pothole", "distance_along_route_m": 6000, "severity": "high"}


class RouteIndexTests(unittest.TestCase):
    def test_total_length_matches_simulator(self):
        _, _, total = _prepare_route(ROUTE)
        self.assertAlmostEqual(INDEX.total_m, total, places=3)

    def test_click_point_round_trips_to_the_simulated_position(self):
        # A defect placed at a clicked distance must be where the car really drives.
        s, _ = simulate_vehicles(1, route=ROUTE, defect_config=POT)
        lat, lon = INDEX.position_at(6000)
        row = s.iloc[(s.distance_along_route_m - 6000).abs().argmin()]
        from geo import haversine_m
        self.assertLess(haversine_m(lat, lon, row.true_latitude, row.true_longitude), 3.0)

    def test_points_cover_the_whole_route(self):
        pts = INDEX.points(20)
        self.assertEqual(pts.distance_m.iloc[0], 0.0)
        self.assertAlmostEqual(pts.distance_m.iloc[-1], INDEX.total_m, places=3)
        self.assertLessEqual(np.diff(pts.distance_m).max(), 20.0 + 1e-6)

    def test_segment_and_clamping(self):
        seg = INDEX.segment(1000, 1080)
        self.assertGreaterEqual(len(seg), 2)
        self.assertEqual(len(INDEX.segment(-50, 1e9)), len(INDEX.lat))  # clamped to the whole route
        with self.assertRaises(ValueError): RouteIndex({"coordinates": [{"latitude": 1, "longitude": 1}]})


class VehicleSpeedBehaviourTests(unittest.TestCase):
    """The requirement: slow down at the defect, then resume the SAME cruise speed."""

    def _profile(self, kmh):
        s, _ = simulate_vehicles(1, route=ROUTE, config={"target_speed_kmh": kmh}, defect_config=POT)
        d, v = s.distance_along_route_m.to_numpy(), s.speed_mps.to_numpy()
        at = lambda x: v[np.argmin(abs(d - x))]
        return at(5600), v[abs(d - 6000) < 6].min(), at(6300), s

    def test_slows_then_resumes_same_speed_at_every_cruise_speed(self):
        for kmh in (18, 36, 54, 72):
            before, minimum, after, _ = self._profile(kmh)
            with self.subTest(kmh=kmh):
                self.assertLess(minimum, 3.5)                        # actually slow at the defect
                self.assertAlmostEqual(after, before, delta=0.02 * before)   # back to the SAME speed

    def test_speed_is_constant_before_the_slowdown_zone(self):
        _, _, _, s = self._profile(36)
        d, v = s.distance_along_route_m.to_numpy(), s.speed_mps.to_numpy()
        steady = v[(d > 3000) & (d < 5800)]
        self.assertLess(steady.max() - steady.min(), 0.05)

    def test_approach_distance_grows_with_speed(self):
        self.assertGreater(approach_distance_m(20), approach_distance_m(10))
        self.assertEqual(approach_distance_m(2), 20.0)

    def test_rough_road_slows_to_six(self):
        rough = {"defect_type": "rough_road", "distance_along_route_m": 6000, "length_m": 80, "severity": "medium"}
        s, _ = simulate_vehicles(1, route=ROUTE, defect_config=rough)
        d, v = s.distance_along_route_m.to_numpy(), s.speed_mps.to_numpy()
        self.assertLess(v[(d > 6020) & (d < 6060)].max(), 6.6)


class AnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s, _ = simulate_vehicles(3, seed=5, route=ROUTE, defect_config=POT)
        cls.tracks = build_tracks(cls.s)
        cls.focus = cls.tracks["VEH-001"]

    def test_frames_are_monotonic_and_denser_near_the_defect(self):
        t = frame_times(self.focus, [POT])
        self.assertTrue((np.diff(t) >= 0).all())
        dist = np.interp(t, self.focus["t"], self.focus["dist"])
        near = ((dist > 5930) & (dist < 6050)).sum() / 120
        far = ((dist > 2000) & (dist < 2120)).sum() / 120
        self.assertGreater(near, 5 * far)                      # slow motion at the defect
        self.assertEqual(len(frame_times(self.focus, [POT], slow_motion=False)) < len(t), True)

    def test_positions_hide_vehicles_not_on_the_road(self):
        self.assertEqual(len(positions_at(self.tracks, -5.0)), 0)
        mid = positions_at(self.tracks, self.focus["t"][len(self.focus["t"]) // 2])
        self.assertGreaterEqual(len(mid), 1)
        self.assertTrue(((mid.ratio >= 0) & (mid.ratio <= 1)).all())

    def test_colour_shows_slowdown(self):
        t = frame_times(self.focus, [POT], fine_step_m=1)
        speeds = [positions_at({"VEH-001": self.focus}, x).speed_kmh.iloc[0] for x in t if True]
        fast = positions_at({"VEH-001": self.focus}, t[np.argmax(speeds)]).ratio.iloc[0]
        slow = positions_at({"VEH-001": self.focus}, t[np.argmin(speeds)]).ratio.iloc[0]
        self.assertGreater(fast, 0.95); self.assertLess(slow, 0.5)

    def test_status_sequence_around_a_defect(self):
        c = self.focus["cruise"]
        self.assertEqual(focus_status(1000, c, c, [POT])[0], "Cruising at constant speed")
        self.assertTrue(focus_status(5985, 5, c, [POT])[0].startswith("Slowing"))
        self.assertEqual(focus_status(6000, 3, c, [POT])[0], "Crossing pothole")
        self.assertTrue(focus_status(6040, 6, c, [POT])[0].startswith("Speeding back"))
        self.assertAlmostEqual(focus_status(5500, c, c, [POT])[1], 500)
        self.assertIsNone(focus_status(7000, c, c, [POT])[1])

if __name__ == "__main__":
    unittest.main()
