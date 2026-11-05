import os, sys, runpy, pathlib, shutil, unittest
from unittest import mock
os.environ["AUTHORITY_EMAIL"] = "roads@city.gov.in"
os.environ["MAIL_MODE"] = "dry_run"
from tests import helpers, streamlit_stub as stub

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTBOX = ROOT / "outbox"


def run_script(pressed=(), overrides=None, click=None):
    stub.PRESSED[:] = list(pressed)
    stub.OVERRIDES.clear(); stub.OVERRIDES.update(overrides or {})
    stub.CLICK.clear(); stub.CLICK.update(click or {})
    stub.CALLS.clear()
    with mock.patch("time.sleep"):
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")


def hud(label):
    return [c[2] for c in stub.CALLS if c[0] == "metric" and c[1] == label]


class AppSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpers.fresh_db(); shutil.rmtree(OUTBOX, ignore_errors=True); cls.st = stub.install()

    @classmethod
    def tearDownClass(cls): shutil.rmtree(OUTBOX, ignore_errors=True)

    def setUp(self):
        self.st.session_state.clear()

    def _place_pothole_by_clicking(self, at=6000.0):
        run_script()
        run_script(pressed=["Use offline demo route"])
        self.assertEqual(self.st.session_state["defects"], [])
        run_script(click={"route_points": [{"distance_m": at, "latitude": 1, "longitude": 1}]})
        self.assertEqual(self.st.session_state["pending_distance"], at)       # click -> position
        run_script(pressed=["Add defect here"], overrides={"Defect type": "pothole", "Severity": "high"},
                   click={"route_points": [{"distance_m": at}]})
        return self.st.session_state["defects"]

    def test_click_adds_defect_at_clicked_position(self):
        defects = self._place_pothole_by_clicking(6000.0)
        self.assertEqual(len(defects), 1)
        self.assertEqual(defects[0]["distance_along_route_m"], 6000.0)
        self.assertEqual(defects[0]["severity"], "high")

    def test_rough_road_length_is_clamped_to_route_end(self):
        run_script(); run_script(pressed=["Use offline demo route"])
        total = self.st.session_state["route_index"].total_m
        run_script(click={"route_points": [{"distance_m": total - 20}]})
        run_script(pressed=["Add defect here"], overrides={"Defect type": "rough_road", "Rough-road length (m)": 300},
                   click={"route_points": [{"distance_m": total - 20}]})
        d = self.st.session_state["defects"][0]
        self.assertLessEqual(d["distance_along_route_m"] + d["length_m"], total + 1e-6)

    def test_remove_and_clear_defects_and_new_route_resets(self):
        self._place_pothole_by_clicking()
        run_script(pressed=["Remove selected"], click={"route_points": [{"distance_m": 6000.0}]})
        self.assertEqual(self.st.session_state["defects"], [])
        self._place_pothole_by_clicking(3000.0)
        run_script(pressed=["Use offline demo route"])                       # new route => old defects dropped
        self.assertEqual(self.st.session_state["defects"], [])

    def test_full_journey_click_run_animate_report(self):
        self._place_pothole_by_clicking(6000.0)
        run_script(pressed=["Start simulation"], click={"route_points": [{"distance_m": 6000.0}]},
                   overrides={"Number of vehicles": 10, "Share of drivers who hit the defect": 1.0})
        run = self.st.session_state["run"]
        self.assertEqual(int(run["hotspots"]["report_eligible"].sum()), 1)
        self.assertEqual(run["defects"][0]["distance_along_route_m"], 6000.0)

        # ---- the animation, frame by frame, as the user would see it ----
        run_script(pressed=["Play animation"], overrides={"Vehicle to follow": "VEH-001"},
                   click={"route_points": [{"distance_m": 6000.0}]})
        frames = [c for c in stub.CALLS if c[0] == "pydeck_chart" and c[1] is None]
        self.assertGreater(len(frames), 100)
        speeds = [float(v.split()[0]) for v in hud("Speed")]
        status = hud("Status")
        order = []
        for st_ in status:
            if not order or order[-1] != st_: order.append(st_)
        self.assertEqual(order, ["Cruising at constant speed", "Slowing down for pothole", "Crossing pothole",
                                 "Speeding back up to cruise speed", "Cruising at constant speed"])
        cruise = float(hud("Normal cruise speed")[0].split()[0])
        self.assertAlmostEqual(speeds[5], cruise, delta=2)                     # constant before
        self.assertLess(min(speeds), 0.4 * cruise)                             # slow at the defect
        self.assertAlmostEqual(speeds[-3], cruise, delta=0.03 * cruise + 1)    # same speed after

        run_script(click={"route_points": [{"distance_m": 6000.0}]})           # all tabs re-render
        run_script(pressed=["Send report(s)"], click={"route_points": [{"distance_m": 6000.0}]})
        self.assertEqual([r["status"] for r in self.st.session_state["mail_results"]], ["dry_run"])

    def test_changed_defects_after_run_does_not_crash(self):
        self._place_pothole_by_clicking(6000.0)
        run_script(pressed=["Start simulation"], overrides={"Number of vehicles": 3},
                   click={"route_points": [{"distance_m": 6000.0}]})
        self.st.session_state["defects"].append(
            {"defect_type": "pothole", "distance_along_route_m": 9000.0, "severity": "low"})
        run_script(click={"route_points": [{"distance_m": 6000.0}]})

    def test_start_without_any_defect_does_not_crash(self):    # the original app's NameError
        run_script(); run_script(pressed=["Use offline demo route"])
        run_script(pressed=["Start simulation"], overrides={"Number of vehicles": 3})
        self.assertEqual(len(self.st.session_state["run"]["events"]), 0)
        run_script(pressed=["Play animation"], overrides={"Vehicle to follow": "VEH-001"})

if __name__ == "__main__":
    unittest.main()
