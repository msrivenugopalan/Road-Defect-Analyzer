import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

_tmp = tempfile.mkdtemp(prefix="roadsense_test_")
os.environ["ROADSENSE_DB"] = os.path.join(_tmp, "test.db")

import database  # noqa: E402

database.DATABASE_PATH = __import__("pathlib").Path(os.environ["ROADSENSE_DB"])


def fresh_db():
    if database.DATABASE_PATH.exists():
        database.DATABASE_PATH.unlink()
    database.initialize_database()


def demo_route():
    from route_service import get_demo_route
    return get_demo_route()


def make_event(vehicle, lat, lon, ts="2026-10-19 10:00:00", kind="pothole", severity="high", peak=6.0):
    return {"vehicle_id": vehicle, "latitude": lat, "longitude": lon,
            "timestamp": pd.Timestamp(ts), "kind": kind, "severity": severity, "peak_mps2": peak}
