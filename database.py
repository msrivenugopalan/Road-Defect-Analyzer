
import sqlite3
from pathlib import Path

import pandas as pd


DATABASE_PATH = Path(__file__).resolve().parent / "roadsenseai.db"



def get_connection():
    """Create a database connection with foreign-key support."""
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )
    connection.execute("PRAGMA foreign_keys = ON")
    return connection



def initialize_database():
    """Create the simulation tables if they do not exist."""

    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS simulation_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                simulation_mode TEXT NOT NULL,
                vehicle_count INTEGER NOT NULL,
                route_start TEXT,
                route_destination TEXT,
                target_speed_kmh REAL NOT NULL,
                max_acceleration_mps2 REAL NOT NULL,
                max_braking_mps2 REAL NOT NULL,
                max_jerk_mps3 REAL NOT NULL,
                timestep_seconds REAL NOT NULL,
                driving_behavior TEXT NOT NULL,
                total_samples INTEGER NOT NULL DEFAULT 0
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS injected_pothole_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                defect_id TEXT NOT NULL,
                defect_type TEXT NOT NULL,
                distance_along_route_m REAL NOT NULL,
                severity TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
                # Store multiple configured defects without changing legacy data.
        
        connection.execute("""
            CREATE TABLE IF NOT EXISTS injected_defect_configs (
                defect_id TEXT PRIMARY KEY,
                defect_type TEXT NOT NULL,
                distance_along_route_m REAL NOT NULL,
                severity TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add fixed GPS coordinates to existing defect configurations.
        defect_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(injected_defect_configs)"
            ).fetchall()
        }

        if "latitude" not in defect_columns:
            connection.execute("""
                ALTER TABLE injected_defect_configs
                ADD COLUMN latitude REAL
            """)

        if "longitude" not in defect_columns:
            connection.execute("""
                ALTER TABLE injected_defect_configs
                ADD COLUMN longitude REAL
            """)


        
        # Track imported CSV files and prevent exact-file re-imports.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS csv_imports (
                import_id INTEGER PRIMARY KEY AUTOINCREMENT,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                original_filename TEXT NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                sample_count INTEGER NOT NULL DEFAULT 0,
                event_count INTEGER NOT NULL DEFAULT 0
            )
        """)

        # Store CSV-detected events separately from simulation events.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS csv_road_defect_events (
                csv_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                import_id INTEGER NOT NULL,
                event_fingerprint TEXT NOT NULL UNIQUE,
                vehicle_id TEXT NOT NULL,
                event_timestamp TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                kind TEXT NOT NULL,
                peak_mps2 REAL,
                rms_mps2 REAL,
                severity TEXT,
                speed_mps REAL,
                duration_s REAL,
                n_samples INTEGER,
                distance_along_route_m REAL,
                detector TEXT,
                FOREIGN KEY (import_id)
                    REFERENCES csv_imports(import_id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_csv_defects_location_time
            ON csv_road_defect_events (
                latitude,
                longitude,
                event_timestamp
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_csv_defects_vehicle
            ON csv_road_defect_events (vehicle_id)
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                defect_type TEXT,
                severity TEXT,
                distinct_vehicles INTEGER,
                event_count INTEGER,
                recipient TEXT,
                subject TEXT,
                message_id TEXT,
                run_id INTEGER,
                source TEXT,
                status TEXT NOT NULL,
                outbox_path TEXT,
                error TEXT,
                attempts INTEGER DEFAULT 0
            )
        """)
        report_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(reports)"
            ).fetchall()
        }

        if "defect_id" not in report_columns:
            connection.execute(
                "ALTER TABLE reports ADD COLUMN defect_id TEXT"
            )

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_reports_defect_id_status
            ON reports(defect_id, status, created_at)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_reports_location_status
            ON reports (latitude, longitude, status, created_at)
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                vehicle_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                distance_along_route_m REAL,
                speed_mps REAL,
                acceleration_mps2 REAL,
                jerk_mps3 REAL,
                throttle REAL,
                brake REAL,
                ground_truth_jerk INTEGER NOT NULL DEFAULT 0,
                ground_truth_defect INTEGER NOT NULL DEFAULT 0,
                defect_id TEXT,
                FOREIGN KEY (run_id)
                    REFERENCES simulation_runs(run_id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_run_vehicle
            ON vehicle_samples (run_id, vehicle_id, timestamp)
        """)
                # Store detected road-defect events across simulation runs.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS road_defect_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                vehicle_id TEXT NOT NULL,
                event_timestamp TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                kind TEXT NOT NULL,
                peak_mps2 REAL,
                rms_mps2 REAL,
                severity TEXT,
                speed_mps REAL,
                duration_s REAL,
                n_samples INTEGER,
                distance_along_route_m REAL,
                detector TEXT,
                FOREIGN KEY (run_id)
                    REFERENCES simulation_runs(run_id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_road_defects_location_time
            ON road_defect_events (
                latitude,
                longitude,
                event_timestamp
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_road_defects_run_vehicle
            ON road_defect_events (run_id, vehicle_id)
        """)
        
        connection.execute("""
            CREATE TABLE IF NOT EXISTS dismissed_hotspots (
                dismissal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dismissed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                restored_at TEXT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                defect_type TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_dismissed_hotspots_active
            ON dismissed_hotspots (restored_at, defect_type)
        """)

        # Add new columns to existing databases without deleting saved data.

        existing_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(vehicle_samples)"
            ).fetchall()
        }

        if "sensor_vertical_acceleration_mps2" not in existing_columns:
            connection.execute("""
                ALTER TABLE vehicle_samples
                ADD COLUMN sensor_vertical_acceleration_mps2 REAL NOT NULL DEFAULT 0.0
            """)

        if "ground_truth_defect_type" not in existing_columns:
            connection.execute("""
                ALTER TABLE vehicle_samples
                ADD COLUMN ground_truth_defect_type TEXT
            """)




def _database_value(value):
    """Convert pandas and NumPy values into SQLite-compatible values."""

    if value is None:
        return None

    if pd.isna(value):
        return None

    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().isoformat()

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, bool):
        return int(value)

    return value

def get_next_injected_defect_id():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS defect_id_sequence (
                sequence_name TEXT PRIMARY KEY,
                next_value INTEGER NOT NULL
            )
        """)

        row = connection.execute(
            """
            SELECT next_value
            FROM defect_id_sequence
            WHERE sequence_name = 'pothole'
            """
        ).fetchone()

        if row is None:
            active_ids = connection.execute(
                """
                SELECT defect_id
                FROM injected_defect_configs
                """
            ).fetchall()

            saved_ids = connection.execute(
                """
                SELECT defect_id
                FROM reports
                WHERE defect_id IS NOT NULL
                """
            ).fetchall()

            import re

            numbers = []

            for item in list(active_ids) + list(saved_ids):
                match = re.fullmatch(
                    r"pothole_(\d+)",
                    str(item[0]),
                )
                if match:
                    numbers.append(int(match.group(1)))

            next_value = max(numbers, default=0) + 1

            connection.execute(
                """
                INSERT INTO defect_id_sequence (
                    sequence_name,
                    next_value
                )
                VALUES ('pothole', ?)
                """,
                (next_value + 1,),
            )
        else:
            next_value = int(row[0])

            connection.execute(
                """
                UPDATE defect_id_sequence
                SET next_value = ?
                WHERE sequence_name = 'pothole'
                """,
                (next_value + 1,),
            )

        return f"pothole_{next_value}"
    
def save_simulation_run(config, samples):
    """
    Save simulation settings and all generated movement samples.
    Returns the newly created run ID.
    """

    required_settings = [
        "simulation_mode",
        "vehicle_count",
        "target_speed_kmh",
        "max_acceleration_mps2",
        "max_braking_mps2",
        "max_jerk_mps3",
        "timestep_seconds",
        "driving_behavior",
    ]

    missing_settings = [
        key for key in required_settings if key not in config
    ]

    if missing_settings:
        raise ValueError(
            f"Missing simulation settings: {', '.join(missing_settings)}"
        )

    sample_columns = [
        "vehicle_id",
        "timestamp",
        "latitude",
        "longitude",
        "distance_along_route_m",
        "speed_mps",
        "acceleration_mps2",
        "jerk_mps3",
        "throttle",
        "brake",
        "ground_truth_jerk",
        "ground_truth_defect",
        "defect_id",
        "sensor_vertical_acceleration_mps2",
        "ground_truth_defect_type",
    ]

    if samples is None:
        samples = pd.DataFrame()

    if not samples.empty:
        missing_columns = [
            column for column in sample_columns
            if column not in samples.columns
        ]

        if missing_columns:
            raise ValueError(
                "Simulation samples are missing columns: "
                + ", ".join(missing_columns)
            )

    run_values = (
        config["simulation_mode"],
        int(config["vehicle_count"]),
        config.get("route_start"),
        config.get("route_destination"),
        float(config["target_speed_kmh"]),
        float(config["max_acceleration_mps2"]),
        float(config["max_braking_mps2"]),
        float(config["max_jerk_mps3"]),
        float(config["timestep_seconds"]),
        config["driving_behavior"],
        len(samples),
    )

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO simulation_runs (
                simulation_mode,
                vehicle_count,
                route_start,
                route_destination,
                target_speed_kmh,
                max_acceleration_mps2,
                max_braking_mps2,
                max_jerk_mps3,
                timestep_seconds,
                driving_behavior,
                total_samples
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, run_values)

        run_id = cursor.lastrowid

        if not samples.empty:
            rows = []

            for record in samples[sample_columns].to_dict("records"):
                rows.append((
                    run_id,
                    *[
                        _database_value(record[column])
                        for column in sample_columns
                    ],
                ))

            connection.executemany("""
                
                INSERT INTO vehicle_samples (
                    run_id,
                    vehicle_id,
                    timestamp,
                    latitude,
                    longitude,
                    distance_along_route_m,
                    speed_mps,
                    acceleration_mps2,
                    jerk_mps3,
                    throttle,
                    brake,
                    ground_truth_jerk,
                    ground_truth_defect,
                    defect_id,
                    sensor_vertical_acceleration_mps2,
                    ground_truth_defect_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, rows)

    return run_id


def load_vehicle_samples(run_id):
    """Load stored movement samples for a particular simulation run."""

    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM vehicle_samples
            WHERE run_id = ?
            ORDER BY timestamp, vehicle_id
            """,
            connection,
            params=(run_id,),
        )


def load_simulation_runs():
    """Return saved simulation runs, newest first."""

    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM simulation_runs
            ORDER BY run_id DESC
            """,
            connection,
        )


def find_recent_report(
    latitude,
    longitude,
    radius_m,
    cooldown_days,
    statuses,
):
    """Find a recent report within the specified radius and cooldown."""
    from geo import haversine_m

    if not statuses:
        return None

    placeholders = ",".join("?" for _ in statuses)

    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT
                report_id,
                created_at,
                latitude,
                longitude,
                outbox_path
            FROM reports
            WHERE status IN ({placeholders})
              AND created_at >= datetime('now', ?)
            ORDER BY created_at DESC
            """,
            (*statuses, f"-{int(cooldown_days)} days"),
        ).fetchall()

    for row in rows:
        distance = haversine_m(
            latitude,
            longitude,
            row[2],
            row[3],
        )

        if distance <= radius_m:
            return {
                "report_id": row[0],
                "created_at": row[1],
                "latitude": row[2],
                "longitude": row[3],
                "outbox_path": row[4],
            }

    return None


def log_report(
    latitude,
    longitude,
    status,
    defect_type=None,
    severity=None,
    distinct_vehicles=None,
    event_count=None,
    recipient=None,
    subject=None,
    message_id=None,
    run_id=None,
    source=None,
    outbox_path=None,
    error=None,
    attempts=0,
    defect_id=None,
):
    """Record a report attempt and return its database ID."""

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO reports (
                latitude,
                longitude,
                defect_type,
                severity,
                distinct_vehicles,
                event_count,
                recipient,
                subject,
                message_id,
                run_id,
                source,
                status,
                outbox_path,
                error,
                attempts,
                defect_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                latitude,
                longitude,
                defect_type,
                severity,
                distinct_vehicles,
                event_count,
                recipient,
                subject,
                message_id,
                run_id,
                source,
                status,
                str(outbox_path) if outbox_path else None,
                error,
                attempts,
                defect_id,
            ),
        )

        return cursor.lastrowid

def find_report_for_defect(defect_id):
    if not defect_id:
        return None

    with get_connection() as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT
                report_id,
                defect_id,
                created_at,
                status,
                outbox_path,
                error
            FROM reports
            WHERE defect_id = ?
              AND status IN ('sent', 'dry_run')
            ORDER BY created_at DESC, report_id DESC
            LIMIT 1
            """,
            (str(defect_id),),
        ).fetchone()

    return dict(row) if row else None
def load_reports(limit=100):
    with get_connection() as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                report_id,
                created_at,
                latitude,
                longitude,
                defect_type,
                severity,
                distinct_vehicles,
                event_count,
                recipient,
                subject,
                message_id,
                run_id,
                source,
                status,
                outbox_path,
                error,
                attempts,
                defect_id
            FROM reports
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return pd.DataFrame([dict(row) for row in rows])



def save_road_defect_events(run_id, events):
    """Save detected road-defect events for a simulation run."""

    if events is None or events.empty:
        return 0

    columns = [
        "vehicle_id",
        "timestamp",
        "latitude",
        "longitude",
        "kind",
        "peak_mps2",
        "rms_mps2",
        "severity",
        "speed_mps",
        "duration_s",
        "n_samples",
        "distance_along_route_m",
        "detector",
    ]

    missing = [column for column in columns if column not in events.columns]
    if missing:
        raise ValueError(
            "Defect events are missing columns: " + ", ".join(missing)
        )

    rows = []
    for record in events[columns].to_dict("records"):
        rows.append((
            int(run_id),
            *[_database_value(record[column]) for column in columns],
        ))

    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO road_defect_events (
                run_id, vehicle_id, event_timestamp, latitude, longitude,
                kind, peak_mps2, rms_mps2, severity, speed_mps,
                duration_s, n_samples, distance_along_route_m, detector
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    return len(rows)


def load_recent_road_defect_events(window_days=30):
    """Load road-defect events from the recent time window."""

    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM road_defect_events
            WHERE event_timestamp >= datetime('now', ?)
            ORDER BY event_timestamp, vehicle_id
            """,
            connection,
            params=(f"-{int(window_days)} days",),
        )


def dismiss_hotspot(latitude, longitude, defect_type, radius_m=10.0):
    """Record a hotspot dismissal without deleting defect evidence."""
    from geo import haversine_m

    defect_type = str(defect_type).strip()
    if not defect_type:
        raise ValueError("Defect type cannot be empty.")

    latitude = float(latitude)
    longitude = float(longitude)

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT dismissal_id, latitude, longitude
            FROM dismissed_hotspots
            WHERE restored_at IS NULL
              AND lower(defect_type) = lower(?)
            """,
            (defect_type,),
        ).fetchall()

        for row in rows:
            if haversine_m(latitude, longitude, row[1], row[2]) <= radius_m:
                return row[0]

        cursor = connection.execute(
            """
            INSERT INTO dismissed_hotspots
                (latitude, longitude, defect_type)
            VALUES (?, ?, ?)
            """,
            (latitude, longitude, defect_type),
        )
        return cursor.lastrowid


def load_dismissed_hotspots(include_restored=False):
    """Load active dismissals or the full dismissal history."""
    query = """
        SELECT dismissal_id, dismissed_at, restored_at,
               latitude, longitude, defect_type
        FROM dismissed_hotspots
    """
    if not include_restored:
        query += " WHERE restored_at IS NULL"
    query += " ORDER BY dismissed_at DESC, dismissal_id DESC"

    with get_connection() as connection:
        return pd.read_sql_query(query, connection)


def restore_dismissed_hotspot(dismissal_id):
    """Restore a marker while retaining its dismissal audit record."""
    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE dismissed_hotspots
            SET restored_at = CURRENT_TIMESTAMP
            WHERE dismissal_id = ?
              AND restored_at IS NULL
            """,
            (int(dismissal_id),),
        )
        return cursor.rowcount == 1


def find_csv_import_by_hash(file_hash):
    """Return an existing CSV import with this hash, if any."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT import_id, original_filename, imported_at,
                   sample_count, event_count
            FROM csv_imports
            WHERE file_hash = ?
            """,
            (file_hash,),
        ).fetchone()

        return dict(row) if row else None

def register_csv_import(
    original_filename,
    file_hash,
    sample_count,
    event_count,
):
    """Register a CSV import and return its ID."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO csv_imports (
                original_filename,
                file_hash,
                sample_count,
                event_count
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                original_filename,
                file_hash,
                sample_count,
                event_count,
            ),
        )
        return cursor.lastrowid


def save_csv_road_defect_events(import_id, events):
    """Save detected CSV events linked to their import record."""
    import hashlib

    if events.empty:
        return 0

    saved_count = 0

    with sqlite3.connect(DATABASE_PATH) as connection:
        for _, event in events.iterrows():
            timestamp = pd.Timestamp(event["timestamp"]).isoformat()

            # Build a stable identifier to prevent duplicate event records.
            fingerprint_data = "|".join(
                [
                    str(import_id),
                    str(event["vehicle_id"]),
                    timestamp,
                    str(event["latitude"]),
                    str(event["longitude"]),
                    str(event["kind"]),
                ]
            )
            fingerprint = hashlib.sha256(
                fingerprint_data.encode("utf-8")
            ).hexdigest()

            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO csv_road_defect_events (
                    import_id,
                    event_fingerprint,
                    vehicle_id,
                    event_timestamp,
                    latitude,
                    longitude,
                    kind,
                    peak_mps2,
                    rms_mps2,
                    severity,
                    speed_mps,
                    duration_s,
                    n_samples,
                    distance_along_route_m,
                    detector
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    import_id,
                    fingerprint,
                    str(event["vehicle_id"]),
                    timestamp,
                    float(event["latitude"]),
                    float(event["longitude"]),
                    str(event["kind"]),
                    _database_value(event.get("peak_mps2")),
                    _database_value(event.get("rms_mps2")),
                    _database_value(event.get("severity")),
                    _database_value(event.get("speed_mps")),
                    _database_value(event.get("duration_s")),
                    _database_value(event.get("n_samples")),
                    _database_value(
                        event.get("distance_along_route_m")
                    ),
                    _database_value(event.get("detector")),
                ),
            )

            saved_count += cursor.rowcount

    return saved_count


def save_injected_pothole_config(config):
    """Persist one or more configured road defects."""
    if config is None:
        configs = []
    elif isinstance(config, dict):
        # Keep compatibility with callers that supply one defect.
        configs = [config]
    elif isinstance(config, list):
        configs = config
    else:
        raise TypeError("Defect configuration must be a dict, list, or None.")

    
    # Validate before modifying the saved configuration.
    for item in configs:
        required = (
            "defect_id",
            "defect_type",
            "distance_along_route_m",
            "severity",
        )
        if any(key not in item for key in required):
            raise ValueError(
                "A defect configuration is missing required fields."
            )
    with get_connection() as connection:
        # Replace only the configuration list, not simulation history.
        connection.execute("DELETE FROM injected_defect_configs")

        
        connection.executemany("""
            INSERT INTO injected_defect_configs (
                defect_id,
                defect_type,
                distance_along_route_m,
                severity,
                latitude,
                longitude,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            (
                item["defect_id"],
                item["defect_type"],
                float(item["distance_along_route_m"]),
                item["severity"],
                item.get("latitude"),
                item.get("longitude"),
            )
            for item in configs
        ])


        # Clear the legacy single-defect setting after the new
        # configuration has been saved successfully.
        connection.execute(
            "DELETE FROM injected_pothole_config WHERE id = 1"
        )


def load_injected_pothole_config():
    """Load all configured defects, migrating the legacy setting if needed."""
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT defect_id, defect_type,
                   distance_along_route_m, severity,
                   latitude, longitude
            FROM injected_defect_configs
            ORDER BY rowid
        """).fetchall()

        if rows:
            return [
                {
                    "defect_id": row[0],
                    "defect_type": row[1],
                    "distance_along_route_m": row[2],
                    "severity": row[3],
                    "latitude": row[4],
                    "longitude": row[5],
                }
                for row in rows
            ]

        # If the new table is empty, preserve compatibility with
        # a configuration saved by the previous version.
        legacy_row = connection.execute("""
            SELECT defect_id, defect_type,
                   distance_along_route_m, severity
            FROM injected_pothole_config
            WHERE id = 1
        """).fetchone()

        if legacy_row is None:
            return []

        legacy_config = {
            "defect_id": legacy_row[0],
            "defect_type": legacy_row[1],
            "distance_along_route_m": legacy_row[2],
            "severity": legacy_row[3],
        }

        connection.execute("""
            INSERT OR IGNORE INTO injected_defect_configs (
                defect_id, defect_type,
                distance_along_route_m, severity
            )
            VALUES (?, ?, ?, ?)
        """, (
            legacy_config["defect_id"],
            legacy_config["defect_type"],
            float(legacy_config["distance_along_route_m"]),
            legacy_config["severity"],
        ))

        return [legacy_config]


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized at: {DATABASE_PATH}")