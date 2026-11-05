"""
Central configuration for RoadSenseAI.

Every tunable number lives here, so the detector, the hotspot engine and
the dashboard never disagree about a threshold.

Secrets (SMTP password etc.) are read from environment variables or a
local `.env` file. They are never hard-coded and never committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE per line). Existing env vars win."""
    path = path or (BASE_DIR / ".env")

    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip(), value.strip().strip('"').strip("'")
        )


load_dotenv()


@dataclass(frozen=True)
class DetectionConfig:
    """Thresholds for the vertical-acceleration defect detector."""

    # A pothole is a short spike. It must exceed this absolute level ...
    pothole_peak_threshold_mps2: float = 1.5
    # ... AND stand out from the local road texture by this many "sigmas".
    pothole_prominence_k: float = 5.0

    # A bad (rough) road is *sustained* vibration, measured as the rolling
    # median of |vertical acceleration|. The median ignores single spikes.
    rough_median_threshold_mps2: float = 0.35
    min_rough_duration_s: float = 2.0

    # Samples closer together than this belong to the same "pass".
    merge_gap_s: float = 1.0

    # Window used for gravity/baseline removal.
    baseline_window_s: float = 5.0

    # Window for the rough-road test. Longer than a slow pothole crossing,
    # so a single pothole is a small fraction of it and cannot make the
    # median look "rough".
    rough_window_s: float = 8.0

    # Window used to estimate normal road texture. It is deliberately much
    # longer than a pothole crossing, so a pothole cannot inflate its own
    # noise estimate and then fail its own prominence test.
    noise_window_s: float = 15.0

    # Never trust a noise estimate below this (avoids divide-by-near-zero).
    min_noise_floor_mps2: float = 0.05

    # Pothole severity by peak vertical acceleration (m/s^2).
    medium_peak_mps2: float = 3.0
    high_peak_mps2: float = 5.0

    # Rough-road severity by local noise level (m/s^2).
    rough_medium_sigma_mps2: float = 0.9
    rough_high_sigma_mps2: float = 1.6


@dataclass(frozen=True)
class HotspotConfig:
    """Rules for turning individual detections into a reportable hotspot."""

    # GPS is never exact (phones: roughly 3-10 m), so "the same coordinate"
    # means "within this radius".
    radius_m: float = 10.0
    # The rule from the project brief: this many DISTINCT vehicles.
    min_vehicles: int = 5
    # Old detections expire so a pothole fixed last year is not reported.
    window_days: int = 30


@dataclass(frozen=True)
class MailConfig:
    """Email settings. Default mode is a safe dry run (no real email)."""

    mode: str = "dry_run"  # "dry_run" or "smtp"

    resend_api_key: str = field(default="", repr=False)
    resend_from: str = ""
    host: str = ""
    port: int = 587
    user: str = ""
    password: str = field(default="", repr=False)
    use_tls: bool = True
    sender: str = ""
    recipients: tuple = ()
    # Do not re-report a hotspot that was already reported nearby ...
    dedupe_radius_m: float = 25.0
    # ... within this many days.
    cooldown_days: int = 30
    # Safety valve against a bug spamming an authority.
    max_emails_per_batch: int = 10
    outbox_dir: Path = BASE_DIR / "outbox"

    @classmethod
    def from_env(cls) -> "MailConfig":
        recipients = tuple(
            address.strip()
            for address in os.environ.get(
                "AUTHORITY_EMAIL", ""
            ).split(",")
            if address.strip()
        )

        
        return cls(
            mode=os.environ.get("MAIL_MODE", "dry_run").strip().lower(),
            resend_api_key=os.environ.get("RESEND_API_KEY", "").strip(),
            resend_from=os.environ.get("RESEND_FROM", "").strip(),
            host=os.environ.get("SMTP_HOST", ""),
            port=int(os.environ.get("SMTP_PORT", "587")),
            user=os.environ.get("SMTP_USER", ""),
            password=os.environ.get("SMTP_PASSWORD", ""),
            use_tls=os.environ.get("SMTP_USE_TLS", "1") != "0",
            sender=os.environ.get(
                "MAIL_FROM", os.environ.get("SMTP_USER", "")
            ),
            recipients=recipients,
        )



# Presets used by the simulator and the dashboard.
REALISM_OFF = {
    "sensor_noise_std_mps2": 0.0,
    "gps_noise_std_m": 0.0,
    "gps_noise_tau_s": 5.0,
    "speed_factor_range": (1.0, 1.0),
    "start_jitter_s": 0.0,
    "defect_hit_probability": 1.0,
}

REALISM_TYPICAL = {
    "sensor_noise_std_mps2": 0.25,   # normal road texture / engine vibration
    "gps_noise_std_m": 3.0,          # typical consumer GPS error
    "gps_noise_tau_s": 5.0,          # GPS error drifts slowly, not white noise
    "speed_factor_range": (0.85, 1.05),
    "start_jitter_s": 120.0,
    "defect_hit_probability": 0.85,  # some drivers swerve around potholes
}
