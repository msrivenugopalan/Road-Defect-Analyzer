"""
Reporting: emails a hotspot to the road authority.

Safety features (an email to a real authority is hard to un-send):

* DRY RUN is the default. Emails are written as .eml files to ./outbox
  and logged, but nothing leaves the machine until MAIL_MODE=smtp.
* DEDUPE. A hotspot already reported nearby within the cooldown is not
  reported again (unless force=True).
* CAP. At most `max_emails_per_batch` emails per batch.
* SIMULATION LABEL. Reports built from simulated data say so in the
  subject and body, so simulated potholes can never be mistaken for real.
* PRIVACY. Vehicle identities are hashed in the email and attachment.
* AUDIT TRAIL. Every attempt (sent / dry_run / failed) is logged to the
  `reports` table.
"""

from __future__ import annotations

import hashlib
import html
import io
import os
import re
import smtplib
import ssl
import time
import uuid
import base64
from datetime import datetime
from email.message import EmailMessage
from email.utils import make_msgid

import pandas as pd
from database import (
    find_recent_report,
    find_report_for_defect,
    log_report,
)
import database
from config import MailConfig
from resend_delivery import send_via_resend
from geo import maps_url, osm_url

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")



def anonymise(vehicle_key: str) -> str:
    """Return a sequential display ID for a simulated vehicle."""
    match = re.search(r"vehicle[_ -]?(\d+)", str(vehicle_key), re.IGNORECASE)

    if match:
        return f"V-{int(match.group(1)):02d}"

    # Fallback for IDs that do not contain a vehicle number.
    return str(vehicle_key)



def validate_config(cfg: MailConfig) -> list:
    """Return a list of human-readable problems (empty list = OK)."""
    problems = []

    if cfg.mode not in ("dry_run", "smtp", "resend"):
        problems.append(
            "MAIL_MODE must be 'dry_run', 'smtp', or 'resend'."
        )

    if not cfg.recipients:
        problems.append("No authority email address is configured.")

    for address in cfg.recipients:
        if not EMAIL_PATTERN.match(address):
            problems.append(f"Invalid recipient address: {address}")

    if cfg.mode == "smtp":
        if not cfg.host:
            problems.append("SMTP_HOST is not set.")
        if not cfg.sender or not EMAIL_PATTERN.match(cfg.sender):
            problems.append("MAIL_FROM / SMTP_USER is not a valid address.")

    if cfg.mode == "resend":
        if not cfg.resend_api_key:
            problems.append("RESEND_API_KEY is not set.")
        if not cfg.resend_from or not EMAIL_PATTERN.match(cfg.resend_from):
            problems.append("RESEND_FROM is not a valid email address.")

    return problems



def _evidence_csv(hotspot: dict, events: pd.DataFrame | None) -> bytes:
    """Create evidence with sequential IDs for simulated vehicles."""
    if events is None or events.empty:
        return b""

    frame = events.copy()
    
    # Format timestamps consistently for reports.
    if "timestamp" in frame.columns:
        frame["timestamp"] = (
            pd.to_datetime(frame["timestamp"], errors="coerce")
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )


    # Use the run-specific key to distinguish vehicles from different runs.
    key = (
        "vehicle_key"
        if "vehicle_key" in frame.columns
        else "vehicle_id"
    )

    # Assign one consistent display ID to each distinct vehicle in this
    # report's evidence. Sorting makes the mapping reproducible.
    unique_vehicles = sorted(
        frame[key].dropna().astype(str).unique()
    )
    vehicle_labels = {
        vehicle: f"V-{index:02d}"
        for index, vehicle in enumerate(unique_vehicles, start=1)
    }

    frame["vehicle"] = (
        frame[key].astype(str).map(vehicle_labels)
    )

    columns = [
        c for c in (
            "vehicle", "timestamp", "latitude", "longitude",
            "kind", "severity", "peak_mps2", "speed_mps",
        ) if c in frame.columns
    ]

    buffer = io.StringIO()
    frame[columns].to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def build_report_email(
    hotspot: dict,
    events: pd.DataFrame | None,
    cfg: MailConfig,
    data_source: str = "simulation",
) -> EmailMessage:
    """Compose the email (plain text + HTML + CSV evidence)."""
    lat, lon = float(hotspot["latitude"]), float(hotspot["longitude"])
    kind = str(hotspot.get("defect_type", "defect")).replace("_", " ")
    severity = str(hotspot.get("max_severity", "unknown"))
    vehicles = int(hotspot["distinct_vehicle_count"])
    count = int(hotspot["event_count"])
    simulated = data_source == "simulation"

    prefix = "[SIMULATED DATA] " if simulated else ""
    subject = (
        f"{prefix}[RoadSenseAI] {kind.title()} reported at "
        f"{lat:.5f}, {lon:.5f} ({vehicles} vehicles)"
    )

    first = pd.to_datetime(
            hotspot["first_seen"], errors="coerce"
    ).strftime("%Y-%m-%d %H:%M:%S")

    last = pd.to_datetime(
            hotspot["last_seen"], errors="coerce"
    ).strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "Road defect report",
        "==================",
        "",
    ]

    if simulated:
        lines += [
            "NOTE: This report was generated from SIMULATED data for",
            "testing. It does not describe a real road. Please ignore.",
            "",
        ]

    lines += [
        f"Defect type     : {kind}",
        f"Severity (max)  : {severity}",
        f"Location        : {lat:.6f}, {lon:.6f}",
        f"Google Maps     : {maps_url(lat, lon)}",
        f"OpenStreetMap   : {osm_url(lat, lon)}",
        f"Distinct vehicles that detected it : {vehicles}",
        f"Total detections                   : {count}",
        f"First detected  : {first}",
        f"Last detected   : {last}",
        f"Position spread : about {float(hotspot.get('spread_m', 0)):.0f} m",
        
    ]
    body = "\n".join(lines)

    
    html_body = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(subject)}</title>"
        "</head><body style='font-family:Arial,sans-serif;line-height:1.5;"
        "max-width:850px;margin:30px auto;padding:0 20px;color:#222'>"
        "<h2>Road defect report</h2>"
        + (
            "<p style='color:#b00'><b>SIMULATED DATA "
        )
        + "<table cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse;width:100%'>"
        f"<tr><td><b>Defect type</b></td><td>{html.escape(kind)}</td></tr>"
        f"<tr><td><b>Severity (max)</b></td><td>{html.escape(severity)}</td></tr>"
        f"<tr><td><b>Location</b></td><td>{lat:.6f}, {lon:.6f}</td></tr>"
        f"<tr><td><b>Distinct vehicles</b></td><td>{vehicles}</td></tr>"
        f"<tr><td><b>Total detections</b></td><td>{count}</td></tr>"
        f"<tr><td><b>First detected</b></td><td>{html.escape(first)}</td></tr>"
        f"<tr><td><b>Last detected</b></td><td>{html.escape(last)}</td></tr>"
        f"<tr><td><b>Position spread</b></td>"
        f"<td>about {float(hotspot.get('spread_m', 0)):.0f} m</td></tr>"
        "</table>"
        f"<p><b>Coordinates:</b> {lat:.6f}, {lon:.6f}</p>"
        f"<p><a href='{html.escape(maps_url(lat, lon), quote=True)}' "
        "target='_blank' rel='noopener noreferrer'>Open in Google Maps</a>"
        
        
        
        "</body></html>"
    )


    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = cfg.sender or "roadsenseai@localhost"
    message["To"] = ", ".join(cfg.recipients)
    message["Date"] = datetime.now().astimezone().strftime(
        "%a, %d %b %Y %H:%M:%S %z"
    )
    message["Message-ID"] = make_msgid(domain="roadsenseai.local")
    message.set_content(body)
    message.add_alternative(html_body, subtype="html")

    evidence = _evidence_csv(hotspot, events)

    if evidence:
        message.add_attachment(
            evidence, maintype="text", subtype="csv",
            filename="evidence_detections.csv",
        )

    return message


def build_report_html(
    hotspot: dict,
    events: pd.DataFrame | None,
    cfg: MailConfig,
    data_source: str = "simulation",
) -> str:
    """Build a standalone HTML report with anonymised evidence."""
    message = build_report_email(hotspot, events, cfg, data_source)

    html_part = message.get_body(preferencelist=("html",))
    if html_part is None:
        raise ValueError("Could not generate the HTML report body.")

    html_body = html_part.get_content()
    evidence = _evidence_csv(hotspot, events)

    if evidence:
        evidence_frame = pd.read_csv(io.BytesIO(evidence))

        if "timestamp" in evidence_frame.columns:
            evidence_frame["timestamp"] = pd.to_datetime(
                evidence_frame["timestamp"], errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M:%S")
        evidence_table = evidence_frame.to_html(
            index=False,
            escape=True,
            border=0,
            classes="evidence-table",
        )

        evidence_csv = base64.b64encode(evidence).decode("ascii")

        evidence_section = f"""
        <hr>
        <h2>Simulated Vehicle Detection Evidence</h2>
        <p>Vehicle simulation is represented with vehicle IDs that can be represented as V-01 to V-20.</p>
        <p>
          <a download="evidence_detections.csv"
             href="data:text/csv;base64,{evidence_csv}">
             Download Evidence CSV
          </a>
        </p>
        {evidence_table}
        """

        html_body = html_body.replace(
            "</body>",
            evidence_section + "</body>",
        )

    return html_body

def _send_via_smtp(message: EmailMessage, cfg: MailConfig, attempts=3):
    """Send with retries and exponential backoff. Returns attempts used."""
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            if cfg.port == 465:
                server = smtplib.SMTP_SSL(
                    cfg.host, cfg.port, timeout=30,
                    context=ssl.create_default_context(),
                )
            else:
                server = smtplib.SMTP(cfg.host, cfg.port, timeout=30)

            try:
                if cfg.port != 465 and cfg.use_tls:
                    server.starttls(context=ssl.create_default_context())

                if cfg.user:
                    server.login(cfg.user, cfg.password)

                server.send_message(message)
            finally:
                try:
                    server.quit()
                except Exception:
                    pass

            return attempt

        except Exception as error:  # network / auth / rejected
            last_error = error

            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"SMTP failed after {attempts} attempts: {last_error}"
    )


def send_report(
    hotspot: dict,
    events: pd.DataFrame | None,
    cfg: MailConfig,
    data_source: str = "simulation",
    run_id=None,
    force: bool = False,
) -> dict:
    """
    Report one hotspot. Returns a result dict with a `status`:
    sent | dry_run | duplicate | invalid_config | failed
    """
    lat, lon = float(hotspot["latitude"]), float(hotspot["longitude"])

    problems = validate_config(cfg)

    if problems:
        return {"status": "invalid_config", "detail": "; ".join(problems)}

    if not force:
        # A dry run only blocks other dry runs; a real send blocks both.
        blocking = ("sent",) if cfg.mode == "smtp" else ("sent", "dry_run")
        previous = database.find_recent_report(
            lat, lon, cfg.dedupe_radius_m, cfg.cooldown_days, blocking
        )
    defect_id = hotspot.get("defect_id") or hotspot.get("hotspot_id")

    defect_id = hotspot.get("defect_id") or hotspot.get("hotspot_id")

    if not force and defect_id:
        existing_defect_report = find_report_for_defect(defect_id)

        if existing_defect_report:
            return {
                "status": "duplicate",
                "report_id": existing_defect_report["report_id"],
                "outbox_path": existing_defect_report.get("outbox_path"),
                "detail": (
                    f"Defect {defect_id} already has a saved report."
                ),
            }

        if not force:
            # A dry run only blocks other dry runs; a real send blocks both.
            blocking = ("sent",) if cfg.mode == "smtp" else ("sent", "dry_run")

            previous = database.find_recent_report(
                lat,
                lon,
                cfg.dedupe_radius_m,
                cfg.cooldown_days,
                blocking,
            )

            if previous:
                return {
                    "status": "duplicate",
                    "report_id": previous.get("report_id"),
                    "outbox_path": previous.get("outbox_path"),
                    "detail": (
                        f"Already reported on {previous['created_at']} "
                        f"(report #{previous['report_id']}). "
                        "Reusing the existing report."
                    ),
                }

    message = build_report_email(hotspot, events, cfg, data_source)
    log_fields = dict(
        latitude=lat, longitude=lon,
        defect_type=hotspot.get("defect_type"),
        severity=hotspot.get("max_severity"),
        distinct_vehicles=hotspot.get("distinct_vehicle_count"),
        event_count=hotspot.get("event_count"),
        recipient=message["To"], subject=message["Subject"],
        message_id=message["Message-ID"], run_id=run_id, source=data_source,
        defect_id=defect_id,
    )

    
    if cfg.mode == "dry_run":
        cfg.outbox_dir.mkdir(parents=True, exist_ok=True)

        path = cfg.outbox_dir / (
            f"report_{datetime.now():%Y%m%d_%H%M%S}_"
            f"{uuid.uuid4().hex[:6]}.html"
        )

        html_report = build_report_html(
            hotspot,
            events,
            cfg,
            data_source,
        )
        path.write_text(html_report, encoding="utf-8")

        report_id = database.log_report(
            status="dry_run",
            outbox_path=path,
            **log_fields,
        )

        return {
            "status": "dry_run",
            "report_id": report_id,
            "outbox_path": str(path),
            "subject": message["Subject"],
        }

    
    
    try:
        if cfg.mode == "resend":
            send_via_resend(message, cfg)
            attempts = 1
        else:
            attempts = _send_via_smtp(message, cfg)

    except Exception as error:
        failed_attempts = 1 if cfg.mode == "resend" else 3

        report_id = database.log_report(
            status="failed",
            error=str(error),
            attempts=failed_attempts,
            **log_fields,
        )
        return {
            "status": "failed",
            "report_id": report_id,
            "detail": str(error),
        }

    report_id = database.log_report(
        status="sent",
        attempts=attempts,
        **log_fields,
    )
    return {
        "status": "sent",
        "report_id": report_id,
        "subject": message["Subject"],
    }


    

def send_reports_for_hotspots(
    hotspots: pd.DataFrame,
    events: pd.DataFrame,
    cfg: MailConfig,
    data_source="simulation",
    run_id=None,
    force=False,
) -> list:
    """Report every eligible hotspot, highest priority first, capped."""
    eligible = hotspots[hotspots["report_eligible"]].sort_values(
        "priority_score", ascending=False
    )
    results = []

    for position, (_, hotspot) in enumerate(eligible.iterrows()):
        if position >= cfg.max_emails_per_batch:
            results.append({
                "hotspot_id": hotspot["hotspot_id"],
                "status": "skipped_cap",
                "detail": (
                    f"Batch cap of {cfg.max_emails_per_batch} emails reached."
                ),
            })
            continue

        # Pass only the events that belong to this hotspot (<= spread).
        from geo import haversine_m

        near = events[
            haversine_m(
                hotspot["latitude"], hotspot["longitude"],
                events["latitude"].to_numpy(),
                events["longitude"].to_numpy(),
            ) <= max(float(hotspot["spread_m"]) + 1.0, 15.0)
        ]

        result = send_report(
            hotspot.to_dict(), near, cfg, data_source, run_id, force
        )
        result["hotspot_id"] = hotspot["hotspot_id"]
        results.append(result)

    return results
