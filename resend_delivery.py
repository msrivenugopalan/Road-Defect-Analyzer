
"""Optional Resend delivery adapter for RoadSenseAI.

No network request is made by build_resend_payload().
Actual delivery requires an explicit resend-mode configuration.
"""

import base64
import json
import urllib.error
import urllib.request


RESEND_ENDPOINT = "https://api.resend.com/emails"


def build_resend_payload(message):
    """Convert an EmailMessage into a Resend API payload locally."""
    html_part = message.get_body(preferencelist=("html",))
    text_part = message.get_body(preferencelist=("plain",))

    payload = {
        "from": str(message["From"]),
        "to": [
            address.strip()
            for address in str(message["To"]).split(",")
            if address.strip()
        ],
        "subject": str(message["Subject"]),
    }

    if text_part is not None:
        payload["text"] = text_part.get_content()

    if html_part is not None:
        payload["html"] = html_part.get_content()

    attachments = []

    for part in message.iter_attachments():
        content = part.get_payload(decode=True)
        if content is None:
            continue

        attachments.append({
            "filename": part.get_filename() or "attachment.bin",
            "content": base64.b64encode(content).decode("ascii"),
        })

    if attachments:
        payload["attachments"] = attachments

    return payload


def send_via_resend(message, cfg, timeout=20):
    """Send through Resend. This function makes a real network request."""
    if cfg.mode != "resend":
        raise RuntimeError(
            "Resend delivery blocked: MAIL_MODE must explicitly be 'resend'."
        )

    if not cfg.resend_api_key:
        raise RuntimeError("RESEND_API_KEY is not configured.")

    if not cfg.resend_from:
        raise RuntimeError("RESEND_FROM is not configured.")

    payload = build_resend_payload(message)
    payload["from"] = cfg.resend_from

    request = urllib.request.Request(
        RESEND_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {cfg.resend_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result
    except urllib.error.HTTPError as error:
        # Avoid exposing request headers or credentials.
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Resend API returned HTTP {error.code}: {detail[:500]}"
        ) from None
