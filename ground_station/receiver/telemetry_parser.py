# receiver/telemetry_parser.py — Parses and stores CubeSat telemetry packets
#
# The CubeSat sends a telemetry JSON blob (type="telemetry" in the header).
# This module:
#   1. Decodes and validates the JSON
#   2. Saves it to data/telemetry/ with a timestamped filename
#   3. Updates the in-memory latest_telemetry dict read by the dashboard

import json
import logging
import os
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)

# In-memory store of the most recent telemetry — read by dashboard/app.py
_latest_telemetry: dict = {}


def get_latest_telemetry() -> dict:
    """Return the most recently received telemetry dict (empty if none yet)."""
    return dict(_latest_telemetry)


def parse_and_save_telemetry(data: bytes, filename: str) -> dict:
    """
    Decode telemetry bytes, validate structure, persist to disk, update cache.

    Args:
        data:     Raw bytes received from the CubeSat (JSON-encoded telemetry).
        filename: Original filename from the transfer header (used for log context).

    Returns:
        Parsed telemetry dict, or empty dict on failure.
    """
    global _latest_telemetry

    # Decode JSON
    try:
        telemetry = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Failed to decode telemetry JSON from '{filename}': {e}")
        return {}

    if not isinstance(telemetry, dict):
        logger.error(f"Telemetry from '{filename}' is not a JSON object — discarding")
        return {}

    # Stamp with ground-receive time if the CubeSat didn't include one
    if "ground_received_utc" not in telemetry:
        telemetry["ground_received_utc"] = datetime.now(timezone.utc).isoformat()

    _log_summary(telemetry, filename)

    # Persist to disk
    _save_to_disk(telemetry, filename)

    # Update in-memory cache for dashboard
    _latest_telemetry = telemetry

    return telemetry


def _save_to_disk(telemetry: dict, source_filename: str):
    """Save telemetry JSON to data/telemetry/ with a timestamped name."""
    os.makedirs(config.TELEMETRY_DIR, exist_ok=True)

    # Build a timestamped filename: telemetry_YYYYMMDD_HHMMSS_SSS.json
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    save_name = f"telemetry_{ts}.json"
    save_path = os.path.join(config.TELEMETRY_DIR, save_name)

    with open(save_path, "w") as f:
        json.dump(telemetry, f, indent=2)

    logger.debug(f"Telemetry from '{source_filename}' saved to '{save_path}'")


def _log_summary(telemetry: dict, source: str):
    """Log a one-line summary of key telemetry fields."""
    state = telemetry.get("state", "?")
    pass_n = telemetry.get("pass_number", "?")
    roll = telemetry.get("roll_deg", "?")
    pitch = telemetry.get("pitch_deg", "?")
    temp = telemetry.get("cpu_temp_c", "?")
    storage_pct = telemetry.get("storage_used_pct", "?")

    logger.info(
        f"Telemetry [{source}]: state={state} pass={pass_n} "
        f"roll={roll}° pitch={pitch}° cpu_temp={temp}°C storage={storage_pct}%"
    )
