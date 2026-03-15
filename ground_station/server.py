# server.py — Ground Station entry point
#
# Run from ground_station/:
#   python server.py
#
# Starts:
#   1. Logging (stdout + file)
#   2. Data directories
#   3. MissionState, Pipeline, Commander
#   4. TCP listener thread (port 5000) — waits for CubeSat to push data
#   5. Flask dashboard (port 3000) — serves the mission ops UI

import logging
import os
import sys
import threading

# ── Make ground_station/ importable as the package root ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import dashboard.app as dash_app
from processing.mission_state import MissionState
from processing.pipeline import Pipeline
from receiver import listener
from uplink.commander import Commander

# ─────────────────────────────────────────────────────────────────────────────
# Directories to create on startup
# ─────────────────────────────────────────────────────────────────────────────
_REQUIRED_DIRS = [
    config.RECEIVED_DIR,
    config.TELEMETRY_DIR,
    os.path.join(config.PROCESSED_DIR, "shadow_masks"),
    os.path.join(config.PROCESSED_DIR, "hazard_maps"),
    os.path.join(config.PROCESSED_DIR, "change_maps"),
    os.path.join(config.PROCESSED_DIR, "mosaics"),
    os.path.join(config.PROCESSED_DIR, "routes"),
    "data/logs",
]


def _create_dirs():
    for d in _REQUIRED_DIRS:
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def _setup_logging():
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler — written to data/logs/gcs.log
    log_path = os.path.join("data", "logs", "gcs.log")
    try:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception as e:
        # Don't crash if log file can't be opened — console logging still works
        print(f"[WARNING] Could not open log file {log_path}: {e}", file=sys.stderr)

    # Silence Flask/Werkzeug access log spam — keep only WARNING+
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1. Logging first so everything below is visible
    _setup_logging()
    logger = logging.getLogger(__name__)

    # 2. Directories
    _create_dirs()
    logger.info("Data directories verified")

    # 3. Core objects
    mission_state = MissionState()
    pipeline      = Pipeline(mission_state)
    commander     = Commander()
    logger.info("MissionState, Pipeline, Commander initialised")

    # 4. Wire up dashboard
    dash_app.set_pipeline(pipeline)
    dash_app.set_mission_state(mission_state)
    dash_app.set_commander(commander)

    # Patch pipeline to push quality entries into the dashboard log
    _patch_pipeline_quality_hook(pipeline)

    # 5. Wire up listener → pipeline callback
    listener.set_pipeline_callback(
        lambda path, meta, quality: pipeline.process(path, meta, quality)
    )

    # 6. Start TCP listener in a background daemon thread
    listener_thread = threading.Thread(
        target=listener.start_listener,
        name="tcp-listener",
        daemon=True,
    )
    listener_thread.start()
    logger.info(f"TCP listener started (port {config.LISTEN_PORT})")

    # 7. Banner
    print()
    print("=" * 60)
    print("  MuraltZ Ground Station Online")
    print(f"  Listening for CubeSat on port {config.LISTEN_PORT}")
    print(f"  Dashboard → http://localhost:{config.DASHBOARD_PORT}")
    if not config.CUBESAT_IP:
        print("  [!] CUBESAT_IP is not set in config.py")
    print("=" * 60)
    print()

    # 8. Flask — blocks here until Ctrl-C
    try:
        dash_app.app.run(
            host="0.0.0.0",
            port=config.DASHBOARD_PORT,
            debug=False,       # debug=True restarts the process, killing the listener thread
            use_reloader=False,
        )
    except KeyboardInterrupt:
        logger.info("Shutdown requested — exiting")


def _patch_pipeline_quality_hook(pipeline: Pipeline):
    """
    Wrap pipeline._process_locked so it appends a quality entry to the
    dashboard log after each image is processed.

    This avoids importing dash_app inside pipeline.py (which would create a
    circular dependency: pipeline → dashboard → pipeline).
    """
    original = pipeline._process_locked

    def _hooked(image_path, metadata, ground_quality):
        original(image_path, metadata, ground_quality)
        _push_quality_entry(image_path, metadata, ground_quality)

    pipeline._process_locked = _hooked


def _push_quality_entry(image_path: str, metadata: dict, ground_quality: dict):
    """Build a quality log entry and push it to the dashboard."""
    import os
    entry = {
        "filename":       os.path.basename(image_path),
        "cubesat_score":  metadata.get("combined_score"),
        "ground_passed":  ground_quality.get("passed", True),
        "notes":          ground_quality.get("notes", []),
        "status":         "flagged" if not ground_quality.get("passed", True) else "ok",
    }
    try:
        dash_app.append_quality_entry(entry)
    except Exception:
        pass  # dashboard not critical path


if __name__ == "__main__":
    main()
