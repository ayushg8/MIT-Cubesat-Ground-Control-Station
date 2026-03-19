#!/usr/bin/env python3
"""mock_pipeline_run.py — Simulate a live CubeSat pass through the GCS.

Feeds existing images from data/received_images/ through the full pipeline,
simulating realistic telemetry, downlink progress bars, and state transitions
so the dashboard looks like a real mission is happening.

Run from ground_station/:
    python3 tools/mock_pipeline_run.py

Then open http://localhost:3000 to watch.
"""

import glob
import json
import logging
import os
import random
import shutil
import sys
import time
import threading
from datetime import datetime, timezone

# Make ground_station/ importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import dashboard.app as dash_app
from processing.mission_state import MissionState
from processing.pipeline import Pipeline
from receiver.quality_check import run_ground_quality_check
from receiver.downlink_state import get_state as get_downlink_state
from receiver import telemetry_parser

# ── Config ──────────────────────────────────────────────────────────────────
PASSES_TO_USE = [1, 2]     # None = all, or e.g. [29, 30]
MAX_IMAGES = 0             # 0 = all matching images
DOWNLINK_SPEED = 4000      # Simulated bytes/sec for progress bar (faster than real 1200)
DELAY_BETWEEN = 0.5        # Seconds between images within a downlink window

# ── Logging ─────────────────────────────────────────────────────────────────
def setup_logging():
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


# ── Find images grouped by pass ─────────────────────────────────────────────
def find_images_by_pass():
    """Return dict: {pass_number: [(jpg_path, meta_path), ...]}"""
    pattern = os.path.join(config.RECEIVED_DIR, "*.jpg")
    all_jpgs = sorted(glob.glob(pattern))

    passes = {}
    for jpg_path in all_jpgs:
        meta_path = jpg_path.replace(".jpg", "_meta.json")
        if not os.path.exists(meta_path):
            continue

        basename = os.path.basename(jpg_path)
        try:
            pass_num = int(basename.split("_")[0].replace("pass", ""))
        except ValueError:
            continue

        if PASSES_TO_USE is not None and pass_num not in PASSES_TO_USE:
            continue

        passes.setdefault(pass_num, []).append((jpg_path, meta_path))

    return passes


# ── Clear old processed data ────────────────────────────────────────────────
def clear_processed_data():
    dirs_to_clear = [
        os.path.join(config.PROCESSED_DIR, "shadow_masks"),
        os.path.join(config.PROCESSED_DIR, "hazard_maps"),
        os.path.join(config.PROCESSED_DIR, "change_maps"),
        os.path.join(config.PROCESSED_DIR, "mosaics"),
        os.path.join(config.PROCESSED_DIR, "routes"),
        os.path.join(config.PROCESSED_DIR, "mosaic_database"),
        os.path.join(config.PROCESSED_DIR, "segmentation_maps"),
        os.path.join(config.PROCESSED_DIR, "yolo_detections"),
    ]
    for d in dirs_to_clear:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    for f in ["image_index.json", "cost_grid.json", "routes.json", "yolo_detections.json"]:
        p = os.path.join(config.PROCESSED_DIR, f)
        if os.path.exists(p):
            os.remove(p)

    if os.path.exists(config.MISSION_STATE_FILE):
        os.remove(config.MISSION_STATE_FILE)

    # Clear old telemetry
    for f in glob.glob(os.path.join(config.TELEMETRY_DIR, "*.json")):
        os.remove(f)


def create_dirs():
    for d in [
        config.RECEIVED_DIR, config.TELEMETRY_DIR,
        os.path.join(config.PROCESSED_DIR, "shadow_masks"),
        os.path.join(config.PROCESSED_DIR, "hazard_maps"),
        os.path.join(config.PROCESSED_DIR, "change_maps"),
        os.path.join(config.PROCESSED_DIR, "mosaics"),
        os.path.join(config.PROCESSED_DIR, "routes"),
        os.path.join(config.PROCESSED_DIR, "mosaic_database"),
        os.path.join(config.PROCESSED_DIR, "segmentation_maps"),
        os.path.join(config.PROCESSED_DIR, "yolo_detections"),
        "data/logs",
    ]:
        os.makedirs(d, exist_ok=True)


# ── Fake telemetry injection ────────────────────────────────────────────────
def inject_telemetry(state, pass_number, images_this_pass=0, total_images=0,
                     rejected=0, queued=0, sent=0, meta=None):
    """Push a fake telemetry packet so the dashboard shows live CubeSat state."""
    imu_data = {}
    cam_data = {}
    if meta:
        imu_data = meta.get("imu", {})
        cam_data = meta.get("camera", {})

    telem = {
        "type": "telemetry",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cubesat_id": "MURALTZ-01",
        "pass_number": pass_number,
        "state": state,
        "uptime_sec": int(time.time()) % 100000,
        "imu": {
            "accel": [0.2, -0.1, 9.78],
            "gyro": [0.01, -0.02, 0.005],
            "angular_rate": imu_data.get("angular_rate", round(random.uniform(0.01, 0.15), 3)),
            "stable": imu_data.get("stable", True),
            "nadir_locked": imu_data.get("nadir_locked", True),
            "nadir_angle_deg": imu_data.get("nadir_angle_deg", round(random.uniform(8, 25), 1)),
            "roll_deg": imu_data.get("roll_deg", round(random.uniform(-15, 15), 1)),
            "pitch_deg": imu_data.get("pitch_deg", round(random.uniform(-15, 15), 1)),
        },
        "camera": {
            "exposure_us": cam_data.get("exposure_us", 5000),
            "analog_gain": cam_data.get("analog_gain", 2.0),
            "lux": cam_data.get("lux", 200),
            "mode": "auto",
        },
        "thermal": {
            "cpu_temp_c": round(random.uniform(48, 58), 1),
            "throttled": False,
        },
        "storage": {
            "used_pct": round(random.uniform(20, 45), 1),
            "free_mb": round(random.uniform(800, 1500), 0),
        },
        "imaging": {
            "captured_this_pass": images_this_pass,
            "captured_total": total_images,
            "rejected_total": rejected,
            "rejection_breakdown": {"blur": rejected, "underexposed": 0, "overexposed": 0, "motion_blur": 0},
        },
        "downlink": {
            "queued": queued,
            "sent_total": sent,
            "bytes_this_pass": sent * 28000,
            "budget_remaining": max(0, 72000 - sent * 28000),
            "gcs_reachable": True,
        },
        "coverage": {
            "cells_filled": min(total_images, 64),
            "cells_total": 64,
            "pct": round(min(total_images, 64) / 64 * 100, 1),
        },
        "errors": [],
        "recent_log": [],
    }

    # Inject into the telemetry parser so dashboard picks it up
    raw = json.dumps(telem).encode("utf-8")
    telemetry_parser.parse_and_save_telemetry(raw, f"mock_pass{pass_number}.json")


# ── Simulate downlink progress bar ─────────────────────────────────────────
def simulate_downlink_progress(filename, file_size):
    """Animate the downlink progress bar in the dashboard."""
    dl = get_downlink_state()
    dl.start_transfer(filename, file_size)

    bytes_sent = 0
    chunk = DOWNLINK_SPEED // 10  # update 10 times per second
    while bytes_sent < file_size:
        bytes_sent = min(file_size, bytes_sent + chunk)
        dl.update_progress(bytes_sent)
        time.sleep(0.1)

    dl.set_status("validating")
    time.sleep(0.3)
    dl.set_status("processing")


def mark_downlink_complete():
    dl = get_downlink_state()
    dl.set_status("complete")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    setup_logging()
    logger = logging.getLogger("mock_run")

    create_dirs()
    clear_processed_data()

    passes = find_images_by_pass()
    if not passes:
        print("\n  No images found in data/received_images/ with metadata sidecars!\n")
        sys.exit(1)

    total_images = sum(len(v) for v in passes.values())
    if MAX_IMAGES > 0:
        # Trim passes to fit
        count = 0
        trimmed = {}
        for pn in sorted(passes):
            remaining = MAX_IMAGES - count
            if remaining <= 0:
                break
            trimmed[pn] = passes[pn][:remaining]
            count += len(trimmed[pn])
        passes = trimmed
        total_images = sum(len(v) for v in passes.values())

    print()
    print("=" * 60)
    print("  MuraltZ GCS — Simulated Mission")
    print("=" * 60)
    print()
    print(f"  Passes:       {sorted(passes.keys())}")
    print(f"  Total images: {total_images}")
    print(f"  Dashboard:    http://localhost:{config.DASHBOARD_PORT}")
    print()
    print("  Starting dashboard...")

    # Initialize core objects
    mission_state = MissionState()
    pipeline = Pipeline(mission_state)
    dash_app.set_pipeline(pipeline)
    dash_app.set_mission_state(mission_state)

    # Patch pipeline quality hook
    original = pipeline._process_locked
    def _hooked(image_path, metadata, ground_quality):
        original(image_path, metadata, ground_quality)
        entry = {
            "filename": os.path.basename(image_path),
            "cubesat_score": metadata.get("combined_score"),
            "ground_passed": ground_quality.get("passed", True),
            "notes": ground_quality.get("notes", []),
            "status": "flagged" if not ground_quality.get("passed", True) else "ok",
        }
        try:
            dash_app.append_quality_entry(entry)
        except Exception:
            pass
    pipeline._process_locked = _hooked

    # Start Flask in background
    flask_thread = threading.Thread(
        target=lambda: dash_app.app.run(
            host="0.0.0.0", port=config.DASHBOARD_PORT,
            debug=False, use_reloader=False,
        ),
        daemon=True,
    )
    flask_thread.start()
    time.sleep(1.5)

    print(f"  Dashboard live at http://localhost:{config.DASHBOARD_PORT}")
    print()

    # ── Simulate mission ────────────────────────────────────────────────────
    global_img_count = 0
    global_sent = 0
    global_rejected = 0

    for pass_idx, pass_number in enumerate(sorted(passes.keys())):
        images = passes[pass_number]
        num_images = len(images)

        # ── BOOT (first pass only) ──
        if pass_idx == 0:
            print("  ┌─ BOOT")
            inject_telemetry("BOOT", pass_number)
            time.sleep(2)
            print("  │  Hardware self-test... OK")
            print("  │")

        # ── WAITING ──
        print(f"  ┌─ WAITING (Pass {pass_number})")
        inject_telemetry("WAITING", pass_number,
                         total_images=global_img_count, sent=global_sent)
        time.sleep(3)
        print(f"  │  Operator: start_pass")
        print(f"  │")

        # ── IMAGING ──
        print(f"  ┌─ IMAGING (Pass {pass_number} — {num_images} images)")
        inject_telemetry("IMAGING", pass_number,
                         images_this_pass=0, total_images=global_img_count,
                         sent=global_sent)
        time.sleep(2)

        # Simulate captures appearing
        for i in range(num_images):
            jpg_path, meta_path = images[i]
            basename = os.path.basename(jpg_path)
            with open(meta_path) as f:
                metadata = json.load(f)

            global_img_count += 1
            inject_telemetry("IMAGING", pass_number,
                             images_this_pass=i + 1,
                             total_images=global_img_count,
                             rejected=global_rejected,
                             sent=global_sent,
                             meta=metadata)

            score = metadata.get("quality", {}).get("combined_score", 0.8)
            status = "PASS" if score > 0.5 else "REJECT"
            if status == "REJECT":
                global_rejected += 1
            print(f"  │  [{i+1:2d}/{num_images}] Captured {basename}  Q={score:.2f}  {status}")
            time.sleep(0.5)

        print(f"  │  Imaging complete: {num_images} captured")
        print(f"  │")

        # ── IDLE ──
        print(f"  ┌─ IDLE (building queue, aging priorities)")
        inject_telemetry("IDLE", pass_number,
                         images_this_pass=num_images,
                         total_images=global_img_count,
                         queued=num_images,
                         sent=global_sent)
        time.sleep(3)
        print(f"  │  Queue built: {num_images} images ready for downlink")
        print(f"  │")

        # ── DOWNLINK ──
        print(f"  ┌─ DOWNLINK (Pass {pass_number})")
        inject_telemetry("DOWNLINK", pass_number,
                         images_this_pass=num_images,
                         total_images=global_img_count,
                         queued=num_images,
                         sent=global_sent)
        time.sleep(1)

        # Set overall session info so dashboard shows total progress
        dl = get_downlink_state()
        session_total_bytes = sum(os.path.getsize(jp) for jp, _ in images)
        dl.start_session(total_images=num_images, total_bytes=session_total_bytes)

        for i, (jpg_path, meta_path) in enumerate(images):
            basename = os.path.basename(jpg_path)
            file_size = os.path.getsize(jpg_path)

            with open(meta_path) as f:
                metadata = json.load(f)

            # Update telemetry
            remaining = num_images - i
            inject_telemetry("DOWNLINK", pass_number,
                             images_this_pass=num_images,
                             total_images=global_img_count,
                             queued=remaining,
                             sent=global_sent + i,
                             meta=metadata)

            # Simulate downlink with progress bar
            print(f"  │  [{i+1:2d}/{num_images}] Downlinking {basename} ({file_size:,} bytes)...", end="", flush=True)

            # Run downlink progress animation in background
            dl_thread = threading.Thread(
                target=simulate_downlink_progress,
                args=(basename, file_size),
            )
            dl_thread.start()

            # While "downloading", wait for it
            dl_thread.join()

            # Now run through pipeline
            quality = run_ground_quality_check(jpg_path)

            t0 = time.monotonic()
            try:
                pipeline.process(jpg_path, metadata, quality)
                elapsed = time.monotonic() - t0
                mark_downlink_complete()
                global_sent += 1
                print(f" ACK ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.monotonic() - t0
                dl = get_downlink_state()
                dl.set_status("failed", str(e))
                print(f" FAIL ({elapsed:.1f}s) — {e}")

            # Brief pause between images
            if i < num_images - 1 and DELAY_BETWEEN > 0:
                time.sleep(DELAY_BETWEEN)

        inject_telemetry("DOWNLINK", pass_number,
                         images_this_pass=num_images,
                         total_images=global_img_count,
                         queued=0,
                         sent=global_sent)

        dl.end_session()
        print(f"  │  Downlink complete: {num_images} images sent")
        print(f"  │")
        print(f"  └─ Pass {pass_number} complete")
        print()

        # Brief pause between passes
        if pass_idx < len(passes) - 1:
            time.sleep(2)

    # ── Final state ──
    inject_telemetry("WAITING", max(passes.keys()),
                     total_images=global_img_count,
                     sent=global_sent,
                     rejected=global_rejected)

    # Check results
    mosaic_path = os.path.join(config.PROCESSED_DIR, "mosaics", "mosaic_latest.png")
    shadow_count = len(glob.glob(os.path.join(config.PROCESSED_DIR, "shadow_masks", "*.png")))
    hazard_count = len(glob.glob(os.path.join(config.PROCESSED_DIR, "hazard_maps", "*.png")))
    seg_count = len(glob.glob(os.path.join(config.PROCESSED_DIR, "segmentation_maps", "*.png")))

    print("=" * 60)
    print("  MISSION SIMULATION COMPLETE")
    print("=" * 60)
    print()
    print(f"  Passes completed:  {len(passes)}")
    print(f"  Images processed:  {global_sent}")
    print(f"  Images rejected:   {global_rejected}")
    print()
    if os.path.exists(mosaic_path):
        size_kb = os.path.getsize(mosaic_path) / 1024
        print(f"  Mosaic:            {size_kb:.0f} KB")
    else:
        print(f"  Mosaic:            not generated")
    print(f"  Shadow masks:      {shadow_count}")
    print(f"  Hazard maps:       {hazard_count}")
    print(f"  Segmentation maps: {seg_count}")
    print()
    print(f"  Dashboard: http://localhost:{config.DASHBOARD_PORT}")
    print("  CubeSat status shows WAITING — ready for next pass")
    print("  Press Ctrl+C to exit.")
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down.")


if __name__ == "__main__":
    main()
