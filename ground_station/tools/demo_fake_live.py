#!/usr/bin/env python3
"""demo_fake_live.py — Run a fully-faked live mission using training dataset images.

Picks real images from the YOLO training dataset (the same images used to train
our terrain/hazard models), copies them into data/received_images/ with proper
naming and metadata sidecars, then runs the full GCS pipeline with:
  - Simulated state transitions (BOOT → IMAGING → IDLE → DOWNLINK → WAITING)
  - Fake but realistic telemetry (IMU, thermal, storage, coverage)
  - Downlink progress bars at configurable speed
  - Full CV pipeline: mosaic stitching, hazard classification, YOLO detection,
    change detection (across passes), route planning, segmentation
  - Live dashboard at http://localhost:3000

This is NOT live — no CubeSat hardware is needed.  The images are real training
data so the YOLO detections and hazard classifications produce meaningful results.

Usage:
    cd ground_station
    python3 tools/demo_fake_live.py                      # 2 passes, 8 images each
    python3 tools/demo_fake_live.py --images-per-pass 5  # 2 passes, 5 images each
    python3 tools/demo_fake_live.py --passes 3           # 3 passes
    python3 tools/demo_fake_live.py --speed 8000         # faster downlink
    python3 tools/demo_fake_live.py --no-clear           # keep previous processed data
"""

import argparse
import glob
import json
import logging
import os
import random
import shutil
import sys
import threading
import time
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

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_PASSES = 2
DEFAULT_IMAGES_PER_PASS = 8
DEFAULT_DOWNLINK_SPEED = 4000   # bytes/sec for progress animation
DEFAULT_DELAY_BETWEEN = 0.5     # seconds between images within downlink

# ── Training dataset paths (relative to repo root) ──────────────────────────
DATASET_DIRS = [
    # Flight software test images (real CubeSat camera captures)
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "MIT-BWSI-Cubesat-Flight-Software", "Images"),
    # YOLO training dataset (clean)
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "yolo_training", "dataset_clean", "train", "images"),
    # Roboflow dataset
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "CubeSat Demo Images.v2i.yolov8", "train", "images"),
    # Validation set (different angles)
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "yolo_training", "dataset_clean", "valid", "images"),
]


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


def find_dataset_images():
    """Gather all JPEG/PNG images from the training datasets."""
    images = []
    for d in DATASET_DIRS:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(glob.glob(os.path.join(d, ext)))
    # Deduplicate and sort
    images = sorted(set(images))
    return images


def prepare_images(all_images, num_passes, images_per_pass):
    """Select and copy images into data/received_images/ with metadata sidecars.

    Returns dict: {pass_number: [(jpg_path, meta_path), ...]}
    """
    total_needed = num_passes * images_per_pass
    if len(all_images) < total_needed:
        # Allow reuse if we don't have enough unique images
        selected = []
        while len(selected) < total_needed:
            remaining = total_needed - len(selected)
            batch = random.sample(all_images, min(remaining, len(all_images)))
            selected.extend(batch)
    else:
        selected = random.sample(all_images, total_needed)

    os.makedirs(config.RECEIVED_DIR, exist_ok=True)

    passes = {}
    img_idx = 0
    for pass_num in range(1, num_passes + 1):
        pass_images = []
        for i in range(images_per_pass):
            src = selected[img_idx]
            img_idx += 1

            # Generate proper filename
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            dst_name = f"pass{pass_num}_img{i:02d}_{ts}.jpg"
            dst_path = os.path.join(config.RECEIVED_DIR, dst_name)

            # Copy image
            shutil.copy2(src, dst_path)

            # Generate metadata sidecar
            # Simulate grid cells that tile a region (overlapping slightly for mosaic)
            row = (i // 4) % 8
            col = (i % 4) + (pass_num - 1) * 2  # shift columns per pass for change detection
            col = col % 8

            blur_score = round(random.uniform(0.65, 0.95), 3)
            exposure_score = round(random.uniform(0.70, 0.98), 3)
            combined_score = round((blur_score + exposure_score) / 2, 3)

            metadata = {
                "grid_cell": [row, col],
                "pass_number": pass_num,
                "capture_time": datetime.now(timezone.utc).isoformat(),
                "blur_score": blur_score,
                "exposure_score": exposure_score,
                "combined_score": combined_score,
                "resolution": [320, 240],
                "source": os.path.basename(src),
                "quality": {
                    "blur_score": blur_score,
                    "exposure_score": exposure_score,
                    "combined_score": combined_score,
                },
                "imu": {
                    "angular_rate": round(random.uniform(0.01, 0.12), 3),
                    "stable": True,
                    "nadir_locked": True,
                    "nadir_angle_deg": round(random.uniform(8, 22), 1),
                    "roll_deg": round(random.uniform(-12, 12), 1),
                    "pitch_deg": round(random.uniform(-12, 12), 1),
                },
                "camera": {
                    "exposure_us": random.choice([5000, 8000, 10000, 12000, 15000]),
                    "analog_gain": round(random.uniform(1.5, 4.0), 1),
                    "lux": random.randint(150, 350),
                },
            }

            meta_path = dst_path.replace(".jpg", "_meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            pass_images.append((dst_path, meta_path))

            # Small time offset so filenames are unique
            time.sleep(0.05)

        passes[pass_num] = pass_images

    return passes


def clear_processed_data():
    """Remove old pipeline outputs so the demo starts fresh."""
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

    for f in glob.glob(os.path.join(config.TELEMETRY_DIR, "*.json")):
        os.remove(f)

    # Clear old received images (demo-generated)
    for ext in ("*.jpg", "*_meta.json"):
        for f in glob.glob(os.path.join(config.RECEIVED_DIR, ext)):
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


def inject_telemetry(state, pass_number, images_this_pass=0, total_images=0,
                     rejected=0, queued=0, sent=0, meta=None):
    """Push fake telemetry so the dashboard shows live CubeSat state."""
    imu_data = meta.get("imu", {}) if meta else {}
    cam_data = meta.get("camera", {}) if meta else {}

    telem = {
        "type": "telemetry",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cubesat_id": "MURALTZ-01",
        "pass_number": pass_number,
        "state": state,
        "uptime_sec": int(time.time()) % 100000,
        "imu": {
            "accel": [
                round(random.uniform(-0.3, 0.3), 2),
                round(random.uniform(-0.3, 0.3), 2),
                round(9.78 + random.uniform(-0.05, 0.05), 3),
            ],
            "gyro": [
                round(random.uniform(-0.03, 0.03), 3),
                round(random.uniform(-0.03, 0.03), 3),
                round(random.uniform(-0.01, 0.01), 3),
            ],
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
            "used_pct": round(20 + total_images * 1.5, 1),
            "free_mb": round(max(500, 1800 - total_images * 25), 0),
        },
        "imaging": {
            "captured_this_pass": images_this_pass,
            "captured_total": total_images,
            "rejected_total": rejected,
            "rejection_breakdown": {
                "blur": max(0, rejected - 1),
                "underexposed": min(1, rejected),
                "overexposed": 0,
                "motion_blur": 0,
            },
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

    raw = json.dumps(telem).encode("utf-8")
    telemetry_parser.parse_and_save_telemetry(raw, f"mock_pass{pass_number}.json")


def simulate_downlink_progress(filename, file_size, speed):
    """Animate the downlink progress bar in the dashboard."""
    dl = get_downlink_state()
    dl.start_transfer(filename, file_size)

    bytes_sent = 0
    chunk = max(1, speed // 10)
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


def main():
    parser = argparse.ArgumentParser(
        description="Run a fully-faked live CubeSat mission using training dataset images"
    )
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES,
                        help=f"Number of passes to simulate (default: {DEFAULT_PASSES})")
    parser.add_argument("--images-per-pass", type=int, default=DEFAULT_IMAGES_PER_PASS,
                        help=f"Images per pass (default: {DEFAULT_IMAGES_PER_PASS})")
    parser.add_argument("--speed", type=int, default=DEFAULT_DOWNLINK_SPEED,
                        help=f"Downlink speed in bytes/sec (default: {DEFAULT_DOWNLINK_SPEED})")
    parser.add_argument("--no-clear", action="store_true",
                        help="Don't clear previous processed data")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible image selection")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    setup_logging()
    logger = logging.getLogger("demo")

    # Find training dataset images
    all_images = find_dataset_images()
    if not all_images:
        print("\n  ERROR: No training dataset images found!")
        print("  Expected images in:")
        for d in DATASET_DIRS:
            print(f"    {os.path.abspath(d)}")
        sys.exit(1)

    print()
    print("=" * 65)
    print("  MuraltZ GCS — DEMO MODE (Fake Live Mission)")
    print("  Images sourced from training dataset — NOT live hardware")
    print("=" * 65)
    print()
    print(f"  Dataset images found: {len(all_images)}")
    print(f"  Passes to simulate:  {args.passes}")
    print(f"  Images per pass:     {args.images_per_pass}")
    print(f"  Downlink speed:      {args.speed} B/s")
    print(f"  Dashboard:           http://localhost:{config.DASHBOARD_PORT}")
    print()

    create_dirs()
    if not args.no_clear:
        print("  Clearing old data...")
        clear_processed_data()

    # Prepare images with metadata sidecars
    print("  Preparing demo images from training dataset...")
    passes = prepare_images(all_images, args.passes, args.images_per_pass)
    total_images = sum(len(v) for v in passes.values())
    print(f"  Prepared {total_images} images across {len(passes)} passes")
    print()

    # Initialize core objects
    mission_state = MissionState()
    pipeline = Pipeline(mission_state)
    dash_app.set_pipeline(pipeline)
    dash_app.set_mission_state(mission_state)

    # Hook quality log into dashboard
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
    print("  Starting dashboard...")
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

    # ── Simulate full mission ────────────────────────────────────────────────
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
            print("  │  Camera init... OK")
            print("  │  IMU calibration... OK")
            print("  │  WiFi link... OK")
            print("  │")

        # ── WAITING ──
        print(f"  ┌─ WAITING (Pass {pass_number})")
        inject_telemetry("WAITING", pass_number,
                         total_images=global_img_count, sent=global_sent)
        time.sleep(3)
        print(f"  │  Operator command received: start_pass")
        print(f"  │")

        # ── IMAGING ──
        print(f"  ┌─ IMAGING (Pass {pass_number} — {num_images} images)")
        inject_telemetry("IMAGING", pass_number,
                         images_this_pass=0, total_images=global_img_count,
                         sent=global_sent)
        time.sleep(2)

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
            cell = metadata.get("grid_cell", [0, 0])
            status = "PASS" if score > 0.5 else "REJECT"
            if status == "REJECT":
                global_rejected += 1
            src_name = metadata.get("source", "?")
            print(f"  │  [{i+1:2d}/{num_images}] Captured {basename}  "
                  f"cell=({cell[0]},{cell[1]})  Q={score:.2f}  {status}  "
                  f"[src: {src_name}]")
            time.sleep(0.3)

        print(f"  │  Imaging complete: {num_images} captured, {global_rejected} rejected total")
        print(f"  │")

        # ── IDLE ──
        print(f"  ┌─ IDLE (building downlink queue)")
        inject_telemetry("IDLE", pass_number,
                         images_this_pass=num_images,
                         total_images=global_img_count,
                         queued=num_images,
                         sent=global_sent)
        time.sleep(2)
        print(f"  │  Priority queue built: {num_images} images")
        print(f"  │")

        # ── DOWNLINK ──
        print(f"  ┌─ DOWNLINK (Pass {pass_number})")
        inject_telemetry("DOWNLINK", pass_number,
                         images_this_pass=num_images,
                         total_images=global_img_count,
                         queued=num_images,
                         sent=global_sent)
        time.sleep(1)

        # Start session tracking
        dl = get_downlink_state()
        session_total_bytes = sum(os.path.getsize(jp) for jp, _ in images)
        dl.start_session(total_images=num_images, total_bytes=session_total_bytes)

        for i, (jpg_path, meta_path) in enumerate(images):
            basename = os.path.basename(jpg_path)
            file_size = os.path.getsize(jpg_path)

            with open(meta_path) as f:
                metadata = json.load(f)

            remaining = num_images - i
            inject_telemetry("DOWNLINK", pass_number,
                             images_this_pass=num_images,
                             total_images=global_img_count,
                             queued=remaining,
                             sent=global_sent + i,
                             meta=metadata)

            print(f"  │  [{i+1:2d}/{num_images}] Downlinking {basename} "
                  f"({file_size:,} bytes)...", end="", flush=True)

            # Animate downlink progress in background
            dl_thread = threading.Thread(
                target=simulate_downlink_progress,
                args=(basename, file_size, args.speed),
            )
            dl_thread.start()
            dl_thread.join()

            # Run through the full CV pipeline
            quality = run_ground_quality_check(jpg_path)

            t0 = time.monotonic()
            try:
                pipeline.process(jpg_path, metadata, quality)
                elapsed = time.monotonic() - t0
                mark_downlink_complete()
                mission_state.record_downlink_bytes(file_size, elapsed, True)
                global_sent += 1
                print(f" ACK ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.monotonic() - t0
                dl = get_downlink_state()
                dl.set_status("failed", str(e))
                mission_state.record_downlink_bytes(file_size, elapsed, False)
                print(f" FAIL ({elapsed:.1f}s) — {e}")

            if i < num_images - 1 and DEFAULT_DELAY_BETWEEN > 0:
                time.sleep(DEFAULT_DELAY_BETWEEN)

        inject_telemetry("DOWNLINK", pass_number,
                         images_this_pass=num_images,
                         total_images=global_img_count,
                         queued=0,
                         sent=global_sent)

        dl.end_session()
        print(f"  │  Downlink complete: {num_images} images transferred")
        print(f"  │")
        print(f"  └─ Pass {pass_number} complete")
        print()

        if pass_idx < len(passes) - 1:
            time.sleep(2)

    # ── Final state ──
    inject_telemetry("WAITING", max(passes.keys()),
                     total_images=global_img_count,
                     sent=global_sent,
                     rejected=global_rejected)

    # Results summary
    mosaic_path = os.path.join(config.PROCESSED_DIR, "mosaics", "mosaic_latest.png")
    shadow_count = len(glob.glob(os.path.join(config.PROCESSED_DIR, "shadow_masks", "*.png")))
    hazard_count = len(glob.glob(os.path.join(config.PROCESSED_DIR, "hazard_maps", "*.png")))
    seg_count = len(glob.glob(os.path.join(config.PROCESSED_DIR, "segmentation_maps", "*.png")))
    yolo_count = len(glob.glob(os.path.join(config.PROCESSED_DIR, "yolo_detections", "*.json")))

    print("=" * 65)
    print("  DEMO MISSION COMPLETE")
    print("=" * 65)
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
    print(f"  YOLO detections:   {yolo_count}")
    print()
    print(f"  Dashboard: http://localhost:{config.DASHBOARD_PORT}")
    print(f"  Status: WAITING — all passes complete")
    print()
    print("  The dashboard is still live. Explore the tabs:")
    print("    - Mosaic: see the stitched terrain canvas")
    print("    - Coverage: grid cell coverage heatmap")
    print("    - Routes: fastest/safest/balanced paths")
    print("    - Changes: objects that appeared/disappeared between passes")
    print("    - Quality Log: per-image quality scores")
    print("    - Telemetry: IMU, thermal, storage readings")
    print()
    print("  Press Ctrl+C to exit.")
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down demo.")


if __name__ == "__main__":
    main()
