#!/usr/bin/env python3
"""capture_training_images.py — Capture 100 training images on the CubeSat Pi.

Run this ON the Raspberry Pi. It captures 100 images over 2 minutes using
rpicam-still, with operator prompts to rearrange the terrain between shots.

Usage:
    python3 capture_training_images.py
"""

import os
import subprocess
import sys
import time
from datetime import datetime

TOTAL_IMAGES = 100
DURATION_SEC = 120
INTERVAL = DURATION_SEC / TOTAL_IMAGES  # 1.2 seconds
SAVE_DIR = os.path.expanduser("~/training_images")
CAM_ARGS = ["--width", "640", "--height", "480", "--nopreview",
            "--timeout", "1", "--quality", "85", "--immediate"]

# Which camera command works (detected on first shot)
_cam_cmd = None

MILESTONES = {
    20: "REARRANGE BOULDERS — move rocks to new positions, add/remove some",
    40: "CHANGE SWEEP DIRECTION — rotate the CubeSat 90 degrees",
    60: "TILT CUBESAT SLIGHTLY — angle the camera 10-15 degrees off-nadir",
    80: "FINAL SLOW PASS — move the CubeSat slowly across the full terrain",
}


def setup():
    """Create save directory, print mission brief, run countdown."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("  MURALTZ TRAINING IMAGE CAPTURE")
    print("=" * 60)
    print()
    print(f"  Images:     {TOTAL_IMAGES}")
    print(f"  Duration:   {DURATION_SEC} seconds")
    print(f"  Interval:   {INTERVAL:.1f} seconds between shots")
    print(f"  Resolution: 640 x 480")
    print(f"  Save dir:   {SAVE_DIR}")
    print()
    print("  OPERATOR INSTRUCTIONS:")
    print("  - Place the CubeSat above the terrain sandbox")
    print("  - Slowly sweep the CubeSat across the terrain during capture")
    print("  - You will be prompted to rearrange terrain at milestones")
    print("  - Vary: position, angle, height, rock placement, lighting")
    print()

    # Countdown
    print("  Starting in...")
    for i in range(5, 0, -1):
        print(f"    {i}...")
        time.sleep(1)
    print()
    print("  GO! Capture started.")
    print()


def take_photo(index):
    """Capture one image. Returns filepath on success, None on failure."""
    global _cam_cmd

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # trim to ms
    filename = f"train_{index:03d}_{ts}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)

    commands_to_try = []
    if _cam_cmd is not None:
        commands_to_try = [_cam_cmd]
    else:
        commands_to_try = ["rpicam-still", "libcamera-still"]

    for cmd in commands_to_try:
        full_cmd = [cmd, "--output", filepath] + CAM_ARGS
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0 and os.path.exists(filepath):
                _cam_cmd = cmd  # remember what works
                return filepath
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue

    # Both commands failed
    if _cam_cmd is None:
        print("\n\n  ERROR: Neither rpicam-still nor libcamera-still found!")
        print("  Install with: sudo apt install -y libcamera-apps")
        print()
        sys.exit(1)

    return None


def print_progress(current, total, success, elapsed):
    """Print an inline progress bar."""
    pct = current / total
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "#" * filled + "-" * (bar_len - filled)

    eta = 0.0
    if current > 0:
        eta = (elapsed / current) * (total - current)

    line = (
        f"\r  [{bar}] {current:3d}/{total} | "
        f"OK: {success} | "
        f"Elapsed: {elapsed:5.1f}s | "
        f"ETA: {eta:5.1f}s"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def main():
    setup()

    start_time = time.monotonic()
    success_count = 0

    for i in range(1, TOTAL_IMAGES + 1):
        shot_start = time.monotonic()

        # Milestone prompts
        if i in MILESTONES:
            elapsed = time.monotonic() - start_time
            sys.stdout.write("\r" + " " * 80 + "\r")
            print(f"\n  >>> SHOT {i}/{TOTAL_IMAGES} — {MILESTONES[i]}")
            print()

        # Take photo
        result = take_photo(i)
        if result is not None:
            success_count += 1

        # Progress
        elapsed = time.monotonic() - start_time
        print_progress(i, TOTAL_IMAGES, success_count, elapsed)

        # Sleep for remaining interval
        shot_elapsed = time.monotonic() - shot_start
        sleep_time = max(0, INTERVAL - shot_elapsed)
        if sleep_time > 0 and i < TOTAL_IMAGES:
            time.sleep(sleep_time)

    # Final summary
    total_elapsed = time.monotonic() - start_time
    print()
    print()
    print("=" * 60)
    print("  CAPTURE COMPLETE")
    print("=" * 60)
    print()
    print(f"  Total shots:    {TOTAL_IMAGES}")
    print(f"  Successful:     {success_count}")
    print(f"  Failed:         {TOTAL_IMAGES - success_count}")
    print(f"  Duration:       {total_elapsed:.1f} seconds")
    print(f"  Saved to:       {SAVE_DIR}")
    print()

    # Count files and total size
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")]
    total_size = sum(
        os.path.getsize(os.path.join(SAVE_DIR, f)) for f in files
    )
    print(f"  Files on disk:  {len(files)}")
    print(f"  Total size:     {total_size / 1024 / 1024:.1f} MB")
    print()
    print("  TO COPY IMAGES TO YOUR MAC, run this on your Mac terminal:")
    print()
    print("  scp -r cubesat@cubesat.local:~/training_images/ ~/Desktop/training_images/")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
