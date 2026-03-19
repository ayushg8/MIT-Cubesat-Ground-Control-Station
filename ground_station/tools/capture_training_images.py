#!/usr/bin/env python3
"""capture_training_images.py — Rapid-fire 400 training images on the CubeSat Pi.

Also serves a live MJPEG stream at http://<pi-ip>:8085/stream so you can
watch the capture in real time from your Mac.

Run this ON the Raspberry Pi:
    python3 capture_training_images.py
"""

import os
import subprocess
import sys
import time
import threading
import shutil
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_IMAGES = 400
INTERVAL = 0.4          # 0.4s between shots — rapid fire
SAVE_DIR = os.path.expanduser("~/training_images")
STREAM_PORT = 8085
CAM_ARGS = ["--width", "640", "--height", "480", "--nopreview",
            "--timeout", "1", "--quality", "85", "--immediate"]

_cam_cmd = None         # detected on first shot
_latest_path = None     # path to most recent image (for stream)
_latest_lock = threading.Lock()
_capture_done = False

MILESTONES = {
    80:  "REARRANGE BOULDERS — move rocks to new positions, add/remove some",
    160: "CHANGE SWEEP DIRECTION — rotate the CubeSat 90 degrees",
    240: "TILT CUBESAT SLIGHTLY — angle the camera 10-15 degrees off-nadir",
    320: "FINAL SLOW PASS — move the CubeSat slowly across the full terrain",
}


# ── MJPEG live stream server ─────────────────────────────────────────────────
class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not _capture_done:
                    with _latest_lock:
                        path = _latest_path
                    if path and os.path.exists(path):
                        try:
                            with open(path, "rb") as f:
                                data = f.read()
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(data)}\r\n\r\n".encode()
                            )
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                        except Exception:
                            pass
                    time.sleep(0.3)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/latest":
            with _latest_lock:
                path = _latest_path
            if path and os.path.exists(path):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(204)
                self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<html><body style="background:#000;margin:0;">
<img src="/stream" style="width:100%;height:auto;">
</body></html>""")

    def log_message(self, fmt, *args):
        pass  # silence request logs


def start_stream_server():
    server = HTTPServer(("0.0.0.0", STREAM_PORT), StreamHandler)
    server.serve_forever()


# ── Camera capture ────────────────────────────────────────────────────────────
def take_photo(index):
    global _cam_cmd, _latest_path

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"train_{index:04d}_{ts}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)

    commands_to_try = (
        [_cam_cmd] if _cam_cmd else [
            "/usr/bin/rpicam-still", "rpicam-still",
            "/usr/bin/libcamera-still", "libcamera-still",
        ]
    )

    for cmd in commands_to_try:
        full_cmd = [cmd, "--output", filepath] + CAM_ARGS
        try:
            result = subprocess.run(full_cmd, capture_output=True, timeout=5)
            if result.returncode == 0 and os.path.exists(filepath):
                _cam_cmd = cmd
                with _latest_lock:
                    _latest_path = filepath
                return filepath
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue

    if _cam_cmd is None:
        print("\n\n  ERROR: Neither rpicam-still nor libcamera-still found!")
        print("  Install with: sudo apt install -y libcamera-apps")
        sys.exit(1)

    return None


# ── Progress bar ──────────────────────────────────────────────────────────────
def print_progress(current, total, success, elapsed):
    pct = current / total
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "#" * filled + "-" * (bar_len - filled)
    eta = (elapsed / current) * (total - current) if current > 0 else 0.0

    line = (
        f"\r  [{bar}] {current:3d}/{total} | "
        f"OK: {success} | "
        f"Elapsed: {elapsed:5.1f}s | "
        f"ETA: {eta:5.1f}s"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


# ── Setup & main ──────────────────────────────────────────────────────────────
def get_pi_ip():
    """Best-effort grab of the Pi's LAN IP for the stream URL."""
    try:
        out = subprocess.check_output(
            ["hostname", "-I"], text=True, timeout=2
        ).strip()
        return out.split()[0] if out else "cubesat.local"
    except Exception:
        return "cubesat.local"


def setup():
    # Clear old images from previous run
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)

    ip = get_pi_ip()
    est_duration = TOTAL_IMAGES * INTERVAL

    print("=" * 60)
    print("  MURALTZ RAPID TRAINING CAPTURE")
    print("=" * 60)
    print()
    print(f"  Images:     {TOTAL_IMAGES}")
    print(f"  Interval:   {INTERVAL}s (rapid fire)")
    print(f"  Est time:   ~{est_duration:.0f} seconds")
    print(f"  Resolution: 640 x 480")
    print(f"  Save dir:   {SAVE_DIR}")
    print()
    print(f"  LIVE STREAM:  http://{ip}:{STREAM_PORT}/")
    print(f"  (open in browser on your Mac to watch)")
    print()
    print("  OPERATOR INSTRUCTIONS:")
    print("  - Sweep the CubeSat slowly and steadily across the terrain")
    print("  - Vary height between 15-40 cm above the surface")
    print("  - Move smoothly — the camera fires every 0.4s")
    print("  - You will be prompted to change things at milestones")
    print()

    print("  Starting in...")
    for i in range(5, 0, -1):
        print(f"    {i}...")
        time.sleep(1)
    print()
    print("  GO! Capture started.")
    print()


def main():
    global _capture_done

    setup()

    # Start MJPEG stream server in background
    stream_thread = threading.Thread(target=start_stream_server, daemon=True)
    stream_thread.start()

    start_time = time.monotonic()
    success_count = 0

    for i in range(1, TOTAL_IMAGES + 1):
        shot_start = time.monotonic()

        if i in MILESTONES:
            sys.stdout.write("\r" + " " * 80 + "\r")
            print(f"\n  >>> SHOT {i}/{TOTAL_IMAGES} — {MILESTONES[i]}")
            print()

        result = take_photo(i)
        if result is not None:
            success_count += 1

        elapsed = time.monotonic() - start_time
        print_progress(i, TOTAL_IMAGES, success_count, elapsed)

        shot_elapsed = time.monotonic() - shot_start
        sleep_time = max(0, INTERVAL - shot_elapsed)
        if sleep_time > 0 and i < TOTAL_IMAGES:
            time.sleep(sleep_time)

    _capture_done = True
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

    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")]
    total_size = sum(
        os.path.getsize(os.path.join(SAVE_DIR, f)) for f in files
    )
    print(f"  Files on disk:  {len(files)}")
    print(f"  Total size:     {total_size / 1024 / 1024:.1f} MB")
    print()
    print("  TO COPY IMAGES TO YOUR MAC:")
    print()
    print("  scp -r cubesat@cubesat.local:~/training_images/ ~/Desktop/training_images/")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
