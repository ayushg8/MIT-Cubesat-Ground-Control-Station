#!/usr/bin/env python3
"""simulate_downlink.py — Simulate CubeSat downlink for live demo.

Sends images through the TCP protocol to the GCS listener, exactly like
the real CubeSat would. The GCS pipeline processes each one in real time:
stitch → classify → route → display on dashboard.

Usage:
    python3 tools/simulate_downlink.py                  # 10 images, 4KB/s
    python3 tools/simulate_downlink.py --count 5        # 5 images
    python3 tools/simulate_downlink.py --speed 1200     # real CubeSat speed (slow)
    python3 tools/simulate_downlink.py --speed 0        # max speed (no throttle)
    python3 tools/simulate_downlink.py --images dir/    # use images from a directory
"""

import argparse
import glob
import hashlib
import json
import os
import random
import socket
import sys
import time

# Default to localhost
GCS_HOST = "127.0.0.1"
GCS_PORT = 5000
DEFAULT_SPEED = 4096  # bytes/sec (fast enough to demo, slow enough to see progress)
DEFAULT_COUNT = 10


def send_image(sock, image_path, pass_number, img_index, speed_bps):
    """Send one image through the CubeSat downlink protocol."""
    with open(image_path, "rb") as f:
        data = f.read()

    md5 = hashlib.md5(data).hexdigest()
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"pass{pass_number}_img{img_index:02d}_{ts}.jpg"

    header = {
        "type": "image",
        "filename": filename,
        "file_size": len(data),
        "md5": md5,
        "metadata": {
            "pass_number": pass_number,
            "grid_cell": [random.randint(0, 7), random.randint(0, 7)],
            "capture_time": ts,
            "exposure_us": 15000,
            "source": os.path.basename(image_path),
        },
    }

    # Send header
    header_bytes = json.dumps(header).encode("utf-8") + b"\n"
    sock.sendall(header_bytes)

    # Send image bytes (throttled)
    sent = 0
    chunk_size = min(4096, max(1024, speed_bps)) if speed_bps > 0 else len(data)
    start = time.time()

    while sent < len(data):
        end = min(sent + chunk_size, len(data))
        sock.sendall(data[sent:end])
        sent = end

        if speed_bps > 0 and sent < len(data):
            elapsed = time.time() - start
            expected = sent / speed_bps
            if expected > elapsed:
                time.sleep(expected - elapsed)

    elapsed = time.time() - start
    rate = len(data) / elapsed if elapsed > 0 else 0

    # Wait for ACK/NACK
    resp = sock.recv(1)
    if resp == b"\x06":
        print(f"  ✓ {filename} — {len(data):,} bytes in {elapsed:.1f}s ({rate:.0f} B/s) — ACK")
        return True
    else:
        print(f"  ✗ {filename} — NACK")
        return False


def main():
    parser = argparse.ArgumentParser(description="Simulate CubeSat downlink")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help=f"Number of images to send (default: {DEFAULT_COUNT})")
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED,
                        help=f"Transfer speed in bytes/sec (default: {DEFAULT_SPEED}, 0=max)")
    parser.add_argument("--images", type=str, default="data/received_images",
                        help="Directory with source images")
    parser.add_argument("--pass-number", type=int, default=99,
                        help="Pass number for the simulated downlink")
    parser.add_argument("--host", type=str, default=GCS_HOST)
    parser.add_argument("--port", type=int, default=GCS_PORT)
    args = parser.parse_args()

    # Find images
    patterns = [os.path.join(args.images, "*.jpg"), os.path.join(args.images, "*.png")]
    images = []
    for p in patterns:
        images.extend(glob.glob(p))
    images.sort()

    if not images:
        print(f"No images found in {args.images}")
        sys.exit(1)

    # Pick a random subset
    selected = random.sample(images, min(args.count, len(images)))
    print(f"Simulating downlink: {len(selected)} images at {args.speed} B/s")
    print(f"Connecting to GCS at {args.host}:{args.port}...")
    print(f"Watch the dashboard at http://localhost:3000")
    print()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((args.host, args.port))
        print(f"Connected. Starting transfers...\n")

        acked = 0
        for i, img_path in enumerate(selected):
            print(f"[{i+1}/{len(selected)}] Sending {os.path.basename(img_path)}...")
            ok = send_image(sock, img_path, args.pass_number, i, args.speed)
            if ok:
                acked += 1
            time.sleep(0.5)  # Small gap between transfers

        sock.close()
        print(f"\nDone. {acked}/{len(selected)} images transferred successfully.")
        print("Check the dashboard to see the results!")

    except ConnectionRefusedError:
        print(f"Cannot connect to {args.host}:{args.port} — is the GCS server running?")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.")
        sock.close()


if __name__ == "__main__":
    main()
