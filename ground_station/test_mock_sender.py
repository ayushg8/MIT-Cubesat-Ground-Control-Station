#!/usr/bin/env python3
"""Mock CubeSat sender — sends real sandbox images to GCS over TCP.

Follows the exact protocol from protocol.py:
  1. Connect to GCS on port 5000
  2. Send JSON header + newline
  3. Send raw image bytes
  4. Read 1-byte ACK/NACK
"""

import hashlib
import json
import os
import socket
import sys
import time

GCS_HOST = "127.0.0.1"
GCS_PORT = 5000

# Real sandbox training images
IMAGE_DIR = "/Users/dullet/Ayush/Cubesat/CubeSat Demo Images.v2i.yolov8/train/images"


def send_image(image_path: str, pass_num: int, img_num: int):
    """Send one image following the CubeSat transfer protocol."""
    with open(image_path, "rb") as f:
        data = f.read()

    md5 = hashlib.md5(data).hexdigest()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"pass{pass_num}_img{img_num:02d}_{timestamp}.jpg"

    header = {
        "type": "image",
        "filename": filename,
        "file_size": len(data),
        "md5": md5,
        "metadata": {
            "pass_number": pass_num,
            "image_index": img_num,
            "capture_time": timestamp,
            "grid_cell": [img_num // 4, img_num % 4],
        },
    }

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    try:
        sock.connect((GCS_HOST, GCS_PORT))
        # Send header + newline
        header_bytes = json.dumps(header).encode("utf-8") + b"\n"
        sock.sendall(header_bytes)
        # Send image data
        sock.sendall(data)
        # Wait for ACK/NACK
        response = sock.recv(1)
        if response == b"\x06":
            print(f"  ACK  {filename} ({len(data):,} bytes)")
            return True
        else:
            print(f"  NACK {filename}")
            return False
    except Exception as e:
        print(f"  ERROR {filename}: {e}")
        return False
    finally:
        sock.close()


def main():
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    images = sorted([
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.endswith(".jpg")
    ])

    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return

    images = images[:n_images]
    print(f"Sending {len(images)} images to {GCS_HOST}:{GCS_PORT}")
    print()

    ack_count = 0
    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] Sending {os.path.basename(img_path)}")
        if send_image(img_path, pass_num=1, img_num=i):
            ack_count += 1
        time.sleep(0.5)  # Small delay between images

    print(f"\nDone: {ack_count}/{len(images)} ACKed")


if __name__ == "__main__":
    main()
