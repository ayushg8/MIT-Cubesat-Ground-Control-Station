#!/usr/bin/env python3
"""demo_mock_cubesat.py — Mock CubeSat that sends REAL training dataset images.

Drop-in replacement for mock_cubesat.py that sends actual images from the
training dataset instead of synthetic sand textures. Pair with the regular
GCS server (server.py) for a full end-to-end demo.

Usage:
    # Terminal 1: start GCS server
    cd ground_station && python3 server.py

    # Terminal 2: start this mock CubeSat
    python3 tools/demo_mock_cubesat.py

    # Terminal 3 (optional): open dashboard
    open http://localhost:3000

    # In dashboard: click START PASS → END PASS to trigger imaging + downlink
"""

import argparse
import glob
import hashlib
import json
import os
import random
import socket
import sys
import threading
import time
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────
COMMAND_PORT = 5001
DATA_PORT = 5000
ACK = b'\x06'
NACK = b'\x15'
THROTTLE_BYTES_PER_SEC = 1200

# ── Dataset paths ─────────────────────────────────────────────────────────────
DATASET_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "MIT-BWSI-Cubesat-Flight-Software", "Images"),
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "yolo_training", "dataset_clean", "train", "images"),
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "CubeSat Demo Images.v2i.yolov8", "train", "images"),
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "yolo_training", "dataset_clean", "valid", "images"),
]

# ── Mutable state ─────────────────────────────────────────────────────────────
_lock = threading.Lock()
_state = "WAITING"
_pass_num = 0
_captured = 0
_current_cell = [0, 0]
_cells_imaged_this_pass = []
_images_per_pass = 5
_gcs_ip = "127.0.0.1"
_img_seq = 0
_dataset_images = []
_dataset_idx = 0

VALID_CMDS = {
    "start_pass", "end_pass", "cell", "set_cell", "retransmit",
    "priority_cell", "adjust_exposure", "enter_safe_mode",
    "resume_normal", "status_request", "retry_downlink", "reset_mission",
}


def load_dataset_images():
    """Load all available training dataset images."""
    global _dataset_images
    images = []
    for d in DATASET_DIRS:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(glob.glob(os.path.join(d, ext)))
    _dataset_images = sorted(set(images))
    random.shuffle(_dataset_images)
    print(f"[MOCK] Loaded {len(_dataset_images)} dataset images")


def get_next_image():
    """Get the next training image (cycles through dataset)."""
    global _dataset_idx
    if not _dataset_images:
        return None
    img = _dataset_images[_dataset_idx % len(_dataset_images)]
    _dataset_idx += 1
    return img


def _send_image(gcs_ip, image_path, filename, metadata):
    """Send one real image to the GCS using the transfer protocol."""
    with open(image_path, "rb") as f:
        jpeg_data = f.read()

    md5 = hashlib.md5(jpeg_data).hexdigest()

    header = json.dumps({
        "type": "image",
        "filename": filename,
        "file_size": len(jpeg_data),
        "md5": md5,
        "metadata": metadata,
    }) + "\n"

    try:
        with socket.create_connection((gcs_ip, DATA_PORT), timeout=10) as s:
            s.sendall(header.encode("utf-8"))

            offset = 0
            chunk_size = THROTTLE_BYTES_PER_SEC
            while offset < len(jpeg_data):
                end = min(offset + chunk_size, len(jpeg_data))
                s.sendall(jpeg_data[offset:end])
                offset = end
                if offset < len(jpeg_data):
                    time.sleep(1)

            s.settimeout(10)
            resp = s.recv(1)
            if resp == ACK:
                print(f"[MOCK] Sent: {filename} ({len(jpeg_data):,} bytes) "
                      f"[src: {os.path.basename(image_path)}] → ACK")
                return True
            else:
                print(f"[MOCK] Sent: {filename} → NACK")
                return False

    except ConnectionRefusedError:
        print(f"[MOCK] GCS not listening on {gcs_ip}:{DATA_PORT}")
        return False
    except Exception as e:
        print(f"[MOCK] Send error: {e}")
        return False


def _run_downlink(gcs_ip, pass_n, cells):
    """Send real training images for each cell to the GCS."""
    global _state, _img_seq

    print(f"[MOCK] Starting downlink: {len(cells)} image(s) for pass {pass_n}")

    sent_count = 0
    for idx, (row, col) in enumerate(cells):
        with _lock:
            if _state != "DOWNLINK":
                print("[MOCK] Downlink interrupted")
                break

        image_path = get_next_image()
        if image_path is None:
            print("[MOCK] No dataset images available!")
            break

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"pass{pass_n}_img{idx:02d}_{ts}.jpg"

        blur_score = round(random.uniform(0.6, 0.95), 3)
        exposure_score = round(random.uniform(0.7, 0.98), 3)
        combined_score = round((blur_score + exposure_score) / 2, 3)

        metadata = {
            "grid_cell": [row, col],
            "pass_number": pass_n,
            "capture_time": datetime.now(timezone.utc).isoformat(),
            "blur_score": blur_score,
            "exposure_score": exposure_score,
            "combined_score": combined_score,
            "resolution": [320, 240],
            "source": os.path.basename(image_path),
        }

        success = _send_image(gcs_ip, image_path, filename, metadata)
        if success:
            sent_count += 1
            with _lock:
                _img_seq += 1

        time.sleep(0.5)

    print(f"[MOCK] Downlink complete: {sent_count}/{len(cells)} images sent")

    with _lock:
        if _state == "DOWNLINK":
            _state = "WAITING"
            print("[MOCK] State → WAITING")


def command_listener():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("", COMMAND_PORT))
    srv.listen(5)
    print(f"[MOCK] Command listener on port {COMMAND_PORT}")

    while True:
        try:
            conn, addr = srv.accept()
            threading.Thread(target=_handle_cmd_conn, args=(conn, addr),
                             daemon=True).start()
        except Exception as e:
            print(f"[MOCK] Accept error: {e}")


def _handle_cmd_conn(conn, addr):
    buf = b""
    try:
        conn.settimeout(5.0)
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line.decode())
                except Exception:
                    conn.sendall(NACK)
                    continue

                name = cmd.get("cmd", "")
                _apply_command(name, cmd)

                if name in VALID_CMDS:
                    print(f"[MOCK] ACK  ← {cmd}")
                    conn.sendall(ACK)
                else:
                    print(f"[MOCK] NACK ← unknown cmd '{name}'")
                    conn.sendall(NACK)
    except Exception as e:
        print(f"[MOCK] cmd conn error: {e}")
    finally:
        conn.close()


def _apply_command(name, cmd):
    global _state, _pass_num, _captured, _current_cell, _cells_imaged_this_pass
    with _lock:
        if name == "start_pass" and _state == "WAITING":
            _pass_num += 1
            _state = "IMAGING"
            _captured = 0
            _cells_imaged_this_pass = []
            all_cells = [(r, c) for r in range(8) for c in range(8)]
            random.shuffle(all_cells)
            _cells_imaged_this_pass = all_cells[:_images_per_pass]
            _captured = len(_cells_imaged_this_pass)
            print(f"[MOCK] State → IMAGING  (pass {_pass_num}, {_captured} cells)")

        elif name == "end_pass" and _state == "IMAGING":
            _state = "PROCESSING"
            pass_n = _pass_num
            cells = list(_cells_imaged_this_pass)
            print(f"[MOCK] State → PROCESSING")
            threading.Thread(target=_auto_advance_with_downlink,
                             args=(pass_n, cells), daemon=True).start()

        elif name == "enter_safe_mode":
            _state = "SAFE_MODE"
            print("[MOCK] State → SAFE_MODE")

        elif name == "resume_normal" and _state == "SAFE_MODE":
            _state = "WAITING"
            print("[MOCK] State → WAITING")

        elif name == "reset_mission":
            _state = "WAITING"
            _pass_num = 0
            _captured = 0
            _cells_imaged_this_pass = []
            print("[MOCK] Mission reset → WAITING")

        elif name in ("cell", "set_cell"):
            r, c = cmd.get("row", "?"), cmd.get("col", "?")
            _current_cell = [r, c]
            print(f"[MOCK] Grid cell set to ({r},{c})")


def _auto_advance_with_downlink(pass_n, cells):
    global _state
    time.sleep(2)
    with _lock:
        if _state == "PROCESSING":
            _state = "DOWNLINK"
            print("[MOCK] State → DOWNLINK")
        else:
            return
    _run_downlink(_gcs_ip, pass_n, cells)


def telemetry_sender(gcs_ip):
    print(f"[MOCK] Telemetry sender → {gcs_ip}:{DATA_PORT} every 3s")
    while True:
        time.sleep(3)
        _send_telemetry(gcs_ip)


def _send_telemetry(gcs_ip):
    with _lock:
        state = _state
        pass_n = _pass_num
        captured = _captured

    telem = {
        "state": state,
        "pass_number": pass_n,
        "captured_this_pass": captured,
        "captured_total": captured,
        "rejected_total": 0,
        "images_sent_total": _img_seq,
        "battery_pct": round(random.uniform(80, 95), 1),
        "cpu_temp_c": round(random.uniform(38, 48), 1),
        "angular_rate_rad_s": round(random.uniform(0.01, 0.05), 3),
        "nadir_angle_deg": round(random.uniform(8, 18), 1),
        "nadir_locked": True,
        "storage_used_pct": round(10 + _img_seq * 0.5, 1),
        "storage_free_mb": max(100, 2000 - _img_seq * 10),
        "gcs_reachable": True,
        "consecutive_failures": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    body = json.dumps(telem).encode("utf-8")
    md5 = hashlib.md5(body).hexdigest()

    header = json.dumps({
        "type": "telemetry",
        "filename": "telemetry.json",
        "file_size": len(body),
        "md5": md5,
        "metadata": {},
    }) + "\n"

    try:
        with socket.create_connection((gcs_ip, DATA_PORT), timeout=3) as s:
            s.sendall(header.encode("utf-8"))
            s.sendall(body)
            s.settimeout(3)
            ack = s.recv(1)
            if random.random() < 0.1:
                status = "ACK" if ack == ACK else "NACK"
                print(f"[MOCK] Telemetry sent  state={state}  GCS={status}")
    except ConnectionRefusedError:
        pass
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mock CubeSat using real training dataset images"
    )
    parser.add_argument("--gcs-ip", default="127.0.0.1",
                        help="GCS IP (default: 127.0.0.1)")
    parser.add_argument("--images-per-pass", type=int, default=5,
                        help="Images per pass (default: 5)")
    args = parser.parse_args()

    _gcs_ip = args.gcs_ip
    _images_per_pass = args.images_per_pass

    load_dataset_images()

    if not _dataset_images:
        print("ERROR: No training dataset images found!")
        sys.exit(1)

    print()
    print("=" * 65)
    print("  Mock CubeSat — DEMO MODE (Real Training Images)")
    print("=" * 65)
    print(f"  Commands:        0.0.0.0:{COMMAND_PORT}")
    print(f"  Data → GCS:      {args.gcs_ip}:{DATA_PORT}")
    print(f"  Images/pass:     {_images_per_pass}")
    print(f"  Dataset images:  {len(_dataset_images)}")
    print(f"  Initial state:   WAITING")
    print()
    print("  Use the GCS dashboard to: START PASS → END PASS")
    print("  Images will be real terrain photos from training data")
    print("=" * 65)

    threading.Thread(target=command_listener, daemon=True).start()
    telemetry_sender(args.gcs_ip)
