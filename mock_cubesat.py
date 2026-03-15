#!/usr/bin/env python3
"""
mock_cubesat.py — Simulates the CubeSat on localhost for GCS testing.

Runs three threads:
  1. Command listener on port 5001 — accepts GCS commands, sends ACK, updates state
  2. Telemetry sender — pushes a telemetry packet to GCS on port 5000 every 3 seconds
  3. Image downlinker — when end_pass triggers DOWNLINK, generates and sends images

Usage:
    python3 mock_cubesat.py [--gcs-ip 127.0.0.1] [--images-per-pass 3]

Then in the GCS dashboard: CONNECT to 127.0.0.1
"""

import argparse
import hashlib
import io
import json
import os
import random
import socket
import threading
import time
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────
COMMAND_PORT  = 5001   # listen here for GCS commands
DATA_PORT     = 5000   # send telemetry/images here to GCS
ACK  = b'\x06'
NACK = b'\x15'
THROTTLE_BYTES_PER_SEC = 1200  # match real CubeSat transfer rate

# ── Mutable state (shared between threads) ────────────────────────────────────
_lock      = threading.Lock()
_state     = "WAITING"   # WAITING / IMAGING / PROCESSING / DOWNLINK / SAFE_MODE
_pass_num  = 0
_captured  = 0
_current_cell = [0, 0]
_cells_imaged_this_pass = []  # list of (row, col) captured during IMAGING
_images_per_pass = 3
_gcs_ip = "127.0.0.1"
_img_seq = 0  # global image sequence counter

VALID_CMDS = {
    "start_pass", "end_pass", "cell", "set_cell", "retransmit",
    "priority_cell", "adjust_exposure", "enter_safe_mode",
    "resume_normal", "status_request", "retry_downlink", "reset_mission",
}


# ── Image generation ─────────────────────────────────────────────────────────

def _generate_sand_image(cell_row: int, cell_col: int, pass_n: int) -> bytes:
    """Generate a synthetic sand-textured JPEG image for a grid cell."""
    try:
        import numpy as np
        import cv2
    except ImportError:
        # Fallback: create a minimal valid JPEG
        return _minimal_jpeg()

    np.random.seed(pass_n * 100 + cell_row * 10 + cell_col)

    # Base sand colour with slight per-cell variation
    base_r = 140 + cell_row * 5
    base_g = 170 + cell_col * 3
    base_b = 180 - cell_row * 2
    base = np.full((240, 320, 3), (base_b, base_g, base_r), dtype=np.uint8)

    # Sand-like noise
    noise = np.random.randint(-25, 25, base.shape, dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Grid tape lines (the real surface has tape at cell boundaries)
    cv2.line(img, (0, 120), (320, 120), (60, 60, 60), 2)
    cv2.line(img, (160, 0), (160, 240), (60, 60, 60), 2)

    # Add some random features to make each cell unique
    n_features = random.randint(0, 4)
    for _ in range(n_features):
        cx = random.randint(20, 300)
        cy = random.randint(20, 220)
        radius = random.randint(5, 25)
        shade = random.randint(60, 120)
        cv2.circle(img, (cx, cy), radius, (shade, shade + 10, shade + 5), -1)

    # Occasional shadow region
    if random.random() < 0.3:
        sx = random.randint(0, 200)
        sy = random.randint(0, 150)
        sw = random.randint(40, 100)
        sh = random.randint(30, 80)
        shadow_overlay = img[sy:sy+sh, sx:sx+sw].astype(np.float32) * 0.4
        img[sy:sy+sh, sx:sx+sw] = shadow_overlay.astype(np.uint8)

    # Cell label
    cv2.putText(img, f"({cell_row},{cell_col}) p{pass_n}", (5, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    _, jpeg_data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg_data.tobytes()


def _minimal_jpeg() -> bytes:
    """Create a tiny valid JPEG without opencv (fallback)."""
    try:
        from PIL import Image as PILImage
        buf = io.BytesIO()
        img = PILImage.new("RGB", (320, 240), (180, 170, 140))
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except ImportError:
        # Absolute minimal 1x1 JPEG
        return (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
            b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
            b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342'
            b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
            b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
            b'\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04'
            b'\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa'
            b'\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n'
            b'\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz'
            b'\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99'
            b'\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7'
            b'\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5'
            b'\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1'
            b'\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa'
            b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\x9e\xa7\xa3\xa0\xa0\x02\x80'
            b'\xff\xd9'
        )


# ── Image downlinker ──────────────────────────────────────────────────────────

def _send_image(gcs_ip: str, jpeg_data: bytes, filename: str, metadata: dict):
    """Send one image to the GCS using the transfer protocol."""
    md5 = hashlib.md5(jpeg_data).hexdigest()

    header = json.dumps({
        "type":      "image",
        "filename":  filename,
        "file_size": len(jpeg_data),
        "md5":       md5,
        "metadata":  metadata,
    }) + "\n"

    try:
        with socket.create_connection((gcs_ip, DATA_PORT), timeout=10) as s:
            # Send header
            s.sendall(header.encode("utf-8"))

            # Send image data in throttled chunks (simulates 1200 B/s link)
            offset = 0
            chunk_size = THROTTLE_BYTES_PER_SEC
            while offset < len(jpeg_data):
                end = min(offset + chunk_size, len(jpeg_data))
                s.sendall(jpeg_data[offset:end])
                offset = end
                if offset < len(jpeg_data):
                    time.sleep(1)  # throttle

            # Wait for ACK/NACK
            s.settimeout(10)
            resp = s.recv(1)
            if resp == ACK:
                print(f"[MOCK] Image sent: {filename} ({len(jpeg_data):,} bytes) → ACK")
                return True
            else:
                print(f"[MOCK] Image sent: {filename} → NACK")
                return False

    except ConnectionRefusedError:
        print(f"[MOCK] Image downlink: GCS not listening on {gcs_ip}:{DATA_PORT}")
        return False
    except Exception as e:
        print(f"[MOCK] Image downlink error: {e}")
        return False


def _run_downlink(gcs_ip: str, pass_n: int, cells: list):
    """Send all captured images for this pass to the GCS."""
    global _state, _img_seq

    print(f"[MOCK] Starting downlink: {len(cells)} image(s) for pass {pass_n}")

    sent_count = 0
    for idx, (row, col) in enumerate(cells):
        with _lock:
            if _state != "DOWNLINK":
                print("[MOCK] Downlink interrupted — state changed")
                break

        # Generate image
        jpeg_data = _generate_sand_image(row, col, pass_n)

        # Build filename and metadata matching protocol.py conventions
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"pass{pass_n}_img{idx:02d}_{ts}.jpg"

        # Quality score: simulated CubeSat-side quality assessment
        blur_score = round(random.uniform(0.6, 0.95), 3)
        exposure_score = round(random.uniform(0.7, 0.98), 3)
        combined_score = round((blur_score + exposure_score) / 2, 3)

        metadata = {
            "grid_cell":      [row, col],
            "pass_number":    pass_n,
            "capture_time":   datetime.now(timezone.utc).isoformat(),
            "blur_score":     blur_score,
            "exposure_score": exposure_score,
            "combined_score": combined_score,
            "resolution":     [320, 240],
        }

        success = _send_image(gcs_ip, jpeg_data, filename, metadata)
        if success:
            sent_count += 1
            with _lock:
                _img_seq += 1

        # Small delay between images (simulates processing gap)
        time.sleep(0.5)

    print(f"[MOCK] Downlink complete: {sent_count}/{len(cells)} images sent")

    # Transition to WAITING
    with _lock:
        if _state == "DOWNLINK":
            _state = "WAITING"
            print("[MOCK] State → WAITING")


# ── Command listener ──────────────────────────────────────────────────────────

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
            # Auto-assign cells to image: pick random grid cells
            all_cells = [(r, c) for r in range(8) for c in range(8)]
            random.shuffle(all_cells)
            _cells_imaged_this_pass = all_cells[:_images_per_pass]
            _captured = len(_cells_imaged_this_pass)
            print(f"[MOCK] State → IMAGING  (pass {_pass_num}, {_captured} cells: "
                  f"{_cells_imaged_this_pass})")

        elif name == "end_pass" and _state == "IMAGING":
            _state = "PROCESSING"
            pass_n = _pass_num
            cells = list(_cells_imaged_this_pass)
            print(f"[MOCK] State → PROCESSING")
            # Launch downlink in background thread
            threading.Thread(target=_auto_advance_with_downlink,
                             args=(pass_n, cells), daemon=True).start()

        elif name == "enter_safe_mode":
            _state = "SAFE_MODE"
            print("[MOCK] State → SAFE_MODE")

        elif name == "resume_normal" and _state == "SAFE_MODE":
            _state = "WAITING"
            print("[MOCK] State → WAITING")

        elif name == "status_request":
            print("[MOCK] Status request received — telemetry will be sent next cycle")

        elif name in ("cell", "set_cell"):
            r, c = cmd.get("row", "?"), cmd.get("col", "?")
            _current_cell = [r, c]
            print(f"[MOCK] Grid cell set to ({r},{c})")

        elif name == "adjust_exposure":
            print(f"[MOCK] Exposure → {cmd.get('exposure_us')} µs")

        elif name == "retry_downlink":
            if _state == "DOWNLINK":
                print("[MOCK] retry_downlink: resuming")

        elif name == "reset_mission":
            _state = "WAITING"
            _pass_num = 0
            _captured = 0
            _cells_imaged_this_pass = []
            print("[MOCK] Mission reset → WAITING")


def _auto_advance_with_downlink(pass_n: int, cells: list):
    """After end_pass: PROCESSING (2s) → DOWNLINK (send images) → WAITING."""
    global _state
    time.sleep(2)
    with _lock:
        if _state == "PROCESSING":
            _state = "DOWNLINK"
            print("[MOCK] State → DOWNLINK")
        else:
            return

    # Send images to GCS
    _run_downlink(_gcs_ip, pass_n, cells)


# ── Telemetry sender ──────────────────────────────────────────────────────────

def telemetry_sender(gcs_ip: str):
    print(f"[MOCK] Telemetry sender → {gcs_ip}:{DATA_PORT} every 3s")
    while True:
        time.sleep(3)
        _send_telemetry(gcs_ip)


def _send_telemetry(gcs_ip: str):
    with _lock:
        state    = _state
        pass_n   = _pass_num
        captured = _captured

    telem = {
        "state":               state,
        "pass_number":         pass_n,
        "captured_this_pass":  captured,
        "captured_total":      captured,
        "rejected_total":      0,
        "images_sent_total":   _img_seq,
        "battery_pct":         round(random.uniform(80, 95), 1),
        "cpu_temp_c":          round(random.uniform(38, 48), 1),
        "angular_rate_rad_s":  round(random.uniform(0.01, 0.05), 3),
        "nadir_angle_deg":     round(random.uniform(8, 18), 1),
        "nadir_locked":        True,
        "storage_used_pct":    round(10 + _img_seq * 0.5, 1),
        "storage_free_mb":     max(100, 2000 - _img_seq * 10),
        "gcs_reachable":       True,
        "consecutive_failures":0,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    }

    body = json.dumps(telem).encode("utf-8")
    md5  = hashlib.md5(body).hexdigest()

    header = json.dumps({
        "type":      "telemetry",
        "filename":  "telemetry.json",
        "file_size": len(body),
        "md5":       md5,
        "metadata":  {},
    }) + "\n"

    try:
        with socket.create_connection((gcs_ip, DATA_PORT), timeout=3) as s:
            s.sendall(header.encode("utf-8"))
            s.sendall(body)
            s.settimeout(3)
            ack = s.recv(1)
            status = "ACK" if ack == ACK else "NACK"
            # Only print every ~10th telemetry to reduce spam
            if random.random() < 0.1:
                print(f"[MOCK] Telemetry sent  state={state}  GCS={status}")
    except ConnectionRefusedError:
        print(f"[MOCK] Telemetry: GCS not listening on {gcs_ip}:{DATA_PORT}")
    except Exception as e:
        print(f"[MOCK] Telemetry error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock CubeSat simulator for GCS testing")
    parser.add_argument("--gcs-ip", default="127.0.0.1",
                        help="GCS IP to send data to (default: 127.0.0.1)")
    parser.add_argument("--images-per-pass", type=int, default=3,
                        help="Number of images to capture per pass (default: 3)")
    args = parser.parse_args()

    _gcs_ip = args.gcs_ip
    _images_per_pass = args.images_per_pass

    print("=" * 60)
    print("  Mock CubeSat")
    print(f"  Commands:      0.0.0.0:{COMMAND_PORT}")
    print(f"  Data → GCS:    {args.gcs_ip}:{DATA_PORT}")
    print(f"  Images/pass:   {_images_per_pass}")
    print(f"  Initial state: WAITING")
    print()
    print("  Workflow: START PASS → (auto-captures cells) → END PASS")
    print("           → PROCESSING (2s) → DOWNLINK (sends images) → WAITING")
    print("=" * 60)

    threading.Thread(target=command_listener, daemon=True).start()
    telemetry_sender(args.gcs_ip)   # blocks
