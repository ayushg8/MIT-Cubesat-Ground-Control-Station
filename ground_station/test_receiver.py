#!/usr/bin/env python3
from __future__ import annotations
"""
test_receiver.py — End-to-end test for the TCP receiver.

Starts the listener on a test port (19500, not 5000) so it doesn't
conflict with a running ground station. Acts as a fake CubeSat client,
sends transfers using the real protocol, and verifies the listener
behaves correctly.

Run from ground_station/:
    python test_receiver.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(_HERE)

import hashlib
import json
import logging
import socket
import threading
import time
logging.basicConfig(level=logging.WARNING)   # suppress INFO noise during test

import cv2
import numpy as np

import config
import protocol

# ── Test port — avoids interfering with a live ground station ──────────────
_TEST_PORT = 19500
_TEST_HOST = "127.0.0.1"

# ── Terminal colour helpers ────────────────────────────────────────────────
_GREEN = "\033[92m"
_RED   = "\033[91m"
_YELLOW= "\033[93m"
_RESET = "\033[0m"

_passes = []
_failures = []

def _pass(label: str):
    _passes.append(label)
    print(f"  {_GREEN}PASS{_RESET}  {label}")

def _fail(label: str, reason: str = ""):
    _failures.append(label)
    suffix = f" — {reason}" if reason else ""
    print(f"  {_RED}FAIL{_RESET}  {label}{suffix}")

def _info(msg: str):
    print(f"         {_YELLOW}{msg}{_RESET}")

# ── Setup ──────────────────────────────────────────────────────────────────
def _setup_dirs():
    for d in [
        config.RECEIVED_DIR,
        config.TELEMETRY_DIR,
        os.path.join(config.PROCESSED_DIR, "shadow_masks"),
        os.path.join(config.PROCESSED_DIR, "hazard_maps"),
        os.path.join(config.PROCESSED_DIR, "change_maps"),
        os.path.join(config.PROCESSED_DIR, "mosaics"),
        os.path.join(config.PROCESSED_DIR, "routes"),
    ]:
        os.makedirs(d, exist_ok=True)


def _start_test_listener() -> bool:
    """
    Patch config to use the test port and start the listener daemon thread.
    Returns True if the socket bound successfully.
    """
    config.LISTEN_PORT = _TEST_PORT
    config.LISTEN_HOST = _TEST_HOST

    from receiver import listener as _listener

    ready = threading.Event()
    error_box = []

    def _run():
        try:
            # Build the server socket ourselves so we can signal readiness
            import socket as _socket
            srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            srv.bind((_TEST_HOST, _TEST_PORT))
            srv.listen(1)
            ready.set()
            while True:
                try:
                    conn, addr = srv.accept()
                    t = threading.Thread(
                        target=_listener._handle_connection,
                        args=(conn, addr),
                        daemon=True,
                    )
                    t.start()
                except Exception:
                    pass
        except Exception as e:
            error_box.append(e)
            ready.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    ready.wait(timeout=3.0)

    if error_box:
        print(f"  {_RED}Listener failed to start: {error_box[0]}{_RESET}")
        return False
    return True


# ── Helpers ────────────────────────────────────────────────────────────────
def _make_jpeg(width: int = 160, height: int = 120) -> bytes:
    """Create a minimal synthetic JPEG with a dark region (shadow-like)."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 150
    cv2.circle(img, (width // 2, height // 2), min(width, height) // 4,
               (20, 20, 20), -1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok, "cv2.imencode failed"
    return buf.tobytes()


def _send_image(jpeg_bytes: bytes, filename: str, metadata: dict,
                bad_md5: bool = False, truncate_at: int | None = None) -> bytes | None:
    """
    Send an image transfer to the test listener.

    Args:
        jpeg_bytes:  Raw JPEG data to send.
        filename:    Filename field in the header.
        metadata:    metadata dict for the header.
        bad_md5:     If True, corrupt the MD5 in the header.
        truncate_at: If set, close the connection after sending this many bytes
                     (simulates a dropped connection mid-transfer).

    Returns 1-byte ACK/NACK, or None if the socket was closed before response.
    """
    md5 = hashlib.md5(jpeg_bytes).hexdigest()
    if bad_md5:
        md5 = "deadbeef" * 4

    header = json.dumps({
        "type":      "image",
        "filename":  filename,
        "file_size": len(jpeg_bytes),
        "md5":       md5,
        "metadata":  metadata,
    }) + "\n"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(8.0)
    try:
        sock.connect((_TEST_HOST, _TEST_PORT))
        sock.sendall(header.encode("utf-8"))

        if truncate_at is not None:
            # Send only truncate_at bytes then drop the connection
            sock.sendall(jpeg_bytes[:truncate_at])
            sock.close()
            return None   # no response expected

        # Send full data in 1200-byte chunks (mirrors CubeSat protocol, no sleep in test)
        CHUNK = 1200
        for i in range(0, len(jpeg_bytes), CHUNK):
            sock.sendall(jpeg_bytes[i:i + CHUNK])

        response = sock.recv(1)
        return response
    except socket.timeout:
        return None
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _send_telemetry(telemetry: dict, filename: str) -> bytes | None:
    """Send a telemetry transfer. Returns 1-byte response."""
    data = json.dumps(telemetry).encode("utf-8")
    md5  = hashlib.md5(data).hexdigest()

    header = json.dumps({
        "type":      "telemetry",
        "filename":  filename,
        "file_size": len(data),
        "md5":       md5,
        "metadata":  {},
    }) + "\n"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(8.0)
    try:
        sock.connect((_TEST_HOST, _TEST_PORT))
        sock.sendall(header.encode("utf-8"))
        sock.sendall(data)
        response = sock.recv(1)
        return response
    except socket.timeout:
        return None
    finally:
        try:
            sock.close()
        except Exception:
            pass


# ── Test cases ─────────────────────────────────────────────────────────────

def test_valid_image():
    print("\n[1] Valid image transfer (correct MD5)")
    jpeg     = _make_jpeg()
    filename = "test_pass1_img00_20260315_120000.jpg"
    metadata = {
        "grid_cell":       [2, 3],
        "pass_number":     1,
        "combined_score":  0.82,
        "blur_variance":   95.4,
        "mean_brightness": 138.0,
    }

    dest = os.path.join(config.RECEIVED_DIR, filename)
    if os.path.exists(dest):
        os.remove(dest)

    response = _send_image(jpeg, filename, metadata)

    _info(f"response byte = {response!r}  (ACK={protocol.ACK!r})")
    _info(f"expected file = {dest}")

    time.sleep(0.3)   # let listener finish writing

    ok_ack  = response == protocol.ACK
    ok_file = os.path.exists(dest)
    ok_size = ok_file and os.path.getsize(dest) == len(jpeg)
    ok_meta = os.path.exists(dest.replace(".jpg", "_meta.json"))

    if ok_ack:    _pass("ACK received")
    else:         _fail("ACK expected", f"got {response!r}")

    if ok_file:   _pass("image file saved to received_images/")
    else:         _fail("image file not saved")

    if ok_size:   _pass(f"saved size matches ({len(jpeg)} bytes)")
    else:         _fail("saved size mismatch")

    if ok_meta:   _pass("sidecar _meta.json saved")
    else:         _fail("sidecar _meta.json not found")

    return ok_ack and ok_file and ok_size and ok_meta


def test_bad_md5():
    print("\n[2] Corrupt MD5 (listener must return NACK)")
    jpeg     = _make_jpeg()
    filename = "test_badmd5_img.jpg"
    metadata = {"grid_cell": [0, 0], "pass_number": 1, "combined_score": 0.5}

    dest = os.path.join(config.RECEIVED_DIR, filename)
    if os.path.exists(dest):
        os.remove(dest)

    response = _send_image(jpeg, filename, metadata, bad_md5=True)

    _info(f"response byte = {response!r}  (NACK={protocol.NACK!r})")

    ok_nack     = response == protocol.NACK
    ok_no_file  = not os.path.exists(dest)

    if ok_nack:     _pass("NACK received for bad MD5")
    else:           _fail("expected NACK", f"got {response!r}")

    if ok_no_file:  _pass("corrupt file correctly not saved")
    else:           _fail("corrupt file was saved (should be discarded)")

    return ok_nack and ok_no_file


def test_partial_transfer():
    print("\n[3] Partial transfer (drop connection mid-stream → file must not be saved)")
    jpeg          = _make_jpeg(width=320, height=240)   # slightly bigger for clear truncation
    filename      = "test_partial_img.jpg"
    truncate_at   = len(jpeg) // 3   # send only first third

    dest = os.path.join(config.RECEIVED_DIR, filename)
    if os.path.exists(dest):
        os.remove(dest)

    _info(f"declared size={len(jpeg)} bytes, sending only {truncate_at} bytes then closing")

    _send_image(jpeg, filename, {"grid_cell": [0, 0], "pass_number": 1,
                                 "combined_score": 0.5},
                truncate_at=truncate_at)

    time.sleep(0.5)   # let listener detect the drop and run its discard logic

    ok = not os.path.exists(dest)
    if ok:  _pass("partial transfer correctly discarded (file not saved)")
    else:   _fail("partial transfer file was saved (should have been discarded)")

    return ok


def test_telemetry():
    print("\n[4] Telemetry transfer")
    telemetry = {
        "state":            "DOWNLINK",
        "pass_number":      1,
        "roll_deg":         2.1,
        "pitch_deg":        -1.3,
        "cpu_temp_c":       58.4,
        "storage_used_pct": 34.2,
        "queue_size":       5,
        "nadir_locked":     True,
        "uptime_sec":       320,
    }
    filename = "telemetry_test_20260315_120000.json"

    response = _send_telemetry(telemetry, filename)

    _info(f"response byte = {response!r}  (ACK={protocol.ACK!r})")

    time.sleep(0.3)

    ok_ack   = response == protocol.ACK
    # telemetry_parser saves a file named telemetry_YYYYMMDD_HHMMSS_XXX.json
    saved = [f for f in os.listdir(config.TELEMETRY_DIR) if f.startswith("telemetry_")]
    ok_file  = len(saved) > 0

    if ok_ack:   _pass("ACK received for telemetry")
    else:        _fail("expected ACK", f"got {response!r}")

    if ok_file:  _pass(f"telemetry saved to disk ({saved[-1]})")
    else:        _fail("no telemetry file found in data/telemetry/")

    return ok_ack and ok_file


def test_two_consecutive_transfers():
    """Listener must accept a second connection after the first completes."""
    print("\n[5] Two consecutive transfers (listener re-accepts after first)")
    jpeg = _make_jpeg()
    meta = {"grid_cell": [1, 1], "pass_number": 2, "combined_score": 0.7}

    fn1 = "test_consec_img_a.jpg"
    fn2 = "test_consec_img_b.jpg"

    for dest in [os.path.join(config.RECEIVED_DIR, fn1),
                 os.path.join(config.RECEIVED_DIR, fn2)]:
        if os.path.exists(dest): os.remove(dest)

    r1 = _send_image(jpeg, fn1, meta)
    time.sleep(0.1)
    r2 = _send_image(jpeg, fn2, meta)
    time.sleep(0.3)

    ok1 = r1 == protocol.ACK and os.path.exists(os.path.join(config.RECEIVED_DIR, fn1))
    ok2 = r2 == protocol.ACK and os.path.exists(os.path.join(config.RECEIVED_DIR, fn2))

    if ok1:  _pass("first transfer ACK'd and saved")
    else:    _fail("first transfer failed", f"ack={r1!r}")

    if ok2:  _pass("second transfer ACK'd and saved (listener re-accepted)")
    else:    _fail("second transfer failed", f"ack={r2!r}")

    return ok1 and ok2


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  MuraltZ GCS — Receiver Smoke Test")
    print("=" * 58)

    _setup_dirs()

    print(f"\n  Starting test listener on {_TEST_HOST}:{_TEST_PORT} ...")
    if not _start_test_listener():
        print(f"  {_RED}Cannot start listener — aborting{_RESET}")
        sys.exit(1)
    _pass(f"listener bound on {_TEST_HOST}:{_TEST_PORT}")

    results = [
        test_valid_image(),
        test_bad_md5(),
        test_partial_transfer(),
        test_telemetry(),
        test_two_consecutive_transfers(),
    ]

    print("\n" + "=" * 58)
    total  = len(_passes) + len(_failures)
    passed = len(_passes)
    print(f"  {passed}/{total} checks passed")
    if _failures:
        print(f"  {_RED}Failed:{_RESET}")
        for f in _failures:
            print(f"    • {f}")
    else:
        print(f"  {_GREEN}All checks passed — receiver is functional{_RESET}")
    print("=" * 58)
    return len(_failures) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
