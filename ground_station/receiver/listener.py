# receiver/listener.py — TCP server that receives data pushed by the CubeSat
#
# The CubeSat connects here during its DOWNLINK state and pushes images +
# telemetry. The GCS does NOT pull — it listens and reacts.
#
# Flow per connection:
#   1. Accept connection from CubeSat
#   2. Read newline-terminated JSON header
#   3. Read file bytes (arrive slowly — CubeSat throttles to 1200 B/s,
#      a 28 KB image takes ~23 real seconds)
#   4. Validate via packet_handler (MD5 + size check)
#   5. ACK or NACK, save file, trigger pipeline
#
# Only one CubeSat connection is accepted at a time.

import json
import logging
import os
import socket
import threading

import config
import protocol
from receiver.downlink_state import get_state as get_downlink_state
from receiver.packet_handler import validate_transfer
from receiver.quality_check import run_ground_quality_check
from receiver.telemetry_parser import parse_and_save_telemetry

logger = logging.getLogger(__name__)

# Callback set by server.py so the pipeline can be triggered without a circular import
_pipeline_callback = None
_pipeline_lock = threading.Lock()
_mission_state = None


def set_pipeline_callback(cb):
    """Register the function to call when a validated image is ready."""
    global _pipeline_callback
    _pipeline_callback = cb


def set_mission_state(ms):
    """Register mission_state so downlink bytes can be tracked."""
    global _mission_state
    _mission_state = ms


def _recv_exact(sock, n):
    """Read exactly n bytes from sock. Returns bytes, or raises if socket closes early."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(4096, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"Socket closed after {len(buf)} of {n} bytes")
        buf += chunk
    return buf


def _read_header(sock):
    """Read bytes until newline, parse as JSON. Returns dict."""
    raw = b""
    while True:
        byte = sock.recv(1)
        if not byte:
            raise ConnectionError("Socket closed while reading header")
        if byte == b"\n":
            break
        raw += byte
    return json.loads(raw.decode("utf-8"))


def _handle_connection(conn, addr):
    """Handle one CubeSat connection which may carry multiple transfers
    (e.g. telemetry then one or more images in the same downlink window)."""
    logger.info(f"Connection from {addr}")
    try:
        while True:
            try:
                header = _read_header(conn)
            except ConnectionError:
                # Client closed cleanly — end of downlink window
                break

            transfer_type = header.get("type")
            filename = header.get("filename", "unknown")
            declared_size = header.get("file_size", 0)
            declared_md5 = header.get("md5", "")
            metadata = header.get("metadata", {})

            if transfer_type == "image":
                _handle_image(conn, filename, declared_size, declared_md5, metadata)
            elif transfer_type == "telemetry":
                _handle_telemetry(conn, filename, declared_size, declared_md5)
            elif transfer_type == "science_summary":
                _handle_science_summary(conn, filename, declared_size, declared_md5)
            else:
                logger.warning(f"Unknown transfer type '{transfer_type}' from {addr}")
                conn.sendall(protocol.NACK)

    except json.JSONDecodeError as e:
        logger.error(f"Bad header JSON from {addr}: {e}")
        try:
            conn.sendall(protocol.NACK)
        except Exception:
            pass
    except ConnectionError as e:
        logger.warning(f"Connection error from {addr}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error handling {addr}: {e}", exc_info=True)
    finally:
        conn.close()


def _handle_image(conn, filename, declared_size, declared_md5, metadata):
    """Receive image bytes, validate, save, trigger pipeline."""
    logger.info(f"Receiving image '{filename}' ({declared_size:,} bytes)")
    dl = get_downlink_state()
    dl.start_transfer(filename, declared_size)

    try:
        data = b""
        while len(data) < declared_size:
            chunk = conn.recv(min(4096, declared_size - len(data)))
            if not chunk:
                break
            data += chunk
            dl.update_progress(len(data))
    except Exception as e:
        logger.error(f"Error reading image bytes for '{filename}': {e}")
        dl.set_status("failed", str(e))
        conn.sendall(protocol.NACK)
        return

    # Partial transfer detection
    if len(data) < declared_size:
        logger.warning(
            f"Partial transfer '{filename}': {len(data):,} of {declared_size:,} bytes — discarded"
        )
        dl.set_status("failed", f"Partial: {len(data)}/{declared_size}")
        conn.sendall(protocol.NACK)
        return

    # Validate MD5 and size
    dl.set_status("validating")
    result = validate_transfer(data, declared_size, declared_md5, metadata)
    if not result.valid:
        logger.error(f"Validation failed for '{filename}': {result.reason}")
        dl.set_status("failed", result.reason)
        conn.sendall(protocol.NACK)
        return

    # Save file
    os.makedirs(config.RECEIVED_DIR, exist_ok=True)
    save_path = os.path.join(config.RECEIVED_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(data)
    logger.info(f"Saved '{filename}' ({len(data):,} bytes)")

    # Save sidecar metadata JSON
    meta_filename = filename.replace(".jpg", "_meta.json")
    meta_path = os.path.join(config.RECEIVED_DIR, meta_filename)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    conn.sendall(protocol.ACK)
    logger.info(f"ACK sent for '{filename}'")

    # Ground-side quality check (different from CubeSat checks)
    quality = run_ground_quality_check(save_path)
    if not quality["passed"]:
        logger.warning(
            f"Ground quality flag for '{filename}': {quality['notes']}"
        )
    else:
        logger.info(f"Ground quality OK for '{filename}'")

    # Trigger CV pipeline in background so we don't block the connection
    dl.set_status("processing")
    transfer_size = len(data)
    transfer_start = dl._start_time  # captured at start_transfer

    if _pipeline_callback is not None:
        def _run_pipeline(path, meta, qual):
            with _pipeline_lock:
                try:
                    _pipeline_callback(path, meta, qual)
                except Exception as e:
                    logger.error(f"Pipeline callback failed for '{os.path.basename(path)}': {e}", exc_info=True)
                finally:
                    dl.set_status("complete")
                    # Record cumulative downlink stats
                    import time as _time
                    duration = _time.monotonic() - transfer_start if transfer_start else 0
                    if _mission_state:
                        _mission_state.record_downlink_bytes(transfer_size, duration, True)

        t = threading.Thread(target=_run_pipeline, args=(save_path, metadata, quality), daemon=True)
        t.start()
    else:
        dl.set_status("complete")
        import time as _time
        duration = _time.monotonic() - transfer_start if transfer_start else 0
        if _mission_state:
            _mission_state.record_downlink_bytes(transfer_size, duration, True)


def _handle_telemetry(conn, filename, declared_size, declared_md5):
    """Receive telemetry JSON bytes, validate, hand off to parser."""
    try:
        data = _recv_exact(conn, declared_size)
    except ConnectionError as e:
        logger.warning(f"Partial telemetry '{filename}': {e}")
        conn.sendall(protocol.NACK)
        return

    result = validate_transfer(data, declared_size, declared_md5, {})
    if not result.valid:
        logger.error(f"Telemetry validation failed for '{filename}': {result.reason}")
        conn.sendall(protocol.NACK)
        return

    conn.sendall(protocol.ACK)

    try:
        parse_and_save_telemetry(data, filename)
    except Exception as e:
        logger.error(f"Telemetry parse error for '{filename}': {e}", exc_info=True)


def _handle_science_summary(conn, filename, declared_size, declared_md5):
    """Receive a compact science-summary JSON packet from the CubeSat."""
    try:
        data = _recv_exact(conn, declared_size)
    except ConnectionError as e:
        logger.warning(f"Partial science summary '{filename}': {e}")
        conn.sendall(protocol.NACK)
        return

    result = validate_transfer(data, declared_size, declared_md5, {})
    if not result.valid:
        logger.error(f"Science summary validation failed for '{filename}': {result.reason}")
        conn.sendall(protocol.NACK)
        return

    try:
        summary = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Science summary decode failed for '{filename}': {e}")
        conn.sendall(protocol.NACK)
        return

    conn.sendall(protocol.ACK)
    logger.info(f"Science summary received for '{summary.get('filename', filename)}'")
    if _mission_state:
        _mission_state.record_science_summary(summary)


def start_listener():
    """Start the TCP listener. Blocks forever, handles one connection at a time."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((config.LISTEN_HOST, config.LISTEN_PORT))
    server_sock.listen(1)  # backlog=1: only one CubeSat
    logger.info(f"Listening for CubeSat on {config.LISTEN_HOST}:{config.LISTEN_PORT}")

    while True:
        try:
            conn, addr = server_sock.accept()
            # Handle each connection in its own thread so the accept loop stays live
            t = threading.Thread(target=_handle_connection, args=(conn, addr), daemon=True)
            t.start()
        except Exception as e:
            logger.error(f"Accept error: {e}", exc_info=True)
