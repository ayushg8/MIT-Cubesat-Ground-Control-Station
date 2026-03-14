# protocol.py — Shared interface contract between CubeSat and GCS
#
# This file is IDENTICAL in both the CubeSat and GCS codebases.
# Do not add CubeSat-only or GCS-only logic here.

# === PORTS ===
DATA_PORT = 5000        # CubeSat → GCS: images and telemetry (TCP)
COMMAND_PORT = 5001     # GCS → CubeSat: JSON commands (TCP, persistent listener)

# === ACK / NACK ===
ACK = b'\x06'   # ASCII ACK — GCS sends this after a successful receive + verify
NACK = b'\x15'  # ASCII NAK — GCS sends this if MD5 mismatch or decode failure

# === TRANSFER PROTOCOL ===
#
# Every transfer (image or telemetry) follows this sequence:
#
#   1. CubeSat opens TCP connection to GCS at DATA_PORT
#   2. CubeSat sends a JSON header terminated by a single newline '\n':
#
#      {"type": "image"|"telemetry",
#       "filename": "pass3_img07_20260315_144500.jpg",
#       "file_size": 28400,
#       "md5": "a1b2c3d4e5f6...",
#       "metadata": { ...image metadata dict... }}
#
#   3. For type "image": CubeSat sends raw JPEG bytes in THROTTLE_BYTES_PER_SEC
#      chunks with real time.sleep(1) between chunks (1200-byte chunks, 1 sec sleep).
#      Transfer of a 28 KB image takes ~23 real seconds.
#
#   4. For type "telemetry": CubeSat sends the telemetry JSON as raw bytes
#      (no chunking needed — ~500 bytes, well under one chunk).
#
#   5. GCS verifies MD5, then sends ACK or NACK byte.
#
#   6. CubeSat reads exactly 1 byte. ACK → mark sent. NACK → retry or skip.
#      Socket timeout counts as a link failure (image stays queued).

# === COMMAND PROTOCOL (GCS → CubeSat) ===
#
# GCS connects to CubeSat COMMAND_PORT and sends JSON + '\n'.
# CubeSat command_listener runs as a daemon thread and parses these continuously.
#
# Supported commands:
#
#   {"cmd": "retransmit",    "image_id": "pass3_img07_20260315_144500"}
#       → Move that image to the top of the downlink queue
#
#   {"cmd": "priority_cell", "row": R, "col": C}
#       → Boost novelty score for that grid cell (next capture gets P1)
#
#   {"cmd": "set_cell",      "row": R, "col": C}
#       → Override current_grid_cell (backup for operator terminal input)
#
#   {"cmd": "adjust_exposure", "exposure_us": N}
#       → Manually set camera exposure time in microseconds for next captures
#
#   {"cmd": "enter_safe_mode"}
#       → Immediately enter SAFE_MODE state
#
#   {"cmd": "resume_normal"}
#       → Exit SAFE_MODE → transition to IDLE
#
#   {"cmd": "status_request"}
#       → Send a telemetry packet immediately (out-of-band, outside downlink window)
#
#   {"cmd": "retry_downlink"}
#       → Reset consecutive_failures counter, resume downlink attempts

# === FILE NAMING CONVENTION ===
#
# Image:    pass{N}_img{MM}_{YYYYMMDD_HHMMSS}.jpg
# Sidecar:  pass{N}_img{MM}_{YYYYMMDD_HHMMSS}_meta.json
#
# N  = pass number (1-indexed, increments each full IMAGING→DOWNLINK cycle)
# MM = image sequence within the pass, zero-padded to 2 digits (00–19)
# Timestamp is capture time in UTC.
#
# Examples:
#   pass1_img00_20260315_143000.jpg
#   pass1_img00_20260315_143000_meta.json
#   pass3_img07_20260315_144500.jpg
