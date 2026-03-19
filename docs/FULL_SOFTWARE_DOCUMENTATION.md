# MuraltZ Artemis Lunar Navigator — Complete Software Documentation

## Mission Overview

MuraltZ is a CubeSat prototype built for the MIT BWSI program that demonstrates autonomous terrain mapping and hazard-aware route planning for lunar surface exploration. The system consists of two major software components:

1. **Flight Software** — Runs on a Raspberry Pi inside the CubeSat. Controls the camera, IMU, and communication hardware. Captures terrain images, performs onboard quality filtering, and downlinks data to the ground station over a throttled TCP link simulating a 9600-baud UHF radio.

2. **Ground Control Station (GCS)** — Runs on a laptop. Receives images and telemetry from the CubeSat, processes them through a multi-stage computer vision pipeline (shadow detection, hazard classification, mosaic stitching, change detection, route planning), and presents everything on a real-time web dashboard.

Both systems communicate over WiFi using a shared TCP protocol defined in `protocol.py`, which exists identically in both codebases.

---

# PART 1: FLIGHT SOFTWARE

**Repository:** `MIT-BWSI-Cubesat-Flight-Software/cubesat_flight/`

## 1.1 Architecture Overview

The flight software is a state machine that cycles through five states:

```
BOOT → WAITING → IMAGING → IDLE → DOWNLINK → WAITING → ...
                                                 ↑
                                          SAFE_MODE (error recovery)
```

Every value comes from real hardware — no simulation or mock data. The system is designed for autonomous operation: if the ground station becomes unreachable, the CubeSat continues capturing and queuing images.

### File Structure

```
cubesat_flight/
├── main.py                     — State machine entry point
├── config.py                   — All mission parameters
├── protocol.py                 — Shared GCS/CubeSat protocol contract
├── comms/
│   ├── command_listener.py     — GCS command daemon thread
│   ├── packet.py               — JSON transfer header builder
│   └── transfer.py             — Throttled TCP client
├── sensors/
│   ├── camera.py               — Pi Camera Module 3 wrapper
│   └── imu.py                  — LSM6DSO32 IMU wrapper
├── states/
│   ├── boot.py                 — Hardware self-test
│   ├── imaging.py              — IMU-gated capture loop
│   ├── idle.py                 — Queue building, aging, cleanup
│   ├── downlink.py             — Throttled TCP downlink
│   └── safe_mode.py            — Error recovery mode
├── processing/
│   ├── metadata.py             — Sidecar JSON generator
│   ├── quality.py              — Image quality gate (blur/exposure/motion)
│   ├── coverage.py             — 8×8 coverage grid tracker
│   ├── pipeline.py             — Onboard processing orchestrator
│   ├── mosaic_grid.py          — Dual-resolution cost grids
│   ├── pixel_segmenter.py      — Terrain segmentation
│   └── route_planner.py        — A* route planning
├── storage/
│   └── manager.py              — Queue, index, capacity, aging
├── utils/
│   ├── logger.py               — Rotating file logger
│   ├── telemetry.py            — Telemetry packet builder
│   ├── thermal.py              — CPU temperature monitor
│   └── watchdog.py             — Software watchdog timer
├── dashboard/
│   └── app.py                  — Onboard Flask dashboard
├── test_hardware.py            — Hardware verification script
└── test_quality.py             — Blur threshold calibration tool
```

---

## 1.2 Configuration — `config.py`

All mission parameters are centralized here. Key values:

### State Machine Timing
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `BOOT_TIMEOUT` | 30s | Max time for hardware self-test |
| `IMAGING_WINDOW_SEC` | 75s | Duration of each imaging pass |
| `IDLE_DURATION_SEC` | 30s | Post-imaging processing window |
| `DOWNLINK_WINDOW_SEC` | 60s | Time allowed for data downlink |

### Camera Settings
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `IMAGE_WIDTH` | 640 | Capture resolution width |
| `IMAGE_HEIGHT` | 480 | Capture resolution height |
| `JPEG_QUALITY` | 70 | JPEG compression level |
| `CAPTURE_INTERVAL_SEC` | 3.0 | Seconds between captures |
| `MAX_IMAGES_PER_PASS` | 20 | Hard cap per imaging pass |

### IMU Thresholds
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `ANGULAR_RATE_THRESHOLD` | 1.0 rad/s | Max rotation for stable capture |
| `NADIR_TOLERANCE_DEG` | 45° | Nadir lock engage angle |
| `NADIR_EXIT_DEG` | 55° | Nadir lock release angle (hysteresis) |

### Quality Gate
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `BLUR_THRESHOLD` | 20.0 | Laplacian variance minimum (requires calibration) |
| `EXPOSURE_MIN` | 15 | Min acceptable mean brightness |
| `EXPOSURE_MAX` | 240 | Max acceptable mean brightness |

### Downlink
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `GROUND_STATION_IP` | 192.168.1.225 | GCS IP address |
| `DATA_PORT` | 5000 | CubeSat → GCS TCP port |
| `COMMAND_PORT` | 5001 | GCS → CubeSat TCP port |
| `THROTTLE_BYTES_PER_SEC` | 1200 | Simulated UHF data rate |
| Data Budget per Pass | 72,000 bytes | 1200 B/s × 60s ≈ 2 images |

### Coverage Grid
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `GRID_ROWS` | 8 | Coverage grid rows |
| `GRID_COLS` | 8 | Coverage grid columns |

### Storage Limits
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `STORAGE_WARNING_PCT` | 80% | Triggers P3 cleanup |
| `STORAGE_CRITICAL_PCT` | 98% | Stops imaging entirely |

### Thermal Limits
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `CPU_TEMP_WARNING_C` | 70°C | Doubles capture interval |
| `CPU_TEMP_CRITICAL_C` | 80°C | Enters SAFE_MODE |

### Watchdog
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `WATCHDOG_TIMEOUT_SEC` | 30 | Restart if main loop stalls |

---

## 1.3 Protocol — `protocol.py`

This file is **identical** in both the CubeSat and GCS codebases. It defines the binary protocol:

### Ports
- **DATA_PORT (5000):** CubeSat → GCS. Images and telemetry.
- **COMMAND_PORT (5001):** GCS → CubeSat. JSON commands.

### ACK/NACK
- `ACK = 0x06` — Transfer received and verified (MD5 match).
- `NACK = 0x15` — MD5 mismatch, decode failure, or partial transfer.

### Transfer Sequence (CubeSat → GCS)
1. CubeSat opens TCP connection to GCS on DATA_PORT
2. Sends JSON header terminated by `\n`:
   ```json
   {
     "type": "image",
     "filename": "pass3_img07_20260315_144500.jpg",
     "file_size": 28400,
     "md5": "a1b2c3d4e5f6...",
     "metadata": { ... }
   }
   ```
3. For images: sends raw JPEG bytes in 1200-byte chunks with 1-second sleep between chunks (simulating 9600-baud UHF). A 28 KB image takes ~23 real seconds.
4. For telemetry: sends JSON bytes in one chunk (~500 bytes).
5. GCS verifies MD5, sends ACK or NACK.
6. CubeSat reads 1 byte: ACK = mark sent, NACK = retry or skip.

### Command Protocol (GCS → CubeSat)
GCS connects to COMMAND_PORT and sends JSON + `\n`. Supported commands:

| Command | Parameters | Effect |
|---------|-----------|--------|
| `retransmit` | `image_id` | Move image to top of downlink queue |
| `priority_cell` | `row`, `col` | Boost novelty for that grid cell |
| `set_cell` / `cell` | `row`, `col` | Set current grid cell to image |
| `adjust_exposure` | `exposure_us` | Manual camera exposure override |
| `enter_safe_mode` | — | Immediately enter SAFE_MODE |
| `resume_normal` | — | Exit SAFE_MODE |
| `status_request` | — | Send telemetry immediately |
| `retry_downlink` | — | Reset failure counter, resume downlink |
| `start_pass` | — | WAITING → IMAGING |
| `end_pass` | — | IMAGING → IDLE (stop capture) |
| `reset_mission` | — | Reset everything, start fresh |

### File Naming Convention
- Image: `pass{N}_img{MM}_{YYYYMMDD_HHMMSS}.jpg`
- Sidecar: `pass{N}_img{MM}_{YYYYMMDD_HHMMSS}_meta.json`

### IMU Metadata (Optional)
When available, the metadata dict includes:
```json
{
  "imu": {
    "roll_deg": 2.5,
    "pitch_deg": -1.3,
    "yaw_deg": 45.0,
    "angular_velocity": [0.1, 0.2, 0.05]
  }
}
```
The GCS mosaic stitcher uses `yaw_deg` as an initial rotation estimate. Falls back to pure feature matching if absent.

---

## 1.4 Main Entry Point — `main.py`

The main loop implements the state machine:

```python
BOOT → WAITING → IMAGING → IDLE → DOWNLINK → WAITING → ...
```

### Initialization Sequence
1. Initialize logger, thermal monitor, storage manager
2. Initialize IMU, camera, command listener, metadata builder, quality gate, coverage tracker
3. Start command listener daemon thread
4. Start watchdog timer
5. Start stdin reader thread (for operator terminal input)
6. Load recovery state if restarting after watchdog

### State Machine Flow

**BOOT:** Runs hardware self-test (IMU, camera, storage, GCS ping). On failure → SAFE_MODE. On success → WAITING.

**WAITING:** Blocks until operator types `start_pass` in terminal or GCS sends `start_pass` command. Increments pass counter.

**IMAGING:** Runs for up to 75 seconds. Only captures when:
- IMU angular rate < 1.0 rad/s (stable)
- Camera pointing within 45° of nadir (gravity direction)
- Nadir hysteresis prevents toggling: locks at 45°, releases at 55°

Each capture goes through the quality gate. Accepted images get priority P1. Rejected images are discarded. Hard stops: MAX_IMAGES_PER_PASS (20), storage critical, thermal critical.

**IDLE:** Builds priority downlink queue, ages old P2 images to P3, cleans up P3 if storage is above warning. Processes any pending GCS commands. Runs for 30 seconds.

**DOWNLINK:** Opens TCP connection to GCS. Sends telemetry first, then pops images from priority queue (P1 first, highest quality score first). Each image transfer takes ~23 seconds at 1200 B/s. Budget is 72,000 bytes per downlink window (~2 images). NACK → retry up to 3 times. If GCS unreachable after 3 consecutive failures → suspend downlink, continue autonomous.

### Operator Terminal Commands
The operator can type commands directly into the Pi terminal:
- `start_pass` — Begin imaging
- `end_pass` — Stop imaging
- `cell R C` — Set grid cell
- `status` — Print current state
- `eclipse_on` / `eclipse_off` — Toggle low-light camera mode
- `kill_link` / `resume_link` — Suspend/resume GCS communication
- `shutdown` — Clean exit

### Recovery and Persistence
The watchdog fires `os.execv()` restart after 30s stall. Before restart, the recovery callback saves current pass number, grid cell, and queue to `RECOVERY_FILE`. On next boot, `boot.py` loads recovery state and resumes from where it left off.

---

## 1.5 Sensors

### Camera — `sensors/camera.py`

Wraps the Pi Camera Module 3 (IMX708) via the `picamera2` library.

**Initialization:** Creates `Picamera2` instance with 640×480 resolution, waits 2 seconds for auto-gain control (AGC) and auto-white balance (AWB) to settle.

**Key Methods:**
- `capture(filepath)` — Captures a JPEG at the configured quality level. Returns the real camera metadata (exposure time in microseconds, analog gain, lux estimate).
- `capture_with_recompress(filepath, max_bytes)` — Captures, then re-encodes at progressively lower JPEG quality until the file is under `max_bytes`. Used to stay within the data budget.
- `set_low_light_mode()` — Switches to manual exposure (100ms, gain 8.0) for eclipse/low-light conditions.
- `set_normal_mode()` — Returns to auto-exposure.
- `get_metadata()` — Returns the most recent capture's real metadata: ExposureTime, AnalogueGain, Lux.
- `close()` — Stops the camera cleanly.

**Important:** The camera uses auto-exposure by default. All exposure and gain values are real readings from the sensor, not simulated.

### IMU — `sensors/imu.py`

Wraps the LSM6DSO32 6-axis IMU (accelerometer + gyroscope) via I2C at address `0x6A`.

**Key Methods:**
- `get_acceleration()` → `(x, y, z)` in m/s² (4G range)
- `get_gyro()` → `(x, y, z)` in rad/s
- `get_angular_rate()` → scalar magnitude of rotation in rad/s
- `is_stable()` → `True` if angular rate < 1.0 rad/s
- `get_nadir_angle()` → angle between camera boresight and gravity vector in degrees
- `get_angular_velocity()` → `[rx, ry, rz]` in deg/s (sent to GCS for mosaic stitcher)
- `get_orientation()` → `{roll, pitch, yaw, accel_mag, angular_rate}`

**Hardware Limitation:** The LSM6DSO32 has no magnetometer, so **yaw is always None**. Roll and pitch are computed from the accelerometer. The camera boresight is assumed along the -X axis (or +Z depending on mount orientation).

---

## 1.6 Communications

### Command Listener — `comms/command_listener.py`

A daemon thread that binds a server socket on COMMAND_PORT (5001) and listens for GCS commands.

**Flow:**
1. Accepts one connection at a time from the GCS
2. Reads newline-delimited JSON messages
3. Parses and validates each command
4. Queues valid commands (thread-safe)
5. Sends ACK/NACK back to GCS

**Thread Safety:** Uses a queue internally. `get_pending()` returns all queued commands non-blocking so the main loop can process them without blocking on I/O.

### Packet Builder — `comms/packet.py`

Builds the JSON headers that precede every TCP transfer.

- `build_image_header(filepath, meta_dict)` — Reads the JPEG file, computes its MD5, and builds the header dict with `type: "image"`, `filename`, `file_size`, `md5`, and `metadata`.
- `build_telemetry_header(telem_dict)` — Serializes the telemetry dict to JSON bytes and builds a header with `type: "telemetry"`, `file_size`, and `md5`.
- `md5_file(filepath)` — Computes MD5 in 64 KB chunks to handle large files without loading into memory.

### Transfer Client — `comms/transfer.py`

Manages TCP connections to the GCS and performs throttled data transfers.

**Key Methods:**
- `connect()` — Opens TCP connection to GCS DATA_PORT
- `send_file(filepath, metadata, watchdog)` — The core transfer method:
  1. Builds image header
  2. Sends header + newline
  3. Sends JPEG bytes in 1200-byte chunks with 1-second sleep between chunks
  4. Waits for ACK/NACK
  5. Pets the watchdog between chunks so the transfer doesn't trigger a restart
- `send_telemetry(telem_dict)` — Sends telemetry in a single chunk (no throttling needed for ~500 bytes)

**Error Handling:** Tracks `consecutive_failures`. After 3 consecutive failures → raises `GCSUnreachableError`, which causes the main loop to suspend downlink and switch to autonomous mode.

**Convenience Function:**
- `send_telemetry_now(telem_dict)` — Opens a fresh connection, sends telemetry, closes. Used for out-of-band status updates.

---

## 1.7 State Machine States

### BOOT — `states/boot.py`

Runs hardware self-test before the main loop begins. Tests:

1. **IMU:** Reads acceleration, checks magnitude is between 8–35 m/s² (should be near 9.8 m/s² at rest on Earth).
2. **Camera:** Captures a test image, verifies it's >1 KB and decodable as JPEG.
3. **Storage:** Checks disk capacity. If >98% full → downlink-only mode.
4. **GCS:** Attempts TCP ping. If unreachable → continues in autonomous mode (non-fatal).
5. **Recovery:** Loads `RECOVERY_FILE` if it exists (watchdog restart).

**Failure Policy:**
- IMU or Camera fail → enters SAFE_MODE
- GCS unreachable → non-fatal, autonomous mode
- Storage critical → imaging disabled, downlink-only

### IMAGING — `states/imaging.py`

The capture loop. Runs for up to `IMAGING_WINDOW_SEC` (75 seconds).

**Capture Gating:** Every iteration checks two conditions:
1. `imu.is_stable()` — Angular rate < 1.0 rad/s
2. Nadir lock — Camera pointing within 45° of gravity direction

**Nadir Hysteresis:** Prevents rapid toggling when the CubeSat is near the threshold:
- Lock **engages** when nadir angle drops below 45°
- Lock **releases** when nadir angle rises above 55°
- This 10° hysteresis band prevents capture → skip → capture oscillation

**Per-Capture Flow:**
1. Wait for IMU stability + nadir lock
2. Read IMU orientation and angular velocity
3. Capture image via camera
4. Compute quality score (blur, exposure, motion, novelty)
5. If quality passes → save image + metadata sidecar, update coverage grid
6. If quality fails → log rejection reason, discard image

**Grid Cell Assignment:** The operator sets the current cell via `cell R C` command in the terminal or the GCS sends `set_cell`. There is no automatic trajectory-based cell assignment.

**Hard Stops:**
- `MAX_IMAGES_PER_PASS` (20) reached
- Storage critical (>98%)
- Thermal critical (>80°C)
- Operator types `end_pass`
- GCS sends `end_pass` command

### IDLE — `states/idle.py`

Post-imaging processing phase (30 seconds).

**Actions:**
1. Build priority downlink queue from image index
2. Age P2 images: if an image is P2 and `current_pass - capture_pass ≥ P2_AGING_PASSES` (2), demote to P3
3. Delete oldest P3 images if storage is above warning (80%)
4. Process pending GCS commands (retransmit, priority_cell, adjust_exposure)
5. Save queue, image index, and coverage to disk
6. Sleep for `IDLE_DURATION_SEC`

### DOWNLINK — `states/downlink.py`

Transfers data to the GCS over the throttled TCP link.

**Sequence:**
1. Send telemetry first (always, regardless of image budget)
2. Pop images from priority queue (P1 first, then P2, then P3; within each tier, highest quality score first)
3. Transfer each image at 1200 B/s (~23 seconds per 28 KB image)
4. Budget: 1200 × 60 = 72,000 bytes per window (~2 images)
5. On NACK: retry up to 3 times, then mark corrupted
6. On socket error: stop downlink, image stays queued for next pass
7. On `GCSUnreachableError`: suspend downlink entirely

**Data Budget Enforcement:** After each image, checks if remaining budget allows another image. Stops if not.

### SAFE_MODE — `states/safe_mode.py`

Error recovery mode. Entered on:
- Hardware failure during BOOT
- Thermal critical (CPU >80°C)
- GCS sends `enter_safe_mode`

**Behavior:**
1. Stops camera to reduce power consumption and heat
2. Blocks, waiting for `resume_normal` from GCS or `resume` from operator
3. Continues petting watchdog to prevent restart
4. On resume: attempts to restart camera, returns to IDLE

---

## 1.8 Image Processing (Onboard)

### Quality Gate — `processing/quality.py`

Scores each image on four criteria before accepting it:

| Check | Weight | Hard Fail Condition | Score Formula |
|-------|--------|-------------------|---------------|
| Blur | 0.30 | Laplacian variance < `BLUR_THRESHOLD` | `min(1, variance / (3 × threshold))` |
| Exposure | 0.25 | Mean brightness < 15 or > 240 | `1.0 - abs(mean - 127.5) / 127.5` |
| Motion | 0.20 | Angular rate ≥ 1.0 rad/s | `1.0 - rate / threshold` |
| Novelty | 0.25 | None (never hard fails) | From coverage tracker |

**Combined Score:** `Q = 0.3×blur + 0.25×exposure + 0.2×motion + 0.25×novelty`

A hard fail on any of the first three criteria rejects the image regardless of the combined score.

### Coverage Tracker — `processing/coverage.py`

Tracks an 8×8 grid of which terrain cells have been imaged.

**Novelty Scoring:**
- `1.0` — Cell never captured (maximum priority)
- `0.5` — Cell captured but best quality < 0.7 (room to improve)
- `0.1` — Cell captured with quality ≥ 0.7 (redundant)

This feeds back into the quality gate's novelty score, so the system naturally prioritizes uncovered terrain.

### Metadata Builder — `processing/metadata.py`

Generates comprehensive JSON sidecars for each image:

```json
{
  "filename": "pass1_img03_20260315_143000.jpg",
  "pass_number": 1,
  "image_sequence": 3,
  "timestamp": "2026-03-15T14:30:00.000Z",
  "grid_cell": [2, 5],
  "imu": {
    "roll": 2.5, "pitch": -1.3, "yaw": null,
    "angular_velocity": [0.1, 0.2, 0.05],
    "accel_mag": 9.81, "angular_rate": 0.15,
    "stable": true, "nadir_locked": true, "nadir_angle_deg": 12.3
  },
  "camera": {
    "exposure_us": 5000, "analog_gain": 2.1,
    "lux": 450, "jpeg_quality": 70, "recompressed": false
  },
  "quality": {
    "blur_variance": 45.2, "blur_score": 0.85,
    "exposure_mean": 130.0, "exposure_score": 0.98,
    "motion_score": 0.85, "novelty_score": 1.0,
    "combined_score": 0.92, "passed_gate": true
  },
  "priority_tier": "P1",
  "file_size_bytes": 28400,
  "md5": "a1b2c3d4e5f6...",
  "downlink_status": "pending"
}
```

---

## 1.9 Storage Management — `storage/manager.py`

Manages all persistent data on the Pi's SD card.

### Priority Queue
Images are queued for downlink in priority order:
- **P1** (tier rank 0): Current pass captures. Downloaded first.
- **P2** (tier rank 1): Previous pass captures not yet sent.
- **P3** (tier rank 2): Old captures. Deleted when storage is low.

Within each tier, images are sorted by quality score (highest first).

### Aging
After each pass, images older than `P2_AGING_PASSES` (2) passes are demoted from P2 to P3. P3 images are candidates for deletion.

### Cleanup
When storage exceeds 80%, the oldest P3 images are deleted until usage drops below the warning threshold.

### Recovery State
Before a watchdog restart, the current pass number, grid cell, and queue are saved to `RECOVERY_FILE`. On next boot, this state is loaded to resume seamlessly.

### Persistence Files
- `IMAGE_INDEX_FILE` — Full record of every capture (metadata, priority, status)
- `QUEUE_FILE` — Ordered list of images pending downlink
- `RECOVERY_FILE` — Watchdog restart context

---

## 1.10 Utilities

### Logger — `utils/logger.py`

Time-stamped, state-labeled log lines with automatic rotation at 5 MB.

**Format:** `{ISO_timestamp} [{STATE}] [{LEVEL}] {message}`

**Levels:** INFO, DEBUG, WARN, ERROR

**Features:**
- `get_recent()` returns last 50 lines (included in telemetry)
- Module-level singleton: `log()`, `set_state()`, `get_recent()`

### Telemetry Builder — `utils/telemetry.py`

Assembles comprehensive telemetry packets from all live hardware:

```json
{
  "type": "telemetry",
  "timestamp": "...",
  "cubesat_id": "muraltz",
  "pass_number": 3,
  "state": "IMAGING",
  "uptime_sec": 450,
  "imu": { "accel": [...], "gyro": [...], "angular_rate": 0.15, "stable": true, "nadir_locked": true, "nadir_angle_deg": 12.3 },
  "camera": { "exposure": 5000, "analog_gain": 2.1, "lux": 450, "mode": "auto" },
  "thermal": { "cpu_temp": 55.0, "throttled": false },
  "storage": { "used_pct": 45, "free_mb": 1200 },
  "imaging": { "captured_this_pass": 5, "captured_total": 23, "rejected_total": 4 },
  "downlink": { "queued": 8, "sent_total": 15, "bytes_this_pass": 56000, "budget_remaining": 16000, "gcs_reachable": true },
  "coverage": { "cells_filled": 12, "cells_total": 64, "pct": 18.75 },
  "errors": [],
  "recent_log": ["...last 50 lines..."]
}
```

### Thermal Monitor — `utils/thermal.py`

Background daemon thread that reads `/sys/class/thermal/thermal_zone0/temp` every 10 seconds.

- Warning at 70°C → imaging loop doubles capture interval
- Critical at 80°C → enters SAFE_MODE (stops camera, waits for cooldown)

### Watchdog — `utils/watchdog.py`

Software watchdog that restarts the entire process if the main loop stalls.

- Main loop must call `watchdog.pet()` regularly (at least every 30 seconds)
- Background thread checks elapsed time every 5 seconds
- On timeout: calls `recovery_callback` to flush state, then `os.execv()` to restart the process
- The transfer loop also pets the watchdog between chunks so long downloads don't trigger a restart

---

## 1.11 Testing Tools

### Hardware Test — `test_hardware.py`

Quick verification of all hardware before flight:
1. IMU: acceleration magnitude, gyro readings, angular rate, nadir angle
2. Camera: captures a test image, verifies >10 KB and decodable
3. Temperature: reads sysfs, checks within safe range
4. GCS: attempts TCP connection to DATA_PORT
5. Storage: reports capacity, usage, free space

### Quality Calibration — `test_quality.py`

Interactive tool for calibrating `BLUR_THRESHOLD`:
1. Hold CubeSat steady → capture photos → note minimum blur variance
2. Shake CubeSat → capture photos → note maximum blur variance
3. Set threshold = midpoint
4. Re-test to confirm sharp photos pass, blurry ones fail

Outputs detailed per-photo analysis with visual quality bars.

---

# PART 2: GROUND CONTROL STATION

**Repository:** `MIT-Cubesat-Ground-Control-Station/ground_station/`

## 2.1 Architecture Overview

The GCS is a Flask-based application that:
1. Listens for CubeSat TCP connections on port 5000
2. Receives images and telemetry
3. Validates transfers (MD5 + size)
4. Runs a multi-stage CV pipeline on each image
5. Maintains a live mission state
6. Serves a real-time web dashboard on port 3000
7. Sends commands to the CubeSat on port 5001

### File Structure

```
ground_station/
├── server.py                       — Entry point
├── config.py                       — All configuration
├── protocol.py                     — Shared protocol (identical to flight)
├── receiver/
│   ├── listener.py                 — TCP server
│   ├── packet_handler.py           — MD5 + size validation
│   ├── quality_check.py            — Ground-side quality checks
│   ├── telemetry_parser.py         — Telemetry JSON parser
│   └── downlink_state.py           — Transfer progress tracker
├── uplink/
│   ├── commander.py                — Command sender
│   └── pi_manager.py               — SSH/mDNS Pi management
├── processing/
│   ├── pipeline.py                 — CV pipeline orchestrator
│   ├── shadow_detector.py          — Adaptive threshold shadow detection
│   ├── hazard_classifier.py        — Multi-feature terrain classification
│   ├── change_detector.py          — SSIM-based change detection
│   ├── mosaic_stitcher.py          — SuperPoint + LightGlue mosaic
│   ├── mosaic_grid.py              — Dynamic dual-resolution grid
│   ├── route_planner.py            — A* with 3 strategies + PPO
│   ├── pixel_segmenter.py          — Per-pixel terrain labeling
│   ├── yolo_detector.py            — YOLOv8 object detection
│   ├── traversability_cnn.py       — MobileNetV2 traversability
│   ├── slope_estimator.py          — Shadow-based slope estimation
│   ├── ppo_planner.py              — PPO reinforcement learning planner
│   └── mission_state.py            — Persistent mission state
├── dashboard/
│   ├── app.py                      — Flask routes + API
│   └── templates/index.html        — Single-page dashboard
├── llm/
│   ├── interface.py                — Ollama LLM wrapper
│   └── system_prompt.txt           — Mission assistant prompt
├── models/                         — ML model weights
├── data/                           — Runtime data storage
└── tools/
    └── capture_training_images.py  — Training data capture utility
```

---

## 2.2 Configuration — `config.py`

### Network
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `LISTEN_PORT` | 5000 | Receive images/telemetry from CubeSat |
| `COMMAND_PORT` | 5001 | Send commands to CubeSat |
| `LISTEN_HOST` | 0.0.0.0 | Listen on all interfaces |
| `CUBESAT_IP` | 192.168.1.229 | CubeSat's IP (must be set before demo) |
| `DASHBOARD_PORT` | 3000 | Web dashboard port |

### Storage Paths
| Parameter | Value |
|-----------|-------|
| `RECEIVED_DIR` | `data/received_images` |
| `PROCESSED_DIR` | `data/processed` |
| `TELEMETRY_DIR` | `data/telemetry` |
| `MISSION_STATE_FILE` | `data/mission_state.json` |

### Ground Quality Checks
These are **different** from the CubeSat's checks. CubeSat checks blur/exposure/motion. Ground checks:

| Check | Threshold | Purpose |
|-------|-----------|---------|
| Texture Variance | >20 | Avg local variance across 8×8 patches |
| Contrast Range | >50 | Grayscale histogram span |
| Single Color | <90% | Reject if >90% pixels in narrow band |

### Hazard Costs
| Level | Cost | Meaning |
|-------|------|---------|
| SAFE | 1 | Free to traverse |
| MODERATE | 5 | Slightly risky |
| SHADOW | 15 | Uncertain terrain |
| HAZARD | 20 | Dangerous |
| IMPASSABLE | 999 | Cannot traverse |

### Mosaic Configuration
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MOSAIC_GRID_CELL_PX` | 80 | Coarse grid cell size (pixels) |
| `MOSAIC_INITIAL_CANVAS_PX` | 640 | Starting canvas size |
| `MOSAIC_PX_PER_CM` | 8.0 | Pixels per centimeter (calibration) |
| `MOSAIC_MIN_SIFT_INLIERS` | 6 | Min RANSAC inliers for valid match |
| `MOSAIC_MAX_CANVAS_PX` | 4096 | Memory cap |
| `MOSAIC_MAX_KEYPOINTS` | 1024 | SuperPoint max keypoints |
| `MOSAIC_BUNDLE_ADJUST_INTERVAL` | 3 | Bundle adjust every N images |
| `MOSAIC_BLEND_LEVELS` | 4 | Laplacian pyramid levels |

### Pixel Segmentation
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `SEG_GRID_CELL_PX` | 20 | Fine grid cell size |
| `SEG_SAFETY_DILATION_PX` | 3 | Safety margin around hazards |
| `SEG_ENABLED` | True | Feature flag |

### Feature Flags
| Flag | Default | Purpose |
|------|---------|---------|
| `SEG_ENABLED` | True | Pixel segmentation + fine grid routing |
| `SLOPE_ENABLED` | True | Shadow-based slope estimation |
| `CNN_ENABLED` | True | MobileNetV2 traversability inference |
| `UNCERTAINTY_ENABLED` | True | Uncertainty-weighted cost grid |

---

## 2.3 Entry Point — `server.py`

Starts the entire ground station in sequence:

1. **Logging** — Console + rotating file (`data/logs/gcs.log`)
2. **Directories** — Creates all required data directories
3. **Core Objects** — `MissionState`, `Pipeline`, `Commander`
4. **Dashboard Wiring** — Injects pipeline, mission state, and commander into dashboard app (avoids circular imports)
5. **Pipeline Quality Hook** — Monkey-patches pipeline to push quality entries to dashboard log
6. **Listener Callback** — Wires listener → pipeline so new images trigger processing
7. **TCP Listener Thread** — Starts `listener.start_listener()` in a daemon thread
8. **Flask Dashboard** — Starts on port 3000 (blocks until Ctrl-C)

### Dependency Injection Pattern
The architecture avoids circular imports through dependency injection:
- `server.py` creates all objects
- Passes `pipeline` and `mission_state` to `dashboard/app.py` via `set_pipeline()` and `set_mission_state()`
- Passes a lambda callback to `listener.py` via `set_pipeline_callback()`
- Pipeline and dashboard never import each other directly

---

## 2.4 Receiver System

### TCP Listener — `receiver/listener.py`

Binds a TCP server on port 5000 and handles CubeSat connections.

**Flow per connection:**
1. Accept connection (one at a time via `listen(1)` backlog)
2. Loop: read newline-terminated JSON header
3. Dispatch based on `type` field: "image" or "telemetry"
4. For images: receive bytes with progress tracking, validate, save, trigger pipeline
5. For telemetry: receive exact bytes, validate, parse

**Image Handling (`_handle_image`):**
1. Parse header → get filename, declared size, MD5, metadata
2. Update `DownlinkState` → "receiving"
3. Read bytes in 4096-byte chunks, tracking progress for SSE dashboard
4. Detect partial transfers (socket closed early) → NACK
5. Validate MD5 + size → "validating"
6. Save JPEG to `data/received_images/`
7. Save metadata sidecar JSON
8. Send ACK
9. Run ground-side quality check
10. Trigger pipeline in a background thread (so we don't block the TCP connection)
11. Pipeline lock ensures only one image is processed at a time

**Telemetry Handling (`_handle_telemetry`):**
1. `_recv_exact()` — reads exactly `declared_size` bytes
2. Validate MD5
3. ACK
4. Hand off to `telemetry_parser.parse_and_save_telemetry()`

### Packet Handler — `receiver/packet_handler.py`

Validates incoming transfers. Returns a `ValidationResult(valid: bool, reason: str)`.

Checks:
1. Actual size matches declared size
2. Computed MD5 matches declared MD5
3. Metadata is JSON-serializable

### Ground Quality Check — `receiver/quality_check.py`

Three ground-side checks that are **different** from what the CubeSat checks:

1. **Texture Variance** — Divides image into 8×8 patches, computes local grayscale variance in each, averages. If average variance < 20, the image has insufficient texture for the CV pipeline (e.g., perfectly uniform sand).

2. **Contrast Range** — Computes grayscale histogram, finds the span (max - min brightness level with nonzero pixels). If span < 50, the image is too low-contrast.

3. **Color Validity** — Counts what percentage of pixels fall in a narrow brightness band (10-level window). If >90%, the image is essentially monochrome and useless for analysis.

Returns `{passed: bool, score: 0.0-1.0, notes: ["issue1", "issue2"]}`.

### Telemetry Parser — `receiver/telemetry_parser.py`

Parses incoming telemetry JSON from the CubeSat.

- Handles both nested format (`telemetry.imu.angular_rate`) and flat format (`angular_rate`)
- Saves timestamped JSON files to `data/telemetry/`
- Maintains an in-memory cache of the latest telemetry for the dashboard

### Downlink State — `receiver/downlink_state.py`

Thread-safe singleton tracking the current download progress.

**State Machine:** `idle → receiving → validating → processing → complete/failed`

**Tracked Values:**
- Filename, declared size, bytes received
- Transfer rate (bytes/sec), ETA
- Error messages
- History of last 20 completed transfers

**Used By:** Dashboard SSE endpoint polls `get_snapshot()` to show real-time transfer progress with a progress bar.

---

## 2.5 Uplink System

### Commander — `uplink/commander.py`

Sends JSON commands to the CubeSat over TCP port 5001.

**Connection Pattern:** Opens a fresh TCP connection for each command (connection-per-command). This is robust against CubeSat restarts — no persistent connection to maintain.

**Flow:**
1. Connect to `CUBESAT_IP:5001`
2. Send JSON + newline
3. Wait up to 5 seconds for ACK
4. Close connection

**Named Command Methods:**
| Method | Command Sent |
|--------|-------------|
| `retransmit(image_id)` | `{"cmd": "retransmit", "image_id": "..."}` |
| `priority_cell(row, col)` | `{"cmd": "priority_cell", "row": R, "col": C}` |
| `set_cell(row, col)` | `{"cmd": "set_cell", "row": R, "col": C}` |
| `adjust_exposure(us)` | `{"cmd": "adjust_exposure", "exposure_us": N}` |
| `enter_safe_mode()` | `{"cmd": "enter_safe_mode"}` |
| `resume_normal()` | `{"cmd": "resume_normal"}` |
| `request_status()` | `{"cmd": "status_request"}` |
| `retry_downlink()` | `{"cmd": "retry_downlink"}` |
| `start_pass()` | `{"cmd": "start_pass"}` |
| `end_pass()` | `{"cmd": "end_pass"}` |
| `set_grid_cell(row, col)` | `{"cmd": "cell", "row": R, "col": C}` |

### Pi Manager — `uplink/pi_manager.py`

Manages the Raspberry Pi remotely via SSH.

- **mDNS Discovery** — Resolves `cubesat.local` via `socket.getaddrinfo()` to find the Pi's IP
- **SSH** — Uses `paramiko` library for SSH connections
- `start_flight_software()` — SSHs into the Pi and runs `python3 main.py` in the flight directory
- `stop_flight_software()` — Kills the flight software process
- `get_pi_log()` — Retrieves recent log lines from the Pi

**Configuration:**
- `PI_FLIGHT_DIR = "/home/cubesat/MIT-BWSI-Cubesat/cubesat_flight"`
- `PI_FLIGHT_CMD = "main.py"`

---

## 2.6 CV Pipeline — `processing/pipeline.py`

The heart of the GCS. Processes each received image through a multi-stage pipeline.

### Pipeline Stages (in order)

```
Image Received
    │
    ▼
1. MOSAIC STITCHING
    │  SuperPoint features → LightGlue matching → MAGSAC++ homography
    │  → exposure compensation → multi-band Laplacian blend
    │  → bundle adjustment every 3 images
    │
    ▼
2. GRID UPDATE
    │  Project image footprint onto dynamic mosaic grid
    │
    ▼
3. SHADOW DETECTION
    │  Adaptive threshold → morphological cleanup
    │  → connected components → boundary gradient analysis
    │
    ▼
4. SLOPE ESTIMATION (if SLOPE_ENABLED)
    │  Shadow geometry + sun position → terrain slope angles
    │
    ▼
5. HAZARD CLASSIFICATION
    │  LBP texture + Canny edges + brightness + contours
    │  → decision tree: SAFE/MODERATE/SHADOW/HAZARD/IMPASSABLE
    │
    ▼
6. YOLO DETECTION + FUSION (if model available)
    │  YOLOv8 inference → bounding boxes + confidence
    │  → fusion with classical CV (agreement boosts, disagreement flags)
    │
    ▼
7. PIXEL SEGMENTATION (if SEG_ENABLED)
    │  Shadow mask + YOLO detections → per-pixel label map
    │  → project onto fine grid (20px cells)
    │
    ▼
8. CNN TRAVERSABILITY (if CNN_ENABLED)
    │  MobileNetV2 patch inference → blend with classical costs
    │
    ▼
9. CHANGE DETECTION
    │  SSIM comparison with previous images of same area
    │  → contour extraction → persistence check → classification
    │
    ▼
10. ROUTE PLANNING
    │   A* with 3 strategies (Fastest/Safest/Balanced)
    │   + PPO reinforcement learning route
    │
    ▼
11. SAVE STATE
    Mission state JSON updated atomically
```

### Thread Safety
The pipeline is protected by a `threading.Lock()`. Only one image is processed at a time, even though the listener may accept new connections concurrently. The pipeline callback runs in a background thread so the TCP connection isn't blocked during processing.

### Image Index
Persistent `image_index.json` tracks every processed image: filename, pass number, grid cell, quality scores, pipeline results. Survives restarts.

---

## 2.7 Shadow Detection — `processing/shadow_detector.py`

Identifies shadow regions in terrain images.

### Algorithm
1. Convert to grayscale
2. **Adaptive thresholding** — Gaussian, blockSize=51, C=10. Identifies locally dark regions.
3. **Morphological cleanup** — Open (remove noise) → Close (fill gaps)
4. **Connected components** — Label each shadow region, compute statistics
5. **Shadow vs. Object Discrimination** — For each region, compute Sobel gradient magnitude at the boundary. If gradient > 30, it's likely a dark object (boulder), not a shadow. True shadows have gradual brightness transitions.

### Output
- Binary shadow mask (255=shadow, 0=lit)
- Blue-tinted overlay visualization
- JSON with per-region statistics (area, centroid, boundary gradient)
- Shadow percentage of total image

---

## 2.8 Hazard Classification — `processing/hazard_classifier.py`

Classifies each image into terrain hazard levels using multiple classical CV features.

### Features Extracted
1. **LBP Texture Variance** — Local Binary Pattern computed on grayscale, variance of LBP histogram. High variance = rough/hazardous texture.
2. **Canny Edge Density** — Ratio of edge pixels to total pixels. High density = complex terrain.
3. **Brightness Statistics** — Mean, std dev, dark pixel percentage.
4. **Shadow Percentage** — From shadow detector output.
5. **Contour Analysis** — Count, total area coverage, average circularity.

### Decision Tree
```
Contour coverage > 50%?
  └─ YES → IMPASSABLE (massive obstruction)
Shadow > 40%?
  └─ YES → SHADOW (uncertain terrain)
Circular + dark contours?
  └─ YES → HAZARD-crater
Irregular + bright contours?
  └─ YES → HAZARD-boulder
LBP variance > 5.0 OR edge density > 0.02?
  └─ YES → MODERATE
Otherwise → SAFE
```

### Confidence Scoring
Each classification gets a confidence score (0-1) based on how strongly the features support the classification. Higher feature values = higher confidence.

### Output
- Classification label + confidence
- BGR color-coded hazard map image
- `cost_grid.json` for the dashboard

---

## 2.9 Mosaic Stitching — `processing/mosaic_stitcher.py`

Builds a continuous terrain mosaic from overlapping images.

### Feature Matching: SuperPoint + LightGlue
Classical SIFT fails on low-texture sand surfaces. The system uses **learned** feature detection:

- **SuperPoint** — Neural network that detects keypoints in textureless regions where SIFT finds nothing. Max 1024 keypoints per image.
- **LightGlue** — Neural network feature matcher that replaces brute-force matching. Handles repetitive textures better than ratio tests.

### Homography Estimation: MAGSAC++
Instead of standard RANSAC, uses **MAGSAC++** (Marginalizing Sample Consensus):
- Automatically adapts the inlier threshold
- More robust on noisy matches from sandy terrain
- Minimum 6 inliers required (low threshold for difficult surfaces)

### Exposure Compensation
Before blending, brightness is normalized across overlapping regions:
- Compute mean brightness in the overlap area for both images
- Apply gain to match brightness levels
- Gain clamped to [0.5, 2.0] to prevent extreme corrections

### Multi-Band Laplacian Blending
Eliminates visible seams by blending at different frequency bands:
1. Build 4-level Laplacian pyramid for each image
2. Build Gaussian pyramid for the blend mask
3. Blend each level independently
4. Reconstruct from blended pyramid

Low frequencies (overall brightness) transition gradually; high frequencies (edges, texture) transition sharply. This prevents ghosting while eliminating seam artifacts.

### Bundle Adjustment
Every 3 images, global pose refinement runs:
- Uses `scipy.optimize.least_squares` to minimize reprojection error across all matched feature pairs
- Refines all homography matrices simultaneously
- Prevents accumulated drift from propagating across the mosaic

### Canvas Management
- Starts at 640×640 pixels
- Grows dynamically when new images exceed current bounds
- Padding of 100px added when expanding
- Hard cap at 4096×4096 pixels (memory protection)

### IMU Yaw Hint
If the CubeSat provides `yaw_deg` in the image metadata, it's used as an initial rotation estimate for feature matching. This improves convergence on images with very few features. Falls back to pure feature matching if yaw is absent.

### Persistence
- `mosaic_index.json` — Records all image entries: filename, homography matrix, placement position
- `mosaic_latest.png` — Current mosaic visualization
- NumPy files — Raw canvas and mask arrays for incremental updates

---

## 2.10 Dynamic Mosaic Grid — `processing/mosaic_grid.py`

Maintains a dual-resolution grid over the mosaic canvas.

### Coarse Grid (80px cells)
- Used for hazard classification display
- Each cell has: hazard class, cost, confidence, observation count
- Updated by `apply_hazard()` when hazard classifier produces results

### Fine Grid (20px cells)
- Used for high-resolution route planning
- Each cell has: semantic label, cost, confidence
- Updated by `apply_segmentation_mask()` from pixel segmenter

### Dynamic Growth
Both grids grow when the mosaic canvas expands. Existing cell data is preserved in the overlapping region.

### Uncertainty-Aware Cost Grid
When `UNCERTAINTY_ENABLED=True`:
- Unsurveyed cells get a base cost of 15 (same as SHADOW)
- Confidence increases by 0.1 per additional observation
- Low-confidence cells get a cost multiplier of `UNCERTAINTY_WEIGHT` (3.0)
- Effective cost = `base_cost × (1 + UNCERTAINTY_WEIGHT × (1 - confidence))`

### Slope Cost Multiplier
When `SLOPE_ENABLED=True`, slope angles from the slope estimator modify costs:
| Slope Range | Multiplier |
|-------------|-----------|
| 0°–15° (gentle) | 1× |
| 15°–30° (moderate) | 2× |
| 30°–45° (steep) | 5× |
| >45° | IMPASSABLE (999) |

### CNN Traversability Blending
When `CNN_ENABLED=True`:
- CNN produces a 0-1 traversability score per patch
- Blended with classical cost: `final = 0.4×CNN + 0.6×classical`

---

## 2.11 Route Planning — `processing/route_planner.py`

Computes optimal paths across the hazard grid.

### A* Search
8-connected grid (cardinal + diagonal movement). Diagonal cost = √2. Cells with cost ≥ COST_IMPASSABLE (999) are walls.

### Three Simultaneous Strategies

| Strategy | Shadow Multiplier | Adjacent-to-Hazard Penalty |
|----------|------------------|-----------------------------|
| **Fastest** | 1× | 0 |
| **Safest** | 3× | +10 |
| **Balanced** | 1.5× | +5 |

All three routes are computed simultaneously and displayed on the dashboard. The operator selects which one to follow.

### PPO Reinforcement Learning Route
Alongside the three A* routes, a PPO (Proximal Policy Optimization) route is computed:
- Model trained via stable-baselines3
- Takes an 11×11×4 local observation: cost view, impassable view, distance-to-goal, direction-to-goal
- Outputs one of 9 discrete actions (8 compass directions + stay)
- Includes stuck detection (bail after 8 visits to same cell)
- Includes oscillation trimming (removes repeated tail cells)

### Constraint-Based Planning
The dashboard provides sliders for:
- Maximum hazard clearance distance
- Shadow avoidance weight
- Maximum slope tolerance

These constraints modify the cost grid before A* runs.

### Route Visualization
Color-coded output:
- **Green** path = Fastest
- **Blue** path = Safest
- **Yellow** path = Balanced
- **Magenta** path = PPO
- Start marker (green circle), End marker (red circle)
- Legend with distance and cost for each route

### Physical Distance
Path length is converted from grid cells to centimeters using `GRID_CELL_SIZE_CM` (10 cm per cell) and the grid resolution ratio.

---

## 2.12 Change Detection — `processing/change_detector.py`

Detects terrain changes between passes (e.g., boulder movement, new shadows, rover tracks).

### Why Not ORB?
ORB feature matching fails on sand — too many repetitive features produce incorrect matches. The system uses SSIM (Structural Similarity Index) instead.

### Algorithm
1. **Alignment** — Template matching using corner patches to align the new image with the previous image of the same area. Alignment confidence must exceed 0.7.
2. **SSIM Comparison** — Compute structural similarity map. Low-similarity regions indicate changes.
3. **Thresholding** — Pixels with SSIM difference > `CHANGE_THRESHOLD` (30/255) are marked as changed.
4. **Contour Extraction** — Find contiguous changed regions. Filter by minimum area (`CHANGE_MIN_AREA_PX` = 50 pixels).
5. **Aspect Ratio Filtering** — Regions with aspect ratio > 5:1 are likely shadow edges, not real changes. Filtered out.
6. **Persistence Check** — A change is only confirmed if it appears in 3+ consecutive passes. Single-pass anomalies are noise.
7. **YOLO-Assisted Classification** — If YOLO detects a known object overlapping the change region, the change is classified (e.g., "new boulder", "crater shift").

### Mosaic-Level Change Detection
`detect_mosaic()` uses homography-based overlap comparison between mosaic versions to detect changes across the entire mapped area, not just individual images.

---

## 2.13 Pixel Segmentation — `processing/pixel_segmenter.py`

Produces a per-pixel semantic label map from shadow masks and YOLO detections.

### Label Schema
| Value | Label | Color (BGR) | Traversal Cost |
|-------|-------|-------------|----------------|
| 0 | UNSURVEYED | Dark gray | 1 (assumed safe) |
| 1 | SAND | Light sandy | 1 |
| 2 | PLAIN_SURFACE | Pale green | 1 |
| 3 | SHADOW | Dark blue | 15 |
| 4 | CRATER | Orange | 20 |
| 5 | BOULDER | Red | 999 |

### Algorithm
1. Start with all pixels labeled SAND (safe)
2. Apply shadow mask → label shadow pixels as SHADOW
3. For each YOLO detection:
   - "plain" class → mark as PLAIN_SURFACE (where not already hazard)
   - "crater" or "boulder" class → extract precise contour within bounding box:
     a. Crop grayscale ROI from the bounding box
     b. Adaptive threshold (objects are darker than surrounding sand)
     c. Morphological cleanup (close then open)
     d. Find contours, keep those > 5% of bbox area
     e. Fill contour pixels with the hazard label
     f. Fallback: if no good contours, fill 60% ellipse within bbox
4. Dilate all hazard labels by 3 pixels (safety margin)
   - Dilation order: SHADOW → CRATER → BOULDER (worse labels overwrite)
5. Save color-coded visualization PNG

---

## 2.14 YOLO Object Detection — `processing/yolo_detector.py`

Dual-tier YOLOv8 detection system.

### Model Tiers
1. **Tier 1: Custom Terrain Model** (`models/terrain_detector.pt`) — Trained on sandbox/desert + Pi camera images. Preferred when available.
2. **Tier 2: Lunar Model** (`models/lunar_detector.pt`) — Trained on 5,600+ real LROC (Lunar Reconnaissance Orbiter Camera) images.
3. **Tier 3: COCO Fallback** (`yolov8n.pt`) — Maps common COCO objects to lunar categories as a last resort.

### COCO Class Mapping (Fallback)
| COCO Class | Lunar Category |
|-----------|---------------|
| bowl, cup, vase, frisbee | crater |
| sports ball, apple, orange, donut | boulder |
| bottle, book, cell phone, remote | obstacle |

### Detection Output
```json
{
  "class": "crater",
  "confidence": 0.87,
  "bbox": [120, 80, 200, 160],
  "area_px": 6400,
  "center": [160, 120],
  "original_class": "Impact_crater_10-100m"
}
```

### Full-Frame Filter
Detections covering >50% of the image area are discarded as model noise.

### Classical CV Fusion
The `fuse_classifications()` function combines YOLO detections with classical CV grid classifications:

| YOLO | Classical | Fused Result |
|------|-----------|-------------|
| Hazard | Hazard | High-confidence hazard (boosted) |
| Hazard | Moderate | Upgrade to HAZARD if YOLO conf >0.7 |
| Hazard | Safe | Override to HAZARD if YOLO conf >0.6, else MODERATE |
| Nothing | Hazard | Keep HAZARD but lower confidence |
| Nothing | Safe | Safe (boosted if YOLO found "plain") |

When both systems agree, confidence increases. When they disagree, the cell is flagged for human review.

---

## 2.15 Slope Estimation — `processing/slope_estimator.py`

Estimates terrain slope from shadow geometry.

### Algorithm
For each shadow contour:
1. Fit minimum bounding rectangle
2. Project rectangle dimensions onto sun direction vector
3. **Shadow length** = extent along sun direction
4. **Shadow width** = extent perpendicular to sun direction
5. **Object height** = shadow_length × tan(sun_elevation)
6. **Slope** = arctan(object_height / shadow_width)
7. Paint slope values along shadow boundary (3px thick contour)
8. Extend slope into adjacent cells (7×7 dilation)

### Configuration
- Sun elevation: 30° above horizon
- Sun azimuth: 180° from North (due south)
- These must be set before demo based on lighting setup

### Output
- `slope_map` — Float32 array of slope in degrees (same resolution as image)
- `regions` — List of detected slope regions with centroid, angle, shadow length

---

## 2.16 Traversability CNN — `processing/traversability_cnn.py`

Learned traversability prediction using MobileNetV2.

### Architecture
- **Backbone:** MobileNetV2 (ImageNet pretrained)
- **Head:** Dropout(0.2) → Linear(1280→1) → Sigmoid
- **Output:** Traversability score 0.0 (impassable) → 1.0 (safe)

### Inference
1. Divide mosaic crop into 64×64 pixel patches
2. Resize each patch to 224×224 (MobileNetV2 input size)
3. Standard ImageNet normalization
4. Batch inference (batch size 32)
5. Output grid of traversability scores

### Training (Bootstrap)
The CNN can be trained from existing pipeline classifications:
1. Extract patches from mosaic
2. Label each patch using the fine grid's dominant semantic label
3. Map labels to traversability scores (SAFE=1.0, MODERATE=0.6, SHADOW=0.3, HAZARD=0.1, IMPASSABLE=0.0)
4. Train with MSE loss, Adam optimizer, lr=1e-4, 10 epochs

### Blending
When `CNN_ENABLED=True`:
- `final_cost = 0.4 × CNN_cost + 0.6 × classical_cost`
- CNN fills in gaps where classical features are ambiguous

### Device Selection
Automatically uses CUDA → MPS (Apple Silicon) → CPU, in that order.

---

## 2.17 PPO Route Planner — `processing/ppo_planner.py`

Reinforcement learning-based route planner using Proximal Policy Optimization.

### Model
- Trained via stable-baselines3
- Loaded from `PPO Training/best_model.zip`

### Observation Space
11×11×4 local view around the agent:
1. **Cost channel** — Normalized cost grid (0=free, 1=wall)
2. **Impassable channel** — Binary wall map
3. **Distance channel** — Normalized Euclidean distance to goal
4. **Direction channel** — Normalized angle to goal

All four channels are flattened into a single vector for the model.

### Action Space
9 discrete actions: 8 compass directions + stay in place.

### Safety Features
- **Stuck detection:** If the agent visits the same cell 8+ times, bail out
- **Oscillation trimming:** Remove repeated tail cells from the final path
- **Boundary padding:** Edges of the grid are treated as walls
- **Max steps:** 500 or 4× grid size (whichever is larger)

### Output
```json
{
  "path": [[r1,c1], [r2,c2], ...],
  "path_length": 45,
  "total_cost": 67.5,
  "reached_goal": true,
  "cumulative_slip_risk": 12.3,
  "distance_cm": 450.0,
  "status": "found"
}
```

---

## 2.18 Mission State — `processing/mission_state.py`

Single source of truth for all mission data. Thread-safe, persistent.

### Atomic Writes
Uses `tempfile` + `os.rename()` pattern:
1. Write JSON to a temp file in the same directory
2. Atomic rename over `mission_state.json`
3. This prevents corruption from interrupted writes

### Tracked Data
- Pass count, current pass number
- Total images captured, corrupted, quality scores
- Coverage percentage and per-cell data
- Hazard detections and locations
- Change events with persistence counts
- Active routes (fastest, safest, balanced, PPO)
- Downlink statistics (sent, failed, bytes transferred)
- Uplink command history
- ML detection summaries (YOLO agreement rates)

### Thread Safety
All reads and writes are protected by `threading.Lock()`. The pipeline updates mission state after each processing stage; the dashboard reads it for display.

---

## 2.19 Dashboard — `dashboard/app.py`

Flask web application serving a real-time mission operations interface.

### API Endpoints

| Endpoint | Method | Returns |
|----------|--------|---------|
| `/` | GET | Dashboard HTML page |
| `/api/status` | GET | Merged telemetry + mission state |
| `/api/mosaic_info` | GET | Mosaic dimensions, image count, grid info |
| `/api/mosaic` | GET | Latest mosaic PNG |
| `/api/coverage` | GET | Dynamic coverage grid JSON |
| `/api/latest_image` | GET | Most recent received JPEG |
| `/api/hazard_map` | GET | Latest hazard classification PNG |
| `/api/change_map` | GET | Latest change detection PNG |
| `/api/route_map` | GET | Latest route visualization PNG |
| `/api/routes` | GET | Route data (fastest/safest/balanced/PPO) |
| `/api/quality_log` | GET | Image quality check history |
| `/api/log` | GET | Last 100 application log lines |
| `/api/downlink_stream` | GET | SSE stream for real-time transfer progress |
| `/api/cost_grid` | GET | Hazard cost grid for mosaic overlay |
| `/api/segmentation_map` | GET | Latest segmentation visualization PNG |

### Command Endpoints

| Endpoint | Method | Effect |
|----------|--------|--------|
| `/api/start_pass` | POST | Send start_pass to CubeSat |
| `/api/end_pass` | POST | Send end_pass to CubeSat |
| `/api/set_cell` | POST | Set CubeSat grid cell |
| `/api/safe_mode` | POST | Put CubeSat in safe mode |
| `/api/resume_normal` | POST | Resume from safe mode |
| `/api/request_status` | POST | Request immediate telemetry |
| `/api/find_cubesat` | POST | mDNS discovery of Pi |
| `/api/pi_start` | POST | SSH start flight software |
| `/api/pi_stop` | POST | SSH stop flight software |
| `/api/set_route_endpoints` | POST | Set route start/end from map clicks |
| `/api/select_route` | POST | Choose a route strategy |
| `/api/plan_constrained` | POST | Plan route with constraint sliders |
| `/api/new_session` | POST | Clear all data, start fresh |
| `/api/export_mission` | GET | Download ZIP of all mission data |
| `/api/llm_query` | POST | Ask LLM about mission data |

### Dashboard Panels (6 Tabs)
1. **Overview** — Live stats (images, coverage, pass, connection status), latest image, event log
2. **Mosaic** — Interactive Leaflet map showing stitched mosaic with grid overlay, click-to-set route endpoints, route visualization, hazard overlay
3. **Telemetry** — IMU data (accel, gyro, angular rate, nadir), camera settings, thermal, storage, command buttons (safe mode, resume, etc.)
4. **Routes** — Route cards for each strategy (distance, cost, slip risk), constraint sliders, mosaic-overlaid route visualization
5. **Change Detection** — Timeline slider, change events per cell, before/after comparisons
6. **Coverage** — Canvas-drawn coverage grid, per-cell quality scores, coverage percentage timeline

### SSE (Server-Sent Events)
The `/api/downlink_stream` endpoint provides real-time transfer updates:
- Polls `DownlinkState.get_snapshot()` every 0.5 seconds
- Only sends events when the sequence number changes
- Dashboard shows progress bar, transfer rate, ETA during active downloads

---

## 2.20 LLM Interface — `llm/interface.py`

Optional natural language interface for querying mission data.

### Architecture
- Uses **Ollama** (local LLM runtime) with **llama3.2** model
- Subprocess-based: calls `ollama run llama3.2` with prompt piped to stdin
- 30-second timeout

### How It Works
1. Load `mission_state.json` (all current mission data)
2. Load `system_prompt.txt` template
3. Inject mission data JSON into the template
4. Append user question
5. Send to ollama, return response

### System Prompt
```
You are the ground station mission planning assistant for the MuraltZ Artemis
Lunar Navigator CubeSat mission. Answer questions using ONLY the mission data
provided below. Every number was computed from real satellite imagery and real
sensor readings. Do not invent any information. If you don't know, say so.

CURRENT MISSION DATA:
{mission_state_json}
```

### Dashboard Integration
The dashboard has a chat panel. User types a question, POST to `/api/llm_query`, response displayed. The dashboard also has its own direct ollama REST API path (urllib) that doesn't go through this module.

### Graceful Degradation
- If ollama is not installed → returns helpful install instructions
- If query times out → returns timeout message
- If model not loaded → returns loading message
- Never crashes the application

---

## 2.21 Training Image Capture — `tools/capture_training_images.py`

Utility for rapidly capturing training data on the Pi.

### Features
- Captures 400 images at 0.4-second intervals
- 640×480 resolution, JPEG quality 85
- Serves a live MJPEG stream on port 8085 so you can watch from your Mac
- Camera auto-detection: tries rpicam-still → libcamera-still
- Milestones at images 80, 160, 240, 320 prompt operator to rearrange the scene

### MJPEG Stream
A built-in HTTP server serves two endpoints:
- `/stream` — MJPEG multipart stream (continuous frame updates)
- `/latest` — Single latest frame

### Usage
```bash
# On the Pi:
python3 capture_training_images.py

# On your Mac, open in browser:
http://cubesat.local:8085/

# After capture, copy images to Mac:
scp -r cubesat@cubesat.local:~/training_images/ ~/Desktop/training_images/
```

---

# PART 3: SYSTEM INTEGRATION

## 3.1 End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        CubeSat (Pi)                         │
│                                                             │
│  Camera → Quality Gate → Priority Queue → TCP Transfer      │
│    ↑          ↑                              │              │
│   IMU    Coverage Grid                       │ 1200 B/s     │
│                                              │ throttled    │
└──────────────────────────────────────────────┼──────────────┘
                                               │
                                    WiFi (TCP port 5000)
                                               │
┌──────────────────────────────────────────────┼──────────────┐
│                   GCS (Laptop)               │              │
│                                              ▼              │
│  TCP Listener → Validate → Ground Quality → Pipeline        │
│                                              │              │
│  Pipeline: Mosaic → Shadow → Hazard → YOLO → Segmentation  │
│            → Slope → CNN → Change → Route Planning          │
│                                              │              │
│  Mission State ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘              │
│       │                                                     │
│       ▼                                                     │
│  Flask Dashboard (port 3000)                                │
│       │                                                     │
│  Commander ──── TCP port 5001 ────→ CubeSat                 │
└─────────────────────────────────────────────────────────────┘
```

## 3.2 Dual Quality Gate Architecture

The CubeSat and GCS perform **different** quality checks:

| Check | CubeSat | GCS |
|-------|---------|-----|
| Blur | Laplacian variance | — |
| Exposure | Mean brightness bounds | — |
| Motion | IMU angular rate | — |
| Texture | — | 8×8 patch variance |
| Contrast | — | Histogram span |
| Color | — | Single-color percentage |

**Rationale:** TCP guarantees delivery, so there's no point re-checking blur or exposure on the ground. Instead, the ground checks things that affect the CV pipeline: can the mosaic stitcher find features? Can the hazard classifier see terrain variation?

## 3.3 Connection Architecture

| Link | Direction | Port | Protocol | Speed |
|------|-----------|------|----------|-------|
| Downlink | CubeSat → GCS | 5000 | TCP + JSON header + raw bytes | 1200 B/s |
| Uplink | GCS → CubeSat | 5001 | TCP + JSON | Instant |
| Dashboard | Browser → GCS | 3000 | HTTP + SSE | LAN speed |
| Pi SSH | GCS → CubeSat | 22 | SSH (paramiko) | LAN speed |

The CubeSat **pushes** data to the GCS. The GCS does **not** pull. This simulates a real satellite downlink where the ground station can only listen during a communication window.

## 3.4 Fault Handling

| Fault | CubeSat Response | GCS Response |
|-------|-----------------|--------------|
| GCS unreachable | Continue autonomous, suspend downlink | — |
| CubeSat unreachable | — | Dashboard shows disconnected |
| IMU failure | SAFE_MODE | — |
| Camera failure | SAFE_MODE | — |
| MD5 mismatch | Retry (3×), then mark corrupted | NACK, log error |
| Partial transfer | Image stays queued | Discard, NACK |
| Storage >80% | Delete P3 images | — |
| Storage >98% | Stop imaging, downlink only | — |
| CPU >70°C | Double capture interval | — |
| CPU >80°C | SAFE_MODE (stop camera) | — |
| Watchdog timeout | `os.execv()` restart with recovery | — |
| Pipeline crash | — | Log error, image marked failed |

## 3.5 Dependencies

### Flight Software (Pi)
```
picamera2          — Pi Camera Module 3 control
adafruit-circuitpython-lsm6ds  — LSM6DSO32 IMU via I2C
board, busio       — I2C bus access
opencv-python      — Image quality checks (Laplacian, brightness)
numpy              — Array operations
```

### Ground Control Station (Laptop)
```
flask              — Web dashboard
opencv-python      — Full CV pipeline
numpy              — Array operations
scipy              — Bundle adjustment (least_squares), SSIM
pillow             — Image manipulation
paramiko           — SSH to Pi
ultralytics        — YOLOv8 inference
torch, torchvision — MobileNetV2 CNN, SuperPoint
kornia             — LightGlue feature matching
stable-baselines3  — PPO route planner
ollama             — Local LLM (optional)
```

---

*This document covers the complete software architecture for both the CubeSat flight software and the Ground Control Station as of the MuraltZ Artemis Lunar Navigator mission build.*
