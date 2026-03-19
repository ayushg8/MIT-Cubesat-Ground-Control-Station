# MuraltZ Ground Control Station — Complete Project Documentation

**Artemis Lunar Navigator · MIT BWSI 2025-2026**

This document describes everything that has been built in this project: architecture, code structure, data flow, modules, APIs, configuration, and how to run it.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Data Flow](#3-data-flow)
4. [Module Reference](#4-module-reference)
5. [Configuration](#5-configuration)
6. [Protocol](#6-protocol)
7. [API Reference](#7-api-reference)
8. [Key Algorithms](#8-key-algorithms)
9. [External Dependencies](#9-external-dependencies)
10. [Tools & Utilities](#10-tools--utilities)
11. [Known Limitations](#11-known-limitations)
12. [How to Run](#12-how-to-run)

---

## 1. Project Overview

### What This Is

The **MuraltZ Ground Control Station (GCS)** is laptop software that:

1. **Receives** images and telemetry from a CubeSat over WiFi (TCP)
2. **Processes** images through a multi-stage computer vision pipeline
3. **Sends** commands back to the CubeSat (start pass, end pass, set cell, etc.)
4. **Displays** everything on a real-time web dashboard

All data is **real** — no mock or simulated data in production. The system is designed for the MIT BWSI CubeSat prototype that demonstrates autonomous terrain mapping and hazard-aware route planning for lunar surface exploration.

### High-Level Architecture

```
┌─────────────────┐     TCP 5000      ┌──────────────────────────────────────────┐
│   CubeSat       │ ─────────────────►│  GCS (this project)                      │
│   (Raspberry Pi)│   images +         │  ┌─────────────┐  ┌─────────────────────┐ │
│                 │   telemetry        │  │  Listener   │─►│  Pipeline           │ │
└────────┬────────┘                    │  │  (receiver) │  │  (CV processing)    │ │
         │                             │  └─────────────┘  └──────────┬──────────┘ │
         │     TCP 5001                │         ▲                   │            │
         │◄───────────────────────────│  ┌───────┴───────┐  ┌────────┴──────────┐ │
         │   commands (JSON)          │  │  Commander    │  │  MissionState      │ │
         │                             │  │  (uplink)     │  │  (accumulated data)│ │
         │                             │  └──────────────┘  └────────────────────┘ │
         │                             │         │                   │            │
         │                             │  ┌──────┴───────────────────┴──────────┐ │
         │                             │  │  Flask Dashboard (port 3000)        │ │
         │                             │  │  Serves HTML + JSON API             │ │
         │                             │  └─────────────────────────────────────┘ │
         │                             └──────────────────────────────────────────┘
```

### Key Design Principles

- **Push model:** CubeSat pushes to GCS; GCS does not pull
- **Continuous mosaic:** Images stitched into a growing canvas; grid derived dynamically from mosaic dimensions
- **Dynamic grid:** No fixed 8×8 — grid size = `canvas_size / MOSAIC_GRID_CELL_PX` (80px per cell)
- **Ground quality ≠ CubeSat quality:** Ground checks texture, contrast, color validity; CubeSat checks blur, exposure, motion

---

## 2. Directory Structure

```
MIT-Cubesat-Ground-Control-Station/
├── ground_station/                    # Main GCS package
│   ├── server.py                      # Entry point — starts listener, dashboard
│   ├── config.py                      # All configuration
│   ├── protocol.py                    # Shared with CubeSat (ports, format)
│   ├── generate_sample_outputs.py     # Generate fake data for UI testing
│   ├── test_pipeline.py               # Pipeline unit tests
│   ├── test_receiver.py               # Receiver protocol tests
│   ├── evaluate_accuracy.py           # Evaluate hazard/change/route accuracy
│   ├── calibrate_detection.py         # Interactive calibration for shadow/hazard
│   │
│   ├── receiver/                      # Data reception
│   │   ├── listener.py                # TCP server (port 5000)
│   │   ├── packet_handler.py         # MD5 validation, transfer validation
│   │   ├── quality_check.py          # Ground-side quality checks
│   │   ├── telemetry_parser.py       # Parse and cache telemetry
│   │   └── downlink_state.py         # Downlink progress tracking
│   │
│   ├── uplink/                       # Command uplink
│   │   ├── commander.py              # Send commands to CubeSat (port 5001)
│   │   └── pi_manager.py             # Discover Pi, start/stop flight SW via SSH
│   │
│   ├── processing/                   # CV pipeline
│   │   ├── pipeline.py               # Orchestrator — runs all stages
│   │   ├── mosaic_stitcher.py        # SuperPoint + LightGlue stitching
│   │   ├── mosaic_grid.py            # Dynamic grid over mosaic
│   │   ├── shadow_detector.py        # Otsu threshold, shadow regions
│   │   ├── hazard_classifier.py      # LBP, edges → hazard class
│   │   ├── change_detector.py        # SSIM, template matching (core science)
│   │   ├── route_planner.py          # A* on cost grid
│   │   ├── pixel_segmenter.py        # Semantic segmentation
│   │   ├── slope_estimator.py        # Shadow-based slope
│   │   ├── yolo_detector.py          # ML object detection + fusion
│   │   ├── traversability_cnn.py     # CNN for traversability
│   │   ├── landing_recommender.py    # Recommend landing sites
│   │   ├── ppo_planner.py            # PPO-based route planning
│   │   └── mission_state.py         # Accumulated mission data
│   │
│   ├── dashboard/
│   │   ├── app.py                    # Flask app, 50+ API routes
│   │   └── templates/
│   │       └── index.html            # Single-page dashboard (~2400 lines)
│   │
│   ├── llm/
│   │   ├── interface.py              # Optional Ollama integration
│   │   └── system_prompt.txt
│   │
│   ├── models/                       # ML models
│   │   ├── download_model.py         # Download lunar YOLO from Roboflow
│   │   └── terrain_combined/         # YOLO training data
│   │
│   └── tools/
│       ├── mock_pipeline_run.py     # Simulate full pipeline run
│       ├── simulate_downlink.py     # Send images to GCS
│       ├── capture_training_images.py
│       ├── train_terrain_yolo.py
│       └── download_datasets.py
│
├── mock_cubesat.py                   # Simulates CubeSat for local testing
├── PPO Training/                     # PPO route planner training
│   ├── ppo_ground_station.py
│   └── lunar_ppo_config.json
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ASSUMPTIONS_AND_LIMITATIONS.md
│   └── FULL_SOFTWARE_DOCUMENTATION.md
│
├── FRONTEND_SPEC_FOR_GOOGLE_STITCH.md
├── GOOGLE_STITCH_PROMPT.md
└── CLAUDE.md
```

---

## 3. Data Flow

### Image Reception Flow

```
CubeSat connects (TCP 5000)
    │
    ▼
listener._handle_connection()
    │
    ├─► _read_header() → JSON: {type, filename, file_size, md5, metadata}
    │
    ├─► _handle_image() or _handle_telemetry()
    │       │
    │       ├─► Receive bytes (throttled by CubeSat)
    │       ├─► packet_handler.validate_transfer() — MD5 + size check
    │       ├─► Save to data/received_images/
    │       ├─► quality_check.run_ground_quality_check()
    │       └─► pipeline.process(path, metadata, quality)  ← callback
    │
    └─► Send ACK or NACK
```

### Pipeline Flow (per image)

```
pipeline._process_locked(image_path, metadata, ground_quality)
    │
    ├─► 1. MosaicStitcher.register_image()
    │       • SuperPoint extracts keypoints
    │       • LightGlue matches against existing entries
    │       • MAGSAC++ homography, multi-band blend
    │       • Returns mosaic_bbox (x, y, w, h)
    │
    ├─► 2. MosaicGrid.update_from_mosaic(canvas_w, canvas_h)
    │       • Resize grid to match canvas
    │       • grid_cell = mosaic_px_to_grid(bbox center)
    │
    ├─► 3. ShadowDetector.run()
    │       • Otsu threshold, contour detection
    │       • Returns shadow_mask, shadow_pct, regions
    │
    ├─► 4. HazardClassifier.classify()
    │       • LBP variance, edge density, contour coverage
    │       • Returns hazard_class (SAFE/MODERATE/SHADOW/HAZARD/IMPASSABLE)
    │       • MosaicGrid.apply_hazard()
    │
    ├─► 4b. YOLODetector.detect() + fuse_classifications()
    │       • ML craters/boulders detection
    │       • Fuse with classical classification
    │
    ├─► 5. ChangeDetector (if same cell imaged in prior pass)
    │       • Template matching alignment
    │       • SSIM diff, contour extraction
    │       • Saves change_map, changes.json
    │
    ├─► 6. RoutePlanner (if start/end set)
    │       • A* on cost grid
    │       • plan_multiple_routes → fastest, safest, balanced
    │
    └─► 7. save_cost_grid_json(), mission_state.save()
```

### Dashboard Polling

| Endpoint        | Interval | Purpose                    |
|----------------|----------|----------------------------|
| /api/status    | 2s       | Mission + telemetry        |
| /api/routes    | 3s       | Route cards                |
| /api/cost_grid | 5s       | Heatmap, coverage          |
| /api/changes   | 3s       | Change detection           |
| /api/quality_log | 5s     | Per-image quality          |
| /api/log       | 3s       | Application log            |
| Images         | 3s       | Latest, hazard, mosaic     |

---

## 4. Module Reference

### 4.1 Entry Point — `server.py`

**Functions:**
- `main()` — Setup logging, dirs, MissionState, Pipeline, Commander; start TCP listener thread; run Flask
- `_patch_pipeline_quality_hook()` — Append quality entries to dashboard after each image
- `_reprocess_existing()` — Re-run pipeline on images in `data/received_images/` (with `--reprocess`)

### 4.2 Receiver — `receiver/`

| File              | Key Classes/Functions                         | Purpose                                      |
|-------------------|-----------------------------------------------|----------------------------------------------|
| `listener.py`     | `start_listener()`, `_handle_connection()`, `_handle_image()` | TCP server, receive images/telemetry        |
| `packet_handler.py` | `validate_transfer()`                        | MD5 + size validation                        |
| `quality_check.py` | `run_ground_quality_check()`                 | Texture variance, contrast range, single-color |
| `telemetry_parser.py` | `get_latest_telemetry()`, `parse_and_save_telemetry()` | Parse, cache, persist telemetry          |
| `downlink_state.py` | `DownlinkState`, `get_state()`               | Track downlink progress for UI               |

### 4.3 Uplink — `uplink/`

| File           | Key Classes/Functions              | Purpose                              |
|----------------|-----------------------------------|--------------------------------------|
| `commander.py` | `Commander.send_command()`, `retransmit()`, `set_grid_cell()`, `start_pass()`, `end_pass()` | Send JSON commands to CubeSat |
| `pi_manager.py` | `discover()`, `start_flight_software()`, `stop_flight_software()`, `get_pi_log()` | mDNS discovery, SSH to Pi     |

### 4.4 Processing — `processing/`

| File                 | Key Classes/Functions                    | Purpose                                           |
|----------------------|-----------------------------------------|---------------------------------------------------|
| `pipeline.py`        | `Pipeline.process()`, `_process_locked()` | Orchestrate full CV pipeline                      |
| `mosaic_stitcher.py` | `MosaicStitcher.register_image()`       | SuperPoint + LightGlue, homography, blend         |
| `mosaic_grid.py`     | `MosaicGrid.update_from_mosaic()`, `mosaic_px_to_grid()`, `apply_hazard()` | Dynamic grid, coordinate conversion   |
| `shadow_detector.py` | `ShadowDetector.run()`                  | Otsu threshold, shadow regions                     |
| `hazard_classifier.py` | `HazardClassifier.classify()`         | LBP, edges → SAFE/MODERATE/SHADOW/HAZARD/IMPASSABLE |
| `change_detector.py` | `ChangeDetector.detect()`               | Template alignment, SSIM diff, change events       |
| `route_planner.py`   | `RoutePlanner.plan_multiple_routes()`, `plan_with_constraints()` | A* routing                        |
| `pixel_segmenter.py` | `PixelSegmenter`                        | Semantic segmentation for fine routing            |
| `yolo_detector.py`   | `YOLODetector.detect()`, `fuse_classifications()` | YOLO + classical fusion                    |
| `slope_estimator.py` | `SlopeEstimator.estimate()`             | Shadow geometry → slope                           |
| `landing_recommender.py` | `LandingRecommender.recommend()`     | Score landing candidates                           |
| `mission_state.py`   | `MissionState.record_*()`, `get_snapshot()` | Accumulate mission data, persist to JSON     |
| `traversability_cnn.py` | `infer_grid()`                       | CNN for traversability (optional)                  |
| `ppo_planner.py`     | `plan_ppo_route()`                    | PPO-based route planning (optional)                |

### 4.5 Dashboard — `dashboard/`

| File         | Key Content                                      |
|--------------|---------------------------------------------------|
| `app.py`     | 50+ Flask routes: status, coverage, routes, images, commands, export, etc. |
| `index.html` | Single-page UI: heatmap, route cards, change detection, coverage, telemetry, LLM |

---

## 5. Configuration

**File:** `ground_station/config.py`

### Network
| Parameter     | Default        | Purpose                    |
|---------------|----------------|----------------------------|
| LISTEN_PORT   | 5000           | TCP port for CubeSat data  |
| COMMAND_PORT  | 5001           | TCP port for commands      |
| CUBESAT_IP    | 192.168.1.229  | CubeSat IP (must be set)   |

### Storage
| Parameter       | Value                    |
|-----------------|--------------------------|
| RECEIVED_DIR    | data/received_images     |
| PROCESSED_DIR   | data/processed            |
| TELEMETRY_DIR   | data/telemetry            |
| MISSION_STATE_FILE | data/mission_state.json |

### Hazard Costs
| Class      | Cost |
|------------|------|
| SAFE       | 1    |
| MODERATE   | 5    |
| SHADOW     | 15   |
| HAZARD     | 20   |
| IMPASSABLE | 999  |

### Mosaic
| Parameter              | Value  | Purpose                    |
|------------------------|--------|----------------------------|
| MOSAIC_GRID_CELL_PX    | 80     | Pixels per grid cell       |
| MOSAIC_INITIAL_CANVAS_PX | 640  | Initial canvas size        |
| MOSAIC_MAX_KEYPOINTS   | 1024   | SuperPoint limit           |
| MOSAIC_BUNDLE_ADJUST_INTERVAL | 3 | Bundle adjustment every N images |

### Change Detection
| Parameter          | Value |
|--------------------|-------|
| CHANGE_THRESHOLD   | 30    |
| CHANGE_MIN_AREA_PX | 50    |

### Dashboard
| Parameter           | Value |
|---------------------|-------|
| DASHBOARD_PORT      | 3000  |

---

## 6. Protocol

**File:** `ground_station/protocol.py` (identical in CubeSat codebase)

### Ports
- **DATA_PORT (5000):** CubeSat → GCS (images, telemetry)
- **COMMAND_PORT (5001):** GCS → CubeSat (commands)

### Transfer Format
1. CubeSat sends JSON header + `\n`: `{type, filename, file_size, md5, metadata}`
2. CubeSat sends raw bytes (image or telemetry JSON)
3. GCS verifies MD5, sends ACK (`\x06`) or NACK (`\x15`)

### Commands (GCS → CubeSat)
| Command          | Body                          |
|------------------|-------------------------------|
| start_pass       | (none)                        |
| end_pass         | (none)                        |
| cell / set_cell  | `{row, col}`                  |
| retransmit       | `{image_id}`                  |
| priority_cell    | `{row, col}`                  |
| adjust_exposure  | `{exposure_us}`               |
| enter_safe_mode  | (none)                        |
| resume_normal    | (none)                        |
| status_request   | (none)                        |
| retry_downlink   | (none)                        |
| reset_mission    | (none)                        |

### File Naming
- Image: `pass{N}_img{MM}_{YYYYMMDD_HHMMSS}.jpg`
- Metadata: `pass{N}_img{MM}_{YYYYMMDD_HHMMSS}_meta.json`

---

## 7. API Reference

**Base:** `http://localhost:3000`

### Key GET Endpoints
| Path                    | Returns                          |
|-------------------------|----------------------------------|
| /api/status             | mission + telemetry + server_time |
| /api/coverage           | 8×8 grid (row, col, hazard_class, cost, has_change) |
| /api/routes             | fastest, safest, balanced, selected |
| /api/cost_grid          | grid, classifications, coverage, pass_data, change_cells |
| /api/changes            | events, summary                  |
| /api/quality_log        | entries (filename, cubesat_score, ground_passed) |
| /api/log                | Last 100 log lines               |
| /api/latest_image       | JPEG (204 if none)               |
| /api/hazard_map         | PNG (204 if none)                |
| /api/mosaic             | PNG (204 if none)                |
| /api/export_mission     | PDF download                     |

### Key POST Endpoints
| Path                | Body                                      |
|---------------------|-------------------------------------------|
| /api/plan_routes    | `{start: [r,c], end: [r,c]}`              |
| /api/plan_constrained | `{max_shadow_pct, min_hazard_clearance, start, end}` |
| /api/select_route   | `{route_name: "fastest"|"safest"|"balanced"}` |
| /api/command        | `{cmd, ...}`                              |
| /api/start_pass     | (none)                                    |
| /api/end_pass       | (none)                                    |
| /api/set_cell       | `{row, col}`                              |
| /api/reset_mission  | (none)                                    |
| /api/clear_last_pass| (none)                                    |
| /api/llm_query      | `{question}`                              |

See `FRONTEND_SPEC_FOR_GOOGLE_STITCH.md` for full API schemas.

---

## 8. Key Algorithms

### Mosaic Stitching (mosaic_stitcher.py)
- **SuperPoint:** Learned keypoint detector (handles low-texture sand)
- **LightGlue:** Learned feature matcher (attention-based)
- **MAGSAC++:** Homography estimation
- **Multi-band blending:** Laplacian pyramid for seamless joins
- **Bundle adjustment:** Every N images, globally refine poses

### Hazard Classification (hazard_classifier.py)
- **LBP variance:** Texture roughness
- **Canny edge density:** Terrain roughness
- **Contour coverage:** Hazardous region coverage
- Thresholds → SAFE, MODERATE, SHADOW, HAZARD, IMPASSABLE

### Change Detection (change_detector.py)
- **Template matching:** Corner patch anchor (no ORB — fails on sand)
- **SSIM:** Structural similarity for diff (ignores brightness shifts)
- **Contour extraction:** Changed regions
- **Persistence check:** If change in both pass1→2 and pass1→3 → real

### Route Planning (route_planner.py)
- **A*:** 8-connected, diagonal cost = √2 × cell cost
- **Three variants:** Fastest (min distance), Safest (min hazard), Balanced
- **Constrained:** Max shadow %, min hazard clearance

---

## 9. External Dependencies

```bash
pip install opencv-python numpy flask pillow torch lightglue
# Optional: ultralytics (YOLO), paramiko (SSH for pi_manager), reportlab (PDF export)
```

### Models
- **SuperPoint + LightGlue:** From `lightglue` package
- **YOLO:** Lunar terrain model (crater, boulder) — download via `models/download_model.py`
- **Traversability CNN:** `models/traversability_model.pt` (optional)
- **PPO:** `PPO Training/` (optional)

### Optional
- **Ollama:** For LLM query panel (`ollama pull llama3.2`)

---

## 10. Tools & Utilities

| Script                    | Purpose                                      |
|---------------------------|----------------------------------------------|
| `mock_cubesat.py`         | Simulate CubeSat locally (commands + images) |
| `generate_sample_outputs.py` | Generate fake cost_grid, routes, changes for UI |
| `tools/mock_pipeline_run.py` | Simulate full pipeline with existing images |
| `tools/simulate_downlink.py` | Send images to GCS at throttled rate        |
| `tools/capture_training_images.py` | Capture images from Pi for YOLO training |
| `tools/train_terrain_yolo.py` | Train YOLO on terrain data               |
| `test_pipeline.py`        | Unit tests for pipeline stages               |
| `test_receiver.py`        | Protocol tests for listener                  |
| `evaluate_accuracy.py`    | Evaluate hazard/change/route accuracy         |
| `calibrate_detection.py`  | Interactive shadow/hazard calibration        |

---

## 11. Known Limitations

See `docs/ASSUMPTIONS_AND_LIMITATIONS.md` for full list. Summary:

- **No absolute reference frame:** Cell (0,0) is wherever first image pointed
- **Featureless terrain:** SIFT/CNN struggle on uniform sand — cell misidentification
- **Grid discretization:** One image per cell; real terrain may span cells
- **Cumulative drift:** Spatial stitching errors accumulate
- **One image per cell in mosaic:** Same cell from same pass → only best-quality shown

---

## 12. How to Run

### Start GCS
```bash
cd ground_station
python server.py
```
- Listener: port 5000
- Dashboard: http://localhost:3000

### Reprocess existing images
```bash
python server.py --reprocess
```

### Test with mock CubeSat
```bash
# Terminal 1: Start GCS
cd ground_station && python server.py

# Terminal 2: Start mock CubeSat
python mock_cubesat.py --gcs-ip 127.0.0.1 --images-per-pass 5
```

### Set CUBESAT_IP
Edit `ground_station/config.py` and set `CUBESAT_IP` to your CubeSat's IP, or use the dashboard's FIND button (discovers via mDNS `cubesat.local`).

---

## Related Documents

- **DIAGRAMS.md** — Mermaid diagrams: architecture, data flow, pipeline, protocol
- **docs/ARCHITECTURE.md** — Detailed architecture, config, module specs
- **docs/ASSUMPTIONS_AND_LIMITATIONS.md** — Error sources, limitations
- **docs/FULL_SOFTWARE_DOCUMENTATION.md** — Flight software + GCS (1500+ lines)
- **FRONTEND_SPEC_FOR_GOOGLE_STITCH.md** — Complete frontend API/UI spec
- **GOOGLE_STITCH_PROMPT.md** — Prompt for rebuilding UI with Google Stitch
- **CLAUDE.md** — Quick reference for AI assistants
