# MuraltZ CubeSat — Ground Control Station

Ground software for a CubeSat prototype. It receives throttled image and telemetry downlinks over TCP, stitches the images into a mosaic, detects surface changes between passes, plans hazard-aware routes across the imaged terrain, and drives the whole mission from a single-page operator dashboard. Built at the MIT Beaver Works Summer Institute.

The companion flight software lives in [MIT-BWSI-Cubesat](https://github.com/ayushg8/MIT-BWSI-Cubesat).

## Overview

The spacecraft decides what to send; the ground station decides what it means. As prioritized images arrive over a slow, lossy link, this software reassembles them into a coherent picture of the surface, finds what changed since the last pass, and computes safe routes across the terrain — then lets an operator command the next pass from a live map.

It runs end to end with no hardware: a mock CubeSat (`mock_cubesat.py`) reproduces the flight state machine and the throttled, MD5-checked link, so the full receive → analyze → plan → command loop is demonstrable on one laptop.

## How it works

`ground_station/server.py` starts three things in one process: a TCP receiver, the Flask dashboard, and an uplink commander.

```
 CubeSat ──(TCP 5000, throttled + MD5)──▶ Receiver ──▶ CV pipeline ──▶ data/processed
    ▲                                                                      │
    │                                                                      ▼
 Commander ◀──(TCP 5001, JSON commands)── Operator ◀── Flask dashboard (port 3000)
```

**Receiver** (`receiver/`) — TCP server on port 5000. Reads a JSON header then raw bytes, validates size and MD5, ACK/NACKs each image, writes the JPEG plus a metadata sidecar, and hands it to the pipeline on a worker thread. Filenames are validated against path traversal and transfers are size-bounded.

**CV pipeline** (`processing/pipeline.py`) — runs per image, each stage wrapped so one failure can't kill the rest:
- **Mosaic** (`mosaic_stitcher.py`) — SIFT feature matching + RANSAC homography + distance-weighted feather blending onto an incremental canvas.
- **Change detection** (`change_detector.py`) — the core science. Compares the same area across passes using **SSIM** on aligned grayscale (not raw pixel diff), filters by area and aspect ratio, classifies darkened/brightened regions, and runs a multi-pass persistence check to separate real surface changes from lighting artifacts.
- **Hazard mapping** — shadow detection and slope estimation produce a terrain cost grid.
- **Route planning** (`route_planner.py`) — 8-connected A\* with an octile heuristic over the cost grid, run three times (fastest / safest / balanced) with different cost weightings, blocking impassable cells.

**Uplink** (`uplink/`) — sends validated JSON commands to the spacecraft over TCP 5001; a Pi manager can discover a real Pi over mDNS and start/stop its flight software over SSH.

**Dashboard** (`dashboard/`) — a single-page operator console (Leaflet map, dark theme) served by Flask, exposing the mosaic, hazard map, change overlays, planned routes, live telemetry, downlink progress, and a PDF mission-report export.

### Optional learned models

The pipeline can fuse learned models when their weights are present (`requirements-ml.txt`): a YOLOv8 object detector, a MobileNetV2 traversability classifier, and a pre-trained PPO policy that plans routes alongside A\*. **These weights are not committed to the repo.** Without them the system runs end to end on classical CV alone (mosaic + SSIM change detection + A\* routing); the learned components are optional add-ons, and `PPO Training/` ships a pre-trained policy artifact for inference, not a training environment.

## Tech

- **Python 3.10+**
- **Flask** backend (server-rendered page + JSON API + SSE for downlink progress); vanilla JS + **Leaflet** front end
- **OpenCV** (SIFT mosaic, shadow/slope), **scikit-image** (SSIM), **NumPy** — classical CV is the backbone
- Optional: **Ultralytics YOLOv8**, **Stable-Baselines3** (PPO), **torchvision** (MobileNetV2)
- **paramiko** (SSH to a real Pi), **reportlab** (PDF report)

## Running it

No hardware needed — two terminals:

```bash
pip install -r requirements.txt           # add -r requirements-ml.txt for the learned models
cd ground_station
python server.py                          # terminal 1: dashboard at http://localhost:3000
python mock_cubesat.py --gcs-ip 127.0.0.1 # terminal 2: simulated satellite
```

Then open the dashboard, CONNECT to `127.0.0.1`, and drive a mission with START PASS / END PASS. `python server.py --reprocess` re-runs the pipeline over already-received images.

Tests:

```bash
python -m unittest test_packet_handler.py test_change_detector.py test_pipeline.py
```

## Honest limits

- The learned models (YOLO, CNN, PPO) need external weight files that aren't in the repo; the demonstrable path is the classical-CV pipeline.
- `PPO Training/` contains a pre-trained policy for inference; the training environment that produced it is not included.
- There are no committed benchmark numbers. Change detection and routing have well-defined, inspectable behavior (SSIM threshold, area/aspect filters, A\* cost weights), not measured accuracy claims.

## TODO — visual assets to add

The repo has no screenshots yet, and the dashboard is the thing worth seeing. Add and reference here:

- `docs/img/dashboard.png` — the running operator console with a mosaic and telemetry loaded.
- `docs/img/change-detection.png` — a before/after pair with detected changes highlighted.
- `docs/img/route-plan.gif` — a short clip of the three A\* routes (fastest/safest/balanced) drawn over the hazard map.
- `docs/img/mission-loop.gif` — driving a full pass from the dashboard against the mock CubeSat.
