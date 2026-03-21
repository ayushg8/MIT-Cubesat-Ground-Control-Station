# MuraltZ Ground Station Software

## What This Is
Ground station software for the MuraltZ CubeSat prototype. Runs on a laptop. Receives images and telemetry from the CubeSat over WiFi, sends commands back, processes images through a CV pipeline (shadow detection, hazard classification, change detection, route planning), and displays everything on a Flask dashboard.

## Connection to CubeSat
- **Downlink:** TCP port 5000 — CubeSat pushes images + telemetry here
- **Uplink:** TCP port 5001 — GCS sends commands to CubeSat here
- **Transfer speed:** CubeSat throttles to 1200 B/s. A 28 KB image takes ~23 real seconds to arrive.
- **Integrity:** MD5 hash on both sides. Mismatch = discard + NACK.
- **Protocol:** See `protocol.py` — identical copy must exist in both CubeSat and GCS codebases.

## Architecture Reference
See `docs/ARCHITECTURE.md` for the full ground station software architecture, config parameters, module specs, CV pipeline, dashboard panels, fault handling, and flight equivalent mapping.

## Key Rules
1. Everything is REAL. Every image comes from the real CubeSat camera. Every telemetry value comes from real sensors. No mock data.
2. The GCS does NOT pull from the CubeSat. The CubeSat pushes to the GCS. The GCS just listens and reacts.
3. Ground-side quality checks are DIFFERENT from CubeSat checks. CubeSat checks blur/exposure/motion. Ground checks texture sufficiency, contrast range, color validity — things that affect whether the CV pipeline can work with the image.
4. Change detection uses object-level comparison (YOLO detection lists + ORB alignment) instead of pixel-level SSIM. Requires 2+ passes over the same terrain. Falls back to pixel differencing with histogram matching only when no YOLO data is available.
5. Elevation mapping is NOT performed (photoclinometry subsystem removed by design). Shadow detection still runs for hazard classification.
6. Mosaic uses continuous stitching (SIFT + homography). Images are placed into a growing canvas. No fixed grid — grid is derived dynamically from mosaic dimensions.
7. Hazard classifier produces one classification per image, projected onto the dynamic mosaic grid cells the image covers.
8. `CUBESAT_IP` in config must be filled in with the CubeSat's real IP before running.
9. `MOSAIC_GRID_CELL_PX` in config controls the dynamic grid resolution (pixels per cell). `GRID_CELL_SIZE_CM` is for physical calibration.
10. Route planner computes 3 simultaneous routes (Fastest/Safest/Balanced). Use `/api/select_route` to choose, `/api/plan_constrained` for constraint-based planning. Landing/Target points are set by clicking on the mosaic viewer.
11. IMU data (yaw) from the CubeSat is used as a placement hint for mosaic stitching when available. Falls back to pure SIFT matching if absent.

## Dependencies
```bash
pip install opencv-python numpy flask pillow
```
Optional: **Ollama** for Log tab “Mission Q&A” only (`ollama pull llama3.2` or match `config.LLM_MODEL`). No AI advisor / specialist pipeline.

## File Structure Target
```
ground_station/
├── server.py
├── config.py
├── protocol.py
├── receiver/
├── uplink/
├── processing/
├── dashboard/
├── data/
└── llm/             (optional)
```
See docs/ARCHITECTURE.md for full tree with all modules.
