# MuraltZ Ground Station — Software Architecture v2
## Laptop | Artemis Lunar Navigator | MIT BWSI 2025-2026
## Fixes applied: #6 change detection no ORB, #8 divergent light documented, #9 ground quality different from CubeSat, #12 mosaic one-per-cell documented

---

## 1. What This System Does

Receives real data from the CubeSat, sends commands back, processes real images through CV, detects changes between passes (core science mission), and displays everything on a live dashboard. Every piece of data comes from real CubeSat hardware.

---

## 2. Connection to CubeSat

| Detail | Value |
|--------|-------|
| Connection | Real WiFi, same network |
| Downlink | TCP on port 5000 — images + telemetry from CubeSat |
| Uplink | TCP on port 5001 — commands to CubeSat |
| Transfer speed | CubeSat throttles to 1200 B/s. 28 KB image ≈ 23 real seconds. |
| Integrity | MD5 hash both sides. Mismatch = discard + NACK. |

Uses `protocol.py` (identical copy on both systems) for ports, header format, ACK/NACK bytes, command schemas, file naming.

---

## 3. Directory Structure

```
ground_station/
├── server.py                  # Entry — starts receiver, uplink, dashboard
├── config.py
├── protocol.py                # Shared contract with CubeSat
├── receiver/
│   ├── __init__.py
│   ├── listener.py            # TCP server — accepts data from CubeSat
│   ├── packet_handler.py      # MD5 validation, partial file handling
│   ├── quality_check.py       # Ground-side quality (DIFFERENT checks from CubeSat)
│   └── telemetry_parser.py
├── uplink/
│   ├── __init__.py
│   └── commander.py           # Sends commands to CubeSat
├── processing/
│   ├── __init__.py
│   ├── pipeline.py            # Orchestrates full CV pipeline
│   ├── shadow_detector.py     # Otsu threshold
│   ├── hazard_classifier.py   # Terrain → risk levels
│   ├── change_detector.py     # Compares same cell across passes (CORE SCIENCE)
│   ├── elevation_map.py       # Photoclinometry (with documented error)
│   ├── mosaic_builder.py      # Stitches multi-pass (one image per cell)
│   └── route_planner.py       # A* on cost grid
├── dashboard/
│   ├── __init__.py
│   ├── app.py                 # Flask
│   ├── templates/index.html
│   └── static/
├── data/
│   ├── received_images/
│   ├── processed/
│   │   ├── shadow_masks/
│   │   ├── hazard_maps/
│   │   ├── change_maps/       # Core science output
│   │   ├── elevation_maps/
│   │   ├── mosaics/
│   │   └── routes/
│   ├── telemetry/
│   └── mission_state.json
└── llm/                       # Optional
    ├── __init__.py
    ├── interface.py
    └── system_prompt.txt
```

---

## 4. config.py

```python
# === NETWORK ===
LISTEN_PORT = 5000
COMMAND_PORT = 5001
LISTEN_HOST = "0.0.0.0"
CUBESAT_IP = "0.0.0.0"            # Fill in from network

# === PHYSICAL SETUP (MEASURED before demo) ===
CAMERA_HEIGHT_CM = 0.0             # Ruler: lens to surface
GSD_CM_PER_PIXEL = 0.0             # = (1.4 * CAMERA_HEIGHT_CM) / 4740
FLASHLIGHT_ELEVATION_DEG = 0.0     # Protractor: angle above horizontal
FLASHLIGHT_AZIMUTH_DEG = 0.0       # Direction flashlight points from
FLASHLIGHT_DISTANCE_CM = 0.0       # Distance from flashlight to surface center

# === STORAGE ===
RECEIVED_DIR = "data/received_images"
PROCESSED_DIR = "data/processed"
TELEMETRY_DIR = "data/telemetry"
MISSION_STATE_FILE = "data/mission_state.json"

# === GROUND QUALITY CHECK [FIX #9] ===
# These are DIFFERENT from the CubeSat's checks.
# CubeSat checks: blur (Laplacian), exposure (mean brightness), motion blur (IMU).
# Ground checks: texture sufficiency, contrast, color validity.
# Purpose: catch images the CubeSat passed but that will break the CV pipeline.
GROUND_MIN_TEXTURE_VARIANCE = 20   # Image must have enough texture for hazard classifier
GROUND_MIN_CONTRAST_RANGE = 50     # Histogram must span at least this many levels (0-255)
GROUND_MAX_SINGLE_COLOR_PCT = 90   # If >90% of pixels are similar, something's wrong

# === HAZARD COSTS ===
COST_SAFE = 1
COST_MODERATE = 5
COST_SHADOW = 15
COST_HAZARD = 20
COST_IMPASSABLE = 999

# === CHANGE DETECTION ===
CHANGE_THRESHOLD = 30              # Pixel difference to count as "changed"
CHANGE_MIN_AREA_PX = 50            # Min contiguous changed pixels to report

# === ROUTE PLANNING ===
GRID_ROWS = 8
GRID_COLS = 8
ROUTE_START = (0, 0)
ROUTE_END = (7, 7)

# === DASHBOARD ===
DASHBOARD_PORT = 8080
DASHBOARD_REFRESH_SEC = 2

# === LLM (optional) ===
LLM_MODEL = "llama3.2"
```

---

## 5. Receiver

### 5.1 `receiver/listener.py`

TCP server on `LISTEN_PORT`. When CubeSat connects during DOWNLINK:

1. Receive JSON header (newline-terminated): filename, file_size, md5, metadata
2. Receive file bytes (arrives slowly — CubeSat throttles to 1200 B/s)
3. Compute MD5, compare with header
4. Match → save, send ACK (0x06), trigger pipeline
5. Mismatch → discard, send NACK (0x15), log corruption

**Partial file handling:** If socket drops mid-transfer, received bytes < declared file_size → discard. Log: "Partial transfer: 12,400 of 28,400 bytes — discarded." CubeSat will re-queue automatically.

### 5.2 `receiver/quality_check.py` — Ground-Side (DIFFERENT from CubeSat)

**[FIX #9]** The CubeSat already checked blur, exposure, and motion blur. Running the same checks again on the ground adds nothing — TCP guarantees delivery, and the image already passed. The ground-side check instead validates whether the image is usable by the CV pipeline:

| Check | Method | Why CubeSat can't do this |
|-------|--------|--------------------------|
| **Texture sufficiency** | Compute local variance across 8x8 patches. If average patch variance < `GROUND_MIN_TEXTURE_VARIANCE`, flag. | CubeSat doesn't know what the hazard classifier needs. An image of perfectly smooth sand passes blur check (it IS sharp) but the classifier can't extract terrain features from it. |
| **Contrast range** | Histogram of grayscale image. Check if pixel values span at least `GROUND_MIN_CONTRAST_RANGE` levels. | Low contrast = poor shadow detection. Image might be sharp and well-exposed but flat in contrast due to lighting angle. |
| **Color validity** | Check if >90% of pixels fall in a narrow brightness band. | Catches: camera pointed at the table edge, operator's hand in frame, lens partially covered. These pass blur/exposure checks but are useless for terrain analysis. |

Output per image: `ground_quality_score` + `ground_quality_notes`. Flagged images are still processed (they already transferred — no point discarding) but the flag appears on the dashboard and in mission_state.json as a source of error.

### 5.3 `receiver/telemetry_parser.py`

Parses real telemetry JSON. Stores in `telemetry/`, feeds dashboard.

---

## 6. Uplink / Commanding

### 6.1 `uplink/commander.py`

Sends JSON commands to CubeSat on `COMMAND_PORT`.

| Command | Effect on CubeSat |
|---------|-------------------|
| `retransmit` | Image → top of queue |
| `priority_cell` | Boost novelty for cell |
| `set_cell` | Override operator's grid cell |
| `adjust_exposure` | Set camera exposure |
| `enter_safe_mode` | Immediate safe mode |
| `resume_normal` | Exit safe mode |
| `status_request` | CubeSat sends telemetry immediately |
| `retry_downlink` | Lifts GCS-unreachable suspension |

Dashboard has a command panel. Operator selects command, Flask sends to commander, commander sends to CubeSat, CubeSat ACKs.

**Answers Judge 1's question:** "What happens to images you can't downlink?" → Ground station reviews what's queued (from telemetry) and sends `retransmit` for specific high-value images.

---

## 7. CV Processing Pipeline

Triggered when a new validated image arrives and passes ground quality check.

```
Real image from CubeSat
    │
    ├──► Shadow Detector → shadow_mask, shadow_pct, shadow_regions
    │
    ├──► Hazard Classifier → hazard_map.png, cost_grid, hazard_counts
    │
    ├──► Change Detector (if same cell was imaged in a previous pass)
    │        → change_map.png, change_events
    │
    ├──► Elevation Map → elevation_map.png (with documented error source)
    │
    ├──► Mosaic Builder (if 3+ cells covered) → mosaic.png
    │
    ├──► Route Planner → route_map.png, route_data
    │
    └──► Mission Summary → mission_state.json
```

### 7.1 `processing/shadow_detector.py`

Input: real image. Method: grayscale → Otsu threshold → binary mask → connected components.

Output: shadow_mask.png, shadow_percentage, shadow_regions (each with area, bounding box, centroid, estimated_height_cm).

Height estimation: `height_px = shadow_length × tan(FLASHLIGHT_ELEVATION_DEG)`, then `height_cm = height_px × GSD_CM_PER_PIXEL`. Both config values are measured from the real physical setup.

### 7.2 `processing/hazard_classifier.py`

Input: real image + shadow mask. Divides into NxN grid, classifies each cell.

| Class | Color | Cost | Detection |
|-------|-------|------|-----------|
| SAFE | Green | 1 | Low variance, well-lit |
| MODERATE | Yellow | 5 | Higher texture variance |
| SHADOW | Blue | 15 | >30% shadow from mask |
| HAZARD | Red | 20 | Large dark circular contour (crater) or small bright irregular (boulder) |
| IMPASSABLE | Dark Red | 999 | >50% of cell is hazard |

Output: hazard_map.png, cost_grid (8×8 numpy), hazard_counts dict.

### 7.3 `processing/change_detector.py` — Core Science Mission

**[FIX #6]** Does NOT use ORB feature matching for alignment. ORB fails on textureless surfaces like sand — almost no keypoints, bad affine transform, noise in the diff.

**Instead:** Since both images cover the same grid cell, and the camera is at roughly the same height above a fixed surface, the images are already approximately aligned. We use direct comparison with lightweight alignment:

**Method:**
1. Ground station identifies which cell the new image belongs to using image fingerprinting (SIFT matching + CNN embeddings against known cells)
2. Find the previous image of the same cell (from image index)
3. Load both as grayscale, resize to identical dimensions if needed
4. **Simple alignment via template matching:** Use a small corner patch as an anchor. `cv2.matchTemplate()` with a cropped region. Surface features (rocks, crater edges) provide sufficient texture for reliable matching.
5. Apply the detected offset (translation only — no rotation or scale, since the camera height and angle are consistent)
6. Compute absolute difference: `diff = cv2.absdiff(aligned_img1, aligned_img2)`
7. Threshold: pixels where diff > `CHANGE_THRESHOLD` (30) → marked as changed
8. Find contours of changed regions, filter by `CHANGE_MIN_AREA_PX` (50 pixels)
9. Classify each change event:
   - **Darkened** (new pixel is darker): possible new crater, new boulder shadow, or shadow shift
   - **Brightened** (new pixel is brighter): possible removed obstacle, lighting change
   - Measure area in pixels → convert to cm² using `GSD_CM_PER_PIXEL`

**Fallback:** If template matching confidence is low (<0.7 correlation), skip alignment and do the diff anyway. Flag the result as "alignment uncertain" in the output. Show both images side by side on the dashboard rather than the diff overlay. Log: "Change detection for cell (2,3): alignment confidence 0.52 — results may be unreliable."

**Demo scenario:** Between pass 1 and pass 3, a team member physically moves a rock, presses a new bowl into the sand, or changes the flashlight angle. These are real physical changes. The change detector compares real images and finds real differences.

**Output:**
- `change_map.png` — newer image with red outlines around changed regions + labels
- `change_events` list:
  ```json
  {
      "id": 1,
      "grid_cell": [2, 3],
      "pass_before": 1, "pass_after": 3,
      "area_px": 320, "area_cm2": 4.5,
      "centroid": [234, 156],
      "type": "darkened",
      "mean_difference": 67,
      "alignment_confidence": 0.89,
      "description": "New dark region — possible new boulder or crater"
  }
  ```
- `change_summary`: total events, total changed area, largest event

**When it runs:** Automatically when a new image arrives for a grid cell that was already imaged in a prior pass.

### 7.4 `processing/elevation_map.py`

Photoclinometry from real shadow lengths + measured flashlight angle.

**[FIX #8] Known error source — divergent illumination:** The sun at the Moon is 150 million km away — its rays are effectively parallel. Our flashlight at ~50 cm produces divergent rays that fan out by 15+ degrees across the surface. This means `shadow_length × tan(elevation)` gives correct heights only directly below the flashlight, with increasing systematic error toward the edges.

**How this is handled in the code:**
1. Compute heights using the simple formula (parallel light assumption)
2. **Add an error estimate** to each height based on distance from the flashlight center:
   ```
   distance_from_center = sqrt((x - center_x)² + (y - center_y)²)
   divergence_angle = arctan(distance_from_center / FLASHLIGHT_DISTANCE_CM)
   error_pct = divergence_angle / FLASHLIGHT_ELEVATION_DEG * 100
   ```
3. Include `error_estimate_pct` in the output metadata for each shadow region
4. On the elevation map image, add a note: "Height error increases toward edges (~±15%)"

**For the presentation:** This is a "source of error" slide point:
> "Our photoclinometry assumes parallel illumination. The flashlight at 50 cm produces divergent rays, introducing systematic height errors of ±15% at surface edges. In the flight equivalent, solar illumination at the Moon is effectively parallel, eliminating this error."

Judges love when you identify real physical limitations.

Output: `elevation_map.png` — false-color heightmap with colorbar in real cm, plus error annotation.

### 7.5 `processing/mosaic_builder.py`

Grid-based placement. One image per cell (the highest-quality one).

**[FIX #12] Known limitation — one image per cell:**

Two images of cell (2,3) from different moments in the same pass get stacked at the same position. The mosaic only shows the best-quality one. This means mosaic resolution is limited by grid resolution (8×8 cells).

**How this is handled:**
- For each cell, pick the image with the highest `combined_score`
- Place at the cell's position on the canvas
- Unfilled cells rendered as dark gray
- Label unfilled cells: "Not surveyed"

**For the presentation:** Another "scaling to flight" point:
> "In flight, precise orbital ephemeris data would provide sub-pixel positioning for seamless, high-resolution mosaic generation. Our grid-based approach demonstrates the concept with the positioning accuracy available from our demo setup."

### 7.6 `processing/route_planner.py`

A* on real cost_grid. 8-connected, diagonal cost = √2 × cell cost.

Output: path, total_cost, path_length, shadow_exposure_pct, route_map.png.

If no path → "no viable route found" (real result if terrain is blocked). This is actually a desirable demo outcome — it shows the system correctly identifies impassable terrain.

### 7.7 Mission Summary → `mission_state.json`

All real data:

```json
{
    "last_updated": "2026-03-15T14:45:00Z",
    "total_passes": 3,
    "total_images_received": 10,
    "total_images_corrupted": 1,
    "quality": {
        "avg_cubesat_score": 0.76,
        "ground_flagged": 1,
        "ground_flag_reasons": ["low_texture"]
    },
    "coverage": {"cells_filled": 40, "cells_total": 64, "pct": 62.5},
    "hazards": {"safe": 32, "moderate": 12, "shadow": 8, "hazard": 6, "impassable": 2},
    "changes": {
        "total_events": 3,
        "total_changed_area_cm2": 18.7,
        "largest_change_cm2": 8.2,
        "types": {"darkened": 2, "brightened": 1},
        "cells_with_changes": [[2,3], [4,5]],
        "alignment_warnings": 0
    },
    "route": {
        "start": [0,0], "end": [7,7],
        "path_length": 12, "total_cost": 47,
        "shadow_exposure_pct": 8.3, "status": "found"
    },
    "elevation": {
        "max_height_cm": 3.2,
        "shadow_regions_analyzed": 14,
        "gsd_cm_per_pixel": 0.012,
        "divergence_error_note": "±15% at surface edges due to non-parallel flashlight illumination"
    },
    "downlink": {
        "total_bytes": 280000,
        "total_time_sec": 230,
        "effective_rate_bps": 1217,
        "failed_transfers": 1,
        "retransmit_requests": 1
    },
    "uplink": {"commands_sent": 4, "commands_acked": 4}
}
```

---

## 8. Dashboard

Flask on port 8080. Auto-refresh every 2 sec.

| Panel | Content | Source |
|-------|---------|--------|
| Mission Status | CubeSat state, pass, uptime | Telemetry |
| Telemetry | IMU, camera, thermal, storage | Telemetry JSON |
| Downlink Progress | Real progress bar with B/s and ETA | Receiver byte counter |
| Command Panel | Retransmit, priority, eclipse buttons | → commander.py |
| Latest Image | Most recent raw photo | received_images/ |
| Hazard Map | Color-coded terrain + legend | hazard_classifier |
| **Change Map** | **Highlighted changes between passes — THE CORE OUTPUT** | **change_detector** |
| Route Map | Hazard map + A* path | route_planner |
| Coverage Map | 8×8 grid green/gray/orange(changed) | Coverage data |
| Mosaic | Stitched survey (one per cell, gaps visible) | mosaic_builder |
| Elevation Map | False-color heights in cm + error note | elevation_map |
| Quality Log | Every image: CubeSat score + ground check + status | Both quality gates |
| Event Log | Timestamped events | All modules |

Flask routes: `GET /api/status`, `/api/latest_image`, `/api/hazard_map`, `/api/change_map`, `/api/route_map`, `/api/coverage`, `/api/mosaic`, `/api/elevation`, `/api/quality_log`, `/api/log`, `POST /api/command`, `POST /api/llm_query`.

---

## 9. Fault Handling

| Scenario | Behavior |
|----------|----------|
| Connection drops mid-transfer | Partial bytes discarded (size < declared). Log. CubeSat re-queues. |
| Pipeline crash on one image | Catch exception, log, mark `processing_error`, **skip to next image**. Pipeline does NOT stop. |
| Disk full | Stop saving, alert on dashboard. CubeSat keeps queueing — nothing lost. |
| Change detection alignment fails | Confidence < 0.7 → flag "alignment uncertain", show images side-by-side instead of diff overlay. |
| Elevation math error (div by zero, etc.) | Skip elevation map for that image, log error, continue other pipeline stages. |
| Multiple simultaneous connections | Reject second connection (only one CubeSat). |

---

## 10. LLM Interface (Optional)

Local Llama 3.2 via ollama. No internet. Reads only `mission_state.json` (all real data). System prompt includes the full JSON. LLM describes real computed results, doesn't invent anything.

Dashboard has text input for queries. Flask calls ollama locally, returns response.

---

## 11. Flight Equivalent Mapping (10 pts in rubric)

| Demo | Flight | What changes | What stays |
|------|--------|-------------|-----------|
| Laptop | Mission Ops Center | Hardware, staffing | Same data flow: receive → process → display |
| WiFi TCP | UHF/S-band ground antenna | Physical RF, Doppler correction | Same packet structure, same validation |
| Flask dashboard | COSMOS / OpenMCT | Professional multi-monitor consoles | Same panels: telemetry, hazard, route, coverage |
| TCP uplink | Encrypted RF uplink | Command encryption, link security | Same command types and logic |
| OpenCV on laptop | GPU processing cluster | Massively more compute | Same algorithms |
| `mission_state.json` | Mission database (PostgreSQL) | Persistent DB, trend analysis | Same data schema |
| Local LLM | Fine-tuned mission planning AI | Larger model, mission-specific training | Same architecture: LLM reads data, doesn't decide |
| Change detection (2-3 passes) | Long-baseline monitoring (months) | Statistical significance testing | Same pixel-diff core approach |
| Image fingerprinting for cell identification | Orbital ephemeris for georeferencing | Precise GPS/star tracker positioning | Same concept: known position → alignment |

---

## 12. Design Decision Traceability

| Decision | Reason | Evidence |
|----------|--------|----------|
| Heavy CV on ground | Pi: 2GB RAM, no cooling. Standard satellite arch. | Real memory + thermal limits |
| Change detection on ground | Needs images from multiple passes. CubeSat only sees one pass. | Real multi-pass dependency |
| **[FIX #6]** Template matching not ORB | Sand is textureless — ORB finds no keypoints. Corner patches with surface features (rocks, edges) are used as anchors. | Real surface material properties |
| **[FIX #8]** Documented light divergence error | Flashlight ≠ sun. Heights have ±15% error at edges. | Real physics, measured distance |
| **[FIX #9]** Different ground quality checks | CubeSat checks capture quality. Ground checks pipeline usability. | Different failure modes |
| **[FIX #12]** One image per mosaic cell | Grid-based, not feature-stitched. Flight would use ephemeris. | Real positioning limitation |
| Bidirectional comms | GCS must retarget, request retransmits, respond to findings | Real operational need |
| Real throttled downlink | Judges see real transfer time → understand bandwidth constraint | Real 23-sec transfers |
| A* on real cost grid | Route avoids real obstacles detected in real images | Real pathfinding on real data |

---

## 13. Testing

| Test | Expected |
|------|----------|
| Receiver accepts + validates | File saved, MD5 OK, ACK sent |
| Partial transfer | Bytes discarded, NACK sent, log shows count |
| Ground quality: low texture | Image flagged "low_texture", still processed |
| Ground quality: low contrast | Image flagged, pipeline warned |
| Ground quality: single color | Image flagged "color_invalid" |
| Shadow detection on real image | Mask matches visible dark regions |
| Hazard classification | Rocks/bowls = red, sand = green |
| Change detection: rock moved | Change map highlights real moved rock |
| Change detection: no change | "No significant changes" |
| Change detection: low alignment | Flagged "uncertain", side-by-side shown |
| Elevation map | Taller objects → higher cm values. Error annotation present. |
| Mosaic from 3 passes | Stitched image, gaps visible for unsurveyed cells |
| A* route | Path avoids red cells |
| A* no path | Reports "no viable route" |
| Uplink retransmit | CubeSat re-queues specified image |
| Pipeline crash recovery | Logs error, skips image, continues |
| Dashboard live | All panels update every 2 sec |
| Full end-to-end | 3 passes with surface change → hazard map + change map + route |

---

## 14. Dependencies

```bash
pip install opencv-python numpy flask matplotlib pillow
# Optional: ollama from https://ollama.com, then: ollama pull llama3.2
```

---

## 15. Pre-Demo Measurements

**These MUST be done on the real physical setup before demo day:**

1. Camera height: measure lens-to-surface with ruler → `CAMERA_HEIGHT_CM`
2. Compute GSD: `(1.4 × CAMERA_HEIGHT_CM) / 4740` → `GSD_CM_PER_PIXEL`
3. Flashlight elevation: protractor on the light → `FLASHLIGHT_ELEVATION_DEG`
4. Flashlight azimuth: compass/estimation → `FLASHLIGHT_AZIMUTH_DEG`
5. Flashlight distance: ruler to surface center → `FLASHLIGHT_DISTANCE_CM`
6. Test shadow detection on a real image of the real surface → verify mask looks right
7. Test change detection: take photo, move a rock, take another → verify change map detects it
8. Calibrate CubeSat `BLUR_THRESHOLD` (see CubeSat doc section 10)
