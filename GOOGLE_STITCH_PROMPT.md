# Google Stitch Prompt: MuraltZ Ground Control Station

**Copy this entire document into Google Stitch.** It contains the complete specification plus Palantir-style design guidance.

---

## Design Direction (Palantir-style)

Build a **mission operations dashboard** that looks like **Palantir** — dark, clean, professional, data-dense, highly organized. Think defense/intelligence operations center: sophisticated, minimal, clear information hierarchy, no visual clutter.

- **Dark theme**: Deep charcoal/navy (#0d1117, #161b22), not pure black
- **Restrained accent**: Cyan/teal (#00d4ff) used sparingly for highlights and active states
- **Typography**: Geometric sans (Inter, IBM Plex Sans) for UI; monospace (JetBrains Mono, Fira Code) for data
- **Minimal chrome**: Thin 1px borders, subtle dividers, no heavy shadows
- **Color semantics**: Green = good, amber = caution, red = critical
- **Grid discipline**: 8px, 12px, 16px spacing; strict alignment

---

## SECTION 1: API ENDPOINTS

Base URL: same origin (e.g. `http://localhost:3000`). All paths relative: `/api/status`, etc.

### GET — JSON Data

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/status` | Merged telemetry + mission_state |

**`/api/status` response:**
```json
{
  "mission": {
    "total_passes": 0,
    "total_images_received": 0,
    "total_images_corrupted": 0,
    "coverage": { "cells_filled": 0, "cells_total": 64, "pct": 0.0 },
    "quality": { "avg_cubesat_score": 0.0, "ground_flagged": 0, "ground_flag_reasons": [] },
    "hazards": { "safe": 0, "moderate": 0, "shadow": 0, "hazard": 0, "impassable": 0 },
    "changes": { "total_events": 0, "total_changed_area_cm2": 0, "largest_change_cm2": 0, "cells_with_changes": [], "types": { "darkened": 0, "brightened": 0 } },
    "route": { "start": [0,0], "end": [7,7], "status": "no viable route", "path_length": 0, "total_cost": 0, "shadow_exposure_pct": 0 },
    "routes": { "selected": "safest", "fastest": {...}, "safest": {...}, "balanced": {...} },
    "downlink": { "total_bytes": 0, "total_time_sec": 0, "effective_rate_bps": 0, "failed_transfers": 0, "retransmit_requests": 0 },
    "uplink": { "commands_sent": 0, "commands_acked": 0 },
    "last_updated": "2025-03-15T12:00:00.000Z"
  },
  "telemetry": {
    "state": "WAITING",
    "pass_number": 0,
    "roll_deg": 0.0,
    "pitch_deg": 0.0,
    "cpu_temp_c": 45.0,
    "storage_used_pct": 10.0,
    "queue_size": 0,
    "nadir_locked": true,
    "uptime_sec": 3600
  },
  "server_time": "2025-03-15T12:00:00.000000+00:00"
}
```

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/coverage` | 8×8 coverage grid |

**`/api/coverage` response:**
```json
{
  "grid": [
    [ { "row": 0, "col": 0, "hazard_class": "SAFE", "cost": 1, "has_change": false }, ... ],
    ...
  ],
  "rows": 8,
  "cols": 8
}
```

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/quality_log` | Quality log entries |

**`/api/quality_log` response:**
```json
{
  "entries": [
    {
      "filename": "pass1_001.jpg",
      "cubesat_score": 0.85,
      "ground_passed": true,
      "notes": [],
      "status": "ok"
    }
  ]
}
```

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/log` | Last 100 log lines |

**`/api/log` response:**
```json
{ "lines": [ "2025-03-15 12:00:00  INFO  ...", ... ] }
```

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/routes` | Route data (fastest/safest/balanced) |

**`/api/routes` response:**
```json
{
  "selected": "safest",
  "start": [0, 0],
  "end": [7, 7],
  "constrained": null,
  "fastest": {
    "path": [[0,0],[0,1],...],
    "path_length_cells": 14,
    "distance_cm": 140,
    "total_cost": 45.0,
    "max_shadow_exposure_pct": 12.5,
    "hazards_near_path": 2,
    "risk_level": "MODERATE",
    "status": "found",
    "name": "Fastest",
    "color": "#00ff88"
  },
  "safest": { ... },
  "balanced": { ... }
}
```

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/cost_grid` | Cost grid JSON |

**`/api/cost_grid` response:**
```json
{
  "grid": [[1,1,5,...],...],
  "classifications": [["SAFE","SAFE",...],...],
  "coverage": [[true,false,...],...],
  "pass_data": [[1,0,2,...],...],
  "change_cells": [[2,3],[4,5]],
  "confidences": [[0.9,...],...]
}
```

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/changes` | Change detection events |

**`/api/changes` response:**
```json
{
  "events": [
    {
      "id": 1,
      "cell": [2, 3],
      "type": "darkened",
      "area_px": 150,
      "bbox": [10, 20, 30, 40],
      "confidence": 0.92,
      "pass_before": 1,
      "pass_after": 2,
      "before_image": "pass1_003.jpg",
      "after_image": "pass2_003.jpg"
    }
  ],
  "summary": { "total_events": 1, "total_area": 15.0 }
}
```

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/shadow_data` | Shadow analysis |

**`/api/shadow_data` response:**
```json
{
  "shadow_pct": 12.5,
  "regions": [
    { "id": 1, "type": "shadow", "area_px": 500, "mean_boundary_gradient": 0.5 }
  ]
}
```

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/cost_heatmap` | 8×8 cost cells |

**`/api/cost_heatmap` response:**
```json
{
  "cells": [
    { "row": 0, "col": 0, "cost": 1, "hazard_class": "SAFE" },
    ...
  ],
  "rows": 8,
  "cols": 8
}
```

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/yolo_detections` | YOLO detection results |

**`/api/yolo_detections` response:**
```json
{
  "detections_per_cell": {},
  "fused_classifications": [
    {
      "cell": [0, 0],
      "classical_classification": "SAFE",
      "fused_classification": "SAFE",
      "fused_confidence": 0.95,
      "agreement": true,
      "yolo_detections": [{ "class": "crater", "confidence": 0.8 }]
    }
  ],
  "summary": {
    "total_detections": 10,
    "craters_detected": 5,
    "boulders_detected": 3,
    "cv_agreement_rate": 0.9,
    "cells_analyzed": 8
  }
}
```

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/pi_log` | Last 30 lines of Pi flight log |

**`/api/pi_log` response:** `{ "lines": ["...", ...] }`

### GET — Image Files (binary)

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/latest_image` | Latest JPEG — 204 if none |
| GET | `/api/hazard_map` | Latest hazard map PNG — 204 if none |
| GET | `/api/change_map` | Latest change map PNG — 204 if none |
| GET | `/api/route_map` | Route latest PNG — 204 if none |
| GET | `/api/mosaic` | Mosaic latest PNG — 204 if none |
| GET | `/api/route_comparison_image` | Route comparison PNG — 204 if none |
| GET | `/api/yolo_annotated` | YOLO annotated PNG — 204 if none |
| GET | `/api/image/<path:filename>` | Raw image — 404 if not found |

### POST Endpoints

| Method | Path | Request Body | Returns |
|--------|------|--------------|---------|
| POST | `/api/plan_routes` | `{"start": [0,0], "end": [7,7]}` | Route JSON or `{"error": "..."}` |
| POST | `/api/plan_constrained` | `{"max_shadow_pct": 50, "min_hazard_clearance": 1, "start": [0,0], "end": [7,7]}` | Constrained route JSON or `{"error": "..."}` |
| POST | `/api/select_route` | `{"route_name": "fastest"|"safest"|"balanced"}` | `{"selected": "safest"}` |
| POST | `/api/command` | `{"cmd": "...", ...}` | `{"success": true|false, "error": "..."}` |
| POST | `/api/start_pass` | (none) | `{"success": true|false, "error": "..."}` |
| POST | `/api/end_pass` | (none) | `{"success": true|false, "error": "..."}` |
| POST | `/api/set_cell` | `{"row": 0, "col": 0}` | `{"success": true|false, "error": "..."}` |
| POST | `/api/set_cubesat_ip` | `{"ip": "192.168.1.229"}` | `{"success": true, "ip": "...", "reachable": true|false}` |
| POST | `/api/discover_cubesat` | (none) | `{"found": true, "ip": "...", "ssh_ok": true, "flight_running": true, "error": ""}` |
| POST | `/api/pi_start` | (none) | `{"success": true, "message": "..."}` |
| POST | `/api/pi_stop` | (none) | `{"success": true, "message": "..."}` |
| POST | `/api/llm_query` | `{"question": "What is the coverage?"}` | `{"response": "..."}` |
| POST | `/api/reset_mission` | (none) | `{"status": "ok", "message": "..."}` or error |
| POST | `/api/clear_last_pass` | (none) | `{"status": "ok", "message": "...", "new_total_passes": N}` or error |
| GET | `/api/export_mission` | (none) | PDF file download |

**Command API details:**
- `retransmit`: `{"cmd": "retransmit", "image_id": "pass1_001"}`
- `priority_cell`: `{"cmd": "priority_cell", "row": 0, "col": 0}`
- `set_cell`: `{"cmd": "set_cell", "row": 0, "col": 0}`
- `adjust_exposure`: `{"cmd": "adjust_exposure", "exposure_us": 20000}`
- `enter_safe_mode`: `{"cmd": "enter_safe_mode"}`
- `resume_normal`: `{"cmd": "resume_normal"}`
- `status_request`: `{"cmd": "status_request"}`
- `retry_downlink`: `{"cmd": "retry_downlink"}`

---

## SECTION 2: POLLING INTERVALS

| Endpoint | Interval (ms) |
|----------|---------------|
| /api/status | 2000 |
| /api/routes | 3000 |
| /api/cost_grid | 5000 |
| /api/changes | 3000 |
| /api/quality_log | 5000 |
| /api/log | 3000 |
| /api/shadow_data | 5000 |
| /api/yolo_detections | 5000 |
| /api/latest_image, /api/hazard_map, /api/route_comparison_image, /api/mosaic | 3000 (cache-bust with `?t=Date.now()`) |

---

## SECTION 3: DASHBOARD PANELS

| Panel | Data | API | Refresh | Interactive |
|-------|------|-----|---------|-------------|
| **Top Bar** | State, Pass, connection | `/api/status` | 2s | START PASS, END PASS, SET CELL, FIND, START SW, STOP SW, NEW SESSION, EXPORT |
| **Terrain Cost Map** | 8×8 heatmap, routes | `/api/cost_grid`, `/api/routes` | 5s | Canvas click (L/T), PLAN ROUTES, endpoint inputs |
| **Route Comparison** | Fastest/Safest/Balanced cards | `/api/routes` | 3s | SELECT, SHOW ONLY per route |
| **Constraint Planner** | Shadow %, clearance sliders | — | — | Sliders, RECALCULATE |
| **Selected Route Stats** | Path stats | `/api/routes` | 3s | — |
| **Change Detection** | Before/after slider, events | `/api/changes` | 3s | Cell tabs, slider, event cards |
| **Coverage Grid** | 8×8 surveyed cells | `/api/cost_grid` | 3s | REPLAY |
| **Latest Image** | Most recent image | `/api/latest_image` | 3s | — |
| **Shadow Analysis** | Shadow %, regions | `/api/shadow_data` | 5s | — |
| **ML Object Detection** | YOLO detections | `/api/yolo_detections`, `/api/yolo_annotated` | 5s | — |
| **Downlink** | Bytes, rate, progress | `/api/status` (mission.downlink) | 2s | — |
| **Hazard Density by Quadrant** | NW/NE/SW/SE | `/api/cost_grid` | 5s | — |
| **Mission Status** | State, pass, images, etc. | `/api/status` | 2s | — |
| **Telemetry** (collapsible) | IMU, system, commands | `/api/status` | 2s | SAFE MODE, RESUME, STATUS REQ, RETRY DL, Retransmit, Priority Cell, Exposure |
| **Quality Log** (collapsible) | Per-image quality | `/api/quality_log` | 5s | — |
| **Event Log** (collapsible) | App log lines | `/api/log` | 3s | — |
| **LLM Query** | Input + response | `/api/llm_query` | on submit | Input, ASK button |

---

## SECTION 4: INTERACTIVE ELEMENTS

| Element | Action | API | Success | Failure | Enable When |
|---------|--------|-----|---------|---------|-------------|
| START PASS | WAITING→IMAGING | POST `/api/start_pass` | Green flash | Red flash | WAITING or SAFE_MODE |
| END PASS | IMAGING→PROCESSING | POST `/api/end_pass` | Green flash | Red flash | IMAGING |
| SET CELL | Set next cell | POST `/api/set_cell` `{row,col}` | "CELL R,C SET" | "FAILED" | IMAGING |
| NEW SESSION | Reset all data | POST `/api/reset_mission` | Reload | Error | Always |
| CLEAR LAST PASS | Remove last pass | POST `/api/clear_last_pass` | Reload | Error | Always (add button) |
| EXPORT | Download PDF | GET `/api/export_mission` | PDF download | — | Always |
| Route SELECT | Select route | POST `/api/select_route` `{route_name}` | Re-fetch routes | — | — |
| Route SHOW ONLY | Toggle single route on heatmap | — | Redraw | — | — |
| RECALCULATE | Plan constrained | POST `/api/plan_constrained` | Show constrained card | "NO FEASIBLE PATH" | — |
| Cost heatmap click | 1st=Landing, 2nd=Target+plan, 3rd=reset | — | Update inputs | — | — |
| PLAN ROUTES | Plan routes | POST `/api/plan_routes` `{start,end}` | "ROUTES UPDATED" | Error | — |
| Change slider | Drag to compare before/after | — | clip-path updates | — | — |
| Change event card click | Highlight bbox on image | — | Bbox stroke | — | — |
| Coverage REPLAY | Animate 3.5s | — | Animation | — | — |
| Command buttons | Various | POST `/api/command` | Green flash | Red flash | — |
| FIND | Discover CubeSat | POST `/api/discover_cubesat` | Show IP, START/STOP SW | "NOT FOUND" | — |
| START SW / STOP SW | Start/stop flight software | POST `/api/pi_start` or `/api/pi_stop` | Toggle buttons | — | — |

---

## SECTION 5: LAYOUT & PAGE STRUCTURE

### Top Bar (fixed, 50px)
- Left: "MURALTZ GCS" + subtitle "ARTEMIS LUNAR NAVIGATOR · MIT BWSI"
- Center: conn-dot, conn-label (LIVE/OFFLINE), STATE (large), PASS
- Right: mc-btn-start, mc-btn-end | mc-row, mc-col, mc-btn-cell | mc-btn-discover, mc-btn-pi-start, mc-btn-pi-stop | mc-pi-ip, mc-connect-dot, mc-feedback | NEW SESSION, EXPORT

### Main (two columns)
**Left (60%, min 600px):**
- heatmap-section: route-change-warning, ep-start-r/c, ep-end-r/c, PLAN ROUTES, plan-routes-feedback, cost-heatmap-canvas (564×560), heatmap-tooltip, legend
- Route Comparison: route-cards, constrained-result
- Constraint Planner: slider-shadow, val-shadow, slider-clearance, val-clearance, RECALCULATE
- Selected Route Stats: path-stats-card

**Right (40%):**
- Change Detection: change-summary, change-cell-tabs, change-slider (cs-before-img, cs-after-wrap, cs-after-img, cs-divider, cs-handle, cs-label-before, cs-label-after, cs-bbox-overlay), change-slider-ph, change-events-list, change-low-conf-warning
- Coverage Grid: coverage-pct-badge, coverage-canvas (340×340), cov-timeline (cov-tl-p1, p2, p3, cov-tl-playhead), REPLAY
- Latest Image: img-latest, img-latest-ph
- Shadow Analysis: shadow-summary-text
- ML Object Detection: yolo-model-badge, yolo-summary, img-yolo, img-yolo-ph, yolo-detections-list
- Downlink: dl-bytes, dl-bar, dl-rate, dl-total-bytes, dl-failed, ul-cmds, dl-rate-badge
- Hazard Density: quad-table-body
- Mission Status: sat-state-badge, s-state, s-pass, s-images, s-corrupt, s-coverage, s-route, s-changes, s-updated

### Bottom (collapsible)
- Telemetry: t-state, t-roll, t-pitch, t-nadir, t-temp, t-storage, t-queue, t-uptime + command buttons + in-retransmit, in-pri-r/c, in-exposure
- Quality Log: q-table-body, q-count-badge
- Event Log: event-log, log-count-badge
- LLM: llm-input, ASK, llm-response

---

## SECTION 6: VISUAL SPEC

### Hazard heatmap cell colors
- SAFE: #00ff88 (green)
- MODERATE: #ffdd00 (yellow)
- SHADOW: #4488ff (blue)
- HAZARD: #ff4444 (red)
- IMPASSABLE: #1a1a2e + red crosshatch
- Unsurveyed: #1e2233 (gray with "?")

### Route colors
- Fastest: #00ff88
- Safest: #4488ff
- Balanced: #ffaa00

### Constants
- DL_BUDGET_BYTES: 72000
- HM_ROWS, HM_COLS: 8
- HM_SIZE: 540, HM_CELL: 67.5
- HM_MARGIN_LEFT: 24, HM_MARGIN_TOP: 20
- Route draw animation: 1000ms
- Coverage REPLAY: 3500ms (3.5s)
- COL_LABELS: "ABCDEFGH"

### Change detection
- Before/after images from `/api/image/{before_image}` and `/api/image/{after_image}` (e.g. `pass1_003.jpg`)
- Slider uses clip-path: `inset(0 0 0 ${pct}%)` on after-wrap
- Bbox overlay: SVG rects with stroke #ff4444, highlight #ff8844

---

## SECTION 7: BEHAVIORS TO IMPLEMENT

1. **Heatmap click cycle**: Click 1 = set Landing (ep-start-r, ep-start-c), feedback "Landing set — click Target"; Click 2 = set Target, auto-call plan_routes; Click 3 = reset cycle
2. **Button enable/disable**: START PASS enabled when telemetry.state is WAITING or SAFE_MODE; END PASS and SET CELL enabled when IMAGING
3. **Route change warning**: If mission.changes.cells_with_changes intersects any route path, show amber banner "Change in cell (r,c) — route passes through"
4. **Coverage REPLAY**: Animate pass_data over 3.5s — cells fill in by pass number, then pulse change_cells orange
5. **Image refresh**: Use `url + '?t=' + Date.now()` for cache-bust; only update img if response 200
6. **Auto-scroll**: Quality log and Event log scroll to bottom on update
7. **Collapsible**: Click header toggles collapse-body.open, arrow rotates 90deg

---

## Deliverable

A single-page application (React/Next.js or Stitch's framework) that:
1. Polls all APIs at the specified intervals
2. Renders all panels with the layout and IDs above
3. Implements all interactive behaviors
4. Uses Palantir-style dark theme (#0d1117, #161b22, #00d4ff accent)
5. Handles 204/404 for images, empty states, errors gracefully
