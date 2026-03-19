# MuraltZ Ground Control Station — Complete Frontend Specification for Google Stitch

This document describes every aspect of the frontend so it can be rebuilt from scratch. The dashboard is a single-page application served by Flask at `http://localhost:3000` (DASHBOARD_PORT from config).

---

## SECTION 1: API ENDPOINTS

Every Flask route with HTTP method, URL path, response schema, request body (for POST), and backend function.

### Pages

| Method | Path | Returns | Backend |
|--------|------|---------|---------|
| GET | `/` | Rendered HTML (`index.html`) with `cubesat_ip` template variable | `index()` → `render_template("index.html", cubesat_ip=config.CUBESAT_IP or "")` |

---

### API — JSON Data (GET)

| Method | Path | Returns | Backend |
|--------|------|---------|---------|
| GET | `/api/status` | Merged telemetry + mission_state | `api_status()` |

**Response schema:**
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

---

| Method | Path | Returns | Backend |
|--------|------|---------|---------|
| GET | `/api/coverage` | 8×8 coverage grid for canvas | `api_coverage()` |

**Response schema:**
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

---

| Method | Path | Returns | Backend |
|--------|------|---------|---------|
| GET | `/api/quality_log` | Quality log entries | `api_quality_log()` |

**Response schema:**
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

---

| Method | Path | Returns | Backend |
|--------|------|---------|---------|
| GET | `/api/log` | Last 100 log lines | `api_log()` |

**Response schema:**
```json
{ "lines": [ "2025-03-15 12:00:00  INFO  ...", ... ] }
```

---

| Method | Path | Returns | Backend |
|--------|------|---------|---------|
| GET | `/api/routes` | Route data (fastest/safest/balanced) | `api_routes()` |

**Response schema:**
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

---

| Method | Path | Returns | Backend |
|--------|------|---------|---------|
| GET | `/api/cost_grid` | Cost grid JSON | `api_cost_grid()` |

**Response schema:** Serves `cost_grid.json` or:
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

---

| Method | Path | Returns | Backend |
|--------|------|---------|---------|
| GET | `/api/changes` | Change detection events | `api_changes()` |

**Response schema:** Serves `changes.json` or:
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

---

| Method | Path | Returns | Backend |
|--------|------|---------|---------|
| GET | `/api/shadow_data` | Shadow analysis | `api_shadow_data()` |

**Response schema:** Serves `shadow_data.json` or:
```json
{
  "shadow_pct": 12.5,
  "regions": [
    { "id": 1, "type": "shadow", "area_px": 500, "mean_boundary_gradient": 0.5 }
  ]
}
```

---

| Method | Path | Returns | Backend |
|--------|------|---------|---------|
| GET | `/api/cost_heatmap` | 8×8 cost cells for heatmap | `api_cost_heatmap()` |

**Response schema:**
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

---

| Method | Path | Returns | Backend |
|--------|------|---------|---------|
| GET | `/api/yolo_detections` | YOLO detection results | `api_yolo_detections()` |

**Response schema:** Serves `yolo_detections.json` or:
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

---

| Method | Path | Returns | Backend |
|--------|------|---------|---------|
| GET | `/api/pi_log` | Last 30 lines of Pi flight log | `api_pi_log()` |

**Response schema:** `pi_manager.get_pi_log(30)` — returns `{ "lines": ["...", ...] }` (array of log line strings).

**Discover/Pi responses:**
- `POST /api/discover_cubesat` → `{ "found": true, "ip": "192.168.1.x", "ssh_ok": true, "flight_running": true, "error": "" }`
- `POST /api/pi_start` → `{ "success": true, "message": "...", "ip": "..." }`
- `POST /api/pi_stop` → `{ "success": true, "message": "..." }`

---

### API — Image Files (GET, binary)

| Method | Path | Returns | Backend |
|--------|------|---------|---------|
| GET | `/api/latest_image` | Latest JPEG from `received_images/` | `api_latest_image()` — 204 if none |
| GET | `/api/hazard_map` | Latest `*_hazard.png` from `processed/hazard_maps/` | `api_hazard_map()` — 204 if none |
| GET | `/api/change_map` | Latest `*_change_*.png` from `processed/change_maps/` | `api_change_map()` — 204 if none |
| GET | `/api/route_map` | `processed/routes/route_latest.png` | `api_route_map()` — 204 if none |
| GET | `/api/mosaic` | `processed/mosaics/mosaic_latest.png` | `api_mosaic()` — 204 if none |
| GET | `/api/route_comparison_image` | `processed/routes/route_comparison.png` or fallback to route_latest | `api_route_comparison_image()` — 204 if none |
| GET | `/api/yolo_annotated` | Latest `*_yolo.png` from `processed/yolo_detections/` | `api_yolo_annotated()` — 204 if none |
| GET | `/api/image/<path:filename>` | Raw image from `received_images/` | `api_image(filename)` — 404 if not found |

---

### API — POST Endpoints

| Method | Path | Request Body | Returns | Backend |
|--------|------|--------------|---------|---------|
| POST | `/api/plan_routes` | `{"start": [0,0], "end": [7,7]}` | Route JSON or `{"error": "..."}` | `api_plan_routes()` → `_pipeline._route_planner.plan_multiple_routes()` |
| POST | `/api/plan_constrained` | `{"max_shadow_pct": 50, "min_hazard_clearance": 1, "start": [0,0], "end": [7,7]}` | Constrained route JSON or `{"error": "..."}` | `api_plan_constrained()` → `plan_with_constraints()` |
| POST | `/api/select_route` | `{"route_name": "fastest"|"safest"|"balanced"}` | `{"selected": "safest"}` | `api_select_route()` → updates mission_state |
| POST | `/api/command` | `{"cmd": "retransmit"|"priority_cell"|"set_cell"|"adjust_exposure"|"enter_safe_mode"|"resume_normal"|"status_request"|"retry_downlink", ...}` | `{"success": true|false, "error": "..."}` | `api_command()` → `Commander` |
| POST | `/api/start_pass` | (none) | `{"success": true|false, "error": "..."}` | `api_start_pass()` → `_commander.start_pass()` |
| POST | `/api/end_pass` | (none) | `{"success": true|false, "error": "..."}` | `api_end_pass()` → `_commander.end_pass()` |
| POST | `/api/set_cell` | `{"row": 0, "col": 0}` | `{"success": true|false, "error": "..."}` | `api_set_cell()` → `_commander.set_grid_cell()` |
| POST | `/api/set_cubesat_ip` | `{"ip": "192.168.1.229"}` | `{"success": true, "ip": "...", "reachable": true|false}` | `api_set_cubesat_ip()` |
| POST | `/api/discover_cubesat` | (none) | `{"found": true, "ip": "...", "ssh_ok": true, "flight_running": true, "error": "..."}` | `api_discover_cubesat()` → `pi_manager.discover()` |
| POST | `/api/pi_start` | (none) | `{"success": true, "message": "..."}` | `api_pi_start()` → `pi_manager.start_flight_software()` |
| POST | `/api/pi_stop` | (none) | `{"success": true, "message": "..."}` | `api_pi_stop()` → `pi_manager.stop_flight_software()` |
| POST | `/api/llm_query` | `{"question": "What is the coverage?"}` | `{"response": "..."}` | `api_llm_query()` → `_query_ollama()` |
| POST | `/api/reset_mission` | (none) | `{"status": "ok", "message": "..."}` or error | `api_reset_mission()` — clears all data |
| POST | `/api/clear_last_pass` | (none) | `{"status": "ok", "message": "...", "new_total_passes": N}` or error | `api_clear_last_pass()` — removes last pass data |
| GET | `/api/export_mission` | (none) | PDF file download | `api_export_mission()` → `_build_mission_pdf()` |

**Command API details:**
- `retransmit`: body `{"cmd": "retransmit", "image_id": "pass1_001"}`
- `priority_cell`: body `{"cmd": "priority_cell", "row": 0, "col": 0}`
- `set_cell`: body `{"cmd": "set_cell", "row": 0, "col": 0}`
- `adjust_exposure`: body `{"cmd": "adjust_exposure", "exposure_us": 20000}`

---

## SECTION 2: DASHBOARD PANELS

| Panel Name | Data Displayed | API Endpoints | Refresh | Interactive Elements | Visual Description |
|------------|----------------|---------------|---------|----------------------|-------------------|
| **Top Bar** | State, Pass, connection status, mission control buttons | `/api/status` | 2s | START PASS, END PASS, SET CELL, FIND, START SW, STOP SW, NEW SESSION, EXPORT | Fixed 50px height, dark panel, cyan brand, state/pass in center |
| **Terrain Cost Map** | 8×8 cost heatmap, routes, landing/target markers | `/api/cost_grid`, `/api/routes` | 5s | Canvas click (set L/T), PLAN ROUTES, endpoint inputs | Canvas 564×560, labels A–H, 1–8, color-coded cells, route overlay |
| **Route Comparison** | Fastest/Safest/Balanced route cards | `/api/routes` | 3s | SELECT, SHOW ONLY per route | Cards with distance, shadow bar, risk badge |
| **Constraint Planner** | Max shadow %, min hazard clearance sliders | — | — | Sliders, RECALCULATE | Two sliders + cyan button |
| **Selected Route Stats** | Path stats for selected route | `/api/routes` | 3s | — | KV rows, risk badge |
| **Change Detection** | Before/after slider, event list | `/api/changes` | 3s | Cell tabs, slider handle, event cards | Slider with clip-path, SVG bbox overlay |
| **Coverage Grid** | 8×8 surveyed cells | `/api/cost_grid` | 3s | REPLAY | Canvas 340×340, timeline bar |
| **Latest Image** | Most recent received image | `/api/latest_image` | 3s | — | img or placeholder |
| **Shadow Analysis** | Shadow %, regions | `/api/shadow_data` | 5s | — | Text summary |
| **ML Object Detection** | YOLO detections, fused classifications | `/api/yolo_detections`, `/api/yolo_annotated` | 5s | — | Image + list |
| **Downlink** | Bytes, rate, progress bar | `/api/status` (mission.downlink) | 2s | — | Progress bar, 4-column grid |
| **Hazard Density by Quadrant** | NW/NE/SW/SE hazard counts | `/api/cost_grid` | 5s | — | Table |
| **Mission Status** | State, pass, images, coverage, route, changes | `/api/status` | 2s | — | KV rows |
| **Telemetry** (collapsible) | IMU, system, commands | `/api/status` (telemetry) | 2s | SAFE MODE, RESUME, STATUS REQ, RETRY DL, Retransmit, Priority Cell, Exposure | 4-column grid |
| **Quality Log** (collapsible) | Per-image quality | `/api/quality_log` | 5s | — | Table |
| **Event Log** (collapsible) | Application log lines | `/api/log` | 3s | — | Preformatted text |
| **LLM Query** | Input + response | `/api/llm_query` | on submit | Input, ASK button | Bar with input |

---

## SECTION 3: INTERACTIVE ELEMENTS

| Element | Action | API | Request | Success | Failure | Visual State |
|---------|--------|-----|---------|---------|---------|--------------|
| **START PASS** | Transition WAITING→IMAGING | POST `/api/start_pass` | (none) | Green flash, "PASS STARTED" | Red flash, error text | Disabled unless WAITING/SAFE_MODE |
| **END PASS** | Transition IMAGING→PROCESSING | POST `/api/end_pass` | (none) | Green flash | Red flash | Disabled unless IMAGING |
| **SET CELL** | Set next grid cell to image | POST `/api/set_cell` | `{row, col}` | "CELL R,C SET" | "FAILED" | Disabled unless IMAGING |
| **NEW SESSION** | Reset all mission data | POST `/api/reset_mission` | (none) | Reload page | Error in feedback | Always enabled |
| **CLEAR LAST PASS** | Remove last pass data | POST `/api/clear_last_pass` | (none) | Reload page | Error | *(API and `mcClearLastPass()` exist; add a button with `onclick="mcClearLastPass()"` if desired)* |
| **EXPORT** | Download PDF report | GET `/api/export_mission` | — | PDF download | — | Always enabled |
| **Route SELECT** | Select route | POST `/api/select_route` | `{route_name}` | Re-fetch routes | — | Card gets `selected-card` |
| **Route SHOW ONLY** | Toggle single-route view on heatmap | — | — | Redraw heatmap | — | Button gets `active-show` |
| **Constraint sliders** | Max shadow %, min clearance | — | — | — | — | Live value display |
| **RECALCULATE** | Plan constrained route | POST `/api/plan_constrained` | `{max_shadow_pct, min_hazard_clearance, start, end}` | Show constrained card | Show "NO FEASIBLE PATH" | Disabled during request |
| **Cost heatmap click** | Set Landing (1st), Target (2nd), Reset (3rd) | — | — | Updates inputs, auto-plans on 2nd | — | Feedback text |
| **PLAN ROUTES** | Plan routes | POST `/api/plan_routes` | `{start, end}` | "ROUTES UPDATED", redraw | Error text | Disabled during request |
| **Change before/after slider** | Drag to compare | — | — | clip-path updates | — | Handle + divider move |
| **Change cell tabs** | Switch cell | — | — | Re-render slider/events | — | Tab gets `active` |
| **Change event card click** | Highlight bbox | — | — | Bbox stroke highlight | — | Card gets `active-event` |
| **Coverage REPLAY** | Animate coverage build-up | — | — | 3.5s animation | — | — |
| **Command panel: SAFE MODE** | Enter safe mode | POST `/api/command` | `{cmd:"enter_safe_mode"}` | Green flash | Red flash | — |
| **Command panel: RESUME** | Resume normal | POST `/api/command` | `{cmd:"resume_normal"}` | — | — | — |
| **Command panel: STATUS REQ** | Request status | POST `/api/command` | `{cmd:"status_request"}` | — | — | — |
| **Command panel: RETRY DL** | Retry downlink | POST `/api/command` | `{cmd:"retry_downlink"}` | — | — | — |
| **Retransmit SEND** | Retransmit image | POST `/api/command` | `{cmd:"retransmit", image_id}` | — | — | — |
| **Priority Cell SET** | Set priority cell | POST `/api/command` | `{cmd:"priority_cell", row, col}` | — | — | — |
| **Exposure SET** | Adjust exposure | POST `/api/command` | `{cmd:"adjust_exposure", exposure_us}` | — | — | — |
| **LLM ASK** | Query LLM | POST `/api/llm_query` | `{question}` | Response in div | "LLM error" | — |
| **FIND** | Discover CubeSat | POST `/api/discover_cubesat` | (none) | Show IP, START SW/STOP SW | "NOT FOUND" | Disabled during scan |
| **START SW** | Start flight software | POST `/api/pi_start` | (none) | Show STOP SW | — | — |
| **STOP SW** | Stop flight software | POST `/api/pi_stop` | (none) | Show START SW | — | — |
| **Collapse headers** | Toggle sections | — | — | Arrow rotates, body shows | — | `arrow.open`, `collapse-body.open` |

---

## SECTION 4: DATA FLOW

| Panel | Origin | Format | Update Path |
|-------|--------|--------|-------------|
| Status / Mission / Telemetry / Downlink | `MissionState` + `telemetry_parser` | JSON from `/api/status` | `pollStatus()` → DOM updates |
| Cost heatmap | `cost_grid.json` + pipeline | JSON from `/api/cost_grid` | `pollCostHeatmap()` → `_heatmapData` → `hmDraw()` |
| Routes | `mission_state` or `routes.json` | JSON from `/api/routes` | `pollRoutes()` → `renderRouteCards()`, `renderPathStatsCard()` |
| Coverage | `cost_grid.json` pass_data | JSON from `/api/cost_grid` | `pollCoverage()` → `drawCoverageFinal()` or `covAnimLoop()` |
| Change detection | `changes.json` | JSON from `/api/changes` | `pollChanges()` → `renderChangeDetection()` |
| Shadow | `shadow_data.json` | JSON from `/api/shadow_data` | `pollShadow()` → `shadow-summary-text` |
| YOLO | `yolo_detections.json` + image | JSON + PNG | `pollYolo()` → summary, image, list |
| Latest image | `received_images/*.jpg` | Binary JPEG | `updateImg()` with cache-bust `?t=` |
| Quality log | In-memory `_quality_log` | JSON from `/api/quality_log` | `pollQuality()` → table |
| Event log | File handler | JSON from `/api/log` | `pollLog()` → pre div |
| Quadrant summary | `cost_grid` classifications | From `pollCostHeatmap()` | `renderQuadrantSummary()` → table |

---

## SECTION 5: JAVASCRIPT FUNCTIONS

| Function | Purpose | APIs | DOM Updates | Trigger |
|----------|---------|------|-------------|---------|
| `toggleCollapse(hdr)` | Toggle collapsible section | — | arrow, collapse-body | Click |
| `el(id)` | Get element by ID | — | — | — |
| `setText(id, val)` | Set textContent | — | Element | — |
| `fmtNum(n, dec)` | Format number | — | — | — |
| `flashBtn(btn, ok)` | Flash green/red | — | btn class | — |
| `updateImg(imgId, phId, url)` | Cache-busted image load | fetch url | img, placeholder | — |
| `pollStatus()` | Fetch status | `/api/status` | conn-dot, mission, telemetry, downlink, MC, coverage badge | 2s interval + load |
| `updateMissionControl(state, pass)` | Enable/disable MC buttons | — | mc-state-value, mc-pass-value, buttons, inputs | From pollStatus |
| `mcStartPass(btn)` | Start pass | POST `/api/start_pass` | mc-feedback, btn flash | Click |
| `mcEndPass(btn)` | End pass | POST `/api/end_pass` | mc-feedback, btn flash | Click |
| `mcSetCell(btn)` | Set cell | POST `/api/set_cell` | mc-feedback, btn flash | Click |
| `mcDiscover()` | Discover CubeSat | POST `/api/discover_cubesat` | mc-feedback, mc-pi-ip, mc-connect-dot, START/STOP visibility | Click + load |
| `mcPiStart()` | Start flight SW | POST `/api/pi_start` | mc-feedback, dot, buttons | Click |
| `mcPiStop()` | Stop flight SW | POST `/api/pi_stop` | mc-feedback, dot, buttons | Click |
| `mcNewSession()` | Reset mission | POST `/api/reset_mission` | Reload | Click |
| `mcClearLastPass()` | Clear last pass | POST `/api/clear_last_pass` | Reload | Click (no visible button) |
| `mcExport()` | Export PDF | GET `/api/export_mission` | Navigate | Click |
| `pollRoutes()` | Fetch routes | `/api/routes` | route-cards, path-stats-card, route-change-warning | 3s + load |
| `renderRouteCards(routes)` | Render route cards | — | route-cards, route-status-badge | From pollRoutes |
| `renderPathStatsCard(routes)` | Render path stats | — | path-stats-card | From pollRoutes |
| `toggleShowOnly(name)` | Toggle single route | — | _hmShowOnlyRoute, redraw | Click |
| `selectRoute(name)` | Select route | POST `/api/select_route` | — | Click |
| `runConstrainedPlan(btn)` | Plan constrained | POST `/api/plan_constrained` | constrained-result | Click |
| `getRouteStart()`, `getRouteEnd()` | Read endpoint inputs | — | — | — |
| `setRouteStart(r,c)`, `setRouteEnd(r,c)` | Set endpoint inputs | — | ep-start-r/c, ep-end-r/c | — |
| `planRoutes(btn)` | Plan routes | POST `/api/plan_routes` | plan-routes-feedback, route cards | Click, heatmap 2nd click |
| `hmDraw(now)` | Draw heatmap canvas | — | cost-heatmap-canvas | requestAnimationFrame |
| `hmCellFill()`, `hmCellName()` | Cell colors | — | — | — |
| `_hmDrawEndpoint()` | Draw L/T markers | — | — | — |
| `pollCostHeatmap()` | Fetch cost grid + routes | `/api/cost_grid`, `/api/routes` | _heatmapData, _hmRouteData, quadrant table | 5s + load |
| `renderQuadrantSummary(data)` | Render quadrant table | — | quad-table-body | From pollCostHeatmap |
| `pollChanges()` | Fetch changes | `/api/changes` | _changeData, renderChangeDetection | 3s + load |
| `renderChangeDetection(data)` | Render change UI | — | change-cell-tabs, slider, events, warning | From pollChanges |
| `updateSliderPosition(pct)` | Update slider | — | cs-after-wrap, cs-divider, cs-handle | Drag, click |
| `renderBboxOverlay(events)` | Draw bboxes | — | cs-bbox-overlay | From renderChangeDetection |
| `highlightBbox(evt)` | Highlight one bbox | — | cs-bbox-overlay rects | Event card click |
| `checkChangeRouteImpact()` | Show route warning | — | route-change-warning | From pollStatus/pollRoutes |
| `pollCoverage()` | Fetch cost grid | `/api/cost_grid` | _covPassData, drawCoverageFinal or covAnimLoop | 3s + load |
| `drawCoverageFinal()` | Draw final coverage | — | coverage-canvas, badge, timeline | — |
| `startCoverageAnim()` | Start replay | — | covAnimLoop | Click REPLAY |
| `covAnimLoop(now)` | Animate coverage | — | coverage-canvas, timeline | requestAnimationFrame |
| `pollShadow()` | Fetch shadow | `/api/shadow_data` | shadow-summary-text | 5s + load |
| `pollYolo()` | Fetch YOLO | `/api/yolo_detections`, `/api/yolo_annotated`, `/api/status` | yolo-summary, img-yolo, yolo-detections-list, badge | 5s + load |
| `pollQuality()` | Fetch quality log | `/api/quality_log` | q-table-body, q-count-badge | 5s + load |
| `pollLog()` | Fetch log | `/api/log` | event-log, log-count-badge | 3s + load |
| `sendCmd(body, btn)` | Send command | POST `/api/command` | btn flash | — |
| `cmdRetransmit(btn)` | Retransmit | — | — | Click |
| `cmdPriorityCell(btn)` | Priority cell | — | — | Click |
| `cmdSetCell(btn)` | Set cell (hidden) | — | — | — |
| `cmdExposure(btn)` | Exposure | — | — | Click |
| `sendLLM()` | LLM query | POST `/api/llm_query` | llm-response | Click, Enter |
| `pollImages()` | Fetch images | latest, hazard, route_comparison, mosaic | img elements | 3s + load |
| `costToHeatmapColour()` | Cost→color | — | — | — |
| `fmtFmt(sec)` | Format uptime | — | — | — |
| `escHtml(s)` | Escape HTML | — | — | — |

---

## SECTION 6: CSS STYLING

### Colors (CSS variables)
| Variable | Value | Usage |
|----------|-------|-------|
| `--bg` | `#0a0e17` | Page background |
| `--panel` | `#141a26` | Panel background |
| `--border` | `#3a3f4b` | Borders |
| `--cyan` | `#00d4ff` | Accent, links, active |
| `--green` | `#00ff88` | Safe, success |
| `--amber` | `#ffaa00` | Warning |
| `--red` | `#ff4444` | Danger, error |
| `--dim` | `#888888` | Muted text |
| `--txt` | `#e0e0e0` | Body text |
| `--bright` | `#ffffff` | Emphasized text |
| `--mono` | Consolas, 'Courier New', monospace | Code/numbers |

### Hazard heatmap colors
- SAFE: `#00ff88` (green)
- MODERATE: `#ffdd00` (yellow)
- SHADOW: `#4488ff` (blue)
- HAZARD: `#ff4444` (red)
- IMPASSABLE: `#1a1a2e` + red crosshatch
- Unsurveyed: `#1e2233`

### Buttons
- **Normal:** `background: var(--panel)`, `border: 1px solid var(--border)`, `color: var(--txt)`
- **Hover:** `border-color: var(--cyan)`, `box-shadow: 0 0 8px rgba(0,212,255,0.3)`
- **Disabled:** `opacity: 0.35`, `cursor: not-allowed`
- **ok:** `border-color: var(--green)`, `color: var(--green)`
- **fail:** `border-color: var(--red)`, `color: var(--red)`
- **active-start:** Green tint
- **active-end:** Red tint
- **btn-recalc:** Cyan fill, dark text, full width

### Sliders
- Track: `height: 4px`, `background: var(--border)`
- Thumb: `16×16px` circle, `background: var(--cyan)`, `border: 2px solid var(--bg)`, `box-shadow: 0 0 4px rgba(0,212,255,0.4)`

### Layout
- Top bar: fixed, 50px, flex
- Main: `margin-top: 50px`, flex row
- Left column: 60%, min-width 600px
- Right column: 40%, overflow-y auto
- Panels: `background: var(--panel)`, `border: 1px solid var(--border)`

### Animations
- Route draw: 1000ms linear progress
- Change cell pulse: `sin(now/1000*π)` for glow
- Landing/Target markers: 8–12px pulse radius
- Coverage fade: alpha 0.4→1 over pass

---

## SECTION 7: REAL-TIME BEHAVIORS

| Behavior | Interval/Trigger | Description |
|----------|-----------------|-------------|
| Status poll | 2000 ms | `/api/status` → mission, telemetry, downlink, MC state |
| Images poll | 3000 ms | Latest, hazard, route_comparison, mosaic |
| Coverage poll | 3000 ms | Cost grid → coverage canvas |
| Quality poll | 5000 ms | Quality log table |
| Log poll | 3000 ms | Event log |
| Routes poll | 3000 ms | Route cards, path stats |
| Heatmap poll | 5000 ms | Cost grid + routes |
| Changes poll | 3000 ms | Change detection |
| Shadow poll | 5000 ms | Shadow summary |
| YOLO poll | 5000 ms | YOLO detections + image |
| Heatmap animation | requestAnimationFrame | Continuous redraw, route draw animation |
| Coverage REPLAY | On click | 3.5s animation (1s per pass + 0.5s change pulse) |
| Button enable/disable | From pollStatus | START when WAITING, END when IMAGING, SET when IMAGING |
| Auto-scroll | Quality log, Event log | `scrollTop = scrollHeight` on update |
| Image refresh | Cache-bust `?t=Date.now()` | Only update if 200 |

---

## SECTION 8: EXTERNAL DEPENDENCIES

**None.** The frontend uses:
- No CDN scripts
- No external JS libraries (vanilla JS only)
- No CSS frameworks
- System fonts: `system-ui, -apple-system, 'Segoe UI', sans-serif`
- Monospace: `Consolas, 'Courier New', monospace`

All CSS and JS are inline in `index.html`.

---

## SECTION 9: PAGE STRUCTURE (HTML Outline)

```
html[lang=en]
├── head
│   ├── meta[charset=UTF-8]
│   ├── meta[viewport]
│   ├── title "MuraltZ GCS — Mission Operations"
│   └── style (inline CSS)
│
└── body
    ├── #topbar (fixed)
    │   ├── .brand "MURALTZ GCS"
    │   ├── #conn-dot
    │   ├── #conn-label
    │   ├── .tb-center
    │   │   ├── STATE / #mc-state-value
    │   │   └── PASS / #mc-pass-value
    │   └── .tb-controls
    │       ├── #mc-btn-start, #mc-btn-end
    │       ├── #mc-row, #mc-col, #mc-btn-cell
    │       ├── #mc-btn-discover, #mc-btn-pi-start, #mc-btn-pi-stop
    │       ├── #mc-pi-ip, #mc-connect-dot, #mc-feedback
    │       ├── NEW SESSION, EXPORT
    │
    ├── #main
    │   ├── #col-left
    │   │   ├── .pnl#heatmap-section
    │   │   │   ├── .pnl-hdr (Terrain Cost Map, #route-status-badge)
    │   │   │   └── .pnl-body
    │   │   │       ├── #route-change-warning
    │   │   │       ├── .ep-row (ep-start-r/c, ep-end-r/c, PLAN ROUTES, #plan-routes-feedback)
    │   │   │       ├── #heatmap-wrap
    │   │   │       │   ├── canvas#cost-heatmap-canvas[564×560]
    │   │   │       │   └── #heatmap-tooltip
    │   │   │       └── .heatmap-legend
    │   │   │
    │   │   ├── .pnl (Route Comparison)
    │   │   │   ├── .pnl-hdr
    │   │   │   └── .pnl-body
    │   │   │       ├── #route-cards
    │   │   │       └── #constrained-result
    │   │   │
    │   │   └── div (flex: Constraint Planner + Selected Route Stats)
    │   │       ├── .pnl (Constraint Planner)
    │   │       │   ├── #slider-shadow, #val-shadow
    │   │       │   ├── #slider-clearance, #val-clearance
    │   │       │   └── .btn-recalc RECALCULATE
    │   │       └── .pnl (Selected Route Stats)
    │   │           └── #path-stats-card
    │   │
    │   └── #col-right
    │       ├── .pnl (Change Detection)
    │       │   ├── #change-summary, #change-cell-tabs
    │       │   ├── #change-slider (.cs-wrap)
    │       │   │   ├── #cs-before-img, #cs-after-wrap, #cs-after-img
    │       │   │   ├── #cs-divider, #cs-handle
    │       │   │   ├── #cs-label-before, #cs-label-after
    │       │   │   └── #cs-bbox-overlay (SVG)
    │       │   ├── #change-slider-ph
    │       │   ├── #change-events-list
    │       │   └── #change-low-conf-warning
    │       │
    │       ├── .pnl (Coverage Grid)
    │       │   ├── #coverage-pct-badge
    │       │   ├── canvas#coverage-canvas[340×340]
    │       │   ├── #cov-timeline (#cov-tl-p1, p2, p3, #cov-tl-playhead)
    │       │   └── REPLAY button
    │       │
    │       ├── .pnl (Latest Image)
    │       │   ├── #img-latest
    │       │   └── #img-latest-ph
    │       │
    │       ├── .pnl (Shadow Analysis)
    │       │   └── #shadow-summary-text
    │       │
    │       ├── .pnl (ML Object Detection)
    │       │   ├── #yolo-model-badge
    │       │   ├── #yolo-summary
    │       │   ├── #img-yolo, #img-yolo-ph
    │       │   └── #yolo-detections-list
    │       │
    │       ├── .pnl (Downlink)
    │       │   ├── #dl-bytes, .prog-track #dl-bar
    │       │   ├── #dl-rate, #dl-total-bytes, #dl-failed, #ul-cmds
    │       │   └── #dl-rate-badge
    │       │
    │       ├── .pnl (Hazard Density by Quadrant)
    │       │   └── .quad-table #quad-table-body
    │       │
    │       └── .pnl (Mission Status)
    │           ├── #sat-state-badge
    │           └── .kv (#s-state, #s-pass, #s-images, #s-corrupt, #s-coverage, #s-route, #s-changes, #s-updated)
    │
    ├── #bottom
    │   ├── .collapse-section (Telemetry)
    │   │   ├── .collapse-hdr
    │   │   └── .collapse-body
    │   │       └── .telem-grid
    │   │           ├── .telem-group (IMU: #t-state, #t-roll, #t-pitch, #t-nadir)
    │   │           ├── .telem-group (System: #t-temp, #t-storage, #t-queue, #t-uptime)
    │   │           ├── .telem-group (Commands: SAFE MODE, RESUME, STATUS REQ, RETRY DL)
    │   │           └── .telem-group (Advanced: #in-retransmit, #in-pri-r/c, #in-exposure + buttons)
    │   │
    │   ├── .collapse-section (Quality Log)
    │   │   ├── .collapse-hdr (#q-count-badge)
    │   │   └── .collapse-body
    │   │       └── .q-table #q-table-body
    │   │
    │   ├── .collapse-section (Event Log)
    │   │   ├── .collapse-hdr (#log-count-badge)
    │   │   └── .collapse-body
    │   │       └── #event-log
    │   │
    │   ├── #llm-bar
    │   │   ├── #llm-input
    │   │   └── ASK button
    │   └── #llm-response
    │
    ├── div[display:none] (hidden)
    │   ├── #img-hazard, #img-hazard-ph
    │   ├── #img-route-comparison, #img-route-comparison-ph
    │   ├── #img-mosaic, #img-mosaic-ph
    │   ├── #server-time
    │   └── #in-cell-r, #in-cell-c
    │
    └── script (inline JS)
```

---

## Constants (JS)

```javascript
POLL_STATUS_MS   = 2000;
POLL_IMAGES_MS   = 3000;
POLL_COVERAGE_MS = 3000;
POLL_QUALITY_MS  = 5000;
POLL_LOG_MS      = 3000;
POLL_ROUTES_MS   = 3000;
POLL_HEATMAP_MS  = 5000;
POLL_CHANGES_MS  = 3000;
DL_BUDGET_BYTES  = 72000;
HM_ROWS = 8, HM_COLS = 8;
HM_MARGIN_LEFT = 24, HM_MARGIN_TOP = 20;
HM_SIZE = 540, HM_CELL = 67.5;
_hmRouteAnimDur = 1000;
COV_ANIM_DUR = 3500;
COL_LABELS = 'ABCDEFGH';
```

---

## Template Variable

- `{{ cubesat_ip }}` — Injected from `config.CUBESAT_IP` (or empty string).

---

*End of specification. Rebuild the frontend from this document.*
