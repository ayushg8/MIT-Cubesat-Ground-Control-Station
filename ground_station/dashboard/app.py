from __future__ import annotations
# dashboard/app.py — Flask dashboard on DASHBOARD_PORT (8080)
#
# Serves the mission operations dashboard and a JSON API consumed by the
# single-page frontend. All data is REAL — pulled from live pipeline state,
# live telemetry, and saved image files.
#
# Dependencies (pipeline, mission_state, commander) are injected by server.py
# via the set_*() functions below so this module has no circular imports.

import glob
import io
import json
import logging
import os
import shutil
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request, send_file

import config
from receiver import telemetry_parser
from uplink.commander import Commander
from uplink import pi_manager

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False

# ── Injected at startup by server.py ──
_pipeline = None
_mission_state = None
_commander: Commander | None = None
_quality_log: list = []        # accumulated list of quality dicts, appended by pipeline


def set_pipeline(p):
    global _pipeline
    _pipeline = p


def set_mission_state(ms):
    global _mission_state
    _mission_state = ms


def set_commander(c: Commander):
    global _commander
    _commander = c


def append_quality_entry(entry: dict):
    """Called by pipeline.py after each image is processed."""
    _quality_log.append(entry)


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", cubesat_ip=config.CUBESAT_IP or "")


# ─────────────────────────────────────────────────────────────────────────────
# API — JSON data
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/mosaic_info")
def api_mosaic_info():
    """Return mosaic metadata: dimensions, image count, entries, grid info."""
    if _pipeline is not None:
        return jsonify(_pipeline.get_mosaic_info())
    return jsonify({
        "width": 0, "height": 0, "image_count": 0,
        "entries": [], "grid": {"rows": 0, "cols": 0, "cell_size_px": config.MOSAIC_GRID_CELL_PX,
                                "origin_x": 0, "origin_y": 0},
    })

@app.route("/api/status")
def api_status():
    """Merge latest telemetry + mission_state into one status blob."""
    state = _mission_state.get_snapshot() if _mission_state else {}
    telemetry = telemetry_parser.get_latest_telemetry()

    # Convert set → list for JSON (cells_covered may be a set if not yet saved)
    payload = {
        "mission": state,
        "telemetry": telemetry,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(payload)


@app.route("/api/coverage")
def api_coverage():
    """Return dynamic coverage grid as JSON for the canvas panel."""
    if _pipeline is not None:
        hazard_grid = _pipeline.get_hazard_grid()
        cost_grid   = _pipeline.get_cost_grid()
        rows, cols = cost_grid.shape
    else:
        rows, cols = 1, 1
        hazard_grid = [["SAFE"]]
        cost_grid   = None

    state = _mission_state.get_snapshot() if _mission_state else {}

    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            cost = int(cost_grid[r, c]) if cost_grid is not None else config.COST_SAFE
            row.append({
                "row": r,
                "col": c,
                "hazard_class": hazard_grid[r][c],
                "cost": cost,
                "has_change": [r, c] in state.get("changes", {}).get("cells_with_changes", []),
            })
        grid.append(row)

    return jsonify({"grid": grid, "rows": rows, "cols": cols})


@app.route("/api/quality_log")
def api_quality_log():
    return jsonify({"entries": _quality_log})


@app.route("/api/log")
def api_log():
    """Return last 100 lines from the application log file, if available."""
    log_lines = _read_log_tail(100)
    return jsonify({"lines": log_lines})


# ─────────────────────────────────────────────────────────────────────────────
# API — image files (send latest PNG/JPEG from processed dirs)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/latest_image")
def api_latest_image():
    path = _latest_file(config.RECEIVED_DIR, "*.jpg")
    if path is None:
        return ("No image yet", 204)
    return send_file(os.path.abspath(path), mimetype="image/jpeg")


@app.route("/api/hazard_map")
def api_hazard_map():
    path = _latest_file(os.path.join(config.PROCESSED_DIR, "hazard_maps"), "*_hazard.png")
    if path is None:
        return ("No hazard map yet", 204)
    return send_file(os.path.abspath(path), mimetype="image/png")


@app.route("/api/change_map")
def api_change_map():
    path = _latest_file(os.path.join(config.PROCESSED_DIR, "change_maps"), "*_change_*.png")
    if path is None:
        return ("No change map yet", 204)
    return send_file(os.path.abspath(path), mimetype="image/png")


@app.route("/api/route_map")
def api_route_map():
    fixed = os.path.join(config.PROCESSED_DIR, "routes", "route_latest.png")
    if os.path.exists(fixed):
        return send_file(os.path.abspath(fixed), mimetype="image/png")
    return ("No route map yet", 204)


@app.route("/api/mosaic")
def api_mosaic():
    fixed = os.path.join(config.PROCESSED_DIR, "mosaics", "mosaic_latest.png")
    if os.path.exists(fixed):
        return send_file(os.path.abspath(fixed), mimetype="image/png")
    return ("No mosaic yet", 204)


@app.route("/api/routes")
def api_routes():
    """Return fastest/safest/balanced route dicts from mission state,
    falling back to routes.json file."""
    state = _mission_state.get_snapshot() if _mission_state else {}
    routes = state.get("routes", {})
    # If mission state has route data WITH paths, use it
    fastest = routes.get("fastest") or {}
    if fastest.get("path"):
        return jsonify(routes)
    # Fall back to routes.json (has full path arrays)
    rpath = os.path.join(config.PROCESSED_DIR, "routes.json")
    if os.path.exists(rpath):
        with open(rpath) as f:
            rdata = json.load(f)
        # Convert array format to keyed format expected by dashboard
        if "routes" in rdata and isinstance(rdata["routes"], list):
            out = {
                "selected": rdata.get("selected", "safest"),
                "constrained": rdata.get("constrained"),
                "start": rdata.get("start"),
                "end": rdata.get("end"),
            }
            for r in rdata["routes"]:
                key = r["name"].lower()
                out[key] = {**r.get("stats", {}), "path": r.get("path", []),
                            "name": r["name"], "color": r.get("color", "#fff")}
            return jsonify(out)
        return jsonify(rdata)
    return jsonify(routes)


@app.route("/api/route_comparison_image")
def api_route_comparison_image():
    fixed = os.path.join(config.PROCESSED_DIR, "routes", "route_comparison.png")
    if os.path.exists(fixed):
        return send_file(os.path.abspath(fixed), mimetype="image/png")
    # Fall back to route_latest.png
    fallback = os.path.join(config.PROCESSED_DIR, "routes", "route_latest.png")
    if os.path.exists(fallback):
        return send_file(os.path.abspath(fallback), mimetype="image/png")
    return ("No route comparison yet", 204)


@app.route("/api/cost_heatmap")
def api_cost_heatmap():
    """Return dynamic JSON array: {row, col, cost, hazard_class} per cell."""
    if _pipeline is not None:
        hazard_grid = _pipeline.get_hazard_grid()
        cost_grid   = _pipeline.get_cost_grid()
        rows, cols = cost_grid.shape
    else:
        rows, cols = 1, 1
        hazard_grid = [["SAFE"]]
        cost_grid   = None

    cells = []
    for r in range(rows):
        for c in range(cols):
            cost = int(cost_grid[r, c]) if cost_grid is not None else config.COST_SAFE
            cells.append({
                "row": r,
                "col": c,
                "cost": cost,
                "hazard_class": hazard_grid[r][c],
            })
    return jsonify({"cells": cells, "rows": rows, "cols": cols})


@app.route("/api/cost_grid")
def api_cost_grid():
    """Return cost_grid.json contents."""
    path = os.path.join(config.PROCESSED_DIR, "cost_grid.json")
    if os.path.exists(path):
        return send_file(os.path.abspath(path), mimetype="application/json")
    return jsonify({"grid": [], "rows": 0, "cols": 0, "classifications": [], "coverage": [], "pass_data": [], "change_cells": []})


@app.route("/api/changes")
def api_changes():
    """Return changes.json contents."""
    path = os.path.join(config.PROCESSED_DIR, "changes.json")
    if os.path.exists(path):
        return send_file(os.path.abspath(path), mimetype="application/json")
    return jsonify({"events": [], "summary": {"total_events": 0, "total_area": 0}})


@app.route("/api/shadow_data")
def api_shadow_data():
    """Return shadow_data.json contents."""
    path = os.path.join(config.PROCESSED_DIR, "shadow_data.json")
    if os.path.exists(path):
        return send_file(os.path.abspath(path), mimetype="application/json")
    return jsonify({"shadow_pct": 0, "regions": []})


@app.route("/api/image/<path:filename>")
def api_image(filename):
    """Serve raw image files from received_images/."""
    path = os.path.join(config.RECEIVED_DIR, filename)
    if os.path.exists(path):
        return send_file(os.path.abspath(path), mimetype="image/jpeg")
    return ("Image not found", 404)


@app.route("/api/yolo_detections")
def api_yolo_detections():
    """Return YOLO detection results and fused classifications."""
    path = os.path.join(config.PROCESSED_DIR, "yolo_detections.json")
    if os.path.exists(path):
        return send_file(os.path.abspath(path), mimetype="application/json")
    return jsonify({
        "detections_per_cell": {},
        "fused_classifications": [],
        "summary": {
            "total_detections": 0, "craters_detected": 0,
            "boulders_detected": 0, "cv_agreement_rate": 1.0,
            "cells_analyzed": 0,
        }
    })


@app.route("/api/yolo_annotated")
def api_yolo_annotated():
    """Return the latest YOLO-annotated image."""
    det_dir = os.path.join(config.PROCESSED_DIR, "yolo_detections")
    path = _latest_file(det_dir, "*_yolo.png")
    if path:
        return send_file(os.path.abspath(path), mimetype="image/png")
    return ("No YOLO annotated image yet", 204)


@app.route("/api/cell_map")
def api_cell_map():
    """Return mosaic entry summary (replaces old cell_identifier database)."""
    if _pipeline is not None:
        info = _pipeline.get_mosaic_info()
        summary = {}
        for entry in info.get("entries", []):
            summary[entry["filename"]] = {
                "bbox": entry["bbox"],
            }
        return jsonify(summary)
    return jsonify({})


@app.route("/api/plan_routes", methods=["POST"])
def api_plan_routes():
    """Body: {start_mosaic: [mx,my], end_mosaic: [mx,my]}
    or legacy: {start: [row,col], end: [row,col]}
    → run plan_multiple_routes. Mosaic coords are converted to grid coords internally."""
    from processing.route_planner import grid_path_to_mosaic_path

    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if _pipeline is None:
        return jsonify({"error": "Pipeline not ready"}), 503

    cost_grid = _pipeline.get_cost_grid()
    hazard_grid = _pipeline.get_hazard_grid()
    rows, cols = cost_grid.shape
    hazard_map_path = _pipeline.get_latest_hazard_map_path() if hasattr(_pipeline, 'get_latest_hazard_map_path') else None

    start_mosaic = body.get("start_mosaic")
    end_mosaic = body.get("end_mosaic")

    if start_mosaic and end_mosaic:
        # Convert mosaic pixel coords to grid coords
        from processing.mosaic_grid import MosaicGrid
        grid = _pipeline._mosaic_grid
        start = grid.mosaic_px_to_grid(start_mosaic[0], start_mosaic[1])
        end = grid.mosaic_px_to_grid(end_mosaic[0], end_mosaic[1])
        # Store mosaic endpoints in pipeline
        _pipeline.set_route_endpoints_mosaic(start_mosaic, end_mosaic)
    else:
        # Legacy grid coords
        default_start = [0, 0]
        default_end = [max(0, rows - 1), max(0, cols - 1)]
        start = tuple(body.get("start", default_start))
        end = tuple(body.get("end", default_end))
        start_mosaic = None
        end_mosaic = None

    # Validate coordinates
    for label, pt in [("start", start), ("end", end)]:
        if len(pt) != 2 or not (0 <= pt[0] < rows and 0 <= pt[1] < cols):
            return jsonify({"error": f"Invalid {label}: {pt} (grid is {rows}x{cols})"}), 400

    try:
        routes = _pipeline._route_planner.plan_multiple_routes(
            cost_grid, hazard_grid, start, end, hazard_map_path,
        )

        # Add mosaic_path to each route
        for name in ("fastest", "safest", "balanced"):
            if name in routes and routes[name].get("path"):
                routes[name]["mosaic_path"] = grid_path_to_mosaic_path(routes[name]["path"])

        if _mission_state:
            with _mission_state._lock:
                _mission_state._state["routes"]["fastest"] = routes.get("fastest")
                _mission_state._state["routes"]["safest"] = routes.get("safest")
                _mission_state._state["routes"]["balanced"] = routes.get("balanced")
                _mission_state._state["route"]["start"] = list(start)
                _mission_state._state["route"]["end"] = list(end)
            _mission_state.save()

        routes["start"] = list(start)
        routes["end"] = list(end)
        if start_mosaic:
            routes["start_mosaic"] = start_mosaic
        if end_mosaic:
            routes["end_mosaic"] = end_mosaic

        return jsonify(routes)
    except Exception as e:
        logger.error(f"plan_routes failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/plan_constrained", methods=["POST"])
def api_plan_constrained():
    """Body: {max_shadow_pct, min_hazard_clearance, start?, end?, start_mosaic?, end_mosaic?} → route JSON."""
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    max_shadow_pct = float(body.get("max_shadow_pct", 50.0))
    min_hazard_clearance = int(body.get("min_hazard_clearance", 1))

    # Handle mosaic or grid coords
    start_mosaic = body.get("start_mosaic")
    end_mosaic = body.get("end_mosaic")
    if start_mosaic and end_mosaic and _pipeline:
        grid = _pipeline._mosaic_grid
        start = grid.mosaic_px_to_grid(start_mosaic[0], start_mosaic[1])
        end = grid.mosaic_px_to_grid(end_mosaic[0], end_mosaic[1])
    else:
        default_start = [0, 0]
        default_end = [0, 0]
        start = tuple(body.get("start", default_start))
        end = tuple(body.get("end", default_end))

    if _pipeline is None:
        return jsonify({"error": "Pipeline not ready"}), 503

    cost_grid   = _pipeline.get_cost_grid()
    hazard_grid = _pipeline.get_hazard_grid()

    try:
        result = _pipeline._route_planner.plan_with_constraints(
            cost_grid, hazard_grid,
            start, end,
            max_shadow_pct, min_hazard_clearance,
        )
        if _mission_state:
            with _mission_state._lock:
                _mission_state._state["routes"]["constrained"] = result
        return jsonify(result)
    except Exception as e:
        logger.error(f"plan_constrained failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/select_route", methods=["POST"])
def api_select_route():
    """Body: {route_name} → sets routes.selected in mission_state."""
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    route_name = body.get("route_name", "")
    if route_name not in ("fastest", "safest", "balanced"):
        return jsonify({"error": "route_name must be fastest, safest, or balanced"}), 400

    if _mission_state:
        with _mission_state._lock:
            _mission_state._state["routes"]["selected"] = route_name
        _mission_state.save()

    return jsonify({"selected": route_name})


# ─────────────────────────────────────────────────────────────────────────────
# API — commands
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/command", methods=["POST"])
def api_command():
    """Dispatch operator command to the CubeSat via commander.py."""
    if _commander is None:
        return jsonify({"success": False, "error": "Commander not initialised"}), 503

    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON"}), 400

    if not body or "cmd" not in body:
        return jsonify({"success": False, "error": "Missing 'cmd' field"}), 400

    cmd = body["cmd"]
    success = False

    try:
        if cmd == "retransmit":
            image_id = body.get("image_id", "")
            success = _commander.retransmit(image_id)
            if success and _mission_state:
                _mission_state.record_retransmit_request()

        elif cmd == "priority_cell":
            success = _commander.priority_cell(int(body["row"]), int(body["col"]))

        elif cmd == "set_cell":
            success = _commander.set_cell(int(body["row"]), int(body["col"]))

        elif cmd == "adjust_exposure":
            success = _commander.adjust_exposure(int(body["exposure_us"]))

        elif cmd == "enter_safe_mode":
            success = _commander.enter_safe_mode()

        elif cmd == "resume_normal":
            success = _commander.resume_normal()

        elif cmd == "status_request":
            success = _commander.request_status()

        elif cmd == "retry_downlink":
            success = _commander.retry_downlink()

        else:
            return jsonify({"success": False, "error": f"Unknown cmd '{cmd}'"}), 400

    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"success": False, "error": f"Bad parameters: {e}"}), 400
    except Exception as e:
        logger.error(f"Command dispatch error: {e}", exc_info=True)
        return jsonify({"success": False, "error": "Internal error"}), 500

    if _mission_state:
        _mission_state.record_command(acked=success)

    logger.info(f"Command '{cmd}' → {'ACK' if success else 'FAIL'}")
    return jsonify({"success": success})


@app.route("/api/start_pass", methods=["POST"])
def api_start_pass():
    """Transition CubeSat WAITING → IMAGING."""
    if _commander is None:
        return jsonify({"success": False, "error": "Commander not initialised"}), 503
    success = _commander.start_pass()
    if _mission_state:
        _mission_state.record_command(acked=success)
    logger.info(f"start_pass → {'ACK' if success else 'FAIL'}")
    return jsonify({"success": success, "error": _commander.last_error if not success else ""})


@app.route("/api/end_pass", methods=["POST"])
def api_end_pass():
    """Transition CubeSat IMAGING → PROCESSING."""
    if _commander is None:
        return jsonify({"success": False, "error": "Commander not initialised"}), 503
    success = _commander.end_pass()
    if _mission_state:
        _mission_state.record_command(acked=success)
    logger.info(f"end_pass → {'ACK' if success else 'FAIL'}")
    return jsonify({"success": success, "error": _commander.last_error if not success else ""})


@app.route("/api/set_cell", methods=["POST"])
def api_set_cell():
    """Set the next grid cell to image. Body: {"row": R, "col": C}."""
    if _commander is None:
        return jsonify({"success": False, "error": "Commander not initialised"}), 503
    try:
        body = request.get_json(force=True) or {}
        row = int(body["row"])
        col = int(body["col"])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"success": False, "error": f"Bad parameters: {e}"}), 400
    success = _commander.set_grid_cell(row, col)
    if _mission_state:
        _mission_state.record_command(acked=success)
    logger.info(f"set_cell ({row},{col}) → {'ACK' if success else 'FAIL'}")
    return jsonify({"success": success, "error": _commander.last_error if not success else ""})


@app.route("/api/set_cubesat_ip", methods=["POST"])
def api_set_cubesat_ip():
    """Set CUBESAT_IP at runtime and test TCP reachability. Body: {ip}."""
    import socket as _socket
    try:
        body = request.get_json(force=True) or {}
        ip = body.get("ip", "").strip()
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON"}), 400

    if not ip:
        return jsonify({"success": False, "error": "Missing ip"}), 400

    # Test TCP connectivity to COMMAND_PORT before committing
    reachable = False
    try:
        with _socket.create_connection((ip, config.COMMAND_PORT), timeout=1.5):
            reachable = True
    except Exception:
        pass

    config.CUBESAT_IP = ip
    logger.info(f"CUBESAT_IP updated to {ip} (reachable={reachable})")
    return jsonify({"success": True, "ip": ip, "reachable": reachable})


@app.route("/api/discover_cubesat", methods=["POST"])
def api_discover_cubesat():
    """Discover the Pi via mDNS, test SSH, check if flight software is running."""
    result = pi_manager.discover()
    return jsonify(result)


@app.route("/api/pi_start", methods=["POST"])
def api_pi_start():
    """SSH into the Pi and start flight software (or report it's already running)."""
    result = pi_manager.start_flight_software()
    return jsonify(result)


@app.route("/api/pi_stop", methods=["POST"])
def api_pi_stop():
    """SSH into the Pi and stop the flight software."""
    result = pi_manager.stop_flight_software()
    return jsonify(result)


@app.route("/api/pi_log")
def api_pi_log():
    """Get last 30 lines of the Pi flight log via SSH."""
    result = pi_manager.get_pi_log(30)
    return jsonify(result)


@app.route("/api/llm_query", methods=["POST"])
def api_llm_query():
    """Optional: send a question to local ollama with mission_state as context."""
    try:
        body = request.get_json(force=True)
        question = body.get("question", "").strip()
    except Exception:
        return jsonify({"response": "Invalid request"}), 400

    if not question:
        return jsonify({"response": "No question provided"}), 400

    state = _mission_state.get_snapshot() if _mission_state else {}
    response_text = _query_ollama(question, state)
    return jsonify({"response": response_text})


# ─────────────────────────────────────────────────────────────────────────────
# Mission management (reset / clear / export)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/reset_mission", methods=["POST"])
def api_reset_mission():
    """Delete ALL mission data: received images, processed files, reset state."""
    try:
        # Clear received images
        if os.path.exists(config.RECEIVED_DIR):
            shutil.rmtree(config.RECEIVED_DIR)
            os.makedirs(config.RECEIVED_DIR, exist_ok=True)

        # Clear processed data
        if os.path.exists(config.PROCESSED_DIR):
            shutil.rmtree(config.PROCESSED_DIR)
            os.makedirs(config.PROCESSED_DIR, exist_ok=True)

        # Reset mission state
        if _mission_state:
            _mission_state.reset()

        # Tell CubeSat to reset
        if _commander:
            _commander.send_command({"cmd": "reset_mission"})

        logger.info("Mission reset: all data cleared")
        return jsonify({"status": "ok", "message": "Mission data reset"})
    except Exception as e:
        logger.error(f"Mission reset failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/clear_last_pass", methods=["POST"])
def api_clear_last_pass():
    """Remove images and processed data from the most recent pass."""
    try:
        state = _mission_state.get_snapshot() if _mission_state else {}
        total_passes = state.get("total_passes", 0)
        if total_passes < 1:
            return jsonify({"status": "error", "message": "No passes to clear"}), 400

        last_pass = total_passes
        prefix = f"pass{last_pass}_"

        # Remove received images from last pass
        removed_images = 0
        if os.path.exists(config.RECEIVED_DIR):
            for f in os.listdir(config.RECEIVED_DIR):
                if f.startswith(prefix):
                    os.remove(os.path.join(config.RECEIVED_DIR, f))
                    removed_images += 1

        # Remove processed change maps from last pass
        change_maps_dir = os.path.join(config.PROCESSED_DIR, "change_maps")
        if os.path.exists(change_maps_dir):
            for f in os.listdir(change_maps_dir):
                if f"_p{last_pass - 1}vs{last_pass}" in f or f"_p{last_pass}vs" in f:
                    os.remove(os.path.join(change_maps_dir, f))

        # Remove last pass entries from image_index.json
        idx_path = os.path.join(config.PROCESSED_DIR, "image_index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                idx = json.load(f)
            for key in idx:
                idx[key] = [e for e in idx[key] if e.get("pass") != last_pass]
            with open(idx_path, "w") as f:
                json.dump(idx, f, indent=2)

        # Decrement pass counter in mission state
        if _mission_state:
            snap = _mission_state.get_snapshot()
            # We can't directly modify — reset and rebuild would be complex,
            # so just adjust total_passes via internal state
            with _mission_state._lock:
                _mission_state._state["total_passes"] = max(0, total_passes - 1)
                _mission_state._state["total_images_received"] = max(
                    0, _mission_state._state["total_images_received"] - removed_images)
            _mission_state.save()

        logger.info(f"Cleared last pass (pass {last_pass}): {removed_images} images removed")
        return jsonify({
            "status": "ok",
            "message": f"Pass {last_pass} cleared ({removed_images} images removed)",
            "new_total_passes": max(0, total_passes - 1),
        })
    except Exception as e:
        logger.error(f"Clear last pass failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/export_mission")
def api_export_mission():
    """Export a formatted PDF mission report."""
    try:
        buf = _build_mission_pdf()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return send_file(
            buf, mimetype="application/pdf", as_attachment=True,
            download_name=f"MuraltZ_Mission_Report_{ts}.pdf",
        )
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# PDF Report Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_mission_pdf() -> io.BytesIO:
    """Build a clean, formatted PDF mission report from current mission data."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
        PageBreak, HRFlowable,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=22, spaceAfter=4,
        textColor=colors.HexColor("#0a2540"),
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading2"], fontSize=14,
        textColor=colors.HexColor("#0a2540"), spaceBefore=16, spaceAfter=6,
        borderWidth=0, borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        "SubHead", parent=styles["Heading3"], fontSize=11,
        textColor=colors.HexColor("#333333"), spaceBefore=10, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "BodyText2", parent=styles["BodyText"], fontSize=10,
        textColor=colors.HexColor("#222222"), leading=14,
    ))
    styles.add(ParagraphStyle(
        "SmallGray", parent=styles["BodyText"], fontSize=8,
        textColor=colors.HexColor("#888888"),
    ))

    story = []

    # ── Load data ──
    state = _mission_state.get_snapshot() if _mission_state else {}
    cost_grid_data = {}
    cost_grid_path = os.path.join(config.PROCESSED_DIR, "cost_grid.json")
    if os.path.exists(cost_grid_path):
        with open(cost_grid_path) as f:
            cost_grid_data = json.load(f)

    changes_data = {}
    changes_path = os.path.join(config.PROCESSED_DIR, "changes.json")
    if os.path.exists(changes_path):
        with open(changes_path) as f:
            changes_data = json.load(f)

    shadow_data = {}
    shadow_path = os.path.join(config.PROCESSED_DIR, "shadow_data.json")
    if os.path.exists(shadow_path):
        with open(shadow_path) as f:
            shadow_data = json.load(f)

    ts_now = datetime.now().strftime("%B %d, %Y  %H:%M:%S")

    # ── Title page ──
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("MuraltZ CubeSat", styles["ReportTitle"]))
    story.append(Paragraph("Mission Report", styles["ReportTitle"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="60%", thickness=2, color=colors.HexColor("#00d4ff")))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(f"Generated: {ts_now}", styles["BodyText2"]))
    last_updated = state.get("last_updated", "N/A")
    if last_updated and last_updated != "N/A":
        try:
            dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            last_updated = dt.strftime("%B %d, %Y  %H:%M:%S UTC")
        except Exception:
            pass
    story.append(Paragraph(f"Mission data as of: {last_updated}", styles["BodyText2"]))
    story.append(PageBreak())

    # ── 1. Mission Overview ──
    story.append(Paragraph("1. Mission Overview", styles["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))

    overview_data = [
        ["Parameter", "Value"],
        ["Total Passes", str(state.get("total_passes", 0))],
        ["Images Received", str(state.get("total_images_received", 0))],
        ["Images Corrupted", str(state.get("total_images_corrupted", 0))],
        ["Avg CubeSat Quality Score", f"{state.get('quality', {}).get('avg_cubesat_score', 0):.3f}"],
        ["Ground-Flagged Images", str(state.get("quality", {}).get("ground_flagged", 0))],
    ]
    flag_reasons = state.get("quality", {}).get("ground_flag_reasons", [])
    if flag_reasons:
        overview_data.append(["Flag Reasons", ", ".join(flag_reasons)])

    story.append(_make_table(overview_data))

    # ── 2. Coverage ──
    story.append(Paragraph("2. Survey Coverage", styles["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))

    cov = state.get("coverage", {})
    filled = cov.get("cells_filled", 0)
    total = cov.get("cells_total", 64)
    pct = cov.get("pct", 0.0)
    story.append(Paragraph(
        f"<b>{filled}</b> of <b>{total}</b> grid cells surveyed (<b>{pct}%</b> coverage)",
        styles["BodyText2"],
    ))

    # Coverage grid visual
    classifications = cost_grid_data.get("classifications", [])
    coverage = cost_grid_data.get("coverage", [])
    if classifications:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Grid Classification Map:", styles["SubHead"]))
        grid_table_data = []
        color_map = {
            "SAFE": colors.HexColor("#1a3a1a"),
            "MODERATE": colors.HexColor("#3a3a1a"),
            "SHADOW": colors.HexColor("#1a1a3a"),
            "HAZARD": colors.HexColor("#3a1a1a"),
            "IMPASSABLE": colors.HexColor("#3a0a0a"),
        }
        text_color_map = {
            "SAFE": colors.HexColor("#44cc44"),
            "MODERATE": colors.HexColor("#cccc44"),
            "SHADOW": colors.HexColor("#6688cc"),
            "HAZARD": colors.HexColor("#cc4444"),
            "IMPASSABLE": colors.HexColor("#ff4444"),
        }
        cell_styles = []
        for r_idx, row in enumerate(classifications):
            grid_row = []
            for c_idx, cls in enumerate(row):
                is_covered = True
                if coverage and r_idx < len(coverage) and c_idx < len(coverage[r_idx]):
                    is_covered = coverage[r_idx][c_idx]
                if is_covered:
                    label = cls[:3]
                else:
                    label = "---"
                grid_row.append(label)
                bg = color_map.get(cls, colors.HexColor("#333333")) if is_covered else colors.HexColor("#1a1a1a")
                cell_styles.append(("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), bg))
                tc = text_color_map.get(cls, colors.white) if is_covered else colors.HexColor("#555555")
                cell_styles.append(("TEXTCOLOR", (c_idx, r_idx), (c_idx, r_idx), tc))
            grid_table_data.append(grid_row)

        if grid_table_data:
            col_w = 0.55 * inch
            t = Table(grid_table_data, colWidths=[col_w] * 8, rowHeights=[0.35 * inch] * len(grid_table_data))
            t.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 0), (-1, -1), "Courier-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#555555")),
            ] + cell_styles))
            story.append(t)

        story.append(Spacer(1, 6))
        legend_items = ["SAF = Safe", "MOD = Moderate", "SHA = Shadow", "HAZ = Hazard", "IMP = Impassable", "--- = Not Surveyed"]
        story.append(Paragraph("Legend: " + "  |  ".join(legend_items), styles["SmallGray"]))

    # ── 3. Hazard Summary ──
    story.append(Paragraph("3. Hazard Classification Summary", styles["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))

    hazards = state.get("hazards", {})
    hazard_data = [
        ["Classification", "Count"],
        ["Safe", str(hazards.get("safe", 0))],
        ["Moderate", str(hazards.get("moderate", 0))],
        ["Shadow", str(hazards.get("shadow", 0))],
        ["Hazard", str(hazards.get("hazard", 0))],
        ["Impassable", str(hazards.get("impassable", 0))],
    ]
    story.append(_make_table(hazard_data))

    # Confidence grid
    confidences = cost_grid_data.get("confidences", [])
    if confidences:
        flat = [c for row in confidences for c in row if isinstance(c, (int, float)) and c > 0]
        if flat:
            avg_conf = sum(flat) / len(flat)
            min_conf = min(flat)
            max_conf = max(flat)
            story.append(Paragraph(
                f"Classification confidence: avg={avg_conf:.2f}, min={min_conf:.2f}, max={max_conf:.2f}",
                styles["BodyText2"],
            ))

    # ── 4. Shadow Detection ──
    story.append(Paragraph("4. Shadow Detection", styles["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))

    shadow_pct = shadow_data.get("shadow_pct", 0)
    regions = shadow_data.get("regions", [])
    shadows = [r for r in regions if r.get("type") == "shadow"]
    objects = [r for r in regions if r.get("type") == "object"]

    shadow_summary = [
        ["Metric", "Value"],
        ["Shadow Coverage", f"{shadow_pct:.1f}%"],
        ["Shadow Regions", str(len(shadows))],
        ["Dark Objects", str(len(objects))],
    ]
    if shadows:
        largest = max(shadows, key=lambda r: r.get("area_px", 0))
        shadow_summary.append(["Largest Shadow Region", f"{largest.get('area_px', 0)} px"])
    story.append(_make_table(shadow_summary))

    if regions:
        story.append(Paragraph("Detected Regions:", styles["SubHead"]))
        region_header = ["ID", "Type", "Area (px)", "Gradient"]
        region_rows = [region_header]
        for r in regions[:10]:
            region_rows.append([
                str(r.get("id", "")),
                r.get("type", ""),
                str(r.get("area_px", "")),
                f"{r.get('mean_boundary_gradient', 0):.1f}",
            ])
        story.append(_make_table(region_rows))

    # ── 5. Change Detection ──
    story.append(Paragraph("5. Change Detection", styles["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))

    ch = state.get("changes", {})
    change_summary = [
        ["Metric", "Value"],
        ["Total Change Events", str(ch.get("total_events", 0))],
        ["Total Changed Area", f"{ch.get('total_changed_area_cm2', 0):.1f} cm\u00b2"],
        ["Largest Change", f"{ch.get('largest_change_cm2', 0):.1f} cm\u00b2"],
        ["Darkened Events", str(ch.get("types", {}).get("darkened", 0))],
        ["Brightened Events", str(ch.get("types", {}).get("brightened", 0))],
        ["Alignment Warnings", str(ch.get("alignment_warnings", 0))],
    ]
    cells_with = ch.get("cells_with_changes", [])
    if cells_with:
        change_summary.append(["Affected Cells", ", ".join(f"({c[0]},{c[1]})" for c in cells_with)])
    story.append(_make_table(change_summary))

    # Individual events
    events = changes_data.get("events", [])
    if events:
        story.append(Paragraph("Change Events Detail:", styles["SubHead"]))
        evt_header = ["ID", "Cell", "Type", "Area (px)", "SSIM", "Persist"]
        evt_rows = [evt_header]
        for evt in events:
            cell = evt.get("cell", [])
            evt_rows.append([
                str(evt.get("id", "")),
                f"({cell[0]},{cell[1]})" if len(cell) == 2 else "",
                evt.get("type", ""),
                str(evt.get("area_px", "")),
                f"{evt.get('ssim_score', 0):.3f}" if evt.get("ssim_score") else "--",
                "Yes" if evt.get("persistence") else "No",
            ])
        story.append(_make_table(evt_rows))

    # ── 6. Route Planning ──
    story.append(PageBreak())
    story.append(Paragraph("6. Route Planning", styles["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))

    route = state.get("route", {})
    route_summary = [
        ["Metric", "Value"],
        ["Start Cell", str(route.get("start", []))],
        ["End Cell", str(route.get("end", []))],
        ["Status", route.get("status", "N/A")],
        ["Path Length", f"{route.get('path_length', 0)} cells"],
        ["Total Cost", f"{route.get('total_cost', 0):.1f}"],
        ["Shadow Exposure", f"{route.get('shadow_exposure_pct', 0):.1f}%"],
    ]
    story.append(_make_table(route_summary))

    # Route comparison
    routes = state.get("routes", {})
    route_names = ["fastest", "safest", "balanced"]
    route_comparison = [["", "Fastest", "Safest", "Balanced"]]
    has_routes = any(routes.get(n) for n in route_names)

    if has_routes:
        story.append(Paragraph("Route Comparison:", styles["SubHead"]))
        metrics = [
            ("Path Length", "path_length_cells", "{} cells"),
            ("Distance", "distance_cm", "{:.0f} cm"),
            ("Total Cost", "total_cost", "{:.1f}"),
            ("Shadow Exposure", "max_shadow_exposure_pct", "{:.1f}%"),
            ("Hazards Near Path", "hazards_near_path", "{}"),
            ("Risk Level", "risk_level", "{}"),
        ]
        for label, key, fmt in metrics:
            row = [label]
            for name in route_names:
                rd = routes.get(name) or {}
                val = rd.get(key)
                if val is not None:
                    try:
                        row.append(fmt.format(val))
                    except Exception:
                        row.append(str(val))
                else:
                    row.append("--")
            route_comparison.append(row)

        selected = routes.get("selected", "")
        route_comparison.append(["Selected", "\u2713" if selected == "fastest" else "",
                                  "\u2713" if selected == "safest" else "",
                                  "\u2713" if selected == "balanced" else ""])
        story.append(_make_table(route_comparison))

    # Route map image
    route_img_path = os.path.join(config.PROCESSED_DIR, "routes", "route_comparison.png")
    if not os.path.exists(route_img_path):
        route_img_path = os.path.join(config.PROCESSED_DIR, "routes", "route_latest.png")
    if os.path.exists(route_img_path):
        story.append(Spacer(1, 8))
        story.append(Paragraph("Route Map:", styles["SubHead"]))
        try:
            img = Image(route_img_path)
            img_w = min(5.5 * inch, img.drawWidth)
            scale = img_w / img.drawWidth
            img.drawWidth = img_w
            img.drawHeight = img.drawHeight * scale
            story.append(img)
        except Exception:
            pass

    # ── 7. Downlink / Uplink ──
    story.append(Paragraph("7. Communication Statistics", styles["SectionHead"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))

    dl = state.get("downlink", {})
    ul = state.get("uplink", {})
    comm_data = [
        ["Metric", "Value"],
        ["Total Downlinked", f"{dl.get('total_bytes', 0):,} bytes"],
        ["Transfer Time", f"{dl.get('total_time_sec', 0):.1f} sec"],
        ["Effective Rate", f"{dl.get('effective_rate_bps', 0):.0f} B/s"],
        ["Failed Transfers", str(dl.get("failed_transfers", 0))],
        ["Retransmit Requests", str(dl.get("retransmit_requests", 0))],
        ["Commands Sent", str(ul.get("commands_sent", 0))],
        ["Commands ACK'd", str(ul.get("commands_acked", 0))],
    ]
    story.append(_make_table(comm_data))

    # ── 8. Mosaic ──
    mosaic_path = os.path.join(config.PROCESSED_DIR, "mosaics", "mosaic_latest.png")
    if os.path.exists(mosaic_path):
        story.append(PageBreak())
        story.append(Paragraph("8. Survey Mosaic", styles["SectionHead"]))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
        try:
            img = Image(mosaic_path)
            img_w = min(6.0 * inch, img.drawWidth)
            scale = img_w / img.drawWidth
            img.drawWidth = img_w
            img.drawHeight = img.drawHeight * scale
            story.append(img)
        except Exception:
            pass

    # ── Footer ──
    story.append(Spacer(1, 0.5 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Paragraph(
        f"MuraltZ Ground Control Station | Report generated {ts_now}",
        styles["SmallGray"],
    ))

    doc.build(story)
    buf.seek(0)
    return buf


def _make_table(data: list) -> Table:
    """Build a styled reportlab Table from a list of rows (first row is header)."""
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle

    t = Table(data, hAlign="LEFT")
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0a2540")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _latest_file(directory: str, pattern: str) -> str | None:
    """Return the absolute path of the most recently modified file matching pattern."""
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        return None
    return os.path.abspath(max(matches, key=os.path.getmtime))


def _read_log_tail(n: int) -> list[str]:
    """
    Read the last n lines from the application log. Returns list of strings.
    Falls back to an empty list if no log file is configured or readable.
    """
    # Find the first FileHandler attached to the root logger
    import logging as _logging
    for handler in _logging.root.handlers:
        if isinstance(handler, _logging.FileHandler):
            try:
                with open(handler.baseFilename, "r", errors="replace") as f:
                    lines = f.readlines()
                return [l.rstrip() for l in lines[-n:]]
            except Exception:
                pass
    return []


def _query_ollama(question: str, mission_state: dict) -> str:
    """
    Send question + mission_state JSON to local ollama (Llama 3.2).
    Returns the response string, or an error message if ollama isn't running.
    """
    try:
        import urllib.request
        import urllib.error

        state_json = json.dumps(mission_state, indent=2)
        system_prompt = (
            "You are a mission analyst for the MuraltZ CubeSat. "
            "The following is the current mission state JSON containing real data. "
            "Answer the operator's question based only on this data. "
            "Do not invent numbers or events not present in the JSON.\n\n"
            f"MISSION STATE:\n{state_json}"
        )

        payload = json.dumps({
            "model": config.LLM_MODEL,
            "prompt": question,
            "system": system_prompt,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "No response from model")

    except Exception as e:
        logger.warning(f"LLM query failed: {e}")
        return f"LLM unavailable: {e}"
