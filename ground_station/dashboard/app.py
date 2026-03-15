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
import zipfile
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
    """Return 8×8 coverage grid as JSON for the canvas panel."""
    rows = config.GRID_ROWS
    cols = config.GRID_COLS

    if _pipeline is not None:
        hazard_grid = _pipeline.get_hazard_grid()
        cost_grid   = _pipeline.get_cost_grid()
    else:
        hazard_grid = [["SAFE"] * cols for _ in range(rows)]
        cost_grid   = None

    state = _mission_state.get_snapshot() if _mission_state else {}
    changed_cells = [tuple(c) for c in state.get("changes", {}).get("cells_with_changes", [])]

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
    """Return 8×8 JSON array: {row, col, cost, hazard_class} per cell."""
    rows = config.GRID_ROWS
    cols = config.GRID_COLS

    if _pipeline is not None:
        hazard_grid = _pipeline.get_hazard_grid()
        cost_grid   = _pipeline.get_cost_grid()
    else:
        hazard_grid = [["SAFE"] * cols for _ in range(rows)]
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
    return jsonify({"grid": [], "classifications": [], "coverage": [], "pass_data": [], "change_cells": []})


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


@app.route("/api/plan_routes", methods=["POST"])
def api_plan_routes():
    """Body: {start: [row, col], end: [row, col]} → run plan_multiple_routes."""
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    start = tuple(body.get("start", list(config.ROUTE_START)))
    end = tuple(body.get("end", list(config.ROUTE_END)))

    # Validate coordinates
    for label, pt in [("start", start), ("end", end)]:
        if len(pt) != 2 or not (0 <= pt[0] < config.GRID_ROWS and 0 <= pt[1] < config.GRID_COLS):
            return jsonify({"error": f"Invalid {label}: {pt}"}), 400

    if _pipeline is None:
        return jsonify({"error": "Pipeline not ready"}), 503

    cost_grid = _pipeline.get_cost_grid()
    hazard_grid = _pipeline.get_hazard_grid()
    hazard_map_path = _pipeline.get_latest_hazard_map_path() if hasattr(_pipeline, 'get_latest_hazard_map_path') else None

    try:
        routes = _pipeline._route_planner.plan_multiple_routes(
            cost_grid, hazard_grid, start, end, hazard_map_path,
        )
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
        return jsonify(routes)
    except Exception as e:
        logger.error(f"plan_routes failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/plan_constrained", methods=["POST"])
def api_plan_constrained():
    """Body: {max_shadow_pct, min_hazard_clearance, start?, end?} → route JSON."""
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    max_shadow_pct = float(body.get("max_shadow_pct", 50.0))
    min_hazard_clearance = int(body.get("min_hazard_clearance", 1))
    start = tuple(body.get("start", list(config.ROUTE_START)))
    end = tuple(body.get("end", list(config.ROUTE_END)))

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
    """Export mission_state.json + all processed files as a zip download."""
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # mission_state.json
            if os.path.exists(config.MISSION_STATE_FILE):
                zf.write(config.MISSION_STATE_FILE, "mission_state.json")

            # All processed files
            if os.path.exists(config.PROCESSED_DIR):
                for root, dirs, files in os.walk(config.PROCESSED_DIR):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        arcname = os.path.join("processed",
                                               os.path.relpath(fpath, config.PROCESSED_DIR))
                        zf.write(fpath, arcname)

            # Received images
            if os.path.exists(config.RECEIVED_DIR):
                for fname in os.listdir(config.RECEIVED_DIR):
                    fpath = os.path.join(config.RECEIVED_DIR, fname)
                    if os.path.isfile(fpath):
                        zf.write(fpath, os.path.join("images", fname))

        buf.seek(0)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return send_file(
            buf, mimetype="application/zip", as_attachment=True,
            download_name=f"mission_export_{ts}.zip",
        )
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


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
