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
import json
import logging
import os
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request, send_file

import config
from receiver import telemetry_parser
from uplink.commander import Commander

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
    return render_template("index.html")


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
    """Return fastest/safest/balanced route dicts from mission state."""
    state = _mission_state.get_snapshot() if _mission_state else {}
    routes = state.get("routes", {})
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


@app.route("/api/plan_constrained", methods=["POST"])
def api_plan_constrained():
    """Body: {max_shadow_pct, min_hazard_clearance} → route JSON."""
    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    max_shadow_pct = float(body.get("max_shadow_pct", 50.0))
    min_hazard_clearance = int(body.get("min_hazard_clearance", 1))

    if _pipeline is None:
        return jsonify({"error": "Pipeline not ready"}), 503

    cost_grid   = _pipeline.get_cost_grid()
    hazard_grid = _pipeline.get_hazard_grid()

    try:
        result = _pipeline._route_planner.plan_with_constraints(
            cost_grid, hazard_grid,
            config.ROUTE_START, config.ROUTE_END,
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
