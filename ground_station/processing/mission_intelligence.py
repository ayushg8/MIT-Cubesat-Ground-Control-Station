from __future__ import annotations

from datetime import datetime, timezone

import config


def cell_key(row: int, col: int) -> str:
    return f"{int(row)},{int(col)}"


def parse_cell_key(key: str) -> tuple[int, int]:
    row, col = key.split(",", 1)
    return int(row), int(col)


def empty_cell_state(row: int, col: int) -> dict:
    return {
        "cell": [row, col],
        "observation_count": 0,
        "passes_seen": [],
        "last_pass": 0,
        "last_seen_utc": "",
        "avg_quality": 0.0,
        "hazard_beliefs": {
            "SAFE": 0.0,
            "MODERATE": 0.0,
            "SHADOW": 0.0,
            "HAZARD": 0.0,
            "IMPASSABLE": 0.0,
        },
        "dominant_hazard": "UNSURVEYED",
        "hazard_confidence": 0.0,
        "change_probability": 0.0,
        "change_events": 0,
        "shadow_pct_avg": 0.0,
        "uncertainty": 1.0,
        "route_relevance": 0.0,
        "science_value": 0.0,
        "last_filename": "",
        "last_task_reason": "",
    }


def update_cell_state(existing: dict | None, observation: dict) -> dict:
    row, col = observation["grid_cell"]
    state = dict(existing or empty_cell_state(row, col))

    obs_count = int(state.get("observation_count", 0)) + 1
    quality = float(observation.get("quality_score") or 0.0)
    shadow_pct = float(observation.get("shadow_percentage") or 0.0)
    hazard_class = observation.get("hazard_class") or "SAFE"
    hazard_conf = float(observation.get("hazard_confidence") or 0.5)
    has_change = bool(observation.get("has_change"))
    change_events = int(observation.get("change_events") or 0)
    route_relevance = float(observation.get("route_relevance") or 0.0)
    science_value = float(observation.get("science_value") or 0.0)

    prev_quality = float(state.get("avg_quality", 0.0))
    prev_shadow = float(state.get("shadow_pct_avg", 0.0))
    prev_change_prob = float(state.get("change_probability", 0.0))

    state["observation_count"] = obs_count
    state["avg_quality"] = round(((prev_quality * (obs_count - 1)) + quality) / obs_count, 3)
    state["shadow_pct_avg"] = round(((prev_shadow * (obs_count - 1)) + shadow_pct) / obs_count, 2)
    state["last_pass"] = int(observation.get("pass_number") or state.get("last_pass") or 0)
    state["last_seen_utc"] = observation.get("timestamp") or datetime.now(timezone.utc).isoformat()
    state["last_filename"] = observation.get("filename", "")
    state["last_task_reason"] = observation.get("task_reason", "")
    state["science_value"] = round(max(state.get("science_value", 0.0), science_value), 3)

    passes_seen = set(int(p) for p in state.get("passes_seen", []))
    if state["last_pass"] > 0:
        passes_seen.add(state["last_pass"])
    state["passes_seen"] = sorted(passes_seen)

    hazard_beliefs = dict(state.get("hazard_beliefs", {}))
    for key in ("SAFE", "MODERATE", "SHADOW", "HAZARD", "IMPASSABLE"):
        hazard_beliefs.setdefault(key, 0.0)
        hazard_beliefs[key] *= 0.75
    hazard_beliefs[hazard_class] = round(hazard_beliefs.get(hazard_class, 0.0) + hazard_conf, 3)
    state["hazard_beliefs"] = hazard_beliefs

    dominant_hazard, belief = max(hazard_beliefs.items(), key=lambda item: item[1])
    belief_sum = sum(hazard_beliefs.values()) or 1.0
    state["dominant_hazard"] = dominant_hazard if obs_count > 0 else "UNSURVEYED"
    state["hazard_confidence"] = round(min(1.0, belief / belief_sum), 3)

    change_signal = min(1.0, 0.25 + 0.15 * change_events) if has_change else 0.0
    state["change_probability"] = round(max(prev_change_prob * 0.7, change_signal), 3)
    state["change_events"] = int(state.get("change_events", 0)) + change_events
    state["route_relevance"] = round(max(float(state.get("route_relevance", 0.0)), route_relevance), 3)

    certainty = min(1.0, state["hazard_confidence"] + min(0.4, 0.1 * obs_count))
    if has_change:
        certainty = min(1.0, certainty + 0.1)
    state["uncertainty"] = round(max(0.0, 1.0 - certainty), 3)

    return state


def estimate_route_relevance(state: dict, routes: dict) -> float:
    cell = state.get("cell") or []
    if len(cell) != 2:
        return 0.0

    route_score = 0.0
    for weight, name in ((1.0, "safest"), (0.85, "balanced"), (0.7, "fastest")):
        path = (routes.get(name) or {}).get("path") or []
        if cell in path:
            route_score = max(route_score, weight)
    return round(route_score, 3)


def build_task_queue(snapshot: dict, max_tasks: int | None = None) -> list[dict]:
    max_tasks = max_tasks or config.MISSION_MAX_TASKS
    coverage = snapshot.get("coverage", {})
    rows = int(coverage.get("rows") or 0)
    cols = int(coverage.get("cols") or 0)
    cell_states = snapshot.get("cell_states", {})
    routes = snapshot.get("routes", {})
    tasks: list[dict] = []

    def push_task(cmd: str, row: int, col: int, score: float, reasons: list[str], category: str):
        tasks.append({
            "cmd": cmd,
            "row": row,
            "col": col,
            "score": round(score, 3),
            "reasons": reasons,
            "label": f"{cmd} ({row},{col})",
            "category": category,
        })

    surveyed_cells = {
        tuple(state.get("cell", []))
        for state in cell_states.values()
        if len(state.get("cell", [])) == 2
    }

    def frontier_bonus(row: int, col: int) -> float:
        neighbors = (
            (row - 1, col), (row + 1, col),
            (row, col - 1), (row, col + 1),
        )
        hits = sum(1 for nbr in neighbors if nbr in surveyed_cells)
        return min(0.08, hits * 0.02)

    if rows > 0 and cols > 0:
        for row in range(rows):
            for col in range(cols):
                key = cell_key(row, col)
                state = cell_states.get(key)
                if state is None:
                    score = 0.16 + frontier_bonus(row, col)
                    reasons = ["unsurveyed cell expands coverage"]
                    if frontier_bonus(row, col) > 0:
                        reasons.append("adjacent to an already observed corridor")
                    push_task("observe_cell", row, col, score, reasons, "coverage")
                    continue

                route_relevance = estimate_route_relevance(state, routes)
                uncertainty = float(state.get("uncertainty", 1.0))
                change_prob = float(state.get("change_probability", 0.0))
                hazard = state.get("dominant_hazard", "SAFE")
                obs_count = int(state.get("observation_count", 0))
                quality = float(state.get("avg_quality", 0.0))
                hazard_pressure = 1.0 if hazard in ("HAZARD", "IMPASSABLE", "SHADOW") else 0.2
                revisit_pressure = 0.18 if obs_count <= 1 else 0.08
                quality_penalty = max(0.0, 0.75 - quality)

                score = (
                    0.34 * uncertainty +
                    0.22 * change_prob +
                    0.20 * route_relevance +
                    0.18 * hazard_pressure +
                    revisit_pressure +
                    0.12 * quality_penalty
                )
                reasons = []
                if uncertainty >= 0.4:
                    reasons.append("low-confidence terrain estimate")
                if change_prob >= 0.3:
                    reasons.append("change signal needs confirmation")
                if route_relevance >= 0.5:
                    reasons.append("lies on a likely route corridor")
                if hazard in ("HAZARD", "IMPASSABLE", "SHADOW"):
                    reasons.append(f"hazard belief is {hazard.lower()}")
                if obs_count <= 1:
                    reasons.append("only one observation so far")
                if quality_penalty > 0.1:
                    reasons.append("existing observation quality is weak")
                if not reasons:
                    reasons.append("refresh existing terrain estimate")
                push_task("revisit_cell", row, col, score, reasons, "revisit")

    tasks.sort(key=lambda task: task["score"], reverse=True)
    deduped: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    coverage_count = 0
    for task in tasks:
        sig = (task["cmd"], task["row"], task["col"])
        if sig in seen:
            continue
        if task.get("category") == "coverage" and coverage_count >= 2 and len(tasks) > max_tasks:
            continue
        seen.add(sig)
        if task.get("category") == "coverage":
            coverage_count += 1
        deduped.append(task)
        if len(deduped) >= max_tasks:
            break
    return deduped


def build_mission_metrics(snapshot: dict) -> dict:
    coverage = snapshot.get("coverage", {})
    downlink = snapshot.get("downlink", {})
    changes = snapshot.get("changes", {})
    quality = snapshot.get("quality", {})
    total_images = int(snapshot.get("total_images_received", 0))
    covered = int(coverage.get("cells_filled", 0))
    total = int(coverage.get("cells_total", 0)) or 1
    bytes_sent = int(downlink.get("total_bytes", 0))

    return {
        "coverage_efficiency": round(covered / max(1, total_images), 3) if total_images else 0.0,
        "bytes_per_surveyed_cell": round(bytes_sent / max(1, covered), 1) if covered else 0.0,
        "change_density": round(changes.get("total_events", 0) / total, 3),
        "ground_flag_rate": round(quality.get("ground_flagged", 0) / max(1, total_images), 3),
        "survey_completion": round(covered / total, 3),
    }


def build_briefing(snapshot: dict) -> dict:
    coverage = snapshot.get("coverage", {})
    changes = snapshot.get("changes", {})
    hazards = snapshot.get("hazards", {})
    tasks = snapshot.get("task_queue", [])
    metrics = snapshot.get("mission_metrics", {})
    top_task = tasks[0] if tasks else None

    headline = (
        f"Surveyed {coverage.get('cells_filled', 0)} of {coverage.get('cells_total', 0)} cells "
        f"({coverage.get('pct', 0.0)}%)."
    )
    if top_task:
        headline += f" Next best action is {top_task['cmd']} at ({top_task['row']},{top_task['col']})."

    bullets = [
        f"Top hazard burden: {max(hazards, key=hazards.get) if hazards else 'unknown'}",
        f"Detected {changes.get('total_events', 0)} total change event(s)",
        f"Coverage efficiency is {metrics.get('coverage_efficiency', 0.0):.3f} surveyed cells per image",
    ]
    if top_task:
        bullets.append("Task rationale: " + "; ".join(top_task.get("reasons", [])))

    return {
        "headline": headline,
        "bullets": bullets,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
