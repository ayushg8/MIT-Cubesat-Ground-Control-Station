from __future__ import annotations

from datetime import datetime, timezone

import config


def _label_quality(avg_quality: float) -> str:
    if avg_quality >= 0.82:
        return "strong"
    if avg_quality >= 0.62:
        return "usable"
    return "weak"


def _label_uncertainty(uncertainty: float) -> str:
    if uncertainty >= 0.65:
        return "high"
    if uncertainty >= 0.35:
        return "medium"
    return "low"


def _priority_band(score: float) -> str:
    if score >= 0.44:
        return "critical"
    if score >= 0.32:
        return "high"
    if score >= 0.22:
        return "medium"
    return "low"


def _task_family(task: dict) -> str:
    return task.get("family") or task.get("category") or "review"


def _task_diversity_rank(task: dict) -> tuple[int, float]:
    family_order = {
        "change_confirmation": 0,
        "hazard_confirmation": 1,
        "route_support": 2,
        "uncertainty_reduction": 3,
        "coverage": 4,
        "refresh": 5,
    }
    return (family_order.get(_task_family(task), 99), -float(task.get("score", 0.0)))


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

    def push_task(
        cmd: str,
        row: int,
        col: int,
        score: float,
        reasons: list[str],
        category: str,
        *,
        family: str,
        why_now: str,
        expected_gain: str,
        confidence_label: str,
        observation_strength: str,
        freshness: str,
    ):
        tasks.append({
            "cmd": cmd,
            "row": row,
            "col": col,
            "score": round(score, 3),
            "reasons": reasons,
            "label": f"{cmd} ({row},{col})",
            "category": category,
            "family": family,
            "priority_band": _priority_band(score),
            "why_now": why_now,
            "expected_gain": expected_gain,
            "confidence_label": confidence_label,
            "observation_strength": observation_strength,
            "freshness": freshness,
            "command_preview": f"{cmd} ({row},{col})",
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
                    push_task(
                        "observe_cell",
                        row,
                        col,
                        score,
                        reasons,
                        "coverage",
                        family="coverage",
                        why_now="This is a clean first-look target and expands the surveyed frontier.",
                        expected_gain="Adds new map coverage and reduces blind spots near the current corridor.",
                        confidence_label="unknown",
                        observation_strength="unseen",
                        freshness="not yet observed",
                    )
                    continue

                route_relevance = estimate_route_relevance(state, routes)
                uncertainty = float(state.get("uncertainty", 1.0))
                change_prob = float(state.get("change_probability", 0.0))
                hazard = state.get("dominant_hazard", "SAFE")
                obs_count = int(state.get("observation_count", 0))
                quality = float(state.get("avg_quality", 0.0))
                quality_label = _label_quality(quality)
                uncertainty_label = _label_uncertainty(uncertainty)
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
                if change_prob >= 0.35:
                    family = "change_confirmation"
                    why_now = "This cell has a live change signal that is still based on limited evidence."
                    expected_gain = "A revisit can confirm or clear a suspected surface change before it propagates into route decisions."
                elif hazard in ("HAZARD", "IMPASSABLE", "SHADOW"):
                    family = "hazard_confirmation"
                    why_now = f"This cell is currently believed to be {hazard.lower()} but the evidence is not mature."
                    expected_gain = "A stronger revisit will sharpen the hazard boundary and reduce false route penalties nearby."
                elif route_relevance >= 0.5:
                    family = "route_support"
                    why_now = "This cell sits on a likely route corridor and weak evidence here can distort planning."
                    expected_gain = "Improves route confidence by resolving terrain quality on a high-impact corridor."
                elif uncertainty >= 0.45:
                    family = "uncertainty_reduction"
                    why_now = "This cell remains one of the least certain parts of the current map."
                    expected_gain = "Reduces uncertainty in the fused terrain map and improves confidence calibration."
                else:
                    family = "refresh"
                    why_now = "This cell benefits from a refresh because the current observation is aging or weak."
                    expected_gain = "Improves local map quality and supports more stable downstream planning."

                freshness = "single observation" if obs_count <= 1 else f"{obs_count} observations"
                push_task(
                    "revisit_cell",
                    row,
                    col,
                    score,
                    reasons,
                    "revisit",
                    family=family,
                    why_now=why_now,
                    expected_gain=expected_gain,
                    confidence_label=uncertainty_label,
                    observation_strength=quality_label,
                    freshness=freshness,
                )

    tasks.sort(key=lambda task: task["score"], reverse=True)
    per_family: dict[str, list[dict]] = {}
    for task in tasks:
        per_family.setdefault(_task_family(task), []).append(task)

    selected: list[dict] = []
    seen_cells: set[tuple[int, int]] = set()
    for family, family_tasks in sorted(per_family.items(), key=lambda item: _task_diversity_rank(item[1][0])):
        top = family_tasks[0]
        cell_sig = (top["row"], top["col"])
        if cell_sig in seen_cells:
            continue
        selected.append(top)
        seen_cells.add(cell_sig)
        if len(selected) >= max_tasks:
            break

    if len(selected) < max_tasks:
        for task in tasks:
            cell_sig = (task["row"], task["col"])
            if cell_sig in seen_cells:
                continue
            if task.get("category") == "coverage":
                coverage_already = sum(1 for item in selected if item.get("category") == "coverage")
                if coverage_already >= 1 and len(tasks) > max_tasks:
                    continue
            selected.append(task)
            seen_cells.add(cell_sig)
            if len(selected) >= max_tasks:
                break

    selected.sort(key=lambda task: task["score"], reverse=True)
    return selected


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
    cell_states = snapshot.get("cell_states", {})

    hazard_cells = [
        state for state in cell_states.values()
        if state.get("dominant_hazard") in ("HAZARD", "IMPASSABLE", "SHADOW")
    ]
    weak_cells = [
        state for state in cell_states.values()
        if float(state.get("uncertainty", 1.0)) >= 0.45
    ]
    route_cells = [
        state for state in cell_states.values()
        if float(estimate_route_relevance(state, snapshot.get("routes", {}))) >= 0.5
    ]

    headline = (
        f"Surveyed {coverage.get('cells_filled', 0)} of {coverage.get('cells_total', 0)} cells "
        f"({coverage.get('pct', 0.0)}%)."
    )
    if top_task:
        headline += f" Recommend {top_task['cmd']} at ({top_task['row']},{top_task['col']}) next."

    mission_status = (
        f"Survey phase is {'early' if coverage.get('pct', 0.0) < 20 else 'developing' if coverage.get('pct', 0.0) < 60 else 'mature'}; "
        f"{len(hazard_cells)} cells currently carry elevated hazard belief."
    )
    recommended_action = (
        f"Send `{top_task['command_preview']}` now."
        if top_task else
        "No immediate follow-up task is available."
    )
    why_now = top_task.get("why_now") if top_task else "Continue collecting observations to build mission state."
    expected_payoff = top_task.get("expected_gain") if top_task else "Additional observations will improve map coverage."
    ai_confidence = (
        f"Map uncertainty is concentrated in {len(weak_cells)} cell(s); "
        f"{len(route_cells)} cell(s) are currently route-relevant."
    )

    bullets = [
        mission_status,
        f"Recommended action: {recommended_action}",
        f"Why now: {why_now}",
        f"Expected payoff: {expected_payoff}",
        f"AI confidence: {ai_confidence}",
        f"Mission metrics: {changes.get('total_events', 0)} change event(s), "
        f"{metrics.get('coverage_efficiency', 0.0):.3f} cells/image, "
        f"{metrics.get('bytes_per_surveyed_cell', 0.0):.1f} bytes/surveyed cell.",
    ]

    return {
        "headline": headline,
        "bullets": bullets,
        "recommended_action": recommended_action,
        "why_now": why_now,
        "expected_payoff": expected_payoff,
        "ai_confidence": ai_confidence,
        "suggested_questions": [
            "What changed this pass?",
            "Why is this the top task?",
            "Which cells most affect route safety?",
            "What should we do if bandwidth is limited?",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
