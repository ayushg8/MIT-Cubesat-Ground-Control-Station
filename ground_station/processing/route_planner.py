from __future__ import annotations
# processing/route_planner.py — A* pathfinding on the hazard cost grid
#
# Input:  8×8 numpy cost_grid from hazard_classifier (one cost per cell).
# Output: shortest path from ROUTE_START to ROUTE_END avoiding IMPASSABLE cells,
#         plus route map images.
#
# 8-connected A*: diagonal moves cost sqrt(2) * destination_cell_cost.
# Heuristic: Manhattan distance (fast; slightly inadmissible for diagonals but
# good enough for an 8×8 grid — the path quality difference is negligible).
#
# IMPASSABLE cells (cost=999) are treated as walls and never entered.
# If no path exists → status="no viable route". This is a valid demo result
# (it shows the system correctly identifies blocked terrain).

import heapq
import json
import logging
import math
import os

import cv2
import numpy as np

import config
from processing.hazard_classifier import (
    SAFE, MODERATE, SHADOW, HAZARD, IMPASSABLE,
    _COLOURS as HAZARD_COLOURS,
)

logger = logging.getLogger(__name__)

# Route segment colours (BGR) — drawn on top of the hazard map
_PATH_COLOUR = {
    SAFE:       (0,   220,   0),   # bright green
    MODERATE:   (0,   165, 255),   # orange
    SHADOW:     (220, 100,   0),   # light blue
    HAZARD:     (0,     0, 255),   # red — bad but we went through it
    IMPASSABLE: (0,     0, 128),   # shouldn't appear, guard value
}
_DEFAULT_PATH_COLOUR = (255, 255, 255)

# Multi-route colours (BGR)
_ROUTE_COLOURS_BGR = {
    "fastest":  (136, 255,   0),   # #00ff88 in BGR
    "safest":   (0,   170, 255),   # #ffaa00 in BGR
    "balanced": (68,  68,  255),   # #ff4444 in BGR
}
_ROUTE_COLOURS_HEX = {
    "fastest":  "#00ff88",
    "safest":   "#ffaa00",
    "balanced": "#ff4444",
}
_ROUTE_NAMES = {
    "fastest":  "Fastest",
    "safest":   "Safest",
    "balanced": "Balanced",
}

_SQRT2 = math.sqrt(2)
_CARDINAL_DIRS  = [(-1,0),(1,0),(0,-1),(0,1)]
_DIAGONAL_DIRS  = [(-1,-1),(-1,1),(1,-1),(1,1)]
_ALL_DIRS = _CARDINAL_DIRS + _DIAGONAL_DIRS


class RoutePlanner:

    def plan(
        self,
        cost_grid: np.ndarray,
        hazard_grid: list[list[str]] | None,
        start: tuple,
        end: tuple,
        hazard_map_path: str | None = None,
    ) -> dict:
        """
        Backward-compatible single-route plan. Internally calls plan_multiple_routes
        and returns the fastest route result in legacy format.
        """
        routes = self.plan_multiple_routes(cost_grid, hazard_grid, start, end, hazard_map_path)
        fastest = routes.get("fastest", {})
        raw_status = fastest.get("status", "no_path")
        # Translate internal status to legacy status string
        legacy_status = "found" if raw_status == "found" else "no viable route"
        return {
            "path":               fastest.get("path", []),
            "total_cost":         fastest.get("total_cost", 0.0),
            "path_length":        fastest.get("path_length_cells", 0),
            "shadow_exposure_pct": fastest.get("max_shadow_exposure_pct", 0.0),
            "status":             legacy_status,
            "route_map_path":     fastest.get("route_map_path", ""),
        }

    def plan_multiple_routes(
        self,
        cost_grid: np.ndarray,
        hazard_grid: list[list[str]] | None,
        start: tuple,
        end: tuple,
        hazard_map_path: str | None = None,
    ) -> dict:
        """
        Run A* 3 times with different cost modifiers.

        Returns:
            {
                "fastest":  { ...route_result... },
                "safest":   { ...route_result... },
                "balanced": { ...route_result... },
            }
        """
        rows, cols = cost_grid.shape

        # Build modified cost grids
        fastest_grid  = cost_grid.copy()
        safest_grid   = _build_safest_grid(cost_grid, hazard_grid, rows, cols)
        balanced_grid = _build_balanced_grid(cost_grid, hazard_grid, rows, cols)

        routes = {}
        for name, grid in [("fastest", fastest_grid), ("safest", safest_grid), ("balanced", balanced_grid)]:
            path = _astar(grid, start, end, rows, cols)
            routes[name] = _build_route_result(
                name, path, cost_grid, hazard_grid, rows, cols
            )
            logger.info(
                f"RoutePlanner [{name}]: status={routes[name]['status']} "
                f"length={routes[name]['path_length_cells']} "
                f"cost={routes[name]['total_cost']}"
            )

        # Save comparison image + individual maps
        _save_route_comparison(cost_grid, hazard_grid, routes, start, end, hazard_map_path)
        for name in ("fastest", "safest", "balanced"):
            path = routes[name].get("path", [])
            out = _save_individual_route_map(
                cost_grid, hazard_grid, path, start, end, hazard_map_path, name
            )
            routes[name]["route_map_path"] = out

        # Save JSON data
        _save_routes_json(routes, start, end)

        return routes

    def plan_with_constraints(
        self,
        cost_grid: np.ndarray,
        hazard_grid: list[list[str]] | None,
        start: tuple,
        end: tuple,
        max_shadow_pct: float,
        min_hazard_clearance: int,
    ) -> dict:
        """
        Plan a route with hard constraints.
        - Cells within min_hazard_clearance of a HAZARD cell → IMPASSABLE
        - Reject if shadow_exposure_pct > max_shadow_pct
        Returns route dict with status="found" or status="no_feasible_path".
        """
        rows, cols = cost_grid.shape
        modified = cost_grid.copy()

        # Mark cells within clearance of HAZARD as IMPASSABLE
        if hazard_grid and min_hazard_clearance > 0:
            hazard_cells = [
                (r, c) for r in range(rows) for c in range(cols)
                if hazard_grid[r][c] == HAZARD
            ]
            for hr, hc in hazard_cells:
                for r in range(rows):
                    for c in range(cols):
                        dist = math.sqrt((r - hr)**2 + (c - hc)**2)
                        if dist <= min_hazard_clearance:
                            modified[r, c] = config.COST_IMPASSABLE

        path = _astar(modified, start, end, rows, cols)
        if path is None:
            return {
                "name": "Constrained",
                "path": [],
                "total_cost": 0.0,
                "path_length_cells": 0,
                "distance_cm": 0.0,
                "max_shadow_exposure_pct": 0.0,
                "hazards_near_path": 0,
                "nearest_hazard_distance_cells": 0.0,
                "risk_level": "HIGH",
                "color": "#ff4444",
                "status": "no_feasible_path",
            }

        result = _build_route_result("constrained", path, cost_grid, hazard_grid, rows, cols)

        if result["max_shadow_exposure_pct"] > max_shadow_pct:
            result["status"] = "no_feasible_path"

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Cost grid modifiers
# ─────────────────────────────────────────────────────────────────────────────

def _build_safest_grid(
    cost_grid: np.ndarray,
    hazard_grid: list[list[str]] | None,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Shadow cells: cost*3. Cells adjacent to HAZARD: cost+10."""
    grid = cost_grid.astype(np.float64).copy()
    if hazard_grid:
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] >= config.COST_IMPASSABLE:
                    continue
                if hazard_grid[r][c] == SHADOW:
                    grid[r, c] = grid[r, c] * 3
                # Adjacent to HAZARD
                for dr, dc in _ALL_DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and hazard_grid[nr][nc] == HAZARD:
                        grid[r, c] = grid[r, c] + 10
                        break
    return grid.astype(np.int32)


def _build_balanced_grid(
    cost_grid: np.ndarray,
    hazard_grid: list[list[str]] | None,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Shadow cells: cost*1.5. Cells adjacent to HAZARD: cost+5."""
    grid = cost_grid.astype(np.float64).copy()
    if hazard_grid:
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] >= config.COST_IMPASSABLE:
                    continue
                if hazard_grid[r][c] == SHADOW:
                    grid[r, c] = grid[r, c] * 1.5
                for dr, dc in _ALL_DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and hazard_grid[nr][nc] == HAZARD:
                        grid[r, c] = grid[r, c] + 5
                        break
    return grid.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Route result builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_route_result(
    name: str,
    path: list | None,
    cost_grid: np.ndarray,
    hazard_grid: list[list[str]] | None,
    rows: int,
    cols: int,
) -> dict:
    colour_map = {"fastest": "#00ff88", "safest": "#ffaa00", "balanced": "#ff4444"}
    colour = colour_map.get(name, "#ffffff")
    display_name = _ROUTE_NAMES.get(name, name.capitalize())

    if path is None:
        return {
            "name": display_name,
            "path": [],
            "total_cost": 0.0,
            "path_length_cells": 0,
            "distance_cm": 0.0,
            "max_shadow_exposure_pct": 0.0,
            "hazards_near_path": 0,
            "nearest_hazard_distance_cells": 0.0,
            "risk_level": "HIGH",
            "color": colour,
            "status": "no_path",
            "route_map_path": "",
        }

    total_cost = _path_cost(path, cost_grid)
    path_len = len(path)

    shadow_cells = 0
    if hazard_grid:
        shadow_cells = sum(1 for (r, c) in path if hazard_grid[r][c] == SHADOW)
    shadow_pct = (shadow_cells / path_len * 100.0) if path_len else 0.0

    path_set = set(path)
    hazards_near = 0
    nearest_hazard_dist = float("inf")
    if hazard_grid:
        hazard_positions = [
            (r, c) for r in range(rows) for c in range(cols)
            if hazard_grid[r][c] == HAZARD
        ]
        near_set = set()
        for pr, pc in path:
            for dr, dc in _ALL_DIRS:
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in path_set:
                    if hazard_grid[nr][nc] == HAZARD:
                        near_set.add((nr, nc))
        hazards_near = len(near_set)

        for pr, pc in path:
            for hr, hc in hazard_positions:
                d = math.sqrt((pr - hr)**2 + (pc - hc)**2)
                if d < nearest_hazard_dist:
                    nearest_hazard_dist = d
    if nearest_hazard_dist == float("inf"):
        nearest_hazard_dist = 0.0

    if shadow_pct < 10.0 and hazards_near == 0:
        risk_level = "LOW"
    elif shadow_pct > 25.0 or hazards_near > 3:
        risk_level = "HIGH"
    else:
        risk_level = "MODERATE"

    return {
        "name": display_name,
        "path": [list(cell) for cell in path],
        "total_cost": round(total_cost, 2),
        "path_length_cells": path_len,
        "distance_cm": round(path_len * config.GRID_CELL_SIZE_CM * (config.SEG_GRID_CELL_PX / config.MOSAIC_GRID_CELL_PX if config.SEG_ENABLED else 1.0), 1),
        "max_shadow_exposure_pct": round(shadow_pct, 1),
        "hazards_near_path": hazards_near,
        "nearest_hazard_distance_cells": round(nearest_hazard_dist, 2),
        "risk_level": risk_level,
        "color": colour,
        "status": "found",
        "route_map_path": "",
    }


def _save_routes_json(routes: dict, start: tuple, end: tuple):
    """Save route planning results as JSON for the dashboard."""
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(config.PROCESSED_DIR, "routes.json")

    route_list = []
    for name in ("fastest", "safest", "balanced"):
        rd = routes.get(name, {})
        route_list.append({
            "name": rd.get("name", name.capitalize()),
            "path": rd.get("path", []),
            "stats": {
                "path_length_cells": rd.get("path_length_cells", 0),
                "distance_cm": rd.get("distance_cm", 0),
                "max_shadow_exposure_pct": rd.get("max_shadow_exposure_pct", 0),
                "hazards_near_path": rd.get("hazards_near_path", 0),
                "nearest_hazard_distance_cells": rd.get("nearest_hazard_distance_cells", 0),
                "risk_level": rd.get("risk_level", "LOW"),
                "total_cost": rd.get("total_cost", 0),
                "status": rd.get("status", "no_path"),
            },
            "color": rd.get("color", "#ffffff"),
        })

    data = {
        "routes": route_list,
        "start": list(start),
        "end": list(end),
        "selected": "safest",
        "constrained": None,
    }

    try:
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save routes.json: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# A* implementation
# ─────────────────────────────────────────────────────────────────────────────

def _astar(cost_grid: np.ndarray, start: tuple, end: tuple, rows: int, cols: int):
    """
    A* with 8-connected neighbours.
    Returns ordered list of (row, col) tuples from start to end, or None.
    """
    def h(cell):
        return abs(cell[0] - end[0]) + abs(cell[1] - end[1])

    open_heap = []          # (f, g, cell)
    heapq.heappush(open_heap, (h(start), 0.0, start))

    came_from = {}          # cell → parent
    g_score = {start: 0.0}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == end:
            return _reconstruct(came_from, current)

        # Skip stale heap entries
        if g > g_score.get(current, float("inf")):
            continue

        cr, cc = current
        for dr, dc in _ALL_DIRS:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell_cost = int(cost_grid[nr, nc])
            if cell_cost >= config.COST_IMPASSABLE:
                continue

            diagonal = (dr != 0 and dc != 0)
            move_cost = (_SQRT2 if diagonal else 1.0) * cell_cost
            tentative_g = g + move_cost

            if tentative_g < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = tentative_g
                came_from[(nr, nc)] = current
                f = tentative_g + h((nr, nc))
                heapq.heappush(open_heap, (f, tentative_g, (nr, nc)))

    return None  # no path


def _reconstruct(came_from: dict, current: tuple) -> list:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _path_cost(path: list, cost_grid: np.ndarray) -> float:
    total = 0.0
    for i in range(1, len(path)):
        pr, pc = path[i - 1]
        cr, cc = path[i]
        diagonal = (pr != cr and pc != cc)
        cell_cost = float(cost_grid[cr, cc])
        total += (_SQRT2 if diagonal else 1.0) * cell_cost
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _get_cell_vis_px(rows, cols):
    """Dynamic cell visualization size — scale down for large grids."""
    return max(4, min(64, 512 // max(rows, cols)))

_CELL_VIS_PX = 64   # default, overridden dynamically for fine grids


def grid_path_to_mosaic_path(path: list, cell_px: int = None) -> list:
    """
    Convert a grid path [(r,c),...] to mosaic pixel coordinates [(mx,my),...].
    Each grid cell center is at (c * cell_px + cell_px/2, r * cell_px + cell_px/2).
    """
    if cell_px is None:
        cell_px = config.MOSAIC_GRID_CELL_PX
    mosaic_path = []
    for step in path:
        r, c = step if isinstance(step, (list, tuple)) else (step[0], step[1])
        mx = c * cell_px + cell_px / 2
        my = r * cell_px + cell_px / 2
        mosaic_path.append([mx, my])
    return mosaic_path


def _save_route_comparison(
    cost_grid: np.ndarray,
    hazard_grid: list | None,
    routes: dict,
    start: tuple,
    end: tuple,
    hazard_map_path: str | None,
) -> str:
    """Draw all 3 paths on one image with a legend. Saves route_comparison.png."""
    rows, cols = cost_grid.shape
    cell_px = _get_cell_vis_px(rows, cols)
    base = _build_base_canvas(cost_grid, hazard_grid, hazard_map_path, rows, cols, cell_px)

    for name in ("fastest", "safest", "balanced"):
        route = routes.get(name, {})
        path = route.get("path", [])
        if len(path) < 2:
            continue
        colour_bgr = _ROUTE_COLOURS_BGR[name]
        for i in range(1, len(path)):
            r0, c0 = path[i - 1]
            r1, c1 = path[i]
            p0 = (c0 * cell_px + cell_px // 2, r0 * cell_px + cell_px // 2)
            p1 = (c1 * cell_px + cell_px // 2, r1 * cell_px + cell_px // 2)
            cv2.line(base, p0, p1, colour_bgr, 3, cv2.LINE_AA)

    # Legend box (top-right)
    legend_x = cols * cell_px - 110
    legend_y = 6
    cv2.rectangle(base, (legend_x - 4, legend_y - 2),
                  (cols * cell_px - 4, legend_y + 52), (30, 30, 30), -1)
    for i, name in enumerate(("fastest", "safest", "balanced")):
        colour_bgr = _ROUTE_COLOURS_BGR[name]
        cy = legend_y + 10 + i * 16
        cv2.circle(base, (legend_x + 6, cy), 5, colour_bgr, -1)
        cv2.putText(base, _ROUTE_NAMES[name], (legend_x + 16, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    _draw_start_end_markers(base, start, end, cell_px)

    os.makedirs(os.path.join(config.PROCESSED_DIR, "routes"), exist_ok=True)
    out_path = os.path.join(config.PROCESSED_DIR, "routes", "route_comparison.png")
    cv2.imwrite(out_path, base)
    logger.debug(f"Route comparison saved: {out_path}")
    return out_path


def _save_individual_route_map(
    cost_grid: np.ndarray,
    hazard_grid: list | None,
    path: list,
    start: tuple,
    end: tuple,
    hazard_map_path: str | None,
    name: str,
) -> str:
    rows, cols = cost_grid.shape
    cell_px = _get_cell_vis_px(rows, cols)
    base = _build_base_canvas(cost_grid, hazard_grid, hazard_map_path, rows, cols, cell_px)

    colour_bgr = _ROUTE_COLOURS_BGR.get(name, _DEFAULT_PATH_COLOUR)
    path_set = set(tuple(c) for c in path)

    for i in range(1, len(path)):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        p0 = (c0 * cell_px + cell_px // 2, r0 * cell_px + cell_px // 2)
        p1 = (c1 * cell_px + cell_px // 2, r1 * cell_px + cell_px // 2)
        cv2.line(base, p0, p1, colour_bgr, 3, cv2.LINE_AA)

    for (r, c) in path_set:
        x0, y0 = c * cell_px, r * cell_px
        cv2.rectangle(base, (x0 + 1, y0 + 1), (x0 + cell_px - 2, y0 + cell_px - 2), colour_bgr, 2)

    _draw_start_end_markers(base, start, end, cell_px)

    if path:
        status_txt = f"{_ROUTE_NAMES.get(name, name)}: {len(path)} cells"
    else:
        status_txt = f"{_ROUTE_NAMES.get(name, name)}: NO PATH"
    banner_h = 26
    banner = np.full((banner_h, base.shape[1], 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(banner, status_txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    output = np.vstack([banner, base])

    os.makedirs(os.path.join(config.PROCESSED_DIR, "routes"), exist_ok=True)
    # Also keep route_latest.png for backward compat (use fastest)
    if name == "fastest":
        cv2.imwrite(os.path.join(config.PROCESSED_DIR, "routes", "route_latest.png"), output)
    out_path = os.path.join(config.PROCESSED_DIR, "routes", f"route_{name}.png")
    cv2.imwrite(out_path, output)
    logger.debug(f"Individual route map saved: {out_path}")
    return out_path


def _save_route_map(
    cost_grid: np.ndarray,
    hazard_grid: list | None,
    path: list,
    start: tuple,
    end: tuple,
    hazard_map_path: str | None,
) -> str:
    """Legacy single-route save (kept for compatibility)."""
    return _save_individual_route_map(
        cost_grid, hazard_grid, path, start, end, hazard_map_path, "fastest"
    )


def _draw_start_end_markers(base: np.ndarray, start: tuple, end: tuple, cell_px: int):
    def _marker(cell, label, colour):
        r, c = cell
        cx = c * cell_px + cell_px // 2
        cy = r * cell_px + cell_px // 2
        cv2.circle(base, (cx, cy), cell_px // 3, colour, -1)
        cv2.putText(base, label, (cx - 6, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    _marker(start, "L", (0, 180, 0))    # Landing site (green)
    _marker(end,   "T", (0, 0, 200))    # Target (red)


def _build_base_canvas(
    cost_grid: np.ndarray,
    hazard_grid: list | None,
    hazard_map_path: str | None,
    rows: int, cols: int, cell_px: int
) -> np.ndarray:
    canvas_h = rows * cell_px
    canvas_w = cols * cell_px

    if hazard_map_path and os.path.exists(hazard_map_path):
        img = cv2.imread(hazard_map_path)
        if img is not None:
            return cv2.resize(img, (canvas_w, canvas_h))

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell_px, r * cell_px
            if hazard_grid:
                cls = hazard_grid[r][c]
                colour = HAZARD_COLOURS.get(cls, (60, 60, 60))
            else:
                cost = int(cost_grid[r, c])
                shade = max(30, min(200, 200 - cost))
                colour = (shade, shade, shade)
            canvas[y0:y0 + cell_px, x0:x0 + cell_px] = colour
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1),
                          (80, 80, 80), 1)

    return canvas
