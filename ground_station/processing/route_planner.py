from __future__ import annotations
# processing/route_planner.py — A* pathfinding on the hazard cost grid
#
# Input:  8×8 numpy cost_grid from hazard_classifier (one cost per cell).
# Output: shortest path from ROUTE_START to ROUTE_END avoiding IMPASSABLE cells,
#         plus a route_map.png overlaid on the hazard map.
#
# 8-connected A*: diagonal moves cost sqrt(2) * destination_cell_cost.
# Heuristic: Manhattan distance (fast; slightly inadmissible for diagonals but
# good enough for an 8×8 grid — the path quality difference is negligible).
#
# IMPASSABLE cells (cost=999) are treated as walls and never entered.
# If no path exists → status="no viable route". This is a valid demo result
# (it shows the system correctly identifies blocked terrain).

import heapq
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

_SQRT2 = math.sqrt(2)
_CARDINAL_DIRS  = [(-1,0),(1,0),(0,-1),(0,1)]
_DIAGONAL_DIRS  = [(-1,-1),(-1,1),(1,-1),(1,1)]


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
        Run A* on cost_grid from start to end.

        Args:
            cost_grid:       8×8 numpy int array.
            hazard_grid:     8×8 list of hazard class strings (for route colouring).
                             Pass None to skip colour-coded path.
            start:           (row, col) start cell.
            end:             (row, col) end cell.
            hazard_map_path: Path to an existing hazard_map.png to overlay the
                             route on. If None a blank grey canvas is used.

        Returns dict:
            {
                "path":               list of (row,col) tuples,
                "total_cost":         float,
                "path_length":        int,
                "shadow_exposure_pct": float,
                "status":             "found" | "no viable route",
                "route_map_path":     str,
            }
        """
        rows, cols = cost_grid.shape
        path = _astar(cost_grid, start, end, rows, cols)

        if path is None:
            logger.info(f"RoutePlanner: no viable route from {start} to {end}")
            route_map_path = _save_route_map(
                cost_grid, hazard_grid, [], start, end, hazard_map_path
            )
            return {
                "path": [],
                "total_cost": 0.0,
                "path_length": 0,
                "shadow_exposure_pct": 0.0,
                "status": "no viable route",
                "route_map_path": route_map_path,
            }

        total_cost = _path_cost(path, cost_grid)
        shadow_cells = sum(
            1 for (r, c) in path
            if hazard_grid is not None and hazard_grid[r][c] == SHADOW
        )
        shadow_pct = (shadow_cells / len(path) * 100.0) if path else 0.0

        logger.info(
            f"RoutePlanner: path found {start}→{end}, "
            f"length={len(path)}, total_cost={total_cost:.1f}, "
            f"shadow_exposure={shadow_pct:.1f}%"
        )

        route_map_path = _save_route_map(
            cost_grid, hazard_grid, path, start, end, hazard_map_path
        )

        return {
            "path": [list(cell) for cell in path],
            "total_cost": round(total_cost, 2),
            "path_length": len(path),
            "shadow_exposure_pct": round(shadow_pct, 1),
            "status": "found",
            "route_map_path": route_map_path,
        }


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
        for dr, dc in _CARDINAL_DIRS + _DIAGONAL_DIRS:
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

_CELL_VIS_PX = 64   # pixels per grid cell in the route map


def _save_route_map(
    cost_grid: np.ndarray,
    hazard_grid: list | None,
    path: list,
    start: tuple,
    end: tuple,
    hazard_map_path: str | None,
) -> str:
    rows, cols = cost_grid.shape
    cell_px = _CELL_VIS_PX

    # Base canvas: load existing hazard map or build a coloured grid
    base = _build_base_canvas(cost_grid, hazard_grid, hazard_map_path, rows, cols, cell_px)

    path_set = set(tuple(c) for c in path)

    # Draw path segments
    for i in range(1, len(path)):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        p0 = (c0 * cell_px + cell_px // 2, r0 * cell_px + cell_px // 2)
        p1 = (c1 * cell_px + cell_px // 2, r1 * cell_px + cell_px // 2)

        seg_class = (hazard_grid[r1][c1] if hazard_grid else None)
        colour = _PATH_COLOUR.get(seg_class, _DEFAULT_PATH_COLOUR)
        cv2.line(base, p0, p1, colour, 3, cv2.LINE_AA)

    # Draw path cell highlights
    for (r, c) in path_set:
        x0, y0 = c * cell_px, r * cell_px
        seg_class = (hazard_grid[r][c] if hazard_grid else None)
        colour = _PATH_COLOUR.get(seg_class, _DEFAULT_PATH_COLOUR)
        cv2.rectangle(base, (x0 + 1, y0 + 1), (x0 + cell_px - 2, y0 + cell_px - 2), colour, 2)

    # Start / End markers
    def _marker(cell, label, colour):
        r, c = cell
        cx = c * cell_px + cell_px // 2
        cy = r * cell_px + cell_px // 2
        cv2.circle(base, (cx, cy), cell_px // 3, colour, -1)
        cv2.putText(base, label, (cx - 6, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    _marker(start, "S", (0, 180, 0))
    _marker(end,   "E", (0, 0, 200))

    # Status banner
    if path:
        status_txt = f"Route: {len(path)} cells"
    else:
        status_txt = "NO VIABLE ROUTE"
    banner_h = 26
    banner = np.full((banner_h, base.shape[1], 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(banner, status_txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    output = np.vstack([banner, base])

    os.makedirs(os.path.join(config.PROCESSED_DIR, "routes"), exist_ok=True)
    out_path = os.path.join(config.PROCESSED_DIR, "routes", "route_latest.png")
    cv2.imwrite(out_path, output)
    logger.debug(f"Route map saved: {out_path}")
    return out_path


def _build_base_canvas(
    cost_grid: np.ndarray,
    hazard_grid: list | None,
    hazard_map_path: str | None,
    rows: int, cols: int, cell_px: int
) -> np.ndarray:
    """
    If hazard_map_path exists, resize it to grid canvas size.
    Otherwise build a colour-coded grid from hazard_grid / cost_grid.
    """
    canvas_h = rows * cell_px
    canvas_w = cols * cell_px

    if hazard_map_path and os.path.exists(hazard_map_path):
        img = cv2.imread(hazard_map_path)
        if img is not None:
            return cv2.resize(img, (canvas_w, canvas_h))

    # Fallback: build from hazard classes or cost values
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell_px, r * cell_px
            if hazard_grid:
                cls = hazard_grid[r][c]
                colour = HAZARD_COLOURS.get(cls, (60, 60, 60))
            else:
                # Shade by cost: darker = cheaper
                cost = int(cost_grid[r, c])
                shade = max(30, min(200, 200 - cost))
                colour = (shade, shade, shade)
            canvas[y0:y0 + cell_px, x0:x0 + cell_px] = colour
            # Grid line
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1),
                          (80, 80, 80), 1)

    return canvas
