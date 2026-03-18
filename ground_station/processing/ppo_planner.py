from __future__ import annotations
# processing/ppo_planner.py — PPO-based route planner (runs alongside A*)
#
# Wraps the trained PPO model from stable-baselines3. The model takes an
# 11x11x4 local observation (cost view, impassable view, distance-to-goal,
# direction-to-goal) and outputs one of 9 discrete actions (8 compass + stay).
#
# Call load_ppo_model() once at startup, then plan_ppo_route() per route request.

import logging
import math
import os

import numpy as np

import config

logger = logging.getLogger(__name__)

VIEW_SIZE = 11
MAX_STEPS = 500  # generous budget for larger grids

_ppo_model = None


def load_ppo_model(model_path: str | None = None) -> bool:
    """
    Load the PPO model from disk. Returns True on success.
    Called once at pipeline startup.
    """
    global _ppo_model

    if model_path is None:
        # Default: look in PPO Training folder relative to ground_station/
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "PPO Training", "best_model.zip"
        )

    if not os.path.exists(model_path):
        logger.warning(f"PPO model not found at {model_path} — PPO planner disabled")
        return False

    try:
        from stable_baselines3 import PPO
        _ppo_model = PPO.load(model_path)
        logger.info(f"PPO planner loaded from {model_path}")
        return True
    except ImportError:
        logger.warning("stable-baselines3 not installed — PPO planner disabled")
        return False
    except Exception as e:
        logger.error(f"Failed to load PPO model: {e}")
        return False


def is_available() -> bool:
    """Check if the PPO model is loaded and ready."""
    return _ppo_model is not None


def plan_ppo_route(
    cost_grid: np.ndarray,
    start: tuple,
    goal: tuple,
) -> dict | None:
    """
    Run the PPO model on the cost grid to find a route from start to goal.

    Args:
        cost_grid: (rows, cols) int/float array of traversal costs.
                   Values >= config.COST_IMPASSABLE are treated as walls.
        start: (row, col) tuple
        goal: (row, col) tuple

    Returns:
        dict with keys: path, path_length, total_cost, reached_goal,
                        cumulative_slip_risk, distance_cm
        or None if PPO model is not loaded.
    """
    if _ppo_model is None:
        return None

    rows, cols = cost_grid.shape
    half = VIEW_SIZE // 2

    # Normalize cost grid to [0, 1] where 0=free, 1=wall
    # Training env used 0-1 range with 1.0 padding for boundaries
    max_cost = float(config.COST_IMPASSABLE)
    cost_f = cost_grid.astype(np.float32)
    cost_norm = np.where(
        cost_f >= max_cost,
        1.0,
        (cost_f - config.COST_SAFE) / (max_cost - config.COST_SAFE),
    ).astype(np.float32)

    # Build impassable map (binary: 1.0 = blocked)
    impassable = (cost_grid >= config.COST_IMPASSABLE).astype(np.float32)

    # Pad for edge observations
    cost_pad = np.pad(cost_norm, half, mode="constant", constant_values=1.0)
    imp_pad = np.pad(impassable, half, mode="constant", constant_values=1.0)

    MOVES = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    DISTS = [1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414]

    pos = np.array(start)
    path = [(int(start[0]), int(start[1]))]
    total_cost = 0.0
    gs = max(rows, cols)
    max_dist = math.sqrt(2) * gs

    # Scale max steps with grid size — generous budget
    max_steps = max(MAX_STEPS, gs * 4)

    # Stuck detection: if the agent visits the same cell too many times, bail
    visit_count = {}
    stuck_threshold = 8

    for step in range(max_steps):
        r, c = pos
        rp, cp = r + half, c + half
        cost_view = cost_pad[rp - half:rp + half + 1, cp - half:cp + half + 1]
        imp_view = imp_pad[rp - half:rp + half + 1, cp - half:cp + half + 1]

        dist = math.sqrt((goal[0] - r) ** 2 + (goal[1] - c) ** 2)
        dist_ch = np.full((VIEW_SIZE, VIEW_SIZE), dist / max_dist, dtype=np.float32)
        dr = (goal[0] - r) / (max_dist + 1e-8)
        dc = (goal[1] - c) / (max_dist + 1e-8)
        dir_ch = np.full(
            (VIEW_SIZE, VIEW_SIZE),
            0.5 + 0.5 * math.atan2(dr, dc) / math.pi,
            dtype=np.float32,
        )

        obs = np.stack([cost_view, imp_view, dist_ch, dir_ch], axis=-1).flatten().astype(np.float32)
        action, _ = _ppo_model.predict(obs, deterministic=True)

        if action < 8:
            mv_dr, mv_dc = MOVES[action]
            nr, nc = r + mv_dr, c + mv_dc
            if 0 <= nr < rows and 0 <= nc < cols and impassable[nr, nc] < 0.5:
                pos = np.array([nr, nc])
                pos_tuple = (int(pos[0]), int(pos[1]))
                path.append(pos_tuple)
                total_cost += float(cost_grid[nr, nc]) * DISTS[action]

                # Stuck detection
                visit_count[pos_tuple] = visit_count.get(pos_tuple, 0) + 1
                if visit_count[pos_tuple] >= stuck_threshold:
                    break

        if dist < 3.0:
            break

    reached = math.sqrt((int(pos[0]) - goal[0]) ** 2 + (int(pos[1]) - goal[1]) ** 2) < 3.0

    # Trim oscillation: remove repeated tail cells
    if len(path) > 2:
        seen_tail = set()
        trim_idx = len(path)
        for i in range(len(path) - 1, -1, -1):
            if path[i] in seen_tail:
                trim_idx = i
            else:
                seen_tail.add(path[i])
                if len(seen_tail) > 3:
                    break
        if trim_idx < len(path) - 1:
            path = path[:trim_idx + 1]

    # Compute cumulative slip risk: sum of normalized costs along path
    slip_risk = 0.0
    for r, c in path:
        cell_cost = float(cost_grid[r, c])
        slip_risk += min(cell_cost / float(config.COST_HAZARD), 1.0)
    slip_risk = round(slip_risk, 2)

    # Path length in physical units
    path_len_cells = len(path)
    seg_cell_factor = (config.SEG_GRID_CELL_PX / config.MOSAIC_GRID_CELL_PX
                       if config.SEG_ENABLED else 1.0)
    distance_cm = round(path_len_cells * config.GRID_CELL_SIZE_CM * seg_cell_factor, 1)

    return {
        "path": [list(p) for p in path],
        "path_length": path_len_cells,
        "total_cost": round(total_cost, 2),
        "reached_goal": reached,
        "cumulative_slip_risk": slip_risk,
        "distance_cm": distance_cm,
        "status": "found" if reached else "incomplete",
    }
