"""processing/landing_recommender.py — Autonomous landing site recommendation.

Uses distance transforms and connected components (cv2) to score candidate
landing sites on the fine grid. No new dependencies beyond cv2 + numpy.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Physical scale: fine grid cell size in cm
_FINE_CELL_CM = config.SEG_GRID_CELL_PX / config.MOSAIC_PX_PER_CM  # 20px / 8px/cm = 2.5 cm


class LandingRecommender:
    """Score candidate cells on the fine grid and return top-K landing sites."""

    def recommend(
        self,
        fine_cost_grid: np.ndarray,
        fine_hazard_grid: np.ndarray,
        observation_count: np.ndarray,
        confidence_grid: np.ndarray,
        surveyed_mask: np.ndarray,
        slope_grid: np.ndarray,
        fine_rows: int,
        fine_cols: int,
        coarse_rows: int,
        coarse_cols: int,
    ) -> dict:
        """Return a dict with 'candidates' list and metadata."""

        weights = config.LANDING_WEIGHTS
        stride = config.LANDING_CANDIDATE_STRIDE
        top_k = config.LANDING_TOP_K
        min_clearance_cm = config.LANDING_MIN_CLEARANCE_CM
        min_radius_cm = config.LANDING_MIN_RADIUS_CM
        min_radius_cells = max(1, int(min_radius_cm / _FINE_CELL_CM))

        # ── Precompute ──────────────────────────────────────────────────────

        # Impassable mask: CRATER(4) or BOULDER(5)
        impassable = (fine_hazard_grid == 4) | (fine_hazard_grid == 5)
        safe_mask_u8 = (~impassable).astype(np.uint8)

        # Distance to nearest hazard (in cells)
        dist_cells = cv2.distanceTransform(safe_mask_u8, cv2.DIST_L2, 5)
        dist_cm = dist_cells * _FINE_CELL_CM

        # Connected components of safe areas
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            safe_mask_u8, connectivity=8
        )
        # stats columns: x, y, w, h, area
        component_area_cells = np.zeros(num_labels, dtype=np.int32)
        for i in range(num_labels):
            component_area_cells[i] = stats[i, cv2.CC_STAT_AREA]

        # Precompute which components touch grid edges (for route viability)
        component_edge_count = np.zeros(num_labels, dtype=np.int32)
        for i in range(1, num_labels):  # skip background label 0
            mask_i = (labels == i)
            edges = 0
            if np.any(mask_i[0, :]):        edges += 1  # top
            if np.any(mask_i[-1, :]):       edges += 1  # bottom
            if np.any(mask_i[:, 0]):        edges += 1  # left
            if np.any(mask_i[:, -1]):       edges += 1  # right
            component_edge_count[i] = edges

        # Upsample coarse grids to fine grid dimensions for per-cell lookup
        # observation_count and confidence_grid are on coarse grid
        if observation_count.shape[0] > 1 or observation_count.shape[1] > 1:
            obs_fine = cv2.resize(
                observation_count.astype(np.float32),
                (fine_cols, fine_rows),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            obs_fine = np.ones((fine_rows, fine_cols), dtype=np.float32)

        if confidence_grid.shape[0] > 1 or confidence_grid.shape[1] > 1:
            conf_fine = cv2.resize(
                confidence_grid.astype(np.float32),
                (fine_cols, fine_rows),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            conf_fine = np.zeros((fine_rows, fine_cols), dtype=np.float32)

        if surveyed_mask.shape[0] > 1 or surveyed_mask.shape[1] > 1:
            surv_fine = cv2.resize(
                surveyed_mask.astype(np.uint8),
                (fine_cols, fine_rows),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            surv_fine = np.zeros((fine_rows, fine_cols), dtype=bool)

        # ── Score candidates ────────────────────────────────────────────────

        candidates = []
        total_evaluated = 0

        for r in range(0, fine_rows, stride):
            for c in range(0, fine_cols, stride):
                total_evaluated += 1

                # Hard rejects
                if impassable[r, c]:
                    continue
                if not surv_fine[r, c]:
                    continue
                if dist_cm[r, c] < min_clearance_cm:
                    continue

                comp_id = labels[r, c]
                if comp_id == 0:
                    continue  # background (impassable)

                comp_area = int(component_area_cells[comp_id])
                comp_area_cm2 = comp_area * (_FINE_CELL_CM ** 2)

                # Zone must be large enough for landing footprint
                landing_footprint_cells = int(np.pi * min_radius_cells ** 2)
                if comp_area < landing_footprint_cells:
                    continue

                # Score 1 — Hazard Clearance
                hc_val_cm = float(dist_cm[r, c])
                s_hc = min(1.0, hc_val_cm / (10.0 * min_clearance_cm))

                # Score 2 — Zone Size
                max_zone_cm2 = 500.0  # normalize against reasonable max
                s_zone = min(1.0, comp_area_cm2 / max_zone_cm2)

                # Score 3 — Confidence (avg observations + classification confidence in neighborhood)
                nr = max(0, r - min_radius_cells)
                nr2 = min(fine_rows, r + min_radius_cells + 1)
                nc = max(0, c - min_radius_cells)
                nc2 = min(fine_cols, c + min_radius_cells + 1)

                avg_obs = float(np.mean(obs_fine[nr:nr2, nc:nc2]))
                avg_conf = float(np.mean(conf_fine[nr:nr2, nc:nc2]))
                s_conf = min(1.0, (avg_obs / 5.0 + avg_conf) / 2.0)

                # Score 4 — Flatness (inverse of average cost in landing radius)
                avg_cost = float(np.mean(fine_cost_grid[nr:nr2, nc:nc2]))
                s_flat = max(0.0, 1.0 - avg_cost / float(config.COST_IMPASSABLE))

                # Score 5 — Route Viability (edges reached by connected component)
                edges_reached = int(component_edge_count[comp_id])
                s_route = edges_reached / 4.0

                # Weighted total
                score = (
                    weights["hazard_clearance"] * s_hc
                    + weights["zone_size"] * s_zone
                    + weights["confidence"] * s_conf
                    + weights["flatness"] * s_flat
                    + weights["route_viability"] * s_route
                )

                # Mosaic pixel center for this fine grid cell
                mosaic_x = c * config.SEG_GRID_CELL_PX + config.SEG_GRID_CELL_PX / 2
                mosaic_y = r * config.SEG_GRID_CELL_PX + config.SEG_GRID_CELL_PX / 2
                pos_cm_x = mosaic_x / config.MOSAIC_PX_PER_CM
                pos_cm_y = mosaic_y / config.MOSAIC_PX_PER_CM

                candidates.append({
                    "grid_rc": [r, c],
                    "mosaic_px": [round(mosaic_x, 1), round(mosaic_y, 1)],
                    "position_cm": [round(pos_cm_x, 1), round(pos_cm_y, 1)],
                    "score": round(score, 4),
                    "breakdown": {
                        "hazard_clearance": {"score": round(s_hc, 3), "value_cm": round(hc_val_cm, 1)},
                        "zone_size": {"score": round(s_zone, 3), "value_cm2": round(comp_area_cm2, 1)},
                        "confidence": {"score": round(s_conf, 3), "avg_observations": round(avg_obs, 1)},
                        "flatness": {"score": round(s_flat, 3), "avg_cost": round(avg_cost, 1)},
                        "route_viability": {"score": round(s_route, 3), "edges_reached": edges_reached},
                    },
                })

        # Sort by score descending, take top K
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[:top_k]

        # Add rank
        for i, cand in enumerate(top):
            cand["rank"] = i + 1

        logger.info(
            f"LandingRecommender: evaluated {total_evaluated} candidates, "
            f"passed {len(candidates)}, top score "
            f"{top[0]['score'] if top else 0:.3f}"
        )

        return {
            "candidates": top,
            "total_evaluated": total_evaluated,
            "total_passed": len(candidates),
            "grid_resolution_cm": round(_FINE_CELL_CM, 2),
            "min_radius_cm": min_radius_cm,
        }
