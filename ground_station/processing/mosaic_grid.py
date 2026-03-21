from __future__ import annotations
# processing/mosaic_grid.py — Dynamic rectangular grid derived from the mosaic
#
# Each grid cell represents MOSAIC_GRID_CELL_PX pixels of mosaic space.
# The grid grows automatically as the mosaic expands, preserving existing
# cell values. Provides coordinate conversion between mosaic pixel space
# and grid (row, col) space, and projects per-image hazard results onto
# the grid cells that the image covers.

import json
import logging
import os

import numpy as np

import config

logger = logging.getLogger(__name__)

CELL_PX = config.MOSAIC_GRID_CELL_PX
FINE_CELL_PX = config.SEG_GRID_CELL_PX


class MosaicGrid:
    """
    Dynamic rectangular grid overlaid on the mosaic canvas.
    Grid cell (r, c) covers mosaic pixels:
        x: [c * CELL_PX, (c+1) * CELL_PX)
        y: [r * CELL_PX, (r+1) * CELL_PX)

    Maintains two parallel grids:
        Coarse grid — CELL_PX (80px) cells for backward compatibility
        Fine grid   — FINE_CELL_PX (20px) cells for pixel-level route planning
    """

    def __init__(self):
        # Coarse grid (80px cells)
        self._rows = 0
        self._cols = 0
        # cost_grid[r][c] = int cost
        self._cost_grid: np.ndarray | None = None
        # hazard_grid[r][c] = str class
        self._hazard_grid: list[list[str]] = []
        # confidence_grid[r][c] = float
        self._confidence_grid: np.ndarray | None = None
        # surveyed[r][c] = bool — True if any image covers this cell
        self._surveyed: np.ndarray | None = None
        # observation_count[r][c] = int — number of times this cell has been observed
        self._observation_count: np.ndarray | None = None
        # slope_grid[r][c] = float degrees — estimated terrain slope
        self._slope_grid: np.ndarray | None = None

        # Fine grid (20px cells) — for pixel-level segmentation routing
        self._fine_rows = 0
        self._fine_cols = 0
        self._fine_cost_grid: np.ndarray | None = None
        self._fine_hazard_grid: np.ndarray | None = None  # uint8 label per cell

    # ─────────────────────────────────────────────────────────────────────
    # Update from mosaic canvas size
    # ─────────────────────────────────────────────────────────────────────

    def update_from_mosaic(self, canvas_w: int, canvas_h: int):
        """
        Resize the grid to match the current mosaic canvas dimensions.
        Preserves existing cell values when growing.
        """
        new_rows = max(1, (canvas_h + CELL_PX - 1) // CELL_PX)
        new_cols = max(1, (canvas_w + CELL_PX - 1) // CELL_PX)

        if new_rows == self._rows and new_cols == self._cols:
            return

        old_rows, old_cols = self._rows, self._cols

        # Build new arrays
        new_cost = np.full((new_rows, new_cols), config.COST_SAFE, dtype=np.int32)
        new_hazard = [["SAFE"] * new_cols for _ in range(new_rows)]
        new_conf = np.zeros((new_rows, new_cols), dtype=np.float64)
        new_surv = np.zeros((new_rows, new_cols), dtype=bool)
        new_obs = np.zeros((new_rows, new_cols), dtype=np.int32)
        new_slope = np.zeros((new_rows, new_cols), dtype=np.float32)

        # Copy old values
        copy_r = min(old_rows, new_rows)
        copy_c = min(old_cols, new_cols)
        if self._cost_grid is not None and copy_r > 0 and copy_c > 0:
            new_cost[:copy_r, :copy_c] = self._cost_grid[:copy_r, :copy_c]
            new_conf[:copy_r, :copy_c] = self._confidence_grid[:copy_r, :copy_c]
            new_surv[:copy_r, :copy_c] = self._surveyed[:copy_r, :copy_c]
            if self._observation_count is not None:
                new_obs[:copy_r, :copy_c] = self._observation_count[:copy_r, :copy_c]
            if self._slope_grid is not None:
                new_slope[:copy_r, :copy_c] = self._slope_grid[:copy_r, :copy_c]
            for r in range(copy_r):
                for c in range(copy_c):
                    new_hazard[r][c] = self._hazard_grid[r][c]

        self._rows = new_rows
        self._cols = new_cols
        self._cost_grid = new_cost
        self._hazard_grid = new_hazard
        self._confidence_grid = new_conf
        self._surveyed = new_surv
        self._observation_count = new_obs
        self._slope_grid = new_slope

        # Fine grid
        new_fr = max(1, (canvas_h + FINE_CELL_PX - 1) // FINE_CELL_PX)
        new_fc = max(1, (canvas_w + FINE_CELL_PX - 1) // FINE_CELL_PX)
        if new_fr != self._fine_rows or new_fc != self._fine_cols:
            old_fr, old_fc = self._fine_rows, self._fine_cols
            new_fine_cost = np.full((new_fr, new_fc), config.COST_SAFE, dtype=np.float32)
            new_fine_hazard = np.zeros((new_fr, new_fc), dtype=np.uint8)
            if self._fine_cost_grid is not None:
                cr = min(old_fr, new_fr)
                cc = min(old_fc, new_fc)
                new_fine_cost[:cr, :cc] = self._fine_cost_grid[:cr, :cc]
                new_fine_hazard[:cr, :cc] = self._fine_hazard_grid[:cr, :cc]
            self._fine_rows = new_fr
            self._fine_cols = new_fc
            self._fine_cost_grid = new_fine_cost
            self._fine_hazard_grid = new_fine_hazard

        if new_rows != old_rows or new_cols != old_cols:
            logger.info(
                f"MosaicGrid: resized coarse {old_rows}x{old_cols} → {new_rows}x{new_cols}, "
                f"fine {self._fine_rows}x{self._fine_cols}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Apply hazard results
    # ─────────────────────────────────────────────────────────────────────

    def apply_hazard(self, mosaic_bbox: tuple, hazard_result: dict):
        """
        Project a per-image hazard classification onto the grid cells
        that the image covers.

        Args:
            mosaic_bbox: (x, y, w, h) in mosaic pixel coords
            hazard_result: dict with "hazard_class", "cost", "confidence"
        """
        if self._cost_grid is None:
            return

        x, y, w, h = mosaic_bbox
        hazard_class = hazard_result.get("hazard_class", "SAFE")
        cost = hazard_result.get("cost", config.COST_SAFE)
        confidence = hazard_result.get("confidence", 0.5)

        # Find grid cells covered by this bbox
        r_min = max(0, y // CELL_PX)
        r_max = min(self._rows - 1, (y + h - 1) // CELL_PX)
        c_min = max(0, x // CELL_PX)
        c_max = min(self._cols - 1, (x + w - 1) // CELL_PX)

        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                # Only update if new confidence is higher or cell is unsurveyed
                if not self._surveyed[r, c] or confidence >= self._confidence_grid[r, c]:
                    self._cost_grid[r, c] = cost
                    self._hazard_grid[r][c] = hazard_class
                    self._confidence_grid[r, c] = confidence
                self._surveyed[r, c] = True
                if self._observation_count is not None:
                    self._observation_count[r, c] += 1

    # ─────────────────────────────────────────────────────────────────────
    # Pixel segmentation → fine grid
    # ─────────────────────────────────────────────────────────────────────

    def apply_segmentation_mask(self, mosaic_bbox: tuple, label_map: np.ndarray,
                                cost_map: dict | None = None, confidence: float = 1.0):
        """
        Project a pixel-level label map onto the fine grid.

        For each fine grid cell overlapping the bbox, samples the label_map pixels
        and assigns the worst label covering >=25% of the cell (conservative).
        Otherwise uses majority label.

        Args:
            mosaic_bbox: (x, y, w, h) in mosaic pixel coords.
            label_map: np.ndarray (img_h, img_w) uint8 from PixelSegmenter.
            cost_map: dict label→cost, defaults to config.SEG_COST_MAP.
            confidence: float 0-1 overall confidence.
        """
        if self._fine_cost_grid is None:
            return

        if cost_map is None:
            cost_map = config.SEG_COST_MAP

        mx, my, mw, mh = mosaic_bbox
        img_h, img_w = label_map.shape[:2]

        # Map fine grid cell range
        fr0 = max(0, my // FINE_CELL_PX)
        fc0 = max(0, mx // FINE_CELL_PX)
        fr1 = min(self._fine_rows, (my + mh + FINE_CELL_PX - 1) // FINE_CELL_PX)
        fc1 = min(self._fine_cols, (mx + mw + FINE_CELL_PX - 1) // FINE_CELL_PX)

        hazard_threshold = 0.25  # 25% coverage triggers worst-label assignment

        for r in range(fr0, fr1):
            for c in range(fc0, fc1):
                # Pixel range in mosaic coords
                py0 = r * FINE_CELL_PX
                px0 = c * FINE_CELL_PX
                py1 = py0 + FINE_CELL_PX
                px1 = px0 + FINE_CELL_PX

                # Convert to image-local coords
                iy0 = max(0, py0 - my)
                ix0 = max(0, px0 - mx)
                iy1 = min(img_h, py1 - my)
                ix1 = min(img_w, px1 - mx)

                if iy1 <= iy0 or ix1 <= ix0:
                    continue

                cell_pixels = label_map[iy0:iy1, ix0:ix1]
                total = cell_pixels.size
                if total == 0:
                    continue

                # Find worst (highest cost) label with >= 25% coverage
                unique, counts = np.unique(cell_pixels, return_counts=True)
                assigned_label = None
                assigned_cost = 0

                for lbl, cnt in zip(unique, counts):
                    lbl_cost = cost_map.get(int(lbl), config.COST_SAFE)
                    fraction = cnt / total
                    if lbl_cost > assigned_cost and fraction >= hazard_threshold:
                        assigned_cost = lbl_cost
                        assigned_label = int(lbl)

                # If no hazard meets threshold, use majority label
                if assigned_label is None:
                    majority_idx = np.argmax(counts)
                    assigned_label = int(unique[majority_idx])
                    assigned_cost = cost_map.get(assigned_label, config.COST_SAFE)

                weighted_cost = config.COST_SAFE + (assigned_cost - config.COST_SAFE) * confidence
                self._fine_cost_grid[r, c] = max(
                    self._fine_cost_grid[r, c], weighted_cost
                )
                self._fine_hazard_grid[r, c] = assigned_label

    # ─────────────────────────────────────────────────────────────────────
    # Coordinate conversion
    # ─────────────────────────────────────────────────────────────────────

    def mosaic_px_to_grid(self, mx: float, my: float) -> tuple[int, int]:
        """Convert mosaic pixel coordinates to grid (row, col)."""
        r = int(my // CELL_PX)
        c = int(mx // CELL_PX)
        r = max(0, min(self._rows - 1, r))
        c = max(0, min(self._cols - 1, c))
        return (r, c)

    def grid_to_mosaic_px(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid (row, col) to mosaic pixel center coordinates."""
        mx = col * CELL_PX + CELL_PX / 2
        my = row * CELL_PX + CELL_PX / 2
        return (mx, my)

    # ─────────────────────────────────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────────────────────────────────

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def get_cost_grid(self) -> np.ndarray:
        if self._cost_grid is None:
            return np.full((1, 1), config.COST_SAFE, dtype=np.int32)
        return self._cost_grid.copy()

    def get_hazard_grid(self) -> list[list[str]]:
        if not self._hazard_grid:
            return [["SAFE"]]
        return [row[:] for row in self._hazard_grid]

    def get_confidence_grid(self) -> np.ndarray:
        if self._confidence_grid is None:
            return np.zeros((1, 1), dtype=np.float64)
        return self._confidence_grid.copy()

    def get_surveyed_mask(self) -> np.ndarray:
        if self._surveyed is None:
            return np.zeros((1, 1), dtype=bool)
        return self._surveyed.copy()

    def get_cells_surveyed(self) -> int:
        if self._surveyed is None:
            return 0
        return int(np.sum(self._surveyed))

    def get_cells_total(self) -> int:
        return self._rows * self._cols

    # ─────────────────────────────────────────────────────────────────────
    # Fine grid accessors
    # ─────────────────────────────────────────────────────────────────────

    @property
    def fine_rows(self) -> int:
        return self._fine_rows

    @property
    def fine_cols(self) -> int:
        return self._fine_cols

    def get_fine_cost_grid(self) -> np.ndarray:
        if self._fine_cost_grid is None:
            return np.full((1, 1), config.COST_SAFE, dtype=np.float32)
        return self._fine_cost_grid.copy()

    def get_fine_hazard_grid(self) -> np.ndarray:
        if self._fine_hazard_grid is None:
            return np.zeros((1, 1), dtype=np.uint8)
        return self._fine_hazard_grid.copy()

    def mosaic_px_to_fine_grid(self, mx: float, my: float) -> tuple[int, int]:
        """Convert mosaic pixel coordinates to fine grid (row, col)."""
        r = int(my // FINE_CELL_PX)
        c = int(mx // FINE_CELL_PX)
        r = max(0, min(self._fine_rows - 1, r))
        c = max(0, min(self._fine_cols - 1, c))
        return (r, c)

    def fine_grid_to_mosaic_px(self, row: int, col: int) -> tuple[float, float]:
        """Convert fine grid (row, col) to mosaic pixel center coordinates."""
        mx = col * FINE_CELL_PX + FINE_CELL_PX / 2
        my = row * FINE_CELL_PX + FINE_CELL_PX / 2
        return (mx, my)

    # ─────────────────────────────────────────────────────────────────────
    # Uncertainty-aware effective cost grid
    # ─────────────────────────────────────────────────────────────────────

    def get_effective_cost_grid(self) -> np.ndarray:
        """
        Return cost grid adjusted for uncertainty, slope, and CNN traversability.
        - Unsurveyed cells → UNSURVEYED_COST
        - Surveyed cells → raw_cost × (1 + UNCERTAINTY_WEIGHT × (1 - adjusted_confidence))
        - Slope multiplier applied on top
        """
        if self._cost_grid is None:
            return np.full((1, 1), config.COST_SAFE, dtype=np.float32)

        effective = self._cost_grid.astype(np.float32).copy()

        if config.UNCERTAINTY_ENABLED:
            for r in range(self._rows):
                for c in range(self._cols):
                    if not self._surveyed[r, c]:
                        effective[r, c] = config.UNSURVEYED_COST
                    else:
                        obs = int(self._observation_count[r, c]) if self._observation_count is not None else 1
                        adj_conf = min(1.0, self._confidence_grid[r, c] +
                                       (obs - 1) * config.CONFIDENCE_OBS_BOOST)
                        effective[r, c] = self._cost_grid[r, c] * (
                            1.0 + config.UNCERTAINTY_WEIGHT * (1.0 - adj_conf)
                        )

        if config.SLOPE_ENABLED and self._slope_grid is not None:
            for r in range(self._rows):
                for c in range(self._cols):
                    slope = self._slope_grid[r, c]
                    effective[r, c] *= self._slope_cost_multiplier(slope)

        return effective

    @staticmethod
    def _slope_cost_multiplier(slope_deg: float) -> float:
        """Return cost multiplier based on terrain slope."""
        if slope_deg <= config.SLOPE_GENTLE_DEG:
            return 1.0
        elif slope_deg <= config.SLOPE_MODERATE_DEG:
            return 2.0
        elif slope_deg <= config.SLOPE_STEEP_DEG:
            return 5.0
        else:
            return 999.0

    # ─────────────────────────────────────────────────────────────────────
    # Slope application
    # ─────────────────────────────────────────────────────────────────────

    def apply_slope(self, mosaic_bbox: tuple, slope_map: np.ndarray):
        """
        Project a slope map onto the grid cells within mosaic_bbox.
        slope_map: (h, w) float array of slope in degrees, same pixel dims as mosaic_bbox.
        """
        if self._slope_grid is None or slope_map is None:
            return

        x, y, w, h = mosaic_bbox
        r_min = max(0, y // CELL_PX)
        r_max = min(self._rows - 1, (y + h - 1) // CELL_PX)
        c_min = max(0, x // CELL_PX)
        c_max = min(self._cols - 1, (x + w - 1) // CELL_PX)

        img_h, img_w = slope_map.shape[:2]

        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                # Pixel range in mosaic coords
                py0 = r * CELL_PX - y
                px0 = c * CELL_PX - x
                py1 = py0 + CELL_PX
                px1 = px0 + CELL_PX

                py0 = max(0, py0)
                px0 = max(0, px0)
                py1 = min(img_h, py1)
                px1 = min(img_w, px1)

                if py1 <= py0 or px1 <= px0:
                    continue

                patch = slope_map[py0:py1, px0:px1]
                max_slope = float(np.max(patch))
                # Keep worst (steepest) slope seen
                self._slope_grid[r, c] = max(self._slope_grid[r, c], max_slope)

    # ─────────────────────────────────────────────────────────────────────
    # CNN traversability blending
    # ─────────────────────────────────────────────────────────────────────

    def apply_cnn_traversability(self, mosaic_bbox: tuple, trav_grid: np.ndarray):
        """
        Blend CNN traversability predictions with classical costs.
        trav_grid: (h, w) float array of traversability scores 0.0-1.0.
        final = alpha * CNN_cost + (1-alpha) * classical_cost
        """
        if self._cost_grid is None or trav_grid is None:
            return

        alpha = config.CNN_BLEND_WEIGHT
        x, y, w, h = mosaic_bbox
        r_min = max(0, y // CELL_PX)
        r_max = min(self._rows - 1, (y + h - 1) // CELL_PX)
        c_min = max(0, x // CELL_PX)
        c_max = min(self._cols - 1, (x + w - 1) // CELL_PX)

        trav_h, trav_w = trav_grid.shape[:2]

        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                py0 = r * CELL_PX - y
                px0 = c * CELL_PX - x
                py1 = py0 + CELL_PX
                px1 = px0 + CELL_PX

                py0 = max(0, py0)
                px0 = max(0, px0)
                py1 = min(trav_h, py1)
                px1 = min(trav_w, px1)

                if py1 <= py0 or px1 <= px0:
                    continue

                patch = trav_grid[py0:py1, px0:px1]
                avg_trav = float(np.mean(patch))
                # Convert traversability (0=impassable, 1=safe) to cost
                cnn_cost = config.COST_SAFE + (config.COST_IMPASSABLE - config.COST_SAFE) * (1.0 - avg_trav)
                classical_cost = float(self._cost_grid[r, c])
                blended = alpha * cnn_cost + (1.0 - alpha) * classical_cost
                self._cost_grid[r, c] = int(round(blended))

    # ─────────────────────────────────────────────────────────────────────
    # Additional accessors
    # ─────────────────────────────────────────────────────────────────────

    def get_observation_count(self) -> np.ndarray:
        if self._observation_count is None:
            return np.zeros((1, 1), dtype=np.int32)
        return self._observation_count.copy()

    def get_slope_grid(self) -> np.ndarray:
        if self._slope_grid is None:
            return np.zeros((1, 1), dtype=np.float32)
        return self._slope_grid.copy()

    def apply_roughness_costs(self, roughness_cost_grid: np.ndarray):
        """Apply terrain roughness cost multipliers to the fine cost grid.

        Multipliers are applied to the BASE cost (before any roughness),
        not cumulatively, to avoid exponential cost blowup on repeated calls.
        """
        if self._fine_cost_grid is None:
            return
        # Save base costs on first call so we can reapply cleanly
        if not hasattr(self, '_fine_base_cost_grid') or self._fine_base_cost_grid is None:
            self._fine_base_cost_grid = self._fine_cost_grid.copy()
        r, c = roughness_cost_grid.shape
        fr, fc = self._fine_cost_grid.shape
        mr, mc = min(r, fr), min(c, fc)
        # Reset to base costs, then apply multipliers once
        self._fine_cost_grid[:mr, :mc] = self._fine_base_cost_grid[:mr, :mc]
        for row in range(mr):
            for col in range(mc):
                mult = roughness_cost_grid[row, col]
                if mult > 1.0:
                    self._fine_cost_grid[row, col] = self._fine_base_cost_grid[row, col] * mult

    def restore_from_json(self, path: str):
        """Restore grid state from a persisted cost_grid.json file."""
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            cost_data = data.get("cost_grid")
            if cost_data:
                self._cost_grid = np.array(cost_data, dtype=np.float32)
                self._rows, self._cols = self._cost_grid.shape
            hazard_data = data.get("hazard_grid")
            if hazard_data:
                self._hazard_grid = hazard_data
            logger.info(f"MosaicGrid: restored from {path}")
        except Exception as e:
            logger.warning(f"MosaicGrid: could not restore from {path}: {e}")
