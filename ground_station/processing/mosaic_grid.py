from __future__ import annotations
# processing/mosaic_grid.py — Dynamic rectangular grid derived from the mosaic
#
# Each grid cell represents MOSAIC_GRID_CELL_PX pixels of mosaic space.
# The grid grows automatically as the mosaic expands, preserving existing
# cell values. Provides coordinate conversion between mosaic pixel space
# and grid (row, col) space, and projects per-image hazard results onto
# the grid cells that the image covers.

import logging

import numpy as np

import config

logger = logging.getLogger(__name__)

CELL_PX = config.MOSAIC_GRID_CELL_PX


class MosaicGrid:
    """
    Dynamic rectangular grid overlaid on the mosaic canvas.
    Grid cell (r, c) covers mosaic pixels:
        x: [c * CELL_PX, (c+1) * CELL_PX)
        y: [r * CELL_PX, (r+1) * CELL_PX)
    """

    def __init__(self):
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

        # Copy old values
        copy_r = min(old_rows, new_rows)
        copy_c = min(old_cols, new_cols)
        if self._cost_grid is not None and copy_r > 0 and copy_c > 0:
            new_cost[:copy_r, :copy_c] = self._cost_grid[:copy_r, :copy_c]
            new_conf[:copy_r, :copy_c] = self._confidence_grid[:copy_r, :copy_c]
            new_surv[:copy_r, :copy_c] = self._surveyed[:copy_r, :copy_c]
            for r in range(copy_r):
                for c in range(copy_c):
                    new_hazard[r][c] = self._hazard_grid[r][c]

        self._rows = new_rows
        self._cols = new_cols
        self._cost_grid = new_cost
        self._hazard_grid = new_hazard
        self._confidence_grid = new_conf
        self._surveyed = new_surv

        if new_rows != old_rows or new_cols != old_cols:
            logger.info(f"MosaicGrid: resized from {old_rows}x{old_cols} to {new_rows}x{new_cols}")

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
