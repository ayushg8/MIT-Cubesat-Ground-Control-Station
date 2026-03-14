from __future__ import annotations
# processing/mosaic_builder.py — Grid-based survey mosaic
#
# Places one image per grid cell on an 8×8 canvas. If a cell has images from
# multiple passes, the one with the highest quality_score is used.
#
# [FIX #12] Known limitation — one image per cell (not feature-stitched):
#   Two images of cell (2,3) from different moments stack at the same position;
#   the mosaic only shows the best-quality one. Resolution is limited by grid
#   resolution (8×8 cells), not pixel count.
#   Flight equivalent: precise orbital ephemeris → sub-pixel positioning for
#   seamless high-resolution mosaic. Our grid-based approach demonstrates the
#   concept within the constraints of the demo setup.
#
# Only runs when 3+ cells are covered (checked by pipeline.py).

import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

_UNSURVEYED_COLOUR = (51, 51, 51)    # dark grey for unfilled cells
_UNSURVEYED_LABEL  = "Not surveyed"
_CHANGED_BORDER    = (0, 140, 255)   # orange border on cells with detected changes


class MosaicBuilder:
    """
    Maintains a registry of (image_path, grid_cell, quality_score) entries
    and rebuilds the mosaic whenever update() is called.
    """

    def __init__(self):
        # { (row, col): {"path": str, "score": float} }
        self._best: dict = {}
        # Cells known to have change events — highlighted with orange border
        self._changed_cells: set = set()

    def update(
        self,
        image_path: str,
        grid_cell: tuple,
        quality_score: float,
        has_change: bool = False,
    ) -> dict | None:
        """
        Register a new image for grid_cell. If its quality_score beats the
        current best for that cell, replace it. Then rebuild the mosaic.

        Args:
            image_path:    Path to the validated JPEG.
            grid_cell:     (row, col) tuple.
            quality_score: Float 0.0–1.0 from the CubeSat quality metadata.
            has_change:    True if ChangeDetector found events for this cell.

        Returns mosaic result dict, or None if fewer than 3 cells are covered.
        """
        row, col = grid_cell
        existing = self._best.get((row, col))

        if existing is None or quality_score > existing["score"]:
            self._best[(row, col)] = {"path": image_path, "score": quality_score}

        if has_change:
            self._changed_cells.add((row, col))

        if len(self._best) < 3:
            logger.debug(
                f"MosaicBuilder: {len(self._best)} cell(s) covered — "
                "need 3+ to build mosaic"
            )
            return None

        return self._build()

    def _build(self) -> dict:
        """Construct the mosaic canvas and save it."""
        # Determine cell size from the first available image
        sample_path = next(iter(self._best.values()))["path"]
        sample_img = cv2.imread(sample_path)
        if sample_img is None:
            logger.error(f"MosaicBuilder: cannot read sample image '{sample_path}'")
            return None

        cell_h, cell_w = sample_img.shape[:2]
        rows = config.GRID_ROWS
        cols = config.GRID_COLS

        canvas = np.full((cell_h * rows, cell_w * cols, 3), _UNSURVEYED_COLOUR, dtype=np.uint8)

        # Label unfilled cells
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in self._best:
                    y0, x0 = r * cell_h, c * cell_w
                    _draw_unsurveyed_cell(canvas, x0, y0, cell_w, cell_h)

        # Place best images
        placed = 0
        for (r, c), entry in self._best.items():
            img = cv2.imread(entry["path"])
            if img is None:
                logger.warning(f"MosaicBuilder: cannot read '{entry['path']}' for cell ({r},{c})")
                continue

            # Resize to cell size if needed (shouldn't happen, but guard)
            if img.shape[:2] != (cell_h, cell_w):
                img = cv2.resize(img, (cell_w, cell_h))

            y0 = r * cell_h
            x0 = c * cell_w
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = img

            # Orange border on cells with detected changes
            if (r, c) in self._changed_cells:
                cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1),
                              _CHANGED_BORDER, 3)

            # Cell label (row, col)
            cv2.putText(canvas, f"({r},{c})", (x0 + 3, y0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            placed += 1

        # Grid lines
        for r in range(1, rows):
            cv2.line(canvas, (0, r * cell_h), (cols * cell_w, r * cell_h), (80, 80, 80), 1)
        for c in range(1, cols):
            cv2.line(canvas, (c * cell_w, 0), (c * cell_w, rows * cell_h), (80, 80, 80), 1)

        os.makedirs(os.path.join(config.PROCESSED_DIR, "mosaics"), exist_ok=True)
        out_path = os.path.join(config.PROCESSED_DIR, "mosaics", "mosaic_latest.png")
        cv2.imwrite(out_path, canvas)

        cells_total = rows * cols
        cells_filled = len(self._best)
        coverage_pct = round(cells_filled / cells_total * 100.0, 1)

        logger.info(
            f"MosaicBuilder: mosaic saved ({cells_filled}/{cells_total} cells, "
            f"{coverage_pct}% coverage)"
        )

        return {
            "mosaic_path": out_path,
            "cells_filled": cells_filled,
            "cells_total": cells_total,
            "coverage_pct": coverage_pct,
        }


def _draw_unsurveyed_cell(canvas: np.ndarray, x0: int, y0: int, w: int, h: int):
    """Draw dark grey fill + 'Not surveyed' label for an empty grid cell."""
    canvas[y0:y0 + h, x0:x0 + w] = _UNSURVEYED_COLOUR
    text_x = x0 + max(2, w // 2 - 35)
    text_y = y0 + h // 2
    cv2.putText(canvas, _UNSURVEYED_LABEL, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1, cv2.LINE_AA)
