from __future__ import annotations
# processing/mosaic_builder.py — Grid-based survey mosaic with feathered blending
#
# Places one image per grid cell on an 8×8 canvas. If a cell has images from
# multiple passes, the one with the highest quality_score is used.
#
# Feathered blending: at the boundary between adjacent cells that both have
# images, a 20-pixel overlap zone uses linear gradient weights so one image
# fades smoothly into the next, eliminating the hard visible seam.
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

# Feathering overlap zone width in pixels (each side contributes half)
_FEATHER_PX = 20


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

        # Load and place best images
        cell_images = {}  # (r, c) → BGR ndarray at cell size
        for (r, c), entry in self._best.items():
            img = cv2.imread(entry["path"])
            if img is None:
                logger.warning(f"MosaicBuilder: cannot read '{entry['path']}' for cell ({r},{c})")
                continue
            if img.shape[:2] != (cell_h, cell_w):
                img = cv2.resize(img, (cell_w, cell_h))
            cell_images[(r, c)] = img

            y0 = r * cell_h
            x0 = c * cell_w
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = img

        # ── Feathered blending at seams ──
        try:
            _feather_seams(canvas, cell_images, cell_h, cell_w, rows, cols)
        except Exception as e:
            logger.warning(f"MosaicBuilder: feathering failed, using hard seams: {e}")

        # Orange border on cells with detected changes
        for (r, c) in self._changed_cells:
            if (r, c) in cell_images:
                y0 = r * cell_h
                x0 = c * cell_w
                cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1),
                              _CHANGED_BORDER, 3)

        # Cell labels
        for (r, c) in cell_images:
            y0 = r * cell_h
            x0 = c * cell_w
            cv2.putText(canvas, f"({r},{c})", (x0 + 3, y0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Grid lines
        for r in range(1, rows):
            cv2.line(canvas, (0, r * cell_h), (cols * cell_w, r * cell_h), (80, 80, 80), 1)
        for c in range(1, cols):
            cv2.line(canvas, (c * cell_w, 0), (c * cell_w, rows * cell_h), (80, 80, 80), 1)

        os.makedirs(os.path.join(config.PROCESSED_DIR, "mosaics"), exist_ok=True)
        out_path = os.path.join(config.PROCESSED_DIR, "mosaics", "mosaic_latest.png")
        cv2.imwrite(out_path, canvas)

        cells_total = rows * cols
        cells_filled = len(cell_images)
        coverage_pct = round(cells_filled / cells_total * 100.0, 1)

        logger.info(
            f"MosaicBuilder: mosaic saved ({cells_filled}/{cells_total} cells, "
            f"{coverage_pct}% coverage, feathered)"
        )

        return {
            "mosaic_path": out_path,
            "cells_filled": cells_filled,
            "cells_total": cells_total,
            "coverage_pct": coverage_pct,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feathered blending
# ─────────────────────────────────────────────────────────────────────────────

def _feather_seams(
    canvas: np.ndarray,
    cell_images: dict,
    cell_h: int,
    cell_w: int,
    rows: int,
    cols: int,
):
    """
    Apply linear gradient blending at the seam between each pair of adjacent
    cells that both have images. Modifies canvas in-place.

    For horizontal seams (left↔right neighbours): blends a vertical strip
    centred on the column boundary.
    For vertical seams (top↔bottom neighbours): blends a horizontal strip
    centred on the row boundary.
    """
    half = _FEATHER_PX // 2

    # ── Horizontal seams (left cell ↔ right cell) ──
    for r in range(rows):
        for c in range(cols - 1):
            if (r, c) not in cell_images or (r, c + 1) not in cell_images:
                continue

            left_img  = cell_images[(r, c)]
            right_img = cell_images[(r, c + 1)]

            # The seam sits at x = cell_w between the two cell images.
            # We blend 'half' pixels from the right edge of left_img
            # with 'half' pixels from the left edge of right_img.

            usable_half = min(half, cell_w // 2)
            if usable_half < 2:
                continue

            # Strips from each image (float for blending)
            left_strip  = left_img[:, cell_w - usable_half : cell_w].astype(np.float32)
            right_strip = right_img[:, 0 : usable_half].astype(np.float32)

            # Linear gradient weights: left fades out, right fades in
            alpha = np.linspace(1.0, 0.0, usable_half, dtype=np.float32)
            # Shape to (1, usable_half, 1) for broadcasting over (H, W, 3)
            weight_left  = alpha[np.newaxis, :, np.newaxis]
            weight_right = 1.0 - weight_left

            blended = (left_strip * weight_left + right_strip * weight_right)
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            # Write back into canvas
            # Left side of seam
            y0 = r * cell_h
            x_seam = c * cell_w + cell_w  # column boundary in canvas coords
            canvas[y0:y0 + cell_h, x_seam - usable_half : x_seam] = blended

            # Also write the right half of the blend into the right cell's area
            # (mirror: right_strip blended with left_strip)
            right_strip2 = right_img[:, 0 : usable_half].astype(np.float32)
            left_strip2  = left_img[:, cell_w - usable_half : cell_w].astype(np.float32)
            alpha_r = np.linspace(0.0, 1.0, usable_half, dtype=np.float32)
            weight_right2 = alpha_r[np.newaxis, :, np.newaxis]
            weight_left2  = 1.0 - weight_right2
            blended_r = (left_strip2 * weight_left2 + right_strip2 * weight_right2)
            blended_r = np.clip(blended_r, 0, 255).astype(np.uint8)
            canvas[y0:y0 + cell_h, x_seam : x_seam + usable_half] = blended_r

    # ── Vertical seams (top cell ↔ bottom cell) ──
    for r in range(rows - 1):
        for c in range(cols):
            if (r, c) not in cell_images or (r + 1, c) not in cell_images:
                continue

            top_img    = cell_images[(r, c)]
            bottom_img = cell_images[(r + 1, c)]

            usable_half = min(half, cell_h // 2)
            if usable_half < 2:
                continue

            top_strip    = top_img[cell_h - usable_half : cell_h, :].astype(np.float32)
            bottom_strip = bottom_img[0 : usable_half, :].astype(np.float32)

            alpha = np.linspace(1.0, 0.0, usable_half, dtype=np.float32)
            weight_top    = alpha[:, np.newaxis, np.newaxis]
            weight_bottom = 1.0 - weight_top

            blended = (top_strip * weight_top + bottom_strip * weight_bottom)
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            x0 = c * cell_w
            y_seam = r * cell_h + cell_h
            canvas[y_seam - usable_half : y_seam, x0:x0 + cell_w] = blended

            # Bottom half
            bottom_strip2 = bottom_img[0 : usable_half, :].astype(np.float32)
            top_strip2    = top_img[cell_h - usable_half : cell_h, :].astype(np.float32)
            alpha_b = np.linspace(0.0, 1.0, usable_half, dtype=np.float32)
            weight_bottom2 = alpha_b[:, np.newaxis, np.newaxis]
            weight_top2    = 1.0 - weight_bottom2
            blended_b = (top_strip2 * weight_top2 + bottom_strip2 * weight_bottom2)
            blended_b = np.clip(blended_b, 0, 255).astype(np.uint8)
            canvas[y_seam : y_seam + usable_half, x0:x0 + cell_w] = blended_b


def _draw_unsurveyed_cell(canvas: np.ndarray, x0: int, y0: int, w: int, h: int):
    """Draw dark grey fill + 'Not surveyed' label for an empty grid cell."""
    canvas[y0:y0 + h, x0:x0 + w] = _UNSURVEYED_COLOUR
    text_x = x0 + max(2, w // 2 - 35)
    text_y = y0 + h // 2
    cv2.putText(canvas, _UNSURVEYED_LABEL, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1, cv2.LINE_AA)
