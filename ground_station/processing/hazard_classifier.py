# processing/hazard_classifier.py — Per-grid-cell terrain hazard classification
#
# Each image covers one grid cell (from metadata["grid_cell"]).
# This module classifies that WHOLE image into one hazard class and returns
# a cost value for that cell in the 8×8 cost_grid.
#
# Classification is NOT a sub-grid within the image. One image → one cell → one class.
#
# Detection cascade (in priority order):
#   1. IMPASSABLE — >50% of image covered by hazard contours
#   2. SHADOW     — >30% of image is shadow (from shadow_mask)
#   3. HAZARD     — large dark circular contour (crater) OR small bright irregular (boulder)
#   4. MODERATE   — high local texture variance
#   5. SAFE       — default (well-lit, low variance, no triggers)

import json
import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Hazard class constants
SAFE       = "SAFE"
MODERATE   = "MODERATE"
SHADOW     = "SHADOW"
HAZARD     = "HAZARD"
IMPASSABLE = "IMPASSABLE"

# BGR colours for the hazard map overlay
_COLOURS = {
    SAFE:       (0,   200,   0),    # green
    MODERATE:   (0,   200, 200),    # yellow
    SHADOW:     (180,  60,   0),    # blue
    HAZARD:     (0,    0,  220),    # red
    IMPASSABLE: (0,    0,  120),    # dark red
}

# Thresholds
_SHADOW_PCT_THRESHOLD    = 30.0   # % shadow → SHADOW class
_IMPASSABLE_COVERAGE_PCT = 50.0   # % hazard contour area → IMPASSABLE
_MODERATE_VARIANCE       = 300.0  # local patch variance above this → MODERATE
_CRATER_MIN_AREA         = 500    # px² — minimum dark circular blob for crater
_BOULDER_MAX_AREA        = 300    # px² — maximum bright irregular blob for boulder
_CRATER_CIRCULARITY_MIN  = 0.55   # 4π·area/perimeter² — how round a crater must be
_PATCH_SIZE              = 16     # for local variance check


class HazardClassifier:

    def classify(
        self,
        image_path: str,
        shadow_mask: np.ndarray,
        shadow_percentage: float,
        grid_cell: tuple,
    ) -> dict:
        """
        Classify a single image (= one grid cell) into a hazard class.

        Args:
            image_path:        Path to the saved JPEG.
            shadow_mask:       uint8 numpy array from ShadowDetector (255=shadow).
            shadow_percentage: Float from ShadowDetector (0–100).
            grid_cell:         (row, col) tuple identifying which cell this image covers.

        Returns dict:
            {
                "hazard_class":  str  (SAFE / MODERATE / SHADOW / HAZARD / IMPASSABLE),
                "cost":          int,
                "grid_cell":     (row, col),
                "hazard_map_path": str,
                "details":       dict  (diagnostic info for logs/dashboard),
            }
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"HazardClassifier: cannot read '{image_path}'")
            return _error_result(grid_cell)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        total_px = h * w

        # --- Step 1: find hazard contours (dark circles + bright irregular blobs) ---
        hazard_mask, details = _find_hazard_contours(gray, total_px)
        hazard_coverage_pct = float(np.sum(hazard_mask > 0)) / total_px * 100.0

        # --- Classification cascade ---
        if hazard_coverage_pct >= _IMPASSABLE_COVERAGE_PCT:
            hazard_class = IMPASSABLE
        elif shadow_percentage >= _SHADOW_PCT_THRESHOLD:
            hazard_class = SHADOW
        elif details["has_crater"] or details["has_boulder"]:
            hazard_class = HAZARD
        elif _high_texture_variance(gray):
            hazard_class = MODERATE
        else:
            hazard_class = SAFE

        cost = _cost(hazard_class)

        details.update({
            "shadow_pct": round(shadow_percentage, 1),
            "hazard_coverage_pct": round(hazard_coverage_pct, 1),
        })

        logger.info(
            f"HazardClassifier: cell {grid_cell} → {hazard_class} (cost={cost}) | "
            f"shadow={shadow_percentage:.1f}% hazard_cov={hazard_coverage_pct:.1f}% "
            f"crater={details['has_crater']} boulder={details['has_boulder']}"
        )

        map_path = _save_hazard_map(image_path, img, shadow_mask, hazard_mask, hazard_class, grid_cell)

        return {
            "hazard_class": hazard_class,
            "cost": cost,
            "grid_cell": grid_cell,
            "hazard_map_path": map_path,
            "details": details,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_hazard_contours(gray: np.ndarray, total_px: int) -> tuple:
    """
    Find dark circular contours (craters) and small bright irregular blobs (boulders).
    Returns (hazard_mask uint8, details dict).
    """
    hazard_mask = np.zeros_like(gray, dtype=np.uint8)
    has_crater = False
    has_boulder = False
    crater_count = 0
    boulder_count = 0

    # — Craters: dark, roughly circular —
    # Threshold to dark pixels, find contours, check area + circularity
    _, dark_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in dark_contours:
        area = cv2.contourArea(cnt)
        if area < _CRATER_MIN_AREA:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity >= _CRATER_CIRCULARITY_MIN:
            cv2.drawContours(hazard_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            has_crater = True
            crater_count += 1

    # — Boulders: small bright irregular blobs —
    mean_brightness = float(gray.mean())
    bright_thresh_val = int(min(255, mean_brightness + 40))
    _, bright_thresh = cv2.threshold(gray, bright_thresh_val, 255, cv2.THRESH_BINARY)
    bright_contours, _ = cv2.findContours(bright_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in bright_contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > _BOULDER_MAX_AREA:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < 0.5:  # irregular = not circular → boulder-shaped
            cv2.drawContours(hazard_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            has_boulder = True
            boulder_count += 1

    return hazard_mask, {
        "has_crater": has_crater,
        "has_boulder": has_boulder,
        "crater_count": crater_count,
        "boulder_count": boulder_count,
    }


def _high_texture_variance(gray: np.ndarray) -> bool:
    """Return True if mean local patch variance exceeds MODERATE threshold."""
    h, w = gray.shape
    variances = []
    for r in range(0, h - _PATCH_SIZE + 1, _PATCH_SIZE):
        for c in range(0, w - _PATCH_SIZE + 1, _PATCH_SIZE):
            patch = gray[r:r + _PATCH_SIZE, c:c + _PATCH_SIZE]
            variances.append(float(np.var(patch)))
    if not variances:
        return False
    return float(np.mean(variances)) > _MODERATE_VARIANCE


def _cost(hazard_class: str) -> int:
    return {
        SAFE:       config.COST_SAFE,
        MODERATE:   config.COST_MODERATE,
        SHADOW:     config.COST_SHADOW,
        HAZARD:     config.COST_HAZARD,
        IMPASSABLE: config.COST_IMPASSABLE,
    }[hazard_class]


def _save_hazard_map(
    image_path: str,
    img: np.ndarray,
    shadow_mask: np.ndarray,
    hazard_mask: np.ndarray,
    hazard_class: str,
    grid_cell: tuple,
) -> str:
    """Overlay shadow (blue) and hazard (red) regions on the image + legend banner."""
    os.makedirs(os.path.join(config.PROCESSED_DIR, "hazard_maps"), exist_ok=True)

    overlay = img.copy()

    # Shadow tint (blue-ish)
    if shadow_mask is not None:
        overlay[shadow_mask > 0] = _blend_colour(overlay[shadow_mask > 0], (180, 60, 0), 0.45)

    # Hazard tint (red)
    overlay[hazard_mask > 0] = _blend_colour(overlay[hazard_mask > 0], (0, 0, 220), 0.55)

    # Cell label + class banner at top
    colour = _COLOURS[hazard_class]
    banner_h = 28
    banner = np.full((banner_h, img.shape[1], 3), colour, dtype=np.uint8)
    label = f"Cell ({grid_cell[0]},{grid_cell[1]}) — {hazard_class}"
    cv2.putText(banner, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    result = np.vstack([banner, overlay])

    # Legend strip at bottom
    legend = _make_legend(img.shape[1])
    result = np.vstack([result, legend])

    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(config.PROCESSED_DIR, "hazard_maps", basename + "_hazard.png")
    cv2.imwrite(out_path, result)
    logger.debug(f"Hazard map saved: {out_path}")
    return out_path


def _blend_colour(region: np.ndarray, bgr: tuple, alpha: float) -> np.ndarray:
    """Blend a solid colour into a pixel region array."""
    colour_arr = np.array(bgr, dtype=np.float32)
    blended = region.astype(np.float32) * (1 - alpha) + colour_arr * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _make_legend(width: int) -> np.ndarray:
    """Create a fixed-height legend strip showing all five hazard classes."""
    classes = [SAFE, MODERATE, SHADOW, HAZARD, IMPASSABLE]
    costs   = [config.COST_SAFE, config.COST_MODERATE, config.COST_SHADOW,
               config.COST_HAZARD, config.COST_IMPASSABLE]
    legend_h = 22
    legend = np.zeros((legend_h, width, 3), dtype=np.uint8)
    cell_w = width // len(classes)

    for i, (cls, cost) in enumerate(zip(classes, costs)):
        x0 = i * cell_w
        x1 = x0 + cell_w
        legend[:, x0:x1] = _COLOURS[cls]
        text = f"{cls} ({cost})"
        cv2.putText(legend, text, (x0 + 3, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return legend


def _error_result(grid_cell: tuple) -> dict:
    return {
        "hazard_class": SAFE,
        "cost": config.COST_SAFE,
        "grid_cell": grid_cell,
        "hazard_map_path": None,
        "details": {"error": "could_not_read_image"},
    }


def save_cost_grid_json(cost_grid, hazard_grid, image_index=None, change_cells=None):
    """Save the full cost grid and classifications as JSON for the dashboard.

    Args:
        cost_grid:    numpy int array (GRID_ROWS x GRID_COLS)
        hazard_grid:  list of lists of class strings
        image_index:  dict from pipeline image_index (optional, for pass_data)
        change_cells: list of [r,c] cells with changes (optional)
    """
    rows, cols = cost_grid.shape
    grid = cost_grid.tolist()
    classifications = [[hazard_grid[r][c] for c in range(cols)] for r in range(rows)]

    # coverage: True if the cell has been surveyed (not default SAFE with cost=1 and no index entry)
    coverage = [[False] * cols for _ in range(rows)]
    pass_data = [[0] * cols for _ in range(rows)]

    if image_index:
        for key, entries in image_index.items():
            parts = key.split(",")
            if len(parts) == 2:
                r, c = int(parts[0]), int(parts[1])
                if 0 <= r < rows and 0 <= c < cols:
                    coverage[r][c] = True
                    if entries:
                        pass_data[r][c] = max(e.get("pass", 0) for e in entries)

    data = {
        "grid": grid,
        "classifications": classifications,
        "coverage": coverage,
        "pass_data": pass_data,
        "change_cells": change_cells or [],
    }

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(config.PROCESSED_DIR, "cost_grid.json")
    try:
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save cost_grid.json: {e}")
