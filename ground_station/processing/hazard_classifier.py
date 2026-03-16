# processing/hazard_classifier.py — Per-grid-cell terrain hazard classification
#
# Each image covers one grid cell (from metadata["grid_cell"]).
# This module classifies that WHOLE image into one hazard class and returns
# a cost value for that cell in the 8×8 cost_grid.
#
# Classification is NOT a sub-grid within the image. One image → one cell → one class.
#
# Multi-feature classification using:
#   1. LBP texture variance (sand vs rock discrimination)
#   2. Canny edge density (smooth vs rough terrain)
#   3. Brightness statistics (mean + std)
#   4. Shadow percentage (from shadow detector)
#   5. Contour features (count, size, circularity → crater vs boulder)
#
# Each classification includes a confidence score (0.0–1.0).

import json
import logging
import math
import os

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

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

# Shadow threshold
_SHADOW_PCT_THRESHOLD    = 40.0   # % shadow → SHADOW class
_SHADOW_PCT_MODERATE     = 20.0   # % shadow below this for SAFE

# Contour thresholds
_MIN_SIGNIFICANT_CONTOUR_AREA = 50   # px² — minimum contour to count
_CRATER_CIRCULARITY_MIN       = 0.55
_IMPASSABLE_COVERAGE_PCT      = 50.0


class HazardClassifier:

    def classify(
        self,
        image_path: str,
        shadow_mask: np.ndarray,
        shadow_percentage: float,
        grid_cell: tuple = None,
        mosaic_bbox: tuple = None,
    ) -> dict:
        """
        Classify a single image (= one grid cell) into a hazard class
        using multi-feature analysis.

        Returns dict:
            {
                "hazard_class":    str  (SAFE / MODERATE / SHADOW / HAZARD / IMPASSABLE),
                "cost":            int,
                "grid_cell":       (row, col),
                "hazard_map_path": str,
                "confidence":      float (0.0–1.0),
                "details":         dict  (all extracted features),
            }
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"HazardClassifier: cannot read '{image_path}'")
            return _error_result(grid_cell)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        total_px = h * w

        # ── Extract all features ──────────────────────────────────────────
        features = _extract_features(gray, shadow_mask, shadow_percentage, total_px)

        # ── Classify using decision tree ──────────────────────────────────
        hazard_class, confidence, hazard_mask = _classify_features(
            gray, features, total_px
        )

        cost = _cost(hazard_class)

        logger.info(
            f"HazardClassifier: cell {grid_cell} → {hazard_class} "
            f"(cost={cost}, conf={confidence:.2f}) | "
            f"lbp_var={features['lbp_variance']:.0f} "
            f"edge_den={features['edge_density']:.3f} "
            f"shadow={features['shadow_pct']:.1f}% "
            f"contours={features['significant_contour_count']}"
        )

        map_path = _save_hazard_map(
            image_path, img, shadow_mask, hazard_mask, hazard_class, grid_cell, confidence
        )

        return {
            "hazard_class": hazard_class,
            "cost": cost,
            "grid_cell": grid_cell,
            "mosaic_bbox": mosaic_bbox,
            "hazard_map_path": map_path,
            "confidence": round(confidence, 3),
            "details": features,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(
    gray: np.ndarray,
    shadow_mask: np.ndarray,
    shadow_percentage: float,
    total_px: int,
) -> dict:
    """Extract all five feature groups from the grayscale cell image."""

    # Feature 1: LBP texture variance
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_variance = float(np.var(lbp))

    # Feature 2: Canny edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / total_px

    # Feature 3: Brightness statistics
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))

    # Feature 4: Shadow percentage (passed in from shadow detector)
    shadow_pct = shadow_percentage

    # Feature 5: Contour features from edge detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    significant_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= _MIN_SIGNIFICANT_CONTOUR_AREA:
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
            significant_contours.append({
                "area": area,
                "perimeter": perimeter,
                "circularity": circularity,
            })

    significant_contour_count = len(significant_contours)
    largest_contour_area = max((c["area"] for c in significant_contours), default=0.0)
    largest_contour_circularity = 0.0
    if significant_contours:
        largest = max(significant_contours, key=lambda c: c["area"])
        largest_contour_circularity = largest["circularity"]

    # Total contour area coverage
    total_contour_area = sum(c["area"] for c in significant_contours)
    contour_coverage_pct = total_contour_area / total_px * 100.0

    return {
        "lbp_variance": round(lbp_variance, 1),
        "edge_density": round(edge_density, 4),
        "mean_brightness": round(mean_brightness, 1),
        "std_brightness": round(std_brightness, 1),
        "shadow_pct": round(shadow_pct, 1),
        "significant_contour_count": significant_contour_count,
        "largest_contour_area": round(largest_contour_area, 0),
        "largest_contour_circularity": round(largest_contour_circularity, 3),
        "contour_coverage_pct": round(contour_coverage_pct, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Classification decision tree
# ─────────────────────────────────────────────────────────────────────────────

def _classify_features(
    gray: np.ndarray,
    features: dict,
    total_px: int,
) -> tuple:
    """
    Apply decision tree to extracted features.
    Returns (hazard_class, confidence, hazard_mask).
    """
    lbp_var    = features["lbp_variance"]
    edge_den   = features["edge_density"]
    shadow_pct = features["shadow_pct"]
    n_contours = features["significant_contour_count"]
    largest_area = features["largest_contour_area"]
    largest_circ = features["largest_contour_circularity"]
    contour_cov  = features["contour_coverage_pct"]
    mean_bright  = features["mean_brightness"]

    lbp_high  = config.LBP_VARIANCE_HIGH
    lbp_mod   = config.LBP_VARIANCE_MODERATE
    edge_high = config.EDGE_DENSITY_HIGH
    edge_mod  = config.EDGE_DENSITY_MODERATE

    hazard_mask = np.zeros_like(gray, dtype=np.uint8)

    # Build hazard mask from Canny edges for visualization
    edges = cv2.Canny(gray, 50, 150)

    # ── Decision tree (priority order) ────────────────────────────────

    # 1. IMPASSABLE: massive contour coverage (>50% of cell)
    if contour_cov >= _IMPASSABLE_COVERAGE_PCT:
        hazard_mask = edges
        conf = _confidence_above(contour_cov, _IMPASSABLE_COVERAGE_PCT, 80.0)
        return IMPASSABLE, conf, hazard_mask

    # 2. SHADOW: high shadow coverage (>40%)
    if shadow_pct > _SHADOW_PCT_THRESHOLD:
        conf = _confidence_above(shadow_pct, _SHADOW_PCT_THRESHOLD, 80.0)
        return SHADOW, conf, hazard_mask

    # 3. HAZARD — crater: high edges + high texture + large circular dark contour
    is_crater = (
        edge_den > edge_high
        and lbp_var > lbp_high
        and largest_circ >= _CRATER_CIRCULARITY_MIN
        and largest_area >= 500
        and mean_bright < 140
    )
    if is_crater:
        hazard_mask = edges
        conf = min(
            _confidence_above(edge_den, edge_high, 0.3),
            _confidence_above(lbp_var, lbp_high, 1000),
        )
        return HAZARD, conf, hazard_mask

    # 4. HAZARD — boulder: high edges + high texture + small bright irregular contours
    is_boulder = (
        edge_den > edge_high
        and lbp_var > lbp_high
        and largest_circ < 0.5
        and n_contours >= 3
        and mean_bright > 100
    )
    if is_boulder:
        hazard_mask = edges
        conf = min(
            _confidence_above(edge_den, edge_high, 0.3),
            _confidence_above(lbp_var, lbp_high, 1000),
        )
        return HAZARD, conf, hazard_mask

    # 5. MODERATE: elevated edges or texture
    if edge_den > edge_mod or lbp_var > lbp_mod:
        # Confidence based on how far above the moderate threshold
        conf_edge = _confidence_above(edge_den, edge_mod, edge_high) if edge_den > edge_mod else 0.5
        conf_lbp  = _confidence_above(lbp_var, lbp_mod, lbp_high) if lbp_var > lbp_mod else 0.5
        conf = max(conf_edge, conf_lbp)
        return MODERATE, conf, hazard_mask

    # 6. SAFE: low edges, low texture, low shadow
    if edge_den < edge_mod and lbp_var < lbp_mod and shadow_pct < _SHADOW_PCT_MODERATE:
        # Higher confidence when further from thresholds
        conf_edge = _confidence_below(edge_den, edge_mod)
        conf_lbp  = _confidence_below(lbp_var, lbp_mod)
        conf = min(conf_edge, conf_lbp)
        return SAFE, conf, hazard_mask

    # Fallback SAFE with lower confidence
    return SAFE, 0.5, hazard_mask


# ─────────────────────────────────────────────────────────────────────────────
# Confidence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_above(value: float, threshold: float, max_range: float) -> float:
    """Confidence for a value above a threshold. Higher = more confident."""
    if max_range <= threshold:
        return 0.85
    distance = value - threshold
    normalized = min(distance / (max_range - threshold), 1.0)
    return round(0.55 + 0.45 * normalized, 3)


def _confidence_below(value: float, threshold: float) -> float:
    """Confidence for a value below a threshold. Lower value = more confident."""
    if threshold <= 0:
        return 0.85
    ratio = value / threshold
    return round(0.55 + 0.45 * (1.0 - ratio), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Cost + visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    confidence: float,
) -> str:
    """Overlay shadow (blue) and hazard (red) regions on the image + legend banner."""
    os.makedirs(os.path.join(config.PROCESSED_DIR, "hazard_maps"), exist_ok=True)

    overlay = img.copy()

    # Shadow tint (blue-ish)
    if shadow_mask is not None:
        overlay[shadow_mask > 0] = _blend_colour(overlay[shadow_mask > 0], (180, 60, 0), 0.45)

    # Hazard tint (red)
    overlay[hazard_mask > 0] = _blend_colour(overlay[hazard_mask > 0], (0, 0, 220), 0.55)

    # Cell label + class + confidence banner at top
    colour = _COLOURS[hazard_class]
    banner_h = 28
    banner = np.full((banner_h, img.shape[1], 3), colour, dtype=np.uint8)
    label = f"Cell ({grid_cell[0]},{grid_cell[1]}) — {hazard_class} ({confidence:.0%})"
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
        "confidence": 0.0,
        "details": {"error": "could_not_read_image"},
    }


def save_cost_grid_json(
    cost_grid,
    hazard_grid,
    image_index=None,
    change_cells=None,
    confidence_grid=None,
):
    """Save the full cost grid, classifications, and confidences as JSON.

    Args:
        cost_grid:       numpy int array (rows x cols) — dimensions are dynamic
        hazard_grid:     list of lists of class strings
        image_index:     dict from pipeline image_index (optional, for pass_data)
        change_cells:    list of [r,c] cells or mosaic bboxes with changes (optional)
        confidence_grid: numpy float array (rows x cols) or None
    """
    rows, cols = cost_grid.shape
    grid = cost_grid.tolist()
    classifications = [[hazard_grid[r][c] for c in range(cols)] for r in range(rows)]

    # coverage: derive from confidence grid (> 0 means surveyed) or cost grid
    coverage = [[False] * cols for _ in range(rows)]
    pass_data = [[0] * cols for _ in range(rows)]

    if confidence_grid is not None:
        for r in range(rows):
            for c in range(cols):
                if float(confidence_grid[r][c]) > 0:
                    coverage[r][c] = True

    # Extract pass data from image index (now keyed by filename)
    if image_index:
        for key, entries in image_index.items():
            if entries:
                max_pass = max(e.get("pass", 0) for e in entries)
                # Find the grid cell for this entry from mosaic_bbox
                for entry in entries:
                    bbox = entry.get("mosaic_bbox")
                    if bbox:
                        cell_px = config.MOSAIC_GRID_CELL_PX
                        r = int((bbox[1] + bbox[3] / 2) // cell_px)
                        c = int((bbox[0] + bbox[2] / 2) // cell_px)
                        if 0 <= r < rows and 0 <= c < cols:
                            coverage[r][c] = True
                            pass_data[r][c] = max(pass_data[r][c], max_pass)

    data = {
        "grid": grid,
        "rows": rows,
        "cols": cols,
        "classifications": classifications,
        "coverage": coverage,
        "pass_data": pass_data,
        "change_cells": change_cells or [],
    }

    if confidence_grid is not None:
        data["confidences"] = [[round(float(confidence_grid[r][c]), 3) for c in range(cols)] for r in range(rows)]

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(config.PROCESSED_DIR, "cost_grid.json")
    try:
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save cost_grid.json: {e}")
