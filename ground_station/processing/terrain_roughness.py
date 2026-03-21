from __future__ import annotations
# processing/terrain_roughness.py — Classical CV terrain roughness analysis
#
# Tiles the stitched mosaic into small grid cells and classifies each cell's
# surface roughness using texture metrics (local standard deviation + Laplacian
# energy). No ML model needed — pure classical CV.
#
# Uses adaptive percentile-based thresholds so it self-calibrates to whatever
# terrain is in the mosaic (sandy, rocky, mixed).
#
# YOLO-aware: cells overlapping known craters/boulders are pre-labeled ROUGH
# but still get a raw score computed so users can compare roughness within
# hazard zones (e.g. "which crater is smoothest?").
#
# Roughness levels:
#   -1 = UNSURVEYED (empty/black mosaic area)
#    0 = SMOOTH     (flat sand, safe)
#    1 = MODERATE   (some texture variation)
#    2 = ROUGH      (rocky/uneven terrain)
#
# Feeds into route planning: roughness costs are blended into the fine cost grid.

import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Roughness levels
UNSURVEYED = -1
SMOOTH = 0
MODERATE = 1
ROUGH = 2

ROUGHNESS_NAMES = {UNSURVEYED: "UNSURVEYED", SMOOTH: "SMOOTH", MODERATE: "MODERATE", ROUGH: "ROUGH"}

# Colors for the overlay (BGRA — green, amber, red with transparency)
ROUGHNESS_COLORS_BGRA = {
    SMOOTH:   (0, 200, 0, 90),       # green
    MODERATE: (0, 180, 255, 110),     # amber/orange
    ROUGH:    (0, 0, 220, 130),       # red
}

# Cost multipliers for route planning
ROUGHNESS_COST_MULTIPLIER = {
    SMOOTH:   1.0,
    MODERATE: 1.8,
    ROUGH:    3.5,
}

# Percentile thresholds for adaptive classification
_PERCENTILE_MODERATE = 50   # above p50 → MODERATE
_PERCENTILE_ROUGH = 80      # above p80 → ROUGH

# Default cell size — matches fine segmentation grid (20px)
DEFAULT_CELL_SIZE_PX = 20


def analyze_roughness(
    mosaic_path: str,
    cell_size_px: int = DEFAULT_CELL_SIZE_PX,
    yolo_detections_mosaic: list[dict] | None = None,
) -> dict | None:
    """Analyze terrain roughness across the mosaic image.

    Args:
        mosaic_path: Path to the stitched mosaic PNG/JPEG.
        cell_size_px: Size of each analysis cell in pixels (default 20).
        yolo_detections_mosaic: Optional list of YOLO detections in mosaic
            pixel coords. Each dict needs "class" and "bbox_mosaic" [x1,y1,x2,y2]
            or "contour_mosaic" [[x,y],...]. Cells overlapping craters/boulders
            are pre-labeled ROUGH but still get raw scores computed.

    Returns:
        Dict with grid, scores, rows, cols, stats, overlay_path, cost_grid.
        Or None on failure.
    """
    img = cv2.imread(mosaic_path)
    if img is None:
        logger.error(f"TerrainRoughness: cannot read mosaic at {mosaic_path}")
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows = h // cell_size_px
    cols = w // cell_size_px
    if rows < 1 or cols < 1:
        logger.warning("TerrainRoughness: mosaic too small for grid analysis")
        return None

    # Build a pixel-level hazard mask, then downsample to grid cells
    hazard_mask = np.zeros((rows, cols), dtype=np.uint8)  # 0=no hazard, 1=hazard
    if yolo_detections_mosaic:
        # Rasterize hazard polygons/bboxes at grid resolution directly
        # Use a small canvas (rows x cols) to avoid full-res allocation
        for det in yolo_detections_mosaic:
            cls = (det.get("class") or "").lower()
            if cls not in ("crater", "boulder", "obstacle", "rock"):
                continue

            contour = det.get("contour_mosaic")
            if contour and len(contour) >= 3:
                # Scale contour points from mosaic pixels to grid cells
                pts = np.array(contour, dtype=np.float32)
                pts[:, 0] /= cell_size_px
                pts[:, 1] /= cell_size_px
                pts = pts.astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(hazard_mask, [pts], 1)
            else:
                bbox = det.get("bbox_mosaic")
                if not bbox or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox[:4]
                c0 = max(0, int(x1 / cell_size_px))
                r0 = max(0, int(y1 / cell_size_px))
                c1 = min(cols, int(np.ceil(x2 / cell_size_px)))
                r1 = min(rows, int(np.ceil(y2 / cell_size_px)))
                hazard_mask[r0:r1, c0:c1] = 1

    # Pass 1: compute raw roughness score for each cell
    raw_scores = np.full((rows, cols), -1.0, dtype=np.float32)

    for r in range(rows):
        y1 = r * cell_size_px
        y2 = y1 + cell_size_px
        for c in range(cols):
            x1 = c * cell_size_px
            x2 = x1 + cell_size_px

            patch = gray[y1:y2, x1:x2]

            # Skip near-black patches (unsurveyed/empty mosaic regions)
            if float(np.mean(patch)) < 10:
                continue

            # Metric 1: Local standard deviation (intensity variation)
            local_std = float(np.std(patch.astype(np.float32)))

            # Metric 2: Laplacian energy (edge/roughness detector)
            lap = cv2.Laplacian(patch, cv2.CV_32F)
            lap_energy = float(np.mean(np.abs(lap)))

            # Combined score — std and Laplacian are complementary
            raw_scores[r, c] = 0.5 * local_std + 0.5 * lap_energy

    # Pass 2: adaptive thresholding based on distribution of valid cells
    valid_mask = raw_scores >= 0
    valid_scores = raw_scores[valid_mask]

    if len(valid_scores) < 3:
        logger.warning("TerrainRoughness: too few valid cells")
        return None

    thresh_moderate = float(np.percentile(valid_scores, _PERCENTILE_MODERATE))
    thresh_rough = float(np.percentile(valid_scores, _PERCENTILE_ROUGH))

    logger.info(
        f"TerrainRoughness: adaptive thresholds — "
        f"moderate={thresh_moderate:.1f} (p{_PERCENTILE_MODERATE}), "
        f"rough={thresh_rough:.1f} (p{_PERCENTILE_ROUGH})"
    )

    # Pass 3: classify each cell
    grid = np.full((rows, cols), UNSURVEYED, dtype=np.int8)
    cost_grid = np.ones((rows, cols), dtype=np.float32)  # multiplier, default 1.0

    for r in range(rows):
        for c in range(cols):
            score = raw_scores[r, c]
            if score < 0:
                continue  # unsurveyed

            # Classify by texture
            if score >= thresh_rough:
                level = ROUGH
            elif score >= thresh_moderate:
                level = MODERATE
            else:
                level = SMOOTH

            # Override: YOLO hazard cells are at least ROUGH
            if hazard_mask[r, c] > 0 and level < ROUGH:
                level = ROUGH

            grid[r, c] = level
            cost_grid[r, c] = ROUGHNESS_COST_MULTIPLIER[level]

    # Generate overlay image (same size as mosaic, BGRA)
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for r in range(rows):
        y1 = r * cell_size_px
        y2 = y1 + cell_size_px
        for c in range(cols):
            x1 = c * cell_size_px
            x2 = x1 + cell_size_px
            level = int(grid[r, c])
            if level < 0:
                continue
            color = ROUGHNESS_COLORS_BGRA[level]
            overlay[y1:y2, x1:x2] = color

    # Draw subtle grid lines
    for r in range(rows + 1):
        y = r * cell_size_px
        if y < h:
            overlay[y:min(y + 1, h), :, :3] = 60
            overlay[y:min(y + 1, h), :, 3] = 40
    for c in range(cols + 1):
        x = c * cell_size_px
        if x < w:
            overlay[:, x:min(x + 1, w), :3] = 60
            overlay[:, x:min(x + 1, w), 3] = 40

    # Save overlay
    out_dir = os.path.join(config.PROCESSED_DIR, "roughness")
    os.makedirs(out_dir, exist_ok=True)
    overlay_path = os.path.join(out_dir, "roughness_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    # Stats
    valid_cells = grid[grid >= 0]
    total = len(valid_cells)
    smooth_count = int(np.sum(valid_cells == SMOOTH))
    moderate_count = int(np.sum(valid_cells == MODERATE))
    rough_count = int(np.sum(valid_cells == ROUGH))
    hazard_cells = int(np.sum(hazard_mask > 0))

    stats = {
        "total_cells": total,
        "smooth": smooth_count,
        "moderate": moderate_count,
        "rough": rough_count,
        "smooth_pct": round(100 * smooth_count / max(1, total), 1),
        "moderate_pct": round(100 * moderate_count / max(1, total), 1),
        "rough_pct": round(100 * rough_count / max(1, total), 1),
        "yolo_hazard_cells": hazard_cells,
    }

    logger.info(
        f"TerrainRoughness: {rows}x{cols} grid ({cell_size_px}px cells), "
        f"smooth={stats['smooth_pct']}% moderate={stats['moderate_pct']}% "
        f"rough={stats['rough_pct']}% ({hazard_cells} YOLO-hazard cells)"
    )

    return {
        "grid": grid.tolist(),
        "scores": np.round(raw_scores, 2).tolist(),
        "cost_grid": cost_grid.tolist(),
        "rows": rows,
        "cols": cols,
        "cell_size_px": cell_size_px,
        "mosaic_width": w,
        "mosaic_height": h,
        "stats": stats,
        "overlay_path": overlay_path,
    }
