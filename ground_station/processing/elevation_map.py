from __future__ import annotations
# processing/elevation_map.py — Photoclinometry from shadow lengths
#
# Estimates object heights from shadow lengths using the physical flashlight
# geometry measured and recorded in config before demo.
#
# Formula (parallel-light assumption):
#   height_px  = shadow_length_px * tan(FLASHLIGHT_ELEVATION_DEG)
#   height_cm  = height_px * GSD_CM_PER_PIXEL
#
# [FIX #8] Known error — divergent illumination:
#   The sun at the Moon is 150 million km away — rays are effectively parallel.
#   Our flashlight at ~50 cm produces divergent rays that fan out by 15+ degrees.
#   This causes systematic over-estimation at the surface edges.
#   An error estimate is added per region and annotated on the output image.
#   This is documented, not hidden.

import logging
import math
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

_DIVERGENCE_ERROR_NOTE = "Height error increases toward edges (~\u00b115%)"


class ElevationMapper:

    def compute(self, shadow_mask: np.ndarray, shadow_regions: list, image_path: str) -> dict | None:
        """
        Compute height estimates for all shadow regions from a real image.

        Args:
            shadow_mask:    uint8 numpy array from ShadowDetector (255=shadow).
            shadow_regions: List of region dicts from ShadowDetector.
            image_path:     Source image path — used for output filename only.

        Returns dict:
            {
                "elevation_map_path": str,
                "max_height_cm":      float,
                "shadow_regions_analyzed": int,
                "gsd_cm_per_pixel":   float,
                "divergence_error_note": str,
                "regions":            list of dicts with height + error_pct,
            }
        Returns None if config values are not yet filled in.
        """
        if config.FLASHLIGHT_ELEVATION_DEG <= 0.0:
            logger.warning(
                "ElevationMapper: FLASHLIGHT_ELEVATION_DEG is 0.0 — "
                "elevation map skipped. Measure and fill in config before demo."
            )
            return None

        if config.GSD_CM_PER_PIXEL <= 0.0:
            logger.warning(
                "ElevationMapper: GSD_CM_PER_PIXEL is 0.0 — "
                "elevation map skipped. Measure and fill in config before demo."
            )
            return None

        elevation_rad = math.radians(config.FLASHLIGHT_ELEVATION_DEG)
        tan_elev = math.tan(elevation_rad)

        h_img, w_img = shadow_mask.shape
        center_x = w_img / 2.0
        center_y = h_img / 2.0

        # Build a float height map — zero everywhere, filled in over shadow pixels
        height_map = np.zeros((h_img, w_img), dtype=np.float32)

        enriched_regions = []
        max_height_cm = 0.0

        for region in shadow_regions:
            shadow_length_px = region.get("shadow_length_px", 0)
            if shadow_length_px <= 0:
                continue

            try:
                height_px = shadow_length_px * tan_elev
                height_cm = height_px * config.GSD_CM_PER_PIXEL
            except (ZeroDivisionError, ValueError, OverflowError) as e:
                logger.warning(f"ElevationMapper: math error on region {region.get('label')}: {e}")
                continue

            # Divergence error estimate for this region
            cx = region["centroid"]["x"]
            cy = region["centroid"]["y"]
            dist_from_center = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

            try:
                if config.FLASHLIGHT_DISTANCE_CM > 0.0:
                    divergence_angle_rad = math.atan(
                        (dist_from_center * config.GSD_CM_PER_PIXEL) / config.FLASHLIGHT_DISTANCE_CM
                    )
                    error_pct = (math.degrees(divergence_angle_rad) / config.FLASHLIGHT_ELEVATION_DEG) * 100.0
                else:
                    error_pct = None
            except (ZeroDivisionError, ValueError):
                error_pct = None

            # Paint this region's estimated height onto the height map
            bbox = region["bbox"]
            x, y, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            region_mask_slice = shadow_mask[y:y + bh, x:x + bw]
            height_map[y:y + bh, x:x + bw][region_mask_slice > 0] = height_cm

            if height_cm > max_height_cm:
                max_height_cm = height_cm

            enriched = dict(region)
            enriched["height_cm"] = round(height_cm, 3)
            enriched["error_estimate_pct"] = round(error_pct, 1) if error_pct is not None else None
            enriched_regions.append(enriched)

        if not enriched_regions:
            logger.info("ElevationMapper: no shadow regions with valid height estimates")
            return None

        logger.info(
            f"ElevationMapper: {len(enriched_regions)} regions, "
            f"max_height={max_height_cm:.2f} cm"
        )

        map_path = _save_elevation_map(height_map, max_height_cm, image_path)

        return {
            "elevation_map_path": map_path,
            "max_height_cm": round(max_height_cm, 3),
            "shadow_regions_analyzed": len(enriched_regions),
            "gsd_cm_per_pixel": config.GSD_CM_PER_PIXEL,
            "divergence_error_note": _DIVERGENCE_ERROR_NOTE,
            "regions": enriched_regions,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_elevation_map(height_map: np.ndarray, max_height_cm: float, image_path: str) -> str:
    """
    Save a false-colour elevation map PNG with a colourbar and error annotation.
    Uses matplotlib for the colourbar; falls back to a plain OpenCV image if
    matplotlib is unavailable.
    """
    os.makedirs(os.path.join(config.PROCESSED_DIR, "elevation_maps"), exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(config.PROCESSED_DIR, "elevation_maps", basename + "_elevation.png")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        vmax = max(max_height_cm, 0.01)  # avoid vmax=0
        im = ax.imshow(height_map, cmap="plasma", vmin=0, vmax=vmax, origin="upper")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Estimated height (cm)", fontsize=10)
        ax.set_title(f"Elevation Map — {os.path.basename(image_path)}", fontsize=11)
        ax.axis("off")
        fig.text(
            0.5, 0.01, _DIVERGENCE_ERROR_NOTE,
            ha="center", fontsize=8, color="red",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
        )
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    except ImportError:
        # Fallback: normalise height_map to 0-255, apply COLORMAP_PLASMA, add text
        if max_height_cm > 0:
            normalised = (height_map / max_height_cm * 255).astype(np.uint8)
        else:
            normalised = height_map.astype(np.uint8)
        coloured = cv2.applyColorMap(normalised, cv2.COLORMAP_PLASMA)
        cv2.putText(
            coloured, _DIVERGENCE_ERROR_NOTE,
            (10, coloured.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
        )
        cv2.imwrite(out_path, coloured)

    logger.debug(f"Elevation map saved: {out_path}")
    return out_path
