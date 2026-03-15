# processing/shadow_detector.py — Shadow detection via adaptive thresholding
#
# Method: grayscale → adaptive threshold (handles uneven flashlight brightness)
# → morphological cleanup → connected components → shadow vs dark-object
# discrimination using boundary edge gradient analysis.
#
# Why adaptive? A single global threshold (Otsu) fails when the flashlight
# creates uneven brightness — one side bright, the other dim. Adaptive
# thresholding computes a local threshold per pixel neighborhood.

import json
import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Minimum shadow region area in pixels — smaller blobs are noise
_MIN_REGION_AREA_PX = 100

# Adaptive threshold parameters
_BLOCK_SIZE = 51   # neighborhood size (must be odd)
_C_OFFSET = 10     # constant subtracted from local mean

# Morphological kernel sizes
_OPEN_KERNEL_SIZE = 5   # noise removal
_CLOSE_KERNEL_SIZE = 9  # gap filling

# Boundary gradient threshold for shadow vs object discrimination
_GRADIENT_THRESHOLD = 30  # mean edge gradient along boundary


class ShadowDetector:

    def run(self, image_path: str) -> dict:
        """
        Detect shadow regions in a real image using adaptive thresholding.

        Args:
            image_path: Absolute or relative path to the saved JPEG.

        Returns dict:
            {
                "shadow_mask":       numpy uint8 array (255=shadow, 0=lit),
                "shadow_mask_path":  str path to saved mask image,
                "shadow_percentage": float (0–100),
                "shadow_regions":    list of region dicts,
            }
        Returns None if the image cannot be read.
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"ShadowDetector: cannot read '{image_path}'")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Step 1: Adaptive threshold ────────────────────────────────────
        # Gaussian-weighted local mean handles uneven flashlight illumination
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=_BLOCK_SIZE,
            C=_C_OFFSET,
        )

        # ── Step 2: Morphological opening — remove small noise spots ──────
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_OPEN_KERNEL_SIZE, _OPEN_KERNEL_SIZE)
        )
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # ── Step 3: Morphological closing — fill gaps in shadow regions ───
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_CLOSE_KERNEL_SIZE, _CLOSE_KERNEL_SIZE)
        )
        shadow_mask = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        # ── Step 4: Connected components ──────────────────────────────────
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            shadow_mask, connectivity=8
        )

        # Precompute Sobel gradients for edge-strength analysis
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        shadow_regions = []

        for label_idx in range(1, num_labels):  # skip label 0 = background
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < _MIN_REGION_AREA_PX:
                # Remove small regions from the mask as well
                shadow_mask[labels == label_idx] = 0
                continue

            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h_bbox = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            cx = float(centroids[label_idx, 0])
            cy = float(centroids[label_idx, 1])

            # ── Step 5: Shadow vs dark-object discrimination ──────────
            # Extract the binary region for this label
            region_mask = (labels == label_idx).astype(np.uint8) * 255

            # Find boundary pixels via contour
            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            # Compute mean edge gradient along the boundary
            mean_gradient = 0.0
            if contours:
                boundary_pixels = contours[0].reshape(-1, 2)  # (N, 2) x,y
                bx = np.clip(boundary_pixels[:, 0], 0, gradient_mag.shape[1] - 1)
                by = np.clip(boundary_pixels[:, 1], 0, gradient_mag.shape[0] - 1)
                mean_gradient = float(np.mean(gradient_mag[by, bx]))

            # Soft edge → shadow, hard edge → object (rock, etc.)
            if mean_gradient > _GRADIENT_THRESHOLD:
                region_type = "object"
                # Remove objects from the shadow mask — they aren't shadows
                shadow_mask[labels == label_idx] = 0
            else:
                region_type = "shadow"

            shadow_regions.append({
                "label": label_idx,
                "area_px": area,
                "width_px": w,
                "height_px": h_bbox,
                "centroid": {"x": round(cx, 1), "y": round(cy, 1)},
                "type": region_type,
                "mean_boundary_gradient": round(mean_gradient, 1),
            })

        # Recalculate shadow percentage after removing objects
        shadow_count = len([r for r in shadow_regions if r["type"] == "shadow"])
        object_count = len([r for r in shadow_regions if r["type"] == "object"])
        shadow_percentage = float(np.sum(shadow_mask > 0)) / shadow_mask.size * 100.0

        logger.info(
            f"ShadowDetector: '{os.path.basename(image_path)}' — "
            f"shadow={shadow_percentage:.1f}%, "
            f"regions={shadow_count} shadow + {object_count} object"
        )

        # Save mask image
        mask_path = _save_mask(image_path, shadow_mask, img)

        # Save JSON data
        _save_shadow_json(shadow_percentage, shadow_regions)

        return {
            "shadow_mask": shadow_mask,
            "shadow_mask_path": mask_path,
            "shadow_percentage": round(shadow_percentage, 2),
            "shadow_regions": shadow_regions,
        }


def _save_mask(image_path: str, shadow_mask: np.ndarray, original_bgr: np.ndarray) -> str:
    """
    Save a visual overlay: original image with shadow regions tinted blue.
    Returns the saved path.
    """
    os.makedirs(config.PROCESSED_DIR + "/shadow_masks", exist_ok=True)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(config.PROCESSED_DIR, "shadow_masks", basename + "_shadow.png")

    # Blue tint over shadow pixels
    overlay = original_bgr.copy()
    overlay[shadow_mask > 0] = (180, 60, 0)  # BGR: dark blue
    blended = cv2.addWeighted(original_bgr, 0.5, overlay, 0.5, 0)

    # White border on shadow regions
    contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 255, 255), 1)

    cv2.imwrite(out_path, blended)
    logger.debug(f"Shadow mask saved: {out_path}")
    return out_path


def _save_shadow_json(shadow_pct: float, regions: list):
    """Save shadow analysis results as JSON for the dashboard."""
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(config.PROCESSED_DIR, "shadow_data.json")
    data = {
        "shadow_pct": round(shadow_pct, 2),
        "regions": [
            {
                "id": r.get("label", i + 1),
                "area_px": r["area_px"],
                "width_px": r["width_px"],
                "height_px": r["height_px"],
                "centroid": [r["centroid"]["x"], r["centroid"]["y"]],
                "type": r["type"],
                "mean_boundary_gradient": r["mean_boundary_gradient"],
            }
            for i, r in enumerate(regions)
        ],
    }
    try:
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save shadow_data.json: {e}")
