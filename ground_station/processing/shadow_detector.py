# processing/shadow_detector.py — Shadow detection via Otsu thresholding
#
# Method: grayscale → Otsu threshold → binary mask → connected components.
# Shadow regions in the real image appear as dark areas. Otsu finds the optimal
# threshold separating lit surface from shadow without manual tuning.

import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Minimum shadow region area in pixels — smaller blobs are noise
_MIN_REGION_AREA_PX = 30


class ShadowDetector:

    def run(self, image_path: str) -> dict:
        """
        Detect shadow regions in a real image.

        Args:
            image_path: Absolute or relative path to the saved JPEG.

        Returns dict:
            {
                "shadow_mask":     numpy uint8 array (255=shadow, 0=lit),
                "shadow_mask_path": str path to saved mask image,
                "shadow_percentage": float (0–100),
                "shadow_regions":  list of region dicts,
                "otsu_threshold":  int,
            }
        Returns None if the image cannot be read.
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"ShadowDetector: cannot read '{image_path}'")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Otsu threshold — finds the optimal split between dark (shadow) and light (surface)
        otsu_val, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # binary: 255 where pixel is DARKER than Otsu threshold → shadow

        shadow_mask = binary  # uint8, 255 = shadow

        shadow_percentage = float(np.sum(shadow_mask > 0)) / shadow_mask.size * 100.0

        # Connected components — label individual shadow blobs
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            shadow_mask, connectivity=8
        )

        shadow_regions = []

        for label_idx in range(1, num_labels):  # skip label 0 = background
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < _MIN_REGION_AREA_PX:
                continue

            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h_bbox = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            cx = float(centroids[label_idx, 0])
            cy = float(centroids[label_idx, 1])

            shadow_length_px = max(w, h_bbox)

            shadow_regions.append({
                "label": label_idx,
                "area_px": area,
                "width_px": w,
                "height_px": h_bbox,
                "centroid": {"x": round(cx, 1), "y": round(cy, 1)},
                "shadow_length_px": shadow_length_px,
            })

        logger.info(
            f"ShadowDetector: '{os.path.basename(image_path)}' — "
            f"Otsu={int(otsu_val)}, shadow={shadow_percentage:.1f}%, "
            f"regions={len(shadow_regions)}"
        )

        # Save mask image
        mask_path = _save_mask(image_path, shadow_mask, img)

        return {
            "shadow_mask": shadow_mask,
            "shadow_mask_path": mask_path,
            "shadow_percentage": round(shadow_percentage, 2),
            "shadow_regions": shadow_regions,
            "otsu_threshold": int(otsu_val),
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
