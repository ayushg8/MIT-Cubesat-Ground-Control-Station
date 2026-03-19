from __future__ import annotations
# processing/pixel_segmenter.py — Pixel-level terrain segmentation
#
# Combines existing shadow masks and YOLO bounding-box detections into a
# per-pixel label map. No new ML models required — uses adaptive thresholding
# and morphology to extract precise hazard shapes within YOLO bboxes.
#
# Labels:
#   0 = UNSURVEYED   (not yet imaged)
#   1 = SAND         (safe traversal)
#   2 = PLAIN_SURFACE (safe, featureless)
#   3 = SHADOW       (uncertain — high traversal cost)
#   4 = CRATER       (hazardous)
#   5 = BOULDER      (impassable)

import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Semantic labels
UNSURVEYED = 0
SAND = 1
PLAIN_SURFACE = 2
SHADOW = 3
CRATER = 4
BOULDER = 5

LABEL_NAMES = {
    UNSURVEYED: "UNSURVEYED",
    SAND: "SAND",
    PLAIN_SURFACE: "PLAIN_SURFACE",
    SHADOW: "SHADOW",
    CRATER: "CRATER",
    BOULDER: "BOULDER",
}

# Color map for visualization (BGR)
LABEL_COLORS = {
    UNSURVEYED:    (80, 80, 80),      # dark gray
    SAND:          (200, 220, 240),    # light sandy
    PLAIN_SURFACE: (180, 200, 180),    # pale green
    SHADOW:        (100, 50, 50),      # dark blue-ish
    CRATER:        (0, 100, 255),      # orange
    BOULDER:       (0, 0, 200),        # red
}

# Where segmentation visualizations are saved
SEG_VIS_DIR = os.path.join(config.PROCESSED_DIR, "segmentation_maps")


class PixelSegmenter:
    """Produces a per-pixel label map from shadow masks and YOLO detections."""

    def segment(self, image_path, shadow_mask=None, yolo_detections=None):
        """Generate a pixel-level segmentation label map.

        Args:
            image_path: Path to the source JPEG image.
            shadow_mask: Optional np.ndarray (H, W) uint8 where >0 means shadow.
                         If None, no shadow labeling is applied.
            yolo_detections: Optional list of dicts, each with:
                - "class": str — "crater", "boulder", or "plain"
                - "bbox": [x1, y1, x2, y2] in pixel coords
                - "confidence": float 0-1

        Returns:
            np.ndarray (H, W) uint8 — label map with values from {0..5}.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Start with all pixels as SAND (safe)
        label_map = np.full((h, w), SAND, dtype=np.uint8)

        # Apply shadow mask
        if shadow_mask is not None:
            if shadow_mask.shape[:2] == (h, w):
                label_map[shadow_mask > 0] = SHADOW

        # Process YOLO detections
        if yolo_detections:
            for det in yolo_detections:
                cls = det.get("class", "").lower()
                bbox = det.get("bbox", [])
                if len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Suppress shadow within YOLO segmentation contour
                # (dark pixels on the object itself aren't navigation shadows)
                contour = det.get("contour")
                if contour and len(contour) >= 3:
                    contour_pts = np.array(contour, dtype=np.int32)
                    contour_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(contour_mask, [contour_pts], 255)
                    shadow_in_contour = (label_map == SHADOW) & (contour_mask > 0)
                    label_map[shadow_in_contour] = SAND

                if cls in ("plain", "plain_surface"):
                    # Mark as PLAIN_SURFACE where not already a hazard
                    roi = label_map[y1:y2, x1:x2]
                    safe_mask = (roi == SAND)
                    roi[safe_mask] = PLAIN_SURFACE
                elif cls in ("crater", "boulder", "obstacle"):
                    label = CRATER if cls == "crater" else BOULDER
                    if contour and len(contour) >= 3:
                        # Use precise segmentation contour instead of bbox thresholding
                        contour_pts = np.array(contour, dtype=np.int32)
                        cv2.fillPoly(label_map, [contour_pts], int(label))
                    else:
                        self._segment_object_in_bbox(
                            gray, label_map, x1, y1, x2, y2, label
                        )

        # Dilate all hazard labels by safety margin
        if config.SEG_SAFETY_DILATION_PX > 0:
            label_map = self._dilate_hazards(label_map)

        # Save visualization
        vis_path = self._save_visualization(label_map, image_path)
        logger.info(
            f"PixelSegmenter: '{os.path.basename(image_path)}' — "
            f"shadow={np.sum(label_map == SHADOW) / label_map.size * 100:.1f}% "
            f"crater={np.sum(label_map == CRATER) / label_map.size * 100:.1f}% "
            f"boulder={np.sum(label_map == BOULDER) / label_map.size * 100:.1f}%"
        )

        return label_map

    def _segment_object_in_bbox(self, gray, label_map, x1, y1, x2, y2, label):
        """Extract precise object contour within a YOLO bbox using adaptive threshold."""
        roi_gray = gray[y1:y2, x1:x2]
        bh, bw = roi_gray.shape

        # Adaptive threshold — objects (rocks, craters) are typically darker than sand
        block_size = max(3, (min(bh, bw) // 4) | 1)  # ensure odd
        thresh = cv2.adaptiveThreshold(
            roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 10
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bbox_area = bw * bh
        min_area = bbox_area * (config.SEG_MIN_CONTOUR_AREA_PCT / 100.0)

        # Keep contours larger than minimum area threshold
        good_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if good_contours:
            mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(mask, good_contours, -1, 255, cv2.FILLED)
            label_map[y1:y2, x1:x2][mask > 0] = label
        else:
            # Fallback: fill an ellipse covering SEG_FALLBACK_ELLIPSE_PCT of bbox
            self._fill_fallback_ellipse(label_map, x1, y1, x2, y2, label)

    def _fill_fallback_ellipse(self, label_map, x1, y1, x2, y2, label):
        """Fill an ellipse within the bbox as a conservative fallback."""
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        scale = (config.SEG_FALLBACK_ELLIPSE_PCT / 100.0) ** 0.5
        ax = int((x2 - x1) * scale / 2)
        ay = int((y2 - y1) * scale / 2)
        if ax < 1 or ay < 1:
            return
        cv2.ellipse(label_map, (cx, cy), (ax, ay), 0, 0, 360, int(label), cv2.FILLED)

    def _dilate_hazards(self, label_map):
        """Dilate hazard pixels (SHADOW, CRATER, BOULDER) by the safety margin."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * config.SEG_SAFETY_DILATION_PX + 1, 2 * config.SEG_SAFETY_DILATION_PX + 1)
        )

        result = label_map.copy()

        # Dilate in order of increasing severity so worse labels overwrite
        for hazard_label in (SHADOW, CRATER, BOULDER):
            mask = (label_map == hazard_label).astype(np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            result[dilated > 0] = np.maximum(result[dilated > 0], hazard_label)

        return result

    def _save_visualization(self, label_map, image_path):
        """Save a color-coded segmentation map as a PNG."""
        os.makedirs(SEG_VIS_DIR, exist_ok=True)

        h, w = label_map.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for label_val, color in LABEL_COLORS.items():
            vis[label_map == label_val] = color

        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(SEG_VIS_DIR, f"{base}_seg.png")
        cv2.imwrite(out_path, vis)
        return out_path
