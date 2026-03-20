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
                        self._segment_object_in_bbox(gray, label_map, x1, y1, x2, y2, label)

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
        """Extract an object contour within a bbox using class-specific segmentation."""
        roi_gray = gray[y1:y2, x1:x2]
        bh, bw = roi_gray.shape
        if bh == 0 or bw == 0:
            return

        if label == CRATER:
            if self._segment_crater_in_bbox(roi_gray, label_map, x1, y1, x2, y2):
                return
        elif label == BOULDER:
            if self._segment_boulder_in_bbox(roi_gray, label_map, x1, y1, x2, y2):
                return

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
            self._fill_fallback_ellipse(label_map, x1, y1, x2, y2, label)

    def _segment_crater_in_bbox(self, roi_gray, label_map, x1, y1, x2, y2):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi_gray)
        blur = cv2.GaussianBlur(clahe, (7, 7), 0)
        block_size = max(5, (min(roi_gray.shape[:2]) // 3) | 1)

        adaptive = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 6
        )
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(blur, 40, 120)

        mask = cv2.bitwise_or(adaptive, otsu)
        mask = cv2.bitwise_or(mask, edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contour = self._pick_best_contour(mask, prefer_round=True)
        if contour is None:
            return False

        crater_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
        cv2.drawContours(crater_mask, [contour], -1, 255, cv2.FILLED)
        label_map[y1:y2, x1:x2][crater_mask > 0] = CRATER
        return True

    def _segment_boulder_in_bbox(self, roi_gray, label_map, x1, y1, x2, y2):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi_gray)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        bright = cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT, kernel_large)
        dark = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, kernel_large)

        masks = []
        for feature in (bright, dark):
            thresh_val = max(10.0, float(feature.mean() + feature.std()))
            _, mask = cv2.threshold(feature, thresh_val, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            masks.append(mask)

        combined = masks[0]
        for extra in masks[1:]:
            combined = cv2.bitwise_or(combined, extra)

        contour = self._pick_best_contour(combined, prefer_round=False)
        if contour is None:
            return False

        boulder_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
        cv2.drawContours(boulder_mask, [contour], -1, 255, cv2.FILLED)
        label_map[y1:y2, x1:x2][boulder_mask > 0] = BOULDER
        return True

    def _pick_best_contour(self, mask, prefer_round=True):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = mask.shape[:2]
        cx = w / 2.0
        cy = h / 2.0
        bbox_area = h * w
        min_area = max(30.0, bbox_area * (config.SEG_MIN_CONTOUR_AREA_PCT / 100.0))

        best = None
        best_score = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > bbox_area * 0.95:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            ccx = m["m10"] / m["m00"]
            ccy = m["m01"] / m["m00"]
            distance = ((ccx - cx) ** 2 + (ccy - cy) ** 2) ** 0.5
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if prefer_round and circularity < 0.22:
                continue
            score = (
                (area / max(1.0, bbox_area)) * 0.45
                + circularity * (0.40 if prefer_round else 0.20)
                - distance / max(h, w) * 0.25
            )
            if best is None or score > best_score:
                best = cnt
                best_score = score
        return best

    def _fill_fallback_ellipse(self, label_map, x1, y1, x2, y2, label):
        """Fill an ellipse within the bbox as a conservative fallback."""
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        fallback_pct = config.SEG_FALLBACK_ELLIPSE_PCT
        if label == BOULDER:
            fallback_pct = min(fallback_pct, 45.0)
        elif label == CRATER:
            fallback_pct = min(fallback_pct, 55.0)
        scale = (fallback_pct / 100.0) ** 0.5
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
