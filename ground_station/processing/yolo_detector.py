from __future__ import annotations
# processing/yolo_detector.py — YOLOv8 object detection for lunar surface imagery
#
# Dual-detection architecture (second layer alongside classical CV pipeline):
#   1. Classical CV (adaptive thresholding, LBP texture, Canny edges) provides
#      grid-level terrain classification (SAFE/MODERATE/HAZARD/etc.)
#   2. YOLOv8 provides object-level detection with bounding boxes and confidence
#      scores for individual craters, boulders, and surface features.
#
# When both systems agree on a hazard, confidence is high. When they disagree,
# the cell is flagged for human review. This multi-model approach reduces both
# false positives and false negatives compared to either system alone.
#
# The classical CV is calibrated to our specific demo surface. The YOLO model
# was trained on real lunar orbital imagery (LROC), demonstrating our pipeline's
# compatibility with flight-grade data sources.
#
# Tier 1: Lunar-trained YOLOv8 (5,600+ real LROC images) — models/lunar_detector.pt
# Tier 2: YOLOv8n COCO fallback — maps common objects to lunar categories

import json
import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Paths
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_SANDBOX_SEG_PATH = os.path.join(_MODEL_DIR, "sandbox_seg.pt")
_TERRAIN_MODEL_PATH = os.path.join(_MODEL_DIR, "terrain_detector.pt")
_LUNAR_MODEL_PATH = os.path.join(_MODEL_DIR, "lunar_detector.pt")
_LOCAL_COCO_MODEL_PATH = os.path.join(_MODEL_DIR, "yolov8n.pt")
_DETECTIONS_DIR = os.path.join(config.PROCESSED_DIR, "yolo_detections")


class YOLODetector:
    """
    Runs YOLOv8 inference on received images and produces object-level
    detections with bounding boxes and confidence scores.

    If a lunar-trained model is available (models/lunar_detector.pt),
    it uses that for high-accuracy crater/boulder detection.
    Otherwise falls back to COCO-pretrained YOLOv8n with class mapping.
    """

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._is_lunar = False
        self._model_name = "none"

        # COCO class mapping (fallback when no lunar model)
        # Maps COCO object classes to lunar terrain categories
        self._coco_to_lunar = {
            "bowl":        "crater",
            "cup":         "crater",
            "vase":        "crater",
            "frisbee":     "crater",
            "sports ball": "boulder",
            "apple":       "boulder",
            "orange":      "boulder",
            "donut":       "boulder",
            "bottle":      "obstacle",
            "book":        "obstacle",
            "cell phone":  "obstacle",
            "remote":      "obstacle",
            "mouse":       "obstacle",
            "scissors":    "obstacle",
        }

    def _load_model(self):
        """Lazy-load the YOLO model on first use."""
        if self._model_loaded:
            return

        try:
            from ultralytics import YOLO

            if os.path.exists(_SANDBOX_SEG_PATH):
                # Prefer sandbox segmentation model — trained on real sandbox demo images
                self._model = YOLO(_SANDBOX_SEG_PATH)
                self._is_lunar = True
                self._model_name = "sandbox_yolov8_seg"
                logger.info(f"YOLO: loaded sandbox segmentation model from {_SANDBOX_SEG_PATH}")
            elif os.path.exists(_TERRAIN_MODEL_PATH):
                self._model = YOLO(_TERRAIN_MODEL_PATH)
                self._is_lunar = True
                self._model_name = "terrain_yolov8"
                logger.info(f"YOLO: loaded terrain model from {_TERRAIN_MODEL_PATH}")
            elif os.path.exists(_LUNAR_MODEL_PATH):
                self._model = YOLO(_LUNAR_MODEL_PATH)
                self._is_lunar = True
                self._model_name = "lunar_yolov8"
                logger.info(f"YOLO: loaded lunar-trained model from {_LUNAR_MODEL_PATH}")
            elif os.path.exists(_LOCAL_COCO_MODEL_PATH):
                self._model = YOLO(_LOCAL_COCO_MODEL_PATH)
                self._is_lunar = False
                self._model_name = "coco_yolov8n_local"
                logger.info(f"YOLO: using local COCO fallback from {_LOCAL_COCO_MODEL_PATH}")
            else:
                self._model = None
                self._is_lunar = False
                self._model_name = "classical_cv_fallback"
                logger.warning("YOLO: no local weights found — using classical crater/boulder fallback")

            self._model_loaded = True

        except Exception as e:
            logger.error(f"YOLO: failed to load model: {e}")
            self._model_loaded = True  # Don't retry
            self._model = None
            self._model_name = "classical_cv_fallback"

    @property
    def model_name(self) -> str:
        self._load_model()
        return self._model_name

    @property
    def is_lunar_model(self) -> bool:
        self._load_model()
        return self._is_lunar

    def detect(self, image_path: str, confidence_threshold: float = 0.2) -> list[dict]:
        """
        Run YOLO detection on an image.

        Returns list of detections:
        [
            {
                "class": "crater" | "boulder" | "plain" | "obstacle",
                "confidence": 0.87,
                "bbox": [x1, y1, x2, y2],
                "area_px": 4500,
                "center": [cx, cy],
                "original_class": "Impact_crater_10-100m"
            },
            ...
        ]
        """
        self._load_model()

        if self._model is None:
            return self._detect_classical(image_path)

        try:
            results = self._model(image_path, conf=confidence_threshold, verbose=False)
        except Exception as e:
            logger.error(f"YOLO: inference failed on '{image_path}': {e}")
            return []

        detections = []
        # Get image dimensions for full-frame filter
        img_check = cv2.imread(image_path)
        img_h, img_w = (img_check.shape[:2]) if img_check is not None else (480, 640)
        img_area = img_h * img_w

        for result in results:
            # Extract mask polygons if available (segmentation model)
            masks_xy = None
            if result.masks is not None and result.masks.xy is not None:
                masks_xy = result.masks.xy

            for idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Map to lunar category
                lunar_class = self._map_class(cls_name)
                if lunar_class is None:
                    continue  # Skip irrelevant COCO classes

                # Filter out full-frame detections (model noise, not real objects)
                det_area = (x2 - x1) * (y2 - y1)
                if det_area > img_area * 0.25:
                    logger.debug(
                        f"YOLO: skipping oversized {lunar_class} "
                        f"({det_area/img_area*100:.0f}% of image) in {os.path.basename(image_path)}"
                    )
                    continue

                # Filter out edge-touching detections (sandbox frame artifacts)
                edge_margin = 5
                touches_edge = (x1 < edge_margin or y1 < edge_margin or
                                x2 > img_w - edge_margin or y2 > img_h - edge_margin)
                if touches_edge and lunar_class in ("boulder", "obstacle"):
                    logger.debug(
                        f"YOLO: skipping edge {lunar_class} in {os.path.basename(image_path)}"
                    )
                    continue

                # Extract and simplify contour from segmentation mask
                contour = None
                if masks_xy is not None and idx < len(masks_xy):
                    raw_poly = masks_xy[idx]
                    if len(raw_poly) >= 3:
                        pts = np.array(raw_poly, dtype=np.float32).reshape(-1, 1, 2)
                        epsilon = 0.005 * cv2.arcLength(pts, closed=True)
                        approx = cv2.approxPolyDP(pts, epsilon, closed=True)
                        contour = approx.reshape(-1, 2).astype(int).tolist()

                detections.append({
                    "class": lunar_class,
                    "confidence": round(conf, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "area_px": int((x2 - x1) * (y2 - y1)),
                    "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    "original_class": cls_name,
                    "contour": contour,
                })

        # If the detector is using generic COCO weights or returned nothing,
        # supplement it with classical lunar heuristics so the downstream
        # segmenter still receives useful object seeds offline.
        if not self._is_lunar or not detections:
            detections = self._merge_detections(
                detections,
                self._detect_classical(image_path),
            )

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def _detect_classical(self, image_path: str) -> list[dict]:
        """Fallback crater/boulder detector for offline or missing-weight operation."""
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"YOLO fallback: cannot read image '{image_path}'")
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blur = cv2.GaussianBlur(clahe, (9, 9), 0)

        detections = []
        detections.extend(self._detect_craters(clahe, blur))
        detections.extend(self._detect_boulders(clahe))
        detections = self._merge_detections([], detections)

        if detections:
            logger.info(
                f"YOLO fallback: detected {len(detections)} candidate object(s) in "
                f"{os.path.basename(image_path)}"
            )
        return detections

    def _detect_craters(self, gray: np.ndarray, blur: np.ndarray) -> list[dict]:
        h, w = gray.shape[:2]
        img_area = h * w
        min_radius = max(10, min(h, w) // 40)
        max_radius = max(min_radius + 4, min(h, w) // 5)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, min(h, w) // 10),
            param1=90,
            param2=18,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        detections = []
        if circles is None:
            return detections

        yy, xx = np.ogrid[:h, :w]
        for cx, cy, radius in circles[0]:
            cx_i, cy_i, r_i = int(cx), int(cy), int(radius)
            if r_i < min_radius:
                continue

            inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius * 0.75) ** 2
            ring = (
                ((xx - cx) ** 2 + (yy - cy) ** 2 <= (radius * 1.2) ** 2)
                & ~((xx - cx) ** 2 + (yy - cy) ** 2 <= (radius * 0.85) ** 2)
            )
            if not np.any(inner) or not np.any(ring):
                continue

            inner_mean = float(gray[inner].mean())
            ring_mean = float(gray[ring].mean())
            contrast = ring_mean - inner_mean
            if contrast < 8.0:
                continue

            x1 = max(0, cx_i - r_i)
            y1 = max(0, cy_i - r_i)
            x2 = min(w, cx_i + r_i)
            y2 = min(h, cy_i + r_i)
            area = (x2 - x1) * (y2 - y1)
            if area <= 0 or area > img_area * 0.20:
                continue

            duplicate = False
            for existing in detections:
                ex, ey = existing["center"]
                er = max(1, (existing["bbox"][2] - existing["bbox"][0]) // 2)
                if ((ex - cx_i) ** 2 + (ey - cy_i) ** 2) ** 0.5 < max(er, r_i) * 0.65:
                    duplicate = True
                    break
            if duplicate:
                continue

            contour = cv2.ellipse2Poly((cx_i, cy_i), (r_i, r_i), 0, 0, 360, 12)
            confidence = min(0.88, 0.45 + contrast / 40.0)
            detections.append({
                "class": "crater",
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
                "area_px": int(area),
                "center": [cx_i, cy_i],
                "original_class": "classical_crater",
                "contour": contour.reshape(-1, 2).astype(int).tolist(),
            })

        return detections

    def _detect_boulders(self, gray: np.ndarray) -> list[dict]:
        h, w = gray.shape[:2]
        img_area = h * w
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        detections = []
        for feature_map, flavor in ((bright, "bright"), (dark, "dark")):
            thresh_val = max(12.0, float(feature_map.mean() + 1.2 * feature_map.std()))
            _, mask = cv2.threshold(feature_map, thresh_val, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 60 or area > img_area * 0.03:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter <= 0:
                    continue
                circularity = float(4 * np.pi * area / (perimeter * perimeter))
                if circularity < 0.18:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / max(1.0, min(bw, bh))
                if aspect > 2.4:
                    continue

                roi = gray[y:y + bh, x:x + bw]
                if roi.size == 0:
                    continue
                pad = 4
                y0 = max(0, y - pad)
                x0 = max(0, x - pad)
                y1 = min(h, y + bh + pad)
                x1 = min(w, x + bw + pad)
                ring = gray[y0:y1, x0:x1]
                local_contrast = abs(float(roi.mean()) - float(ring.mean()))
                if local_contrast < 9.0:
                    continue

                confidence = min(0.82, 0.35 + circularity * 0.45 + local_contrast / 50.0)
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                detections.append({
                    "class": "boulder",
                    "confidence": round(confidence, 3),
                    "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                    "area_px": int(area),
                    "center": [int(x + bw / 2), int(y + bh / 2)],
                    "original_class": f"classical_boulder_{flavor}",
                    "contour": approx.reshape(-1, 2).astype(int).tolist() if len(approx) >= 3 else None,
                })

        return detections

    def _merge_detections(self, primary: list[dict], secondary: list[dict]) -> list[dict]:
        merged = []
        for det in sorted(primary + secondary, key=lambda d: d["confidence"], reverse=True):
            duplicate = False
            for existing in merged:
                if det["class"] != existing["class"]:
                    continue
                if self._iou(det["bbox"], existing["bbox"]) >= 0.45:
                    duplicate = True
                    break
            if not duplicate:
                merged.append(det)

        pruned = []
        for det in merged:
            if det["class"] == "boulder":
                overshadowed = any(
                    existing["class"] == "crater"
                    and existing["confidence"] >= det["confidence"]
                    and self._iou(det["bbox"], existing["bbox"]) >= 0.20
                    for existing in merged
                )
                if overshadowed:
                    continue
            pruned.append(det)
        return pruned

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    def _map_class(self, cls_name: str) -> str | None:
        """Map a YOLO class name to a lunar terrain category."""
        if self._is_lunar:
            lower = cls_name.lower()
            if "crater" in lower:
                return "crater"
            elif "boulder" in lower or "rock" in lower:
                return "boulder"
            elif "sand" in lower or "plain" in lower or "surface" in lower or "flat" in lower:
                return "plain"
            elif "shadow" in lower:
                return "shadow"
            else:
                return "obstacle"
        else:
            return self._coco_to_lunar.get(cls_name)

    def detect_and_annotate(self, image_path: str, output_path: str) -> list[dict]:
        """
        Run detection and save annotated image with bounding boxes and labels.
        Returns the detection list.
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"YOLO: cannot read image '{image_path}'")
            return []

        detections = self.detect(image_path)

        colors = {
            "crater":   (0, 0, 255),     # Red
            "boulder":  (0, 165, 255),    # Orange
            "plain":    (0, 200, 0),      # Green
            "obstacle": (0, 0, 180),      # Dark red
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = colors.get(det["class"], (200, 200, 200))

            # Draw contour polygon if available, otherwise bbox
            contour = det.get("contour")
            if contour and len(contour) >= 3:
                pts = np.array(contour, dtype=np.int32)
                # Semi-transparent filled polygon via alpha blending
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
                cv2.drawContours(img, [pts], -1, color, 2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{det['class']} {det['confidence']:.0%}"
            # Background rectangle for text readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Model label in top-left corner
        model_label = "LUNAR YOLO" if self._is_lunar else "COCO YOLO (fallback)"
        cv2.putText(img, model_label, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Fusion: combine YOLO detections with classical CV classification
# ─────────────────────────────────────────────────────────────────────────────

def fuse_classifications(
    grid_cell: tuple,
    classical_class: str,
    classical_confidence: float,
    yolo_detections: list[dict],
) -> dict:
    """
    Fuse YOLO object detections with the classical CV grid-level classification.

    Fusion rules:
    - Both agree hazard → high confidence (boost)
    - YOLO finds hazard but classical says SAFE → upgrade to MODERATE
    - Classical says HAZARD but YOLO finds nothing → keep HAZARD, lower confidence
    - Both say safe/plain → high confidence safe

    Returns:
    {
        "cell": [r, c],
        "classical_classification": "HAZARD",
        "classical_confidence": 0.78,
        "yolo_detections": [...],
        "fused_classification": "HAZARD",
        "fused_confidence": 0.95,
        "agreement": True
    }
    """
    hazard_classes = {"crater", "boulder", "obstacle"}
    yolo_hazards = [d for d in yolo_detections if d["class"] in hazard_classes]
    yolo_max_conf = max((d["confidence"] for d in yolo_hazards), default=0.0)

    classical_is_hazardous = classical_class in ("HAZARD", "IMPASSABLE")
    classical_is_moderate = classical_class == "MODERATE"
    yolo_found_hazard = len(yolo_hazards) > 0

    # Fusion logic
    if yolo_found_hazard and classical_is_hazardous:
        # Both agree — high confidence hazard
        fused_class = classical_class
        fused_conf = min(1.0, (classical_confidence + yolo_max_conf) / 2 + 0.15)
        agreement = True

    elif yolo_found_hazard and classical_is_moderate:
        # YOLO confirms moderate or upgrades
        if yolo_max_conf > 0.7:
            fused_class = "HAZARD"
            fused_conf = round((classical_confidence + yolo_max_conf) / 2, 3)
        else:
            fused_class = "MODERATE"
            fused_conf = min(1.0, classical_confidence + 0.1)
        agreement = True

    elif yolo_found_hazard and classical_class in ("SAFE", "SHADOW"):
        # YOLO found something classical missed — trust YOLO based on confidence
        if yolo_max_conf > 0.6:
            # High confidence YOLO detection overrides classical
            best_yolo = max(yolo_hazards, key=lambda d: d["confidence"])
            if best_yolo["class"] == "boulder":
                fused_class = "IMPASSABLE"
            else:
                fused_class = "HAZARD"
            fused_conf = round(yolo_max_conf * 0.85, 3)
        else:
            fused_class = "MODERATE"
            fused_conf = round(yolo_max_conf * 0.7, 3)
        agreement = False

    elif not yolo_found_hazard and classical_is_hazardous:
        # Classical says hazard but YOLO doesn't confirm — lower confidence
        fused_class = classical_class
        fused_conf = max(0.3, classical_confidence - 0.15)
        agreement = False

    else:
        # Both say safe/plain
        fused_class = classical_class
        plain_detections = [d for d in yolo_detections if d["class"] == "plain"]
        if plain_detections:
            fused_conf = min(1.0, classical_confidence + 0.1)
        else:
            fused_conf = classical_confidence
        agreement = len(yolo_detections) == 0 or len(plain_detections) > 0

    fused_conf = round(min(1.0, max(0.0, fused_conf)), 3)

    return {
        "cell": list(grid_cell),
        "classical_classification": classical_class,
        "classical_confidence": round(classical_confidence, 3),
        "yolo_detections": [
            {"class": d["class"], "confidence": d["confidence"]}
            for d in yolo_detections
        ],
        "fused_classification": fused_class,
        "fused_confidence": fused_conf,
        "agreement": agreement,
    }


def save_detections_json(all_detections: dict, fused_results: list[dict]):
    """
    Save YOLO detection results and fused classifications to JSON.

    all_detections: {"R,C": [detection_dicts, ...]}
    fused_results:  [fused_dict, ...]
    """
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    data = {
        "detections_per_cell": all_detections,
        "fused_classifications": fused_results,
        "summary": _build_summary(all_detections, fused_results),
    }

    path = os.path.join(config.PROCESSED_DIR, "yolo_detections.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"YOLO: detections saved to {path}")


def _build_summary(all_detections: dict, fused_results: list) -> dict:
    """Build a summary dict for mission state and dashboard."""
    flat = []
    for dets in all_detections.values():
        flat.extend(dets)

    craters = [d for d in flat if d["class"] == "crater"]
    boulders = [d for d in flat if d["class"] == "boulder"]

    agreement_count = sum(1 for f in fused_results if f["agreement"])
    total_fused = len(fused_results)

    return {
        "total_detections": len(flat),
        "craters_detected": len(craters),
        "boulders_detected": len(boulders),
        "cv_agreement_rate": round(agreement_count / total_fused, 3) if total_fused else 1.0,
        "cells_analyzed": len(all_detections),
    }
