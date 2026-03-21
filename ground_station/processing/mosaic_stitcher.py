from __future__ import annotations
# processing/mosaic_stitcher.py — SIFT-based mosaic stitcher
#
# Incrementally builds a mosaic canvas. Each new image is matched against
# the existing canvas using SIFT features + homography, then warped and
# blended into the growing canvas with distance-weighted feathering.

import json
import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

_DB_DIR = os.path.join(config.PROCESSED_DIR, "mosaic_database")
_DB_INDEX_FILE = os.path.join(_DB_DIR, "mosaic_index.json")
_CANVAS_FILE = os.path.join(_DB_DIR, "mosaic_canvas.png")

# SIFT matching parameters
_LOWE_RATIO = 0.7
_MIN_GOOD_MATCHES = 8
_RANSAC_THRESH = 5.0
_MAX_ROTATION_DEG = 90.0   # allow large rotations (satellite tumble)
_MAX_SCALE_CHANGE = 0.4    # reject if scale changes more than 40%


def _distance_weight_map(h, w):
    """Create a weight map where center pixels have weight 1.0 and edges fade to 0."""
    ys = np.linspace(0, 1, h, dtype=np.float32)
    xs = np.linspace(0, 1, w, dtype=np.float32)
    # Distance from each edge, clamped to [0, 1]
    wy = np.minimum(ys, 1.0 - ys) * 2.0  # 0 at edges, 1 at center
    wx = np.minimum(xs, 1.0 - xs) * 2.0
    # Combine: minimum of x and y distance (so corners fade too)
    weight = np.outer(wy, wx)
    # Sharpen: make the center region (inner 60%) fully opaque
    weight = np.clip(weight * 3.0, 0, 1)
    return weight


class MosaicEntry:
    """Metadata for one registered image in the mosaic."""
    __slots__ = ("filename", "image_path", "homography", "bbox")

    def __init__(self, filename, image_path, homography, bbox):
        self.filename = filename
        self.image_path = image_path
        self.homography = homography
        self.bbox = bbox


class MosaicStitcher:
    """
    Builds a continuous mosaic from incoming images using SIFT feature
    matching and homography warping with distance-weighted blending.
    """

    def __init__(self):
        self._canvas: np.ndarray | None = None
        self._canvas_weights: np.ndarray | None = None  # accumulated weight map
        self._entries: list[MosaicEntry] = []
        self._source_paths: list[str] = []
        self._source_filenames: list[str] = []
        self._origin_x = 0
        self._origin_y = 0

        self._sift = cv2.SIFT_create(nfeatures=3000)
        self._canvas_kp = None
        self._canvas_desc = None

        os.makedirs(_DB_DIR, exist_ok=True)
        self._load()

    def register_image(self, image_path: str, metadata: dict | None = None) -> dict:
        """Register and stitch a new image into the mosaic canvas."""
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"MosaicStitcher: cannot read '{image_path}'")
            return self._error_result()

        h, w = img.shape[:2]
        basename = os.path.basename(image_path)
        self._source_paths.append(image_path)
        self._source_filenames.append(basename)

        # First image: base canvas
        if self._canvas is None:
            self._canvas = img.copy()
            self._canvas_weights = np.ones((h, w), dtype=np.float32)
            self._update_canvas_features()

            entry = MosaicEntry(
                filename=basename, image_path=image_path,
                homography=np.eye(3, dtype=np.float64),
                bbox=(0, 0, w, h),
            )
            self._entries.append(entry)
            self._save_canvas()
            self._save_index()

            logger.info(f"MosaicStitcher: '{basename}' as base canvas ({w}x{h})")
            return {
                "mosaic_bbox": (0, 0, w, h),
                "entry_index": 0,
                "method": "base",
                "canvas_size": (w, h),
                "image_count": 1,
            }

        # Match new image against canvas
        H, method = self._find_homography(img)

        if H is None:
            logger.warning(f"MosaicStitcher: no match for '{basename}', overlaying at origin")
            H = np.eye(3, dtype=np.float64)
            method = "overlay"

        # Apply origin offset
        T_origin = np.array([
            [1, 0, self._origin_x],
            [0, 1, self._origin_y],
            [0, 0, 1],
        ], dtype=np.float64)
        H_canvas = T_origin @ H

        # Compute where corners land
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H_canvas).reshape(-1, 2)

        # Compute new canvas bounds
        ch, cw = self._canvas.shape[:2]
        all_x = np.concatenate([warped_corners[:, 0], [0, cw]])
        all_y = np.concatenate([warped_corners[:, 1], [0, ch]])
        min_x = int(np.floor(min(all_x.min(), 0)))
        min_y = int(np.floor(min(all_y.min(), 0)))
        max_x = int(np.ceil(max(all_x.max(), cw)))
        max_y = int(np.ceil(max(all_y.max(), ch)))

        dx = max(0, -min_x)
        dy = max(0, -min_y)
        new_w = min(max_x + dx, 4000)
        new_h = min(max_y + dy, 4000)

        # Expand canvas if needed
        if dx > 0 or dy > 0 or new_w > cw or new_h > ch:
            expanded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            expanded[dy:dy + ch, dx:dx + cw] = self._canvas
            self._canvas = expanded

            expanded_w = np.zeros((new_h, new_w), dtype=np.float32)
            if self._canvas_weights is not None:
                expanded_w[dy:dy + ch, dx:dx + cw] = self._canvas_weights
            self._canvas_weights = expanded_w

            self._origin_x += dx
            self._origin_y += dy
            H_canvas[0, 2] += dx
            H_canvas[1, 2] += dy

        out_size = (self._canvas.shape[1], self._canvas.shape[0])

        # Warp new image and its weight map
        warped = cv2.warpPerspective(img, H_canvas, out_size)
        img_weight = _distance_weight_map(h, w)
        warped_weight = cv2.warpPerspective(img_weight, H_canvas, out_size)

        # Distance-weighted blending: each pixel = weighted average of
        # canvas and new image based on their distance-from-edge weights
        mask = warped_weight > 0.001
        canvas_f = self._canvas.astype(np.float32)
        warped_f = warped.astype(np.float32)
        cw_map = self._canvas_weights  # existing accumulated weights

        total_weight = cw_map + warped_weight
        total_weight_safe = np.where(total_weight > 0, total_weight, 1.0)

        for c in range(3):
            blended = (canvas_f[:, :, c] * cw_map + warped_f[:, :, c] * warped_weight) / total_weight_safe
            # Only update where new image has content
            self._canvas[:, :, c] = np.where(mask, blended.astype(np.uint8), self._canvas[:, :, c])

        # Update accumulated weights (cap to prevent old images dominating forever)
        self._canvas_weights = np.minimum(total_weight, 5.0)

        # Compute bbox
        wc = warped_corners + np.array([dx, dy])
        bx, by = int(wc[:, 0].min()), int(wc[:, 1].min())
        bw, bh = int(wc[:, 0].max()) - bx, int(wc[:, 1].max()) - by

        entry = MosaicEntry(
            filename=basename, image_path=image_path,
            homography=H_canvas, bbox=(bx, by, bw, bh),
        )
        self._entries.append(entry)

        self._update_canvas_features()
        self._save_canvas()
        self._save_index()

        cw_f, ch_f = self._canvas.shape[1], self._canvas.shape[0]
        logger.info(
            f"MosaicStitcher: '{basename}' stitched via {method}, "
            f"canvas {cw_f}x{ch_f}, images={len(self._entries)}"
        )

        return {
            "mosaic_bbox": (bx, by, bw, bh),
            "entry_index": len(self._entries) - 1,
            "method": method,
            "canvas_size": (cw_f, ch_f),
            "image_count": len(self._entries),
        }

    def _find_homography(self, img: np.ndarray):
        """Find homography from img → canvas.

        Strategy:
        1. Try matching against the full canvas
        2. If that fails, try matching against each source image individually
           and chain the homography through that image's known placement
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp_img, desc_img = self._sift.detectAndCompute(gray, None)

        if desc_img is None:
            return None, "no_features"

        # --- Attempt 1: match against canvas directly ---
        H, method = self._match_against(kp_img, desc_img, self._canvas_kp, self._canvas_desc)
        if H is not None:
            return H, method

        # --- Attempt 2: match against each source image, chain homographies ---
        best_H = None
        best_inliers = 0
        best_method = "no_match"

        for entry in reversed(self._entries):  # try newest first
            src_img = cv2.imread(entry.image_path)
            if src_img is None:
                continue
            src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
            src_kp, src_desc = self._sift.detectAndCompute(src_gray, None)
            if src_desc is None:
                continue

            H_to_src, method_src = self._match_against(kp_img, desc_img, src_kp, src_desc)
            if H_to_src is not None:
                # Chain: img → source_image → canvas
                H_chained = entry.homography @ H_to_src
                # Count inliers from the method string
                try:
                    inliers = int(method_src.split("_")[2].replace("i", ""))
                except (IndexError, ValueError):
                    inliers = _MIN_GOOD_MATCHES

                if inliers > best_inliers:
                    best_H = H_chained
                    best_inliers = inliers
                    best_method = f"chain_{entry.filename}_{method_src}"

        return best_H, best_method

    def _match_against(self, kp_img, desc_img, kp_ref, desc_ref):
        """Try to find homography from img features to reference features."""
        if desc_ref is None or kp_ref is None:
            return None, "no_ref_features"
        if len(kp_img) < _MIN_GOOD_MATCHES or len(kp_ref) < _MIN_GOOD_MATCHES:
            return None, "too_few_features"

        bf = cv2.BFMatcher()
        try:
            raw_matches = bf.knnMatch(desc_img, desc_ref, k=2)
        except cv2.error:
            return None, "match_error"

        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < _LOWE_RATIO * n.distance:
                    good.append(m)

        if len(good) < _MIN_GOOD_MATCHES:
            return None, "insufficient_matches"

        src_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, _RANSAC_THRESH)
        if H is None:
            return None, "homography_failed"

        inliers = int(mask.sum()) if mask is not None else 0

        # Validate
        sx = np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
        sy = np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)
        rotation = np.degrees(np.arctan2(H[1, 0], H[0, 0]))

        if abs(rotation) > _MAX_ROTATION_DEG:
            return None, "too_much_rotation"
        if abs(sx - 1.0) > _MAX_SCALE_CHANGE or abs(sy - 1.0) > _MAX_SCALE_CHANGE:
            return None, "too_much_scale"

        det = np.linalg.det(H[:2, :2])
        if det < 0.3 or det > 3.0:
            return None, "degenerate"

        logger.info(f"MosaicStitcher: {len(good)} matches, {inliers} inliers, rot={rotation:.1f}°")
        return H, f"sift_{len(good)}m_{inliers}i"

    def _update_canvas_features(self):
        if self._canvas is None:
            return
        gray = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        self._canvas_kp, self._canvas_desc = self._sift.detectAndCompute(gray, None)

    # -----------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------

    def get_canvas(self) -> np.ndarray | None:
        return self._canvas

    def get_canvas_size(self) -> tuple[int, int]:
        if self._canvas is None:
            return (0, 0)
        return (self._canvas.shape[1], self._canvas.shape[0])

    def get_entries(self) -> list[MosaicEntry]:
        return self._entries

    def get_image_count(self) -> int:
        return len(self._entries)

    def get_overlapping_entries(self, bbox: tuple) -> list[MosaicEntry]:
        bx, by, bw, bh = bbox
        results = []
        for entry in self._entries:
            ex, ey, ew, eh = entry.bbox
            if bx < ex + ew and bx + bw > ex and by < ey + eh and by + bh > ey:
                results.append(entry)
        return results if results else (self._entries[-1:] if self._entries else [])

    def get_entry_by_filename(self, filename: str) -> MosaicEntry | None:
        for entry in self._entries:
            if entry.filename == filename:
                return entry
        return None

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def _save_canvas(self):
        if self._canvas is not None:
            os.makedirs(_DB_DIR, exist_ok=True)
            cv2.imwrite(_CANVAS_FILE, self._canvas)
            mosaic_dir = os.path.join(config.PROCESSED_DIR, "mosaics")
            os.makedirs(mosaic_dir, exist_ok=True)
            cv2.imwrite(os.path.join(mosaic_dir, "mosaic_latest.png"), self._canvas)

    def _save_index(self):
        os.makedirs(_DB_DIR, exist_ok=True)
        index = {
            "source_paths": self._source_paths,
            "source_filenames": self._source_filenames,
            "origin_x": self._origin_x,
            "origin_y": self._origin_y,
            "entries": [],
        }
        for i, entry in enumerate(self._entries):
            h_file = f"entry_{i}_H.npy"
            np.save(os.path.join(_DB_DIR, h_file), entry.homography)
            index["entries"].append({
                "filename": entry.filename,
                "image_path": entry.image_path,
                "bbox": list(entry.bbox),
                "h_file": h_file,
            })
        with open(_DB_INDEX_FILE, "w") as f:
            json.dump(index, f, indent=2)

    def _load(self):
        if not os.path.exists(_DB_INDEX_FILE):
            return
        if os.path.exists(_CANVAS_FILE):
            self._canvas = cv2.imread(_CANVAS_FILE)
            if self._canvas is not None:
                self._canvas_weights = np.ones(
                    (self._canvas.shape[0], self._canvas.shape[1]), dtype=np.float32
                )
                self._update_canvas_features()
        try:
            with open(_DB_INDEX_FILE) as f:
                index = json.load(f)
        except Exception as e:
            logger.warning(f"MosaicStitcher: could not load index: {e}")
            return

        self._source_paths = index.get("source_paths", [])
        self._source_filenames = index.get("source_filenames", [])
        self._origin_x = index.get("origin_x", 0)
        self._origin_y = index.get("origin_y", 0)

        for edata in index.get("entries", []):
            H = np.eye(3, dtype=np.float64)
            h_path = os.path.join(_DB_DIR, edata.get("h_file", ""))
            if os.path.exists(h_path):
                H = np.load(h_path)
            entry = MosaicEntry(
                filename=edata["filename"],
                image_path=edata["image_path"],
                homography=H,
                bbox=tuple(edata["bbox"]),
            )
            self._entries.append(entry)

        logger.info(f"MosaicStitcher: loaded {len(self._entries)} entries")

    def _error_result(self) -> dict:
        return {
            "mosaic_bbox": (0, 0, 0, 0),
            "entry_index": -1,
            "method": "error",
            "canvas_size": self.get_canvas_size(),
            "image_count": len(self._entries),
        }
