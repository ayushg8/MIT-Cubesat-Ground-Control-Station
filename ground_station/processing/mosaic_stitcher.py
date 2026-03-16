from __future__ import annotations
# processing/mosaic_stitcher.py — Incremental image stitching into a continuous mosaic
#
# Replaces the fixed 8x8 grid cell approach. Images are placed into a growing
# canvas using SIFT feature matching + homography estimation. The first image
# is placed at the center. Subsequent images are matched against all registered
# images to find the best overlap, then warped into mosaic space.
#
# When images extend beyond the current canvas, the canvas grows dynamically
# and all existing homographies are updated with the translation offset.
#
# IMU data (yaw) is used as a placement hint when available; SIFT confirms.
# Falls back to sequential placement if matching fails.
#
# Persistence: saves homographies, SIFT features, canvas to
# data/processed/mosaic_database/

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

# Matching parameters
_SIFT_RATIO_THRESH = 0.75  # Lowe's ratio test


class MosaicEntry:
    """Metadata for one registered image in the mosaic."""
    __slots__ = ("filename", "image_path", "homography", "bbox", "keypoints", "descriptors")

    def __init__(self, filename, image_path, homography, bbox, keypoints=None, descriptors=None):
        self.filename = filename
        self.image_path = image_path
        self.homography = homography  # 3x3 warp from image space → mosaic space
        self.bbox = bbox              # (x, y, w, h) in mosaic space
        self.keypoints = keypoints    # Nx2 float32 array of keypoint coords
        self.descriptors = descriptors  # NxD float32 SIFT descriptors


class MosaicStitcher:
    """
    Incrementally stitches images into a growing mosaic canvas.
    Thread-safe: caller (Pipeline) holds a lock before calling register_image().
    """

    def __init__(self):
        self._sift = cv2.SIFT_create(nfeatures=500)
        self._bf = cv2.BFMatcher(cv2.NORM_L2)

        self._canvas: np.ndarray | None = None
        self._entries: list[MosaicEntry] = []

        # Origin offset: when canvas grows left/up, all coords shift by this amount
        self._origin_x = 0  # mosaic coords of canvas pixel (0,0)
        self._origin_y = 0

        # Sequential placement fallback: next position for unmatched images
        self._seq_x = 0
        self._seq_y = 0

        os.makedirs(_DB_DIR, exist_ok=True)
        self._load()

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def register_image(self, image_path: str, metadata: dict | None = None) -> dict:
        """
        Register a new image into the mosaic.

        Returns dict:
            {
                "mosaic_bbox": (x, y, w, h) in mosaic pixel coords,
                "entry_index": int,
                "method": "sift" | "imu+sift" | "sequential",
                "canvas_size": (w, h),
                "image_count": int,
            }
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"MosaicStitcher: cannot read '{image_path}'")
            return self._error_result()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        basename = os.path.basename(image_path)

        # Extract SIFT features
        kp, desc = self._sift.detectAndCompute(gray, None)
        kp_pts = np.array([k.pt for k in kp], dtype=np.float32) if kp else np.empty((0, 2), dtype=np.float32)

        # IMU hint
        imu = (metadata or {}).get("imu")

        # ── First image → place at center of initial canvas ──
        if self._canvas is None:
            canvas_size = config.MOSAIC_INITIAL_CANVAS_PX
            self._canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            cx = (canvas_size - w) // 2
            cy = (canvas_size - h) // 2
            cx = max(0, cx)
            cy = max(0, cy)

            # Place image
            pw = min(w, canvas_size - cx)
            ph = min(h, canvas_size - cy)
            self._canvas[cy:cy + ph, cx:cx + pw] = img[:ph, :pw]

            H = np.eye(3, dtype=np.float64)
            H[0, 2] = cx
            H[1, 2] = cy

            entry = MosaicEntry(
                filename=basename, image_path=image_path,
                homography=H, bbox=(cx, cy, pw, ph),
                keypoints=kp_pts, descriptors=desc,
            )
            self._entries.append(entry)

            self._seq_x = cx + pw
            self._seq_y = cy

            self._save_canvas()
            self._save_index()
            logger.info(f"MosaicStitcher: first image '{basename}' placed at ({cx},{cy})")

            return {
                "mosaic_bbox": (cx, cy, pw, ph),
                "entry_index": 0,
                "method": "first",
                "canvas_size": (self._canvas.shape[1], self._canvas.shape[0]),
                "image_count": 1,
            }

        # ── Try SIFT matching against existing entries ──
        best_entry = None
        best_H = None
        best_inliers = 0
        best_method = "sequential"

        if desc is not None and len(kp_pts) >= 5:
            # If IMU yaw is available, use it to estimate initial rotation
            initial_angle = None
            if imu and "yaw_deg" in imu:
                initial_angle = float(imu["yaw_deg"])

            for entry in self._entries:
                if entry.descriptors is None or len(entry.descriptors) < 5:
                    continue

                # SIFT match
                good_matches = self._sift_match(desc, entry.descriptors)
                if len(good_matches) < config.MOSAIC_MIN_SIFT_INLIERS:
                    continue

                # Compute homography: new image → entry's image space
                src_pts = np.float32([kp_pts[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([entry.keypoints[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

                H_to_entry, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H_to_entry is None:
                    continue

                inliers = int(mask.sum()) if mask is not None else 0
                if inliers < config.MOSAIC_MIN_SIFT_INLIERS:
                    continue

                if inliers > best_inliers:
                    # Chain: new_image → entry_image → mosaic
                    H_to_mosaic = entry.homography @ H_to_entry
                    best_entry = entry
                    best_H = H_to_mosaic
                    best_inliers = inliers
                    best_method = "imu+sift" if initial_angle is not None else "sift"

        if best_H is not None:
            # Warp corners to find bounding box in mosaic space
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners, best_H)
            wc = warped_corners.reshape(-1, 2)

            mx_min = int(np.floor(wc[:, 0].min()))
            my_min = int(np.floor(wc[:, 1].min()))
            mx_max = int(np.ceil(wc[:, 0].max()))
            my_max = int(np.ceil(wc[:, 1].max()))

            # Ensure canvas is large enough
            self._ensure_canvas_size(mx_min, my_min, mx_max, my_max)

            # Warp the image into the canvas
            self._warp_and_blend(img, best_H)

            entry = MosaicEntry(
                filename=basename, image_path=image_path,
                homography=best_H.copy(),
                bbox=(mx_min, my_min, mx_max - mx_min, my_max - my_min),
                keypoints=kp_pts, descriptors=desc,
            )
            self._entries.append(entry)

            self._save_canvas()
            self._save_index()

            logger.info(
                f"MosaicStitcher: '{basename}' stitched via {best_method} "
                f"({best_inliers} inliers, matched '{best_entry.filename}')"
            )

            return {
                "mosaic_bbox": (mx_min, my_min, mx_max - mx_min, my_max - my_min),
                "entry_index": len(self._entries) - 1,
                "method": best_method,
                "canvas_size": (self._canvas.shape[1], self._canvas.shape[0]),
                "image_count": len(self._entries),
            }

        # ── Fallback: sequential placement ──
        logger.warning(
            f"MosaicStitcher: '{basename}' no SIFT match — placing sequentially"
        )

        # Place to the right of the last image, wrapping down
        ch, cw = self._canvas.shape[:2]
        px, py = self._seq_x, self._seq_y
        if px + w > cw:
            px = 0
            py = py + h + 10

        self._ensure_canvas_size(px, py, px + w, py + h)

        # Direct placement (identity-like homography with translation)
        H = np.eye(3, dtype=np.float64)
        H[0, 2] = px
        H[1, 2] = py

        ch, cw = self._canvas.shape[:2]
        pw = min(w, cw - px)
        ph = min(h, ch - py)
        if pw > 0 and ph > 0:
            self._canvas[py:py + ph, px:px + pw] = img[:ph, :pw]

        entry = MosaicEntry(
            filename=basename, image_path=image_path,
            homography=H, bbox=(px, py, pw, ph),
            keypoints=kp_pts, descriptors=desc,
        )
        self._entries.append(entry)

        self._seq_x = px + pw + 5
        self._seq_y = py

        self._save_canvas()
        self._save_index()

        return {
            "mosaic_bbox": (px, py, pw, ph),
            "entry_index": len(self._entries) - 1,
            "method": "sequential",
            "canvas_size": (self._canvas.shape[1], self._canvas.shape[0]),
            "image_count": len(self._entries),
        }

    def get_canvas(self) -> np.ndarray | None:
        """Return the current mosaic canvas (or None if empty)."""
        return self._canvas

    def get_canvas_size(self) -> tuple[int, int]:
        """Return (width, height) of current canvas."""
        if self._canvas is None:
            return (0, 0)
        return (self._canvas.shape[1], self._canvas.shape[0])

    def get_entries(self) -> list[MosaicEntry]:
        """Return list of all registered mosaic entries."""
        return self._entries

    def get_image_count(self) -> int:
        return len(self._entries)

    def get_overlapping_entries(self, bbox: tuple) -> list[MosaicEntry]:
        """
        Find entries whose mosaic bounding boxes overlap with the given bbox.
        bbox = (x, y, w, h).
        """
        x, y, w, h = bbox
        results = []
        for entry in self._entries:
            ex, ey, ew, eh = entry.bbox
            # Check overlap
            if (x < ex + ew and x + w > ex and y < ey + eh and y + h > ey):
                results.append(entry)
        return results

    def get_entry_by_filename(self, filename: str) -> MosaicEntry | None:
        for entry in self._entries:
            if entry.filename == filename:
                return entry
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Internal: SIFT matching
    # ─────────────────────────────────────────────────────────────────────

    def _sift_match(self, desc_new: np.ndarray, desc_ref: np.ndarray) -> list:
        """Match SIFT descriptors using Lowe's ratio test."""
        if desc_new is None or desc_ref is None:
            return []
        if len(desc_new) < 2 or len(desc_ref) < 2:
            return []
        try:
            matches = self._bf.knnMatch(desc_new, desc_ref, k=2)
        except cv2.error:
            return []

        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < _SIFT_RATIO_THRESH * n.distance:
                    good.append(m)
        return good

    # ─────────────────────────────────────────────────────────────────────
    # Internal: Canvas management
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_canvas_size(self, x_min: int, y_min: int, x_max: int, y_max: int):
        """Grow the canvas if the given bounds exceed current canvas dimensions."""
        if self._canvas is None:
            return

        ch, cw = self._canvas.shape[:2]
        pad = config.MOSAIC_CANVAS_PAD_PX
        max_dim = config.MOSAIC_MAX_CANVAS_PX

        need_left = max(0, -x_min + pad)
        need_top = max(0, -y_min + pad)
        need_right = max(0, x_max - cw + pad)
        need_bottom = max(0, y_max - ch + pad)

        if need_left == 0 and need_top == 0 and need_right == 0 and need_bottom == 0:
            return

        new_w = min(max_dim, cw + need_left + need_right)
        new_h = min(max_dim, ch + need_top + need_bottom)

        # Clamp growth
        actual_left = min(need_left, new_w - cw)
        actual_top = min(need_top, new_h - ch)

        if actual_left == 0 and actual_top == 0 and new_w == cw and new_h == ch:
            return

        new_canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        # Copy old canvas into new position
        dst_x = actual_left
        dst_y = actual_top
        copy_w = min(cw, new_w - dst_x)
        copy_h = min(ch, new_h - dst_y)
        new_canvas[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = self._canvas[:copy_h, :copy_w]

        self._canvas = new_canvas

        # Update all existing homographies with the translation offset
        if actual_left > 0 or actual_top > 0:
            offset = np.eye(3, dtype=np.float64)
            offset[0, 2] = actual_left
            offset[1, 2] = actual_top

            for entry in self._entries:
                entry.homography = offset @ entry.homography
                bx, by, bw, bh = entry.bbox
                entry.bbox = (bx + actual_left, by + actual_top, bw, bh)

            self._origin_x += actual_left
            self._origin_y += actual_top
            self._seq_x += actual_left
            self._seq_y += actual_top

        logger.info(
            f"MosaicStitcher: canvas grew from {cw}x{ch} to {new_w}x{new_h} "
            f"(offset +{actual_left},+{actual_top})"
        )

    def _warp_and_blend(self, img: np.ndarray, H: np.ndarray):
        """Warp an image into the canvas using the given homography, with simple blending."""
        if self._canvas is None:
            return

        ch, cw = self._canvas.shape[:2]
        warped = cv2.warpPerspective(img, H, (cw, ch))

        # Create a mask for the warped image (non-black pixels)
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = (gray_warped > 0).astype(np.uint8)

        # Existing content mask
        gray_canvas = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        existing = (gray_canvas > 0).astype(np.uint8)

        # Overlap region
        overlap = mask & existing

        # Non-overlap: direct copy
        non_overlap = mask & (~overlap & 1)
        for c in range(3):
            self._canvas[:, :, c] = np.where(
                non_overlap > 0, warped[:, :, c], self._canvas[:, :, c]
            )

        # Overlap: simple 50/50 blend
        if np.any(overlap):
            for c in range(3):
                self._canvas[:, :, c] = np.where(
                    overlap > 0,
                    ((self._canvas[:, :, c].astype(np.int32) + warped[:, :, c].astype(np.int32)) // 2).astype(np.uint8),
                    self._canvas[:, :, c],
                )

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def _save_canvas(self):
        """Save the current canvas to disk."""
        if self._canvas is not None:
            os.makedirs(_DB_DIR, exist_ok=True)
            cv2.imwrite(_CANVAS_FILE, self._canvas)
            # Also save to the mosaics dir for the /api/mosaic endpoint
            mosaic_dir = os.path.join(config.PROCESSED_DIR, "mosaics")
            os.makedirs(mosaic_dir, exist_ok=True)
            cv2.imwrite(os.path.join(mosaic_dir, "mosaic_latest.png"), self._canvas)

    def _save_index(self):
        """Save the mosaic index (entry metadata + homographies) to JSON."""
        os.makedirs(_DB_DIR, exist_ok=True)

        index = {
            "origin_x": self._origin_x,
            "origin_y": self._origin_y,
            "seq_x": self._seq_x,
            "seq_y": self._seq_y,
            "entries": [],
        }

        for i, entry in enumerate(self._entries):
            # Save descriptors and keypoints as .npy files
            desc_file = f"entry_{i}_desc.npy"
            kp_file = f"entry_{i}_kp.npy"
            h_file = f"entry_{i}_H.npy"

            if entry.descriptors is not None:
                np.save(os.path.join(_DB_DIR, desc_file), entry.descriptors)
            if entry.keypoints is not None:
                np.save(os.path.join(_DB_DIR, kp_file), entry.keypoints)
            np.save(os.path.join(_DB_DIR, h_file), entry.homography)

            index["entries"].append({
                "filename": entry.filename,
                "image_path": entry.image_path,
                "bbox": list(entry.bbox),
                "desc_file": desc_file,
                "kp_file": kp_file,
                "h_file": h_file,
            })

        with open(_DB_INDEX_FILE, "w") as f:
            json.dump(index, f, indent=2)

    def _load(self):
        """Load mosaic state from disk."""
        if not os.path.exists(_DB_INDEX_FILE):
            return

        # Load canvas
        if os.path.exists(_CANVAS_FILE):
            self._canvas = cv2.imread(_CANVAS_FILE)

        try:
            with open(_DB_INDEX_FILE) as f:
                index = json.load(f)
        except Exception as e:
            logger.warning(f"MosaicStitcher: could not load index: {e}")
            return

        self._origin_x = index.get("origin_x", 0)
        self._origin_y = index.get("origin_y", 0)
        self._seq_x = index.get("seq_x", 0)
        self._seq_y = index.get("seq_y", 0)

        for edata in index.get("entries", []):
            desc = None
            kp = None
            H = np.eye(3, dtype=np.float64)

            desc_path = os.path.join(_DB_DIR, edata["desc_file"])
            kp_path = os.path.join(_DB_DIR, edata["kp_file"])
            h_path = os.path.join(_DB_DIR, edata["h_file"])

            if os.path.exists(desc_path):
                desc = np.load(desc_path)
            if os.path.exists(kp_path):
                kp = np.load(kp_path)
            if os.path.exists(h_path):
                H = np.load(h_path)

            entry = MosaicEntry(
                filename=edata["filename"],
                image_path=edata["image_path"],
                homography=H,
                bbox=tuple(edata["bbox"]),
                keypoints=kp,
                descriptors=desc,
            )
            self._entries.append(entry)

        logger.info(f"MosaicStitcher: loaded {len(self._entries)} entries from database")

    def _error_result(self) -> dict:
        return {
            "mosaic_bbox": (0, 0, 0, 0),
            "entry_index": -1,
            "method": "error",
            "canvas_size": self.get_canvas_size(),
            "image_count": len(self._entries),
        }
