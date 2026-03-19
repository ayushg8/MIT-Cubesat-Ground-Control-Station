from __future__ import annotations
# processing/mosaic_stitcher.py — Mosaic stitcher using OpenCV Stitcher (SCANS mode)
#
# Uses cv2.Stitcher_create(cv2.Stitcher_SCANS) which is specifically designed
# for flat/planar scene stitching (overhead nadir imagery). Internally it runs
# SIFT features, bundle adjustment, seam finding, and multi-band blending.
#
# On each new image:
#   1. Re-stitch all images from scratch using cv2.Stitcher
#   2. Recover per-image bounding boxes via template matching against the result
#   3. Save the stitched mosaic for dashboard display
#
# With 6-12 images this takes 2-5 seconds — acceptable for a ground station
# that receives one image every ~23 seconds.
#
# Public API is identical to previous implementations.

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

# Margin trimmed from each image when doing template matching for bbox recovery
_MATCH_MARGIN = 30


class MosaicEntry:
    """Metadata for one registered image in the mosaic."""
    __slots__ = ("filename", "image_path", "homography", "bbox")

    def __init__(self, filename, image_path, homography, bbox):
        self.filename = filename
        self.image_path = image_path
        self.homography = homography  # 3x3 translation matrix (image -> mosaic)
        self.bbox = bbox              # (x, y, w, h) in mosaic space


class MosaicStitcher:
    """
    Stitches images into a mosaic using OpenCV's Stitcher in SCANS mode.
    Re-stitches all images on each new addition for optimal alignment.
    Thread-safe: caller (Pipeline) holds a lock before calling register_image().
    """

    def __init__(self):
        self._canvas: np.ndarray | None = None
        self._entries: list[MosaicEntry] = []

        # Keep all source images in memory for re-stitching
        self._source_images: list[np.ndarray] = []
        self._source_paths: list[str] = []
        self._source_filenames: list[str] = []

        # Origin offset (for API compatibility with pipeline.get_mosaic_info)
        self._origin_x = 0
        self._origin_y = 0

        os.makedirs(_DB_DIR, exist_ok=True)
        self._load()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def register_image(self, image_path: str, metadata: dict | None = None) -> dict:
        """Register a new image into the mosaic.

        Re-stitches all images using OpenCV Stitcher for optimal results.
        Falls back to simple side-by-side placement if stitching fails.
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"MosaicStitcher: cannot read '{image_path}'")
            return self._error_result()

        h, w = img.shape[:2]
        basename = os.path.basename(image_path)

        # Add to source collection
        self._source_images.append(img)
        self._source_paths.append(image_path)
        self._source_filenames.append(basename)

        n = len(self._source_images)

        if n == 1:
            # First image — just use it directly
            self._canvas = img.copy()
            entry = MosaicEntry(
                filename=basename, image_path=image_path,
                homography=np.eye(3, dtype=np.float64),
                bbox=(0, 0, w, h),
            )
            self._entries = [entry]
            self._save_canvas()
            self._save_index()
            logger.info(f"MosaicStitcher: '{basename}' placed as mosaic base ({w}x{h})")
            return {
                "mosaic_bbox": (0, 0, w, h),
                "entry_index": 0,
                "method": "base",
                "canvas_size": (w, h),
                "image_count": 1,
            }

        # Re-stitch all images
        method = "stitcher"
        stitched = self._stitch_all()

        if stitched is None:
            # Stitcher failed — fall back to simple horizontal strip
            logger.warning("MosaicStitcher: OpenCV Stitcher failed, using strip fallback")
            method = "strip_fallback"
            stitched = self._strip_fallback()

        self._canvas = stitched

        # Recover per-image bounding boxes via template matching
        self._entries = self._recover_bboxes()

        self._save_canvas()
        self._save_index()

        # Find bbox for the latest image
        latest_entry = None
        for e in self._entries:
            if e.filename == basename:
                latest_entry = e
                break

        bbox = latest_entry.bbox if latest_entry else (0, 0, w, h)
        idx = self._entries.index(latest_entry) if latest_entry else n - 1

        logger.info(
            f"MosaicStitcher: '{basename}' — re-stitched {n} images via {method}, "
            f"canvas {self._canvas.shape[1]}x{self._canvas.shape[0]}"
        )

        return {
            "mosaic_bbox": bbox,
            "entry_index": idx,
            "method": method,
            "canvas_size": (self._canvas.shape[1], self._canvas.shape[0]),
            "image_count": n,
        }

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
        x, y, w, h = bbox
        results = []
        for entry in self._entries:
            ex, ey, ew, eh = entry.bbox
            if (x < ex + ew and x + w > ex and y < ey + eh and y + h > ey):
                results.append(entry)
        return results

    def get_entry_by_filename(self, filename: str) -> MosaicEntry | None:
        for entry in self._entries:
            if entry.filename == filename:
                return entry
        return None

    # -----------------------------------------------------------------
    # Stitching
    # -----------------------------------------------------------------

    def _stitch_all(self) -> np.ndarray | None:
        """Stitch all source images using OpenCV Stitcher (SCANS mode)."""
        if len(self._source_images) < 2:
            return self._source_images[0].copy() if self._source_images else None

        try:
            stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
            status, result = stitcher.stitch(self._source_images)

            if status == cv2.Stitcher_OK:
                return result

            status_names = {
                cv2.Stitcher_ERR_NEED_MORE_IMGS: "NEED_MORE_IMGS",
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "HOMOGRAPHY_FAIL",
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "CAMERA_PARAMS_FAIL",
            }
            logger.warning(
                f"MosaicStitcher: SCANS stitcher failed: "
                f"{status_names.get(status, status)}"
            )

            # Try PANORAMA mode as backup
            stitcher2 = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            status2, result2 = stitcher2.stitch(self._source_images)
            if status2 == cv2.Stitcher_OK:
                logger.info("MosaicStitcher: PANORAMA mode succeeded as fallback")
                return result2

            logger.warning(
                f"MosaicStitcher: PANORAMA mode also failed: "
                f"{status_names.get(status2, status2)}"
            )
            return None

        except Exception as e:
            logger.error(f"MosaicStitcher: stitcher exception: {e}")
            return None

    def _strip_fallback(self) -> np.ndarray:
        """Simple horizontal strip layout when stitcher fails."""
        gap = 4
        imgs = self._source_images
        if not imgs:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        max_h = max(im.shape[0] for im in imgs)
        total_w = sum(im.shape[1] for im in imgs) + gap * (len(imgs) - 1)

        canvas = np.zeros((max_h, total_w, 3), dtype=np.uint8)
        x = 0
        for im in imgs:
            h, w = im.shape[:2]
            canvas[:h, x:x + w] = im
            x += w + gap

        return canvas

    # -----------------------------------------------------------------
    # Bounding box recovery
    # -----------------------------------------------------------------

    def _recover_bboxes(self) -> list[MosaicEntry]:
        """Find where each source image sits in the stitched canvas."""
        if self._canvas is None:
            return []

        entries = []
        canvas_gray = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        m = _MATCH_MARGIN

        for i, img in enumerate(self._source_images):
            h, w = img.shape[:2]
            filename = self._source_filenames[i]
            image_path = self._source_paths[i]

            # Use a trimmed center region as template to avoid edge artifacts
            # from blending/cropping
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if h > m * 2 + 20 and w > m * 2 + 20:
                template = gray[m:-m, m:-m]
            else:
                template = gray

            try:
                result = cv2.matchTemplate(canvas_gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                mx = max_loc[0] - m
                my = max_loc[1] - m

                # Clamp to canvas bounds
                mx = max(0, mx)
                my = max(0, my)

                bbox = (mx, my, w, h)

                H = np.eye(3, dtype=np.float64)
                H[0, 2] = mx
                H[1, 2] = my

                entries.append(MosaicEntry(
                    filename=filename, image_path=image_path,
                    homography=H, bbox=bbox,
                ))
            except Exception as e:
                logger.warning(f"MosaicStitcher: bbox recovery failed for {filename}: {e}")
                # Fallback: use sequential position
                entries.append(MosaicEntry(
                    filename=filename, image_path=image_path,
                    homography=np.eye(3, dtype=np.float64),
                    bbox=(0, 0, w, h),
                ))

        return entries

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

        try:
            with open(_DB_INDEX_FILE) as f:
                index = json.load(f)
        except Exception as e:
            logger.warning(f"MosaicStitcher: could not load index: {e}")
            return

        # Reload source images for future re-stitching
        self._source_paths = index.get("source_paths", [])
        self._source_filenames = index.get("source_filenames", [])
        self._source_images = []

        for path in self._source_paths:
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    self._source_images.append(img)
                    continue
            # Image not found — can't reload, clear everything
            logger.warning(f"MosaicStitcher: source image missing: {path}")
            self._source_images = []
            self._source_paths = []
            self._source_filenames = []
            self._entries = []
            self._canvas = None
            return

        # Reload entries
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

        logger.info(
            f"MosaicStitcher: loaded {len(self._entries)} entries, "
            f"{len(self._source_images)} source images"
        )

    def _error_result(self) -> dict:
        return {
            "mosaic_bbox": (0, 0, 0, 0),
            "entry_index": -1,
            "method": "error",
            "canvas_size": self.get_canvas_size(),
            "image_count": len(self._entries),
        }
