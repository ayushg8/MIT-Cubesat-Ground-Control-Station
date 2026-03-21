from __future__ import annotations
# processing/mosaic_stitcher.py — Single-image mosaic display
#
# Each new image replaces the canvas. No stitching — the dashboard always
# shows the latest received image. All images are still tracked as entries
# so the pipeline (hazard classification, YOLO, segmentation, routing) can
# reference them.

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


class MosaicEntry:
    """Metadata for one registered image in the mosaic."""
    __slots__ = ("filename", "image_path", "homography", "bbox")

    def __init__(self, filename, image_path, homography, bbox):
        self.filename = filename
        self.image_path = image_path
        self.homography = homography
        self.bbox = bbox              # (x, y, w, h) in mosaic space


class MosaicStitcher:
    """
    Displays the latest image as the mosaic canvas.
    Tracks all received images as entries for the CV pipeline.
    Thread-safe: caller (Pipeline) holds a lock before calling register_image().
    """

    def __init__(self):
        self._canvas: np.ndarray | None = None
        self._entries: list[MosaicEntry] = []

        self._source_paths: list[str] = []
        self._source_filenames: list[str] = []

        self._origin_x = 0
        self._origin_y = 0

        os.makedirs(_DB_DIR, exist_ok=True)
        self._load()

    def register_image(self, image_path: str, metadata: dict | None = None) -> dict:
        """Register a new image — it becomes the displayed mosaic."""
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"MosaicStitcher: cannot read '{image_path}'")
            return self._error_result()

        h, w = img.shape[:2]
        basename = os.path.basename(image_path)

        # Use this image as the canvas
        self._canvas = img.copy()

        self._source_paths.append(image_path)
        self._source_filenames.append(basename)

        # Entry bbox covers the full image at (0,0)
        entry = MosaicEntry(
            filename=basename, image_path=image_path,
            homography=np.eye(3, dtype=np.float64),
            bbox=(0, 0, w, h),
        )
        self._entries.append(entry)

        self._save_canvas()
        self._save_index()

        logger.info(f"MosaicStitcher: '{basename}' set as canvas ({w}x{h}), total images={len(self._entries)}")

        return {
            "mosaic_bbox": (0, 0, w, h),
            "entry_index": len(self._entries) - 1,
            "method": "latest",
            "canvas_size": (w, h),
            "image_count": len(self._entries),
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
        """Return the latest entry (since canvas = latest image)."""
        if self._entries:
            return [self._entries[-1]]
        return []

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

        self._source_paths = index.get("source_paths", [])
        self._source_filenames = index.get("source_filenames", [])

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
