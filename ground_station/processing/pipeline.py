from __future__ import annotations
# processing/pipeline.py — CV pipeline orchestrator
#
# Called by receiver/listener.py (via callback) when a validated image arrives.
# Runs each stage in order. If any stage raises, that stage is skipped for this
# image; the pipeline continues with the remaining stages and with future images.
#
# Architecture: Continuous Mosaic
#   Images are stitched into a growing mosaic canvas. A dynamic grid is derived
#   from the mosaic. Hazard classification, change detection, and route planning
#   all operate on this dynamic grid.
#
# Stage order:
#   1. MosaicStitcher.register_image()   — place image in mosaic
#   2. MosaicGrid.update_from_mosaic()   — resize grid to match canvas
#   3. ShadowDetector
#   4. HazardClassifier → MosaicGrid.apply_hazard()
#   4b. YOLODetector + Fusion
#   5. ChangeDetector (if overlapping prior images exist)
#   6. RoutePlanner (if start/end are set)
#   7. Save cost_grid.json, mission state
#
# The image index (keyed by filename) is persisted to
# data/processed/image_index.json so it survives restarts.

import json
import logging
import os
import threading

import numpy as np

import config
from processing.change_detector import ChangeDetector
from processing.hazard_classifier import HazardClassifier, save_cost_grid_json
from processing.mission_state import MissionState
from processing.mosaic_stitcher import MosaicStitcher
from processing.mosaic_grid import MosaicGrid
from processing.route_planner import RoutePlanner
from processing.shadow_detector import ShadowDetector
from processing.pixel_segmenter import PixelSegmenter
from processing.yolo_detector import YOLODetector, fuse_classifications, save_detections_json
from processing.slope_estimator import SlopeEstimator
from processing.landing_recommender import LandingRecommender
from processing import ppo_planner

logger = logging.getLogger(__name__)

_IMAGE_INDEX_FILE = os.path.join(config.PROCESSED_DIR, "image_index.json")


class Pipeline:
    """
    Stateful pipeline. Holds one instance of each CV module and the shared
    mosaic stitcher + dynamic grid. Designed to run in a single background thread
    (the listener spawns a thread per connection; pipeline calls are serialised
    by the _lock so the grids and mosaic stay consistent).
    """

    def __init__(self, mission_state: MissionState):
        self._mission_state = mission_state
        self._lock = threading.Lock()

        self._stitcher           = MosaicStitcher()
        self._mosaic_grid        = MosaicGrid()
        self._shadow_detector    = ShadowDetector()
        self._hazard_classifier  = HazardClassifier()
        self._yolo_detector      = YOLODetector()
        self._change_detector    = ChangeDetector()
        self._route_planner      = RoutePlanner()
        self._pixel_segmenter    = PixelSegmenter()
        self._slope_estimator    = SlopeEstimator()
        self._landing_recommender = LandingRecommender()

        # YOLO detection state (accumulated across images)
        self._yolo_detections: dict = {}   # {"filename": [detection_dicts]}
        self._fused_results: list = []     # [fused_classification_dicts]
        self._shadow_percentages: dict = {}  # {"filename": float}
        self._image_metadata: list = []      # accumulated metadata dicts

        # Latest hazard map path (used as base for route overlay)
        self._latest_hazard_map_path: str | None = None

        # image_index: { "filename": [{"pass": int, "path": str, "score": float, "mosaic_bbox": [...]}, ...] }
        self._image_index: dict = self._load_image_index()

        # Route endpoints — set from dashboard clicks (mosaic pixel coords)
        self._route_start_mosaic: list | None = None
        self._route_end_mosaic: list | None = None

        # Load PPO planner model
        ppo_planner.load_ppo_model()

        # Initialize grid from existing mosaic canvas
        cw, ch = self._stitcher.get_canvas_size()
        if cw > 0 and ch > 0:
            self._mosaic_grid.update_from_mosaic(cw, ch)

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, image_path: str, metadata: dict, ground_quality: dict):
        """
        Run the full CV pipeline for one received image.
        Called from receiver/listener.py (in its per-connection thread).
        Acquires _lock so only one image is processed at a time.
        """
        with self._lock:
            self._process_locked(image_path, metadata, ground_quality)

    def _process_locked(self, image_path: str, metadata: dict, ground_quality: dict):
        basename = os.path.basename(image_path)

        quality_score = float(
            metadata.get("combined_score")
            or metadata.get("cubesat_quality_score")
            or 0.5
        )

        logger.info(f"Pipeline: starting for '{basename}'")

        # ── Record in mission state + accumulate metadata ──
        self._mission_state.record_image_received(basename, metadata, ground_quality)
        self._image_metadata.append(metadata)

        mosaic_result   = None
        shadow_result   = None
        hazard_result   = None
        change_result   = None
        mosaic_bbox     = (0, 0, 0, 0)

        # ── 1. Register image in mosaic ──
        try:
            mosaic_result = self._stitcher.register_image(image_path, metadata)
            mosaic_bbox = mosaic_result["mosaic_bbox"]
            logger.info(
                f"Pipeline [{basename}] mosaic: placed via {mosaic_result['method']} "
                f"bbox={mosaic_bbox} images={mosaic_result['image_count']}"
            )
        except Exception as e:
            logger.error(f"Pipeline [{basename}] mosaic_stitcher FAILED: {e}", exc_info=True)

        # ── 2. Update dynamic grid from mosaic ──
        try:
            cw, ch = self._stitcher.get_canvas_size()
            if cw > 0 and ch > 0:
                self._mosaic_grid.update_from_mosaic(cw, ch)

            # Update mission state with dynamic coverage
            self._mission_state.record_mosaic_update(
                self._mosaic_grid.get_cells_surveyed(),
                self._mosaic_grid.get_cells_total(),
            )
        except Exception as e:
            logger.error(f"Pipeline [{basename}] mosaic_grid update FAILED: {e}", exc_info=True)

        # ── 3. Shadow detection ──
        try:
            shadow_result = self._shadow_detector.run(image_path)
            if shadow_result is None:
                raise RuntimeError("ShadowDetector returned None")
            self._shadow_percentages[basename] = shadow_result['shadow_percentage']
            logger.info(
                f"Pipeline [{basename}] shadow: "
                f"{shadow_result['shadow_percentage']:.1f}%, "
                f"{len(shadow_result['shadow_regions'])} regions"
            )
        except Exception as e:
            logger.error(f"Pipeline [{basename}] shadow_detector FAILED: {e}", exc_info=True)

        # ── 3b. Slope estimation (from shadow geometry) ──
        if config.SLOPE_ENABLED and shadow_result is not None:
            try:
                shadow_mask = shadow_result.get("shadow_mask")
                if shadow_mask is not None:
                    slope_result = self._slope_estimator.estimate(shadow_mask)
                    slope_map = slope_result.get("slope_map")
                    if slope_map is not None:
                        self._mosaic_grid.apply_slope(mosaic_bbox, slope_map)
                        logger.info(
                            f"Pipeline [{basename}] slope: "
                            f"{len(slope_result.get('regions', []))} region(s)"
                        )
            except Exception as e:
                logger.error(f"Pipeline [{basename}] slope_estimator FAILED: {e}", exc_info=True)

        # ── 4. Hazard classification ──
        # Derive a grid_cell from mosaic_bbox center for labeling
        grid_cell = self._mosaic_grid.mosaic_px_to_grid(
            mosaic_bbox[0] + mosaic_bbox[2] / 2,
            mosaic_bbox[1] + mosaic_bbox[3] / 2,
        )

        try:
            shadow_mask = shadow_result["shadow_mask"] if shadow_result else None
            shadow_pct  = shadow_result["shadow_percentage"] if shadow_result else 0.0

            hazard_result = self._hazard_classifier.classify(
                image_path, shadow_mask, shadow_pct,
                grid_cell=grid_cell, mosaic_bbox=mosaic_bbox,
            )

            # Apply hazard to dynamic grid
            self._mosaic_grid.apply_hazard(mosaic_bbox, hazard_result)
            self._latest_hazard_map_path = hazard_result.get("hazard_map_path")

            self._mission_state.record_hazard_result(grid_cell, hazard_result["hazard_class"])
            logger.info(
                f"Pipeline [{basename}] hazard: "
                f"cell {grid_cell} → {hazard_result['hazard_class']} "
                f"(cost={hazard_result['cost']}, conf={hazard_result.get('confidence', 0):.2f})"
            )
        except Exception as e:
            logger.error(f"Pipeline [{basename}] hazard_classifier FAILED: {e}", exc_info=True)

        # ── 4b. YOLO ML detection + Fusion ──
        try:
            yolo_dets = self._yolo_detector.detect(image_path)

            # Save annotated image
            yolo_out_dir = os.path.join(config.PROCESSED_DIR, "yolo_detections")
            os.makedirs(yolo_out_dir, exist_ok=True)
            annotated_path = os.path.join(yolo_out_dir, f"{os.path.splitext(basename)[0]}_yolo.png")
            if yolo_dets:
                self._yolo_detector.detect_and_annotate(image_path, annotated_path)

            # Store per-image detections (keyed by filename)
            self._yolo_detections[basename] = yolo_dets

            # Fuse with classical classification
            classical_class = hazard_result["hazard_class"] if hazard_result else "SAFE"
            classical_conf = hazard_result.get("confidence", 0.5) if hazard_result else 0.5

            fused = fuse_classifications(grid_cell, classical_class, classical_conf, yolo_dets)

            # Update fused results
            self._fused_results = [
                f for f in self._fused_results if f["cell"] != list(grid_cell)
            ]
            self._fused_results.append(fused)

            # Apply fused classification back to grid if it changed
            if fused["fused_classification"] != classical_class:
                fused_class = fused["fused_classification"]
                cost_map = {
                    "SAFE": config.COST_SAFE, "MODERATE": config.COST_MODERATE,
                    "SHADOW": config.COST_SHADOW, "HAZARD": config.COST_HAZARD,
                    "IMPASSABLE": config.COST_IMPASSABLE,
                }
                fused_result = {
                    "hazard_class": fused_class,
                    "cost": cost_map.get(fused_class, config.COST_SAFE),
                    "confidence": fused["fused_confidence"],
                }
                self._mosaic_grid.apply_hazard(mosaic_bbox, fused_result)
                logger.info(
                    f"Pipeline [{basename}] fusion: {classical_class} → {fused_class} "
                    f"(conf {classical_conf:.2f} → {fused['fused_confidence']:.2f})"
                )

            # Update mission state with YOLO data
            self._mission_state.record_yolo_result(
                self._yolo_detector.model_name,
                self._yolo_detections,
                self._fused_results,
            )

            # Save detections JSON
            save_detections_json(self._yolo_detections, self._fused_results)

            n_dets = len(yolo_dets)
            agreement = "agree" if fused["agreement"] else "DISAGREE"
            logger.info(
                f"Pipeline [{basename}] yolo: {n_dets} detection(s), "
                f"fused={fused['fused_classification']} conf={fused['fused_confidence']:.2f} "
                f"({agreement} with classical CV)"
            )
        except Exception as e:
            logger.error(f"Pipeline [{basename}] yolo_detector FAILED: {e}", exc_info=True)

        # ── 4c. Pixel segmentation (if enabled) ──
        if config.SEG_ENABLED:
            try:
                import time as _time
                t_seg = _time.monotonic()
                shadow_mask_seg = shadow_result["shadow_mask"] if shadow_result else None

                # Collect YOLO detections for segmenter
                yolo_dets_for_seg = self._yolo_detections.get(basename, [])

                label_map = self._pixel_segmenter.segment(
                    image_path, shadow_mask_seg, yolo_dets_for_seg
                )

                # Project onto fine grid
                self._mosaic_grid.apply_segmentation_mask(
                    mosaic_bbox, label_map, config.SEG_COST_MAP, confidence=1.0
                )

                seg_ms = (_time.monotonic() - t_seg) * 1000
                logger.info(
                    f"Pipeline [{basename}] segmentation: {seg_ms:.0f}ms, "
                    f"fine grid {self._mosaic_grid.fine_rows}x{self._mosaic_grid.fine_cols}"
                )
            except Exception as e:
                logger.error(f"Pipeline [{basename}] pixel_segmenter FAILED: {e}", exc_info=True)

        # ── 4d. CNN traversability (if enabled) ──
        if config.CNN_ENABLED:
            try:
                import cv2
                from processing import traversability_cnn
                img = cv2.imread(image_path)
                if img is not None:
                    trav_grid = traversability_cnn.infer_grid(img)
                    if trav_grid is not None:
                        self._mosaic_grid.apply_cnn_traversability(mosaic_bbox, trav_grid)
                        logger.info(f"Pipeline [{basename}] cnn: traversability applied")
            except Exception as e:
                logger.error(f"Pipeline [{basename}] traversability_cnn FAILED: {e}", exc_info=True)

        # ── Update image index ──
        pass_number = metadata.get("pass_number", 1)
        prev_entry = self._get_prev_entry(basename, pass_number)
        self._record_in_index(basename, pass_number, image_path, quality_score, mosaic_bbox)

        # ── 5. Change detection (YOLO-assisted) ──
        has_change = False
        yolo_after = self._yolo_detections.get(basename, [])
        if prev_entry is not None:
            try:
                prev_path = prev_entry["path"]
                prev_pass = prev_entry["pass"]
                prev_basename = os.path.basename(prev_path)
                yolo_before = self._yolo_detections.get(prev_basename, [])
                all_entries = self._image_index.get(basename, [])
                change_result = self._change_detector.detect(
                    prev_path, image_path, grid_cell, prev_pass, pass_number,
                    all_cell_entries=all_entries,
                    yolo_before=yolo_before,
                    yolo_after=yolo_after,
                )
                if change_result and change_result["change_summary"]["total_events"] > 0:
                    has_change = True
                    self._mission_state.record_change_result(
                        change_result["change_summary"],
                        change_result["change_events"],
                    )
                    logger.info(
                        f"Pipeline [{basename}] change: "
                        f"{change_result['change_summary']['total_events']} event(s)"
                    )
                else:
                    logger.info(f"Pipeline [{basename}] change: no significant changes")
            except Exception as e:
                logger.error(f"Pipeline [{basename}] change_detector FAILED: {e}", exc_info=True)
        else:
            # Also check for overlapping entries from different filenames
            overlapping = self._stitcher.get_overlapping_entries(mosaic_bbox)
            prior_overlaps = [
                e for e in overlapping
                if e.filename != basename
            ]
            if prior_overlaps:
                try:
                    prev = prior_overlaps[-1]  # most recent overlapping entry
                    prev_basename_ovl = os.path.basename(prev.image_path)
                    yolo_before_ovl = self._yolo_detections.get(prev_basename_ovl, [])
                    change_result = self._change_detector.detect(
                        prev.image_path, image_path, grid_cell, 0, pass_number,
                        yolo_before=yolo_before_ovl,
                        yolo_after=yolo_after,
                    )
                    if change_result and change_result["change_summary"]["total_events"] > 0:
                        has_change = True
                        self._mission_state.record_change_result(
                            change_result["change_summary"],
                            change_result["change_events"],
                        )
                        logger.info(
                            f"Pipeline [{basename}] change (overlap): "
                            f"{change_result['change_summary']['total_events']} event(s)"
                        )
                except Exception as e:
                    logger.error(f"Pipeline [{basename}] change_detector (overlap) FAILED: {e}", exc_info=True)

        # ── 6. Route planning ──
        if self._route_start_mosaic is not None and self._route_end_mosaic is not None:
            try:
                if config.SEG_ENABLED:
                    cost_grid = self._mosaic_grid.get_fine_cost_grid()
                    # Build a string hazard grid from fine label grid for route planner compat
                    from processing.pixel_segmenter import LABEL_NAMES
                    fine_hg = self._mosaic_grid.get_fine_hazard_grid()
                    _label_to_hazard = {0: "SAFE", 1: "SAFE", 2: "SAFE",
                                        3: "SHADOW", 4: "HAZARD", 5: "IMPASSABLE"}
                    hazard_grid = [
                        [_label_to_hazard.get(int(fine_hg[r, c]), "SAFE")
                         for c in range(fine_hg.shape[1])]
                        for r in range(fine_hg.shape[0])
                    ]
                    start = self._mosaic_grid.mosaic_px_to_fine_grid(
                        self._route_start_mosaic[0], self._route_start_mosaic[1]
                    )
                    end = self._mosaic_grid.mosaic_px_to_fine_grid(
                        self._route_end_mosaic[0], self._route_end_mosaic[1]
                    )
                else:
                    cost_grid = self._mosaic_grid.get_effective_cost_grid() if config.UNCERTAINTY_ENABLED else self._mosaic_grid.get_cost_grid()
                    hazard_grid = self._mosaic_grid.get_hazard_grid()
                    start = self._mosaic_grid.mosaic_px_to_grid(
                        self._route_start_mosaic[0], self._route_start_mosaic[1]
                    )
                    end = self._mosaic_grid.mosaic_px_to_grid(
                        self._route_end_mosaic[0], self._route_end_mosaic[1]
                    )

                routes = self._route_planner.plan_multiple_routes(
                    cost_grid, hazard_grid, start, end,
                    hazard_map_path=self._latest_hazard_map_path,
                )
                self._mission_state.record_route_comparison(routes)
                fastest = routes.get("fastest", {})
                self._mission_state.record_route_result({
                    "path":               fastest.get("path", []),
                    "path_length":        fastest.get("path_length_cells", 0),
                    "total_cost":         fastest.get("total_cost", 0.0),
                    "shadow_exposure_pct": fastest.get("max_shadow_exposure_pct", 0.0),
                    "status":             fastest.get("status", "unknown"),
                })
                # Log PPO route if present
                ppo_route = routes.get("ppo")
                if ppo_route:
                    logger.info(
                        f"Pipeline [{basename}] PPO route: "
                        f"status={ppo_route.get('status')} "
                        f"length={ppo_route.get('path_length_cells')} "
                        f"cost={ppo_route.get('total_cost')} "
                        f"slip_risk={ppo_route.get('cumulative_slip_risk')}"
                    )
                logger.info(
                    f"Pipeline [{basename}] routes: "
                    f"fastest={fastest.get('status')} "
                    f"length={fastest.get('path_length_cells')} "
                    f"cost={fastest.get('total_cost')}"
                )
            except Exception as e:
                logger.error(f"Pipeline [{basename}] route_planner FAILED: {e}", exc_info=True)
        else:
            logger.debug(f"Pipeline [{basename}] routes: start/end not set — skipped")

        # ── 7. Save cost_grid.json ──
        try:
            cost_grid = self._mosaic_grid.get_cost_grid()
            hazard_grid = self._mosaic_grid.get_hazard_grid()
            confidence_grid = self._mosaic_grid.get_confidence_grid()
            surveyed = self._mosaic_grid.get_surveyed_mask()

            change_cells = []
            if change_result and change_result["change_summary"]["total_events"] > 0:
                change_cells = [list(grid_cell)]

            save_cost_grid_json(
                cost_grid, hazard_grid,
                image_index=self._image_index,
                change_cells=change_cells,
                confidence_grid=confidence_grid,
            )
        except Exception as e:
            logger.error(f"Pipeline [{basename}] save_cost_grid_json FAILED: {e}", exc_info=True)

        # ── 8. Persist mission state ──
        try:
            self._mission_state.save()
        except Exception as e:
            logger.error(f"Pipeline [{basename}] mission_state.save() FAILED: {e}", exc_info=True)

        logger.info(f"Pipeline: completed for '{basename}'")

    # ─────────────────────────────────────────────────────────────────────────
    # Image index helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_prev_entry(self, filename: str, current_pass: int) -> dict | None:
        """
        Find the most recent entry for the same filename from a different (earlier) pass.
        """
        entries = self._image_index.get(filename, [])
        prior = [e for e in entries if e["pass"] < current_pass]
        if not prior:
            return None
        return max(prior, key=lambda e: e["pass"])

    def _record_in_index(self, filename: str, pass_number: int, image_path: str,
                         score: float, mosaic_bbox: tuple):
        if filename not in self._image_index:
            self._image_index[filename] = []
        for existing in self._image_index[filename]:
            if existing["pass"] == pass_number and existing["path"] == image_path:
                return
        self._image_index[filename].append({
            "pass": pass_number, "path": image_path, "score": score,
            "mosaic_bbox": list(mosaic_bbox),
        })
        self._save_image_index()

    def _save_image_index(self):
        os.makedirs(config.PROCESSED_DIR, exist_ok=True)
        try:
            with open(_IMAGE_INDEX_FILE, "w") as f:
                json.dump(self._image_index, f, indent=2)
        except Exception as e:
            logger.error(f"Pipeline: failed to save image_index.json: {e}")

    def _load_image_index(self) -> dict:
        if not os.path.exists(_IMAGE_INDEX_FILE):
            return {}
        try:
            with open(_IMAGE_INDEX_FILE) as f:
                idx = json.load(f)
            logger.info(f"Pipeline: loaded image index ({len(idx)} entries)")
            return idx
        except Exception as e:
            logger.warning(f"Pipeline: could not load image_index.json: {e} — starting fresh")
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # Route endpoint setters (called from dashboard)
    # ─────────────────────────────────────────────────────────────────────────

    def set_route_endpoints_mosaic(self, start_mosaic: list, end_mosaic: list):
        """Set route start/end in mosaic pixel coordinates."""
        with self._lock:
            self._route_start_mosaic = start_mosaic
            self._route_end_mosaic = end_mosaic

    def get_route_endpoints_mosaic(self) -> tuple:
        """Return (start_mosaic, end_mosaic) or (None, None)."""
        return (self._route_start_mosaic, self._route_end_mosaic)

    # ─────────────────────────────────────────────────────────────────────────
    # Accessors for dashboard
    # ─────────────────────────────────────────────────────────────────────────

    def get_cost_grid(self):
        """Return a copy of the current cost_grid numpy array."""
        with self._lock:
            return self._mosaic_grid.get_cost_grid()

    def get_hazard_grid(self):
        """Return a copy of the hazard class string grid."""
        with self._lock:
            return self._mosaic_grid.get_hazard_grid()

    def get_confidence_grid(self):
        """Return a copy of the confidence grid."""
        with self._lock:
            return self._mosaic_grid.get_confidence_grid()

    def get_latest_hazard_map_path(self) -> str | None:
        """Return path to the most recent hazard map image."""
        with self._lock:
            return self._latest_hazard_map_path

    def get_fine_cost_grid(self):
        """Return a copy of the fine cost_grid numpy array (if SEG_ENABLED)."""
        with self._lock:
            return self._mosaic_grid.get_fine_cost_grid()

    def get_fine_hazard_grid(self):
        """Return a copy of the fine hazard label grid (if SEG_ENABLED)."""
        with self._lock:
            return self._mosaic_grid.get_fine_hazard_grid()

    def get_segmentation_overlay(self):
        """Generate a color-coded BGRA segmentation overlay for the mosaic."""
        with self._lock:
            hazard_grid = self._mosaic_grid.get_fine_hazard_grid()
            if hazard_grid is None or hazard_grid.shape == (1, 1):
                return None

            from processing.pixel_segmenter import LABEL_COLORS

            rows, cols = hazard_grid.shape
            grid_img = np.zeros((rows, cols, 4), dtype=np.uint8)

            for label_val, bgr_color in LABEL_COLORS.items():
                mask = (hazard_grid == label_val)
                grid_img[mask, 0] = bgr_color[0]
                grid_img[mask, 1] = bgr_color[1]
                grid_img[mask, 2] = bgr_color[2]
                grid_img[mask, 3] = 160 if label_val > 0 else 0

            # Upscale to mosaic pixel dimensions
            cw, ch = self._stitcher.get_canvas_size()
            if cw > 0 and ch > 0:
                import cv2
                overlay = cv2.resize(grid_img, (cw, ch), interpolation=cv2.INTER_NEAREST)
                return overlay
            return grid_img

    def get_effective_cost_grid(self):
        """Return the uncertainty/slope-adjusted effective cost grid."""
        with self._lock:
            return self._mosaic_grid.get_effective_cost_grid()

    def get_slope_grid(self):
        """Return the slope grid (degrees)."""
        with self._lock:
            return self._mosaic_grid.get_slope_grid()

    def get_yolo_detections_mosaic(self) -> list:
        """
        Project YOLO detections from image pixel space to mosaic pixel space
        using each entry's homography matrix.
        """
        import cv2
        with self._lock:
            detections = []
            entries = self._stitcher.get_entries()
            entry_map = {e.filename: e for e in entries}

            for filename, dets in self._yolo_detections.items():
                entry = entry_map.get(filename)
                if entry is None or entry.homography is None:
                    continue

                H = entry.homography
                for det in dets:
                    # Skip shadow detections — not useful on the map
                    if det.get("class", "").lower() == "shadow":
                        continue

                    bbox = det.get("bbox") or det.get("box")
                    if not bbox:
                        continue

                    # bbox is [x1, y1, x2, y2] in image space
                    x1, y1, x2, y2 = bbox[:4]
                    corners = np.array([
                        [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]
                    ], dtype=np.float64)

                    try:
                        mosaic_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
                        mx1 = float(mosaic_corners[:, 0].min())
                        my1 = float(mosaic_corners[:, 1].min())
                        mx2 = float(mosaic_corners[:, 0].max())
                        my2 = float(mosaic_corners[:, 1].max())
                        mcx = (mx1 + mx2) / 2
                        mcy = (my1 + my2) / 2
                    except Exception:
                        continue

                    # Project contour polygon to mosaic space
                    contour_mosaic = None
                    raw_contour = det.get("contour")
                    if raw_contour and len(raw_contour) >= 3:
                        try:
                            cpts = np.array(raw_contour, dtype=np.float64).reshape(-1, 1, 2)
                            cpts_mosaic = cv2.perspectiveTransform(cpts, H).reshape(-1, 2)
                            contour_mosaic = [[round(float(p[0]), 1), round(float(p[1]), 1)]
                                              for p in cpts_mosaic]
                        except Exception:
                            contour_mosaic = None

                    detections.append({
                        "class": det.get("class", "unknown"),
                        "confidence": det.get("confidence", 0.0),
                        "bbox_mosaic": [round(mx1, 1), round(my1, 1), round(mx2, 1), round(my2, 1)],
                        "center_mosaic": [round(mcx, 1), round(mcy, 1)],
                        "contour_mosaic": contour_mosaic,
                        "source_image": filename,
                        "original_class": det.get("original_class", det.get("class", "")),
                    })

            return detections

    def get_analyzer_context(self) -> dict:
        """Gather all data MissionAnalyzer needs in one lock-acquire."""
        with self._lock:
            return {
                "observation_count": self._mosaic_grid.get_observation_count(),
                "surveyed_mask": self._mosaic_grid.get_surveyed_mask(),
                "fine_hazard_grid": self._mosaic_grid.get_fine_hazard_grid(),
                "yolo_detections": dict(self._yolo_detections),
                "fused_results": list(self._fused_results),
                "shadow_percentages": dict(self._shadow_percentages),
                "image_metadata": list(self._image_metadata),
            }

    def recommend_landing_sites(self) -> dict:
        """Run the landing recommender on current grid state."""
        with self._lock:
            fine_cost = self._mosaic_grid.get_fine_cost_grid()
            fine_hazard = self._mosaic_grid.get_fine_hazard_grid()
            obs_count = self._mosaic_grid.get_observation_count()
            conf_grid = self._mosaic_grid.get_confidence_grid()
            surv_mask = self._mosaic_grid.get_surveyed_mask()
            slope = self._mosaic_grid.get_slope_grid()
            fr = self._mosaic_grid.fine_rows
            fc = self._mosaic_grid.fine_cols
            cr = self._mosaic_grid.rows
            cc = self._mosaic_grid.cols

        return self._landing_recommender.recommend(
            fine_cost, fine_hazard, obs_count, conf_grid,
            surv_mask, slope, fr, fc, cr, cc,
        )

    def get_mission_summary(self) -> dict:
        """Aggregate mission data into a summary dict."""
        state = self._mission_state.get_snapshot() if self._mission_state else {}
        mosaic_info = self.get_mosaic_info()

        total_images = mosaic_info.get("image_count", 0)

        # Coverage
        with self._lock:
            cells_surveyed = self._mosaic_grid.get_cells_surveyed()
            cells_total = self._mosaic_grid.get_cells_total()
        coverage_pct = round(100.0 * cells_surveyed / max(1, cells_total), 1)

        # Hazard counts from YOLO detections
        craters = 0
        boulders = 0
        for dets in self._yolo_detections.values():
            for d in dets:
                cls = (d.get("class") or "").lower()
                if "crater" in cls:
                    craters += 1
                elif "boulder" in cls or "rock" in cls:
                    boulders += 1

        # Landing recommendation
        try:
            landing = self.recommend_landing_sites()
            top_candidate = landing["candidates"][0] if landing["candidates"] else None
        except Exception:
            landing = None
            top_candidate = None

        # Route stats from mission state
        routes_state = state.get("routes", {})
        route_stats = {}
        for key in ("safest", "fastest", "balanced"):
            r = routes_state.get(key)
            if r and r.get("status") == "found":
                dist_cells = r.get("path_length_cells", 0)
                dist_cm = round(dist_cells * config.GRID_CELL_SIZE_CM, 1)
                route_stats[key] = {
                    "distance_cm": dist_cm,
                    "cost": r.get("total_cost", 0),
                }

        # Assessment text
        landing_score_pct = round(top_candidate["score"] * 100) if top_candidate else 0
        exit_routes = top_candidate["breakdown"]["route_viability"]["edges_reached"] if top_candidate else 0
        if coverage_pct >= 50 and landing_score_pct >= 60:
            assessment = (
                f"NOMINAL — {coverage_pct}% surveyed, landing score {landing_score_pct}%, "
                f"{exit_routes} exit routes confirmed."
            )
        elif coverage_pct >= 30:
            assessment = (
                f"MARGINAL — {coverage_pct}% surveyed, landing score {landing_score_pct}%. "
                f"More survey data recommended."
            )
        else:
            assessment = (
                f"INSUFFICIENT — only {coverage_pct}% surveyed. "
                f"Cannot reliably recommend landing site."
            )

        result = {
            "total_images": total_images,
            "coverage_pct": coverage_pct,
            "hazards_detected": {
                "craters": craters,
                "boulders": boulders,
                "total": craters + boulders,
            },
            "recommended_landing": {
                "position_cm": top_candidate["position_cm"],
                "score": top_candidate["score"],
                "breakdown": top_candidate["breakdown"],
            } if top_candidate else None,
            "route_stats": route_stats,
            "assessment": assessment,
        }
        return result

    def get_mosaic_info(self) -> dict:
        """Return mosaic metadata for the dashboard."""
        with self._lock:
            cw, ch = self._stitcher.get_canvas_size()
            entries = self._stitcher.get_entries()
            grid = self._mosaic_grid
            return {
                "width": cw,
                "height": ch,
                "image_count": len(entries),
                "entries": [
                    {
                        "filename": e.filename,
                        "bbox": list(e.bbox),
                    }
                    for e in entries
                ],
                "grid": {
                    "rows": grid.rows,
                    "cols": grid.cols,
                    "cell_size_px": config.MOSAIC_GRID_CELL_PX,
                    "origin_x": self._stitcher._origin_x,
                    "origin_y": self._stitcher._origin_y,
                },
            }
