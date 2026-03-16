from __future__ import annotations
# processing/pipeline.py — CV pipeline orchestrator
#
# Called by receiver/listener.py (via callback) when a validated image arrives.
# Runs each stage in order. If any stage raises, that stage is skipped for this
# image; the pipeline continues with the remaining stages and with future images.
#
# Dual-detection architecture:
#   1. Classical CV (shadow, hazard, change, mosaic, route) — grid-level
#   2. YOLOv8 ML detection — object-level bounding boxes for craters/boulders
#   3. Fusion — combines both for higher confidence classification
#
# Stage order:
#   1. ShadowDetector
#   2. HazardClassifier
#   2b. YOLODetector + Fusion (ML object detection + CV/ML agreement)
#   3. ChangeDetector  (only if same cell was imaged in a prior pass)
#   4. MosaicBuilder   (only if 3+ cells now covered)
#   5. RoutePlanner
#   6. MissionState.save()
#
# The image index (which cell was imaged in which pass, with which image path)
# is persisted to data/processed/image_index.json so it survives restarts.

import json
import logging
import os
import threading

import config
from processing.change_detector import ChangeDetector
from processing.hazard_classifier import HazardClassifier, save_cost_grid_json
from processing.mission_state import MissionState
from processing.mosaic_builder import MosaicBuilder
from processing.route_planner import RoutePlanner
from processing.shadow_detector import ShadowDetector
from processing.cell_identifier import CellIdentifier
from processing.yolo_detector import YOLODetector, fuse_classifications, save_detections_json

logger = logging.getLogger(__name__)

_IMAGE_INDEX_FILE = os.path.join(config.PROCESSED_DIR, "image_index.json")


class Pipeline:
    """
    Stateful pipeline. Holds one instance of each CV module and the shared
    image index + cost grid. Designed to run in a single background thread
    (the listener spawns a thread per connection; pipeline calls are serialised
    by the _lock so the cost_grid and image index stay consistent).
    """

    def __init__(self, mission_state: MissionState):
        self._mission_state = mission_state
        self._lock = threading.Lock()

        self._cell_identifier    = CellIdentifier()
        self._shadow_detector    = ShadowDetector()
        self._hazard_classifier  = HazardClassifier()
        self._yolo_detector      = YOLODetector()
        self._change_detector    = ChangeDetector()
        self._mosaic_builder     = MosaicBuilder()
        self._route_planner      = RoutePlanner()

        # YOLO detection state (accumulated across images)
        self._yolo_detections: dict = {}   # {"R,C": [detection_dicts]}
        self._fused_results: list = []     # [fused_classification_dicts]

        # cost_grid: 8×8 numpy int array, one cost per cell.
        # Starts at COST_SAFE for unvisited cells.
        import numpy as np
        self._cost_grid = np.full(
            (config.GRID_ROWS, config.GRID_COLS),
            config.COST_SAFE,
            dtype=np.int32
        )

        # hazard_grid: parallel 8×8 list of class strings for route colouring.
        self._hazard_grid = [
            ["SAFE"] * config.GRID_COLS for _ in range(config.GRID_ROWS)
        ]

        # confidence_grid: 8×8 float array, one confidence per cell.
        self._confidence_grid = np.zeros(
            (config.GRID_ROWS, config.GRID_COLS), dtype=np.float64
        )

        # Latest hazard map path (used as base for route overlay)
        self._latest_hazard_map_path: str | None = None

        # image_index: { "R,C": [{"pass": int, "path": str, "score": float}, ...] }
        self._image_index: dict = self._load_image_index()

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

        # ── Cell identification (ground-side, image-based) ──
        # The ground station determines which cell the image belongs to
        # using SIFT + CNN + Delaunay fingerprinting — no CubeSat metadata needed.
        try:
            cell_result = self._cell_identifier.identify(image_path)
            grid_cell = cell_result["cell"]
            pass_number = cell_result["pass_number"]
            is_revisit = cell_result["is_revisit"]
            cell_method = cell_result["method"]
            cell_confidence = cell_result["confidence"]
        except Exception as e:
            logger.error(f"Pipeline [{basename}] cell identification FAILED: {e}", exc_info=True)
            grid_cell = (0, 0)
            pass_number = 1
            is_revisit = False
            cell_method = "error"
            cell_confidence = 0.0

        quality_score = float(
            metadata.get("combined_score")
            or metadata.get("cubesat_quality_score")
            or 0.5
        )

        logger.info(
            f"Pipeline: starting for '{basename}' cell={grid_cell} pass={pass_number} "
            f"({'REVISIT' if is_revisit else 'NEW'} via {cell_method}, conf={cell_confidence:.2f})"
        )

        # ── Record in mission state ──
        self._mission_state.record_image_received(basename, metadata, ground_quality)

        shadow_result    = None
        hazard_result    = None
        change_result    = None

        # ── 1. Shadow detection ──
        try:
            shadow_result = self._shadow_detector.run(image_path)
            if shadow_result is None:
                raise RuntimeError("ShadowDetector returned None")
            logger.info(
                f"Pipeline [{basename}] shadow: "
                f"{shadow_result['shadow_percentage']:.1f}%, "
                f"{len(shadow_result['shadow_regions'])} regions"
            )
        except Exception as e:
            logger.error(f"Pipeline [{basename}] shadow_detector FAILED: {e}", exc_info=True)

        # ── 2. Hazard classification ──
        try:
            shadow_mask = shadow_result["shadow_mask"] if shadow_result else None
            shadow_pct  = shadow_result["shadow_percentage"] if shadow_result else 0.0

            hazard_result = self._hazard_classifier.classify(
                image_path, shadow_mask, shadow_pct, grid_cell
            )
            r, c = grid_cell
            self._cost_grid[r, c] = hazard_result["cost"]
            self._hazard_grid[r][c] = hazard_result["hazard_class"]
            self._confidence_grid[r, c] = hazard_result.get("confidence", 0.0)
            self._latest_hazard_map_path = hazard_result.get("hazard_map_path")

            self._mission_state.record_hazard_result(grid_cell, hazard_result["hazard_class"])
            logger.info(
                f"Pipeline [{basename}] hazard: "
                f"cell {grid_cell} → {hazard_result['hazard_class']} "
                f"(cost={hazard_result['cost']}, conf={hazard_result.get('confidence', 0):.2f})"
            )
        except Exception as e:
            logger.error(f"Pipeline [{basename}] hazard_classifier FAILED: {e}", exc_info=True)

        # ── 2b. YOLO ML detection + Fusion ──
        try:
            yolo_dets = self._yolo_detector.detect(image_path)

            # Save annotated image
            yolo_out_dir = os.path.join(config.PROCESSED_DIR, "yolo_detections")
            os.makedirs(yolo_out_dir, exist_ok=True)
            annotated_path = os.path.join(yolo_out_dir, f"{os.path.splitext(basename)[0]}_yolo.png")
            if yolo_dets:
                self._yolo_detector.detect_and_annotate(image_path, annotated_path)

            # Store per-cell detections
            cell_key = self._cell_key(grid_cell)
            self._yolo_detections[cell_key] = yolo_dets

            # Fuse with classical classification
            classical_class = hazard_result["hazard_class"] if hazard_result else "SAFE"
            classical_conf = hazard_result.get("confidence", 0.5) if hazard_result else 0.5

            fused = fuse_classifications(grid_cell, classical_class, classical_conf, yolo_dets)

            # Update fused results (replace existing for this cell)
            self._fused_results = [
                f for f in self._fused_results if f["cell"] != list(grid_cell)
            ]
            self._fused_results.append(fused)

            # Apply fused classification back to grids if it changed
            if fused["fused_classification"] != classical_class:
                r, c = grid_cell
                fused_class = fused["fused_classification"]
                cost_map = {
                    "SAFE": config.COST_SAFE, "MODERATE": config.COST_MODERATE,
                    "SHADOW": config.COST_SHADOW, "HAZARD": config.COST_HAZARD,
                    "IMPASSABLE": config.COST_IMPASSABLE,
                }
                self._cost_grid[r, c] = cost_map.get(fused_class, config.COST_SAFE)
                self._hazard_grid[r][c] = fused_class
                self._confidence_grid[r, c] = fused["fused_confidence"]
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

        # ── Update image index (for change detection) ──
        prev_entry = self._get_prev_entry(grid_cell, pass_number)
        self._record_in_index(grid_cell, pass_number, image_path, quality_score)

        # ── 3. Change detection ──
        has_change = False
        if prev_entry is not None:
            try:
                prev_path   = prev_entry["path"]
                prev_pass   = prev_entry["pass"]
                # Pass all entries for this cell for persistence checking
                cell_key = self._cell_key(grid_cell)
                all_entries = self._image_index.get(cell_key, [])
                change_result = self._change_detector.detect(
                    prev_path, image_path, grid_cell, prev_pass, pass_number,
                    all_cell_entries=all_entries,
                )
                if change_result and change_result["change_summary"]["total_events"] > 0:
                    has_change = True
                    self._mission_state.record_change_result(
                        change_result["change_summary"],
                        change_result["change_events"],
                    )
                    logger.info(
                        f"Pipeline [{basename}] change: "
                        f"{change_result['change_summary']['total_events']} event(s) in cell {grid_cell}"
                    )
                else:
                    logger.info(f"Pipeline [{basename}] change: no significant changes in cell {grid_cell}")
            except Exception as e:
                logger.error(f"Pipeline [{basename}] change_detector FAILED: {e}", exc_info=True)
        else:
            logger.debug(f"Pipeline [{basename}] change: no prior image for cell {grid_cell} — skipped")

        # ── 4. Mosaic ──  (was 5 — elevation removed)
        try:
            mosaic_result = self._mosaic_builder.update(
                image_path, grid_cell, quality_score, has_change=has_change
            )
            if mosaic_result:
                logger.info(
                    f"Pipeline [{basename}] mosaic: "
                    f"{mosaic_result['cells_filled']}/{mosaic_result['cells_total']} cells "
                    f"({mosaic_result['coverage_pct']}%)"
                )
        except Exception as e:
            logger.error(f"Pipeline [{basename}] mosaic_builder FAILED: {e}", exc_info=True)

        # ── 5. Route planning ──
        try:
            routes = self._route_planner.plan_multiple_routes(
                self._cost_grid,
                self._hazard_grid,
                config.ROUTE_START,
                config.ROUTE_END,
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
            logger.info(
                f"Pipeline [{basename}] routes: "
                f"fastest={fastest.get('status')} "
                f"length={fastest.get('path_length_cells')} "
                f"cost={fastest.get('total_cost')}"
            )
        except Exception as e:
            logger.error(f"Pipeline [{basename}] route_planner FAILED: {e}", exc_info=True)

        # ── 6. Save cost_grid.json ──
        try:
            change_cells = []
            if change_result and change_result["change_summary"]["total_events"] > 0:
                change_cells = [list(grid_cell)]
            save_cost_grid_json(
                self._cost_grid, self._hazard_grid,
                image_index=self._image_index,
                change_cells=change_cells,
                confidence_grid=self._confidence_grid,
            )
        except Exception as e:
            logger.error(f"Pipeline [{basename}] save_cost_grid_json FAILED: {e}", exc_info=True)

        # ── 7. Persist mission state ──
        try:
            self._mission_state.save()
        except Exception as e:
            logger.error(f"Pipeline [{basename}] mission_state.save() FAILED: {e}", exc_info=True)

        logger.info(f"Pipeline: completed for '{basename}'")

    # ─────────────────────────────────────────────────────────────────────────
    # Image index helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _cell_key(self, grid_cell: tuple) -> str:
        return f"{grid_cell[0]},{grid_cell[1]}"

    def _get_prev_entry(self, grid_cell: tuple, current_pass: int) -> dict | None:
        """
        Find the most recent entry for grid_cell from a DIFFERENT (earlier) pass.
        Returns {"pass": int, "path": str, "score": float} or None.
        """
        key = self._cell_key(grid_cell)
        entries = self._image_index.get(key, [])
        prior = [e for e in entries if e["pass"] < current_pass]
        if not prior:
            return None
        # Most recent prior pass
        return max(prior, key=lambda e: e["pass"])

    def _record_in_index(self, grid_cell: tuple, pass_number: int, image_path: str, score: float):
        key = self._cell_key(grid_cell)
        if key not in self._image_index:
            self._image_index[key] = []
        # Avoid duplicate entries (same pass + path)
        for existing in self._image_index[key]:
            if existing["pass"] == pass_number and existing["path"] == image_path:
                return
        self._image_index[key].append({"pass": pass_number, "path": image_path, "score": score})
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
            logger.info(f"Pipeline: loaded image index ({len(idx)} cells)")
            return idx
        except Exception as e:
            logger.warning(f"Pipeline: could not load image_index.json: {e} — starting fresh")
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # Accessors for dashboard
    # ─────────────────────────────────────────────────────────────────────────

    def get_cost_grid(self):
        """Return a copy of the current cost_grid numpy array."""
        with self._lock:
            return self._cost_grid.copy()

    def get_hazard_grid(self):
        """Return a copy of the hazard class string grid."""
        with self._lock:
            return [row[:] for row in self._hazard_grid]

    def get_latest_hazard_map_path(self) -> str | None:
        """Return path to the most recent hazard map image."""
        with self._lock:
            return self._latest_hazard_map_path
