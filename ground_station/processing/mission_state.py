# processing/mission_state.py — Accumulates all mission data and writes mission_state.json
#
# MissionState is the single source of truth read by the dashboard.
# It is updated after every pipeline run and written atomically (temp file → rename)
# so the dashboard never reads a partial file.
#
# Schema matches ARCHITECTURE.md §7.7.
# All data is REAL — no defaults are shown as results.

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone

import config
from processing.hazard_classifier import SHADOW

logger = logging.getLogger(__name__)


class MissionState:

    def __init__(self):
        self._lock = threading.Lock()
        self._state = self._empty_state()
        self._load()

    # ─────────────────────────────────────────────────────────────────────────
    # Public record methods — called by pipeline.py after each step
    # ─────────────────────────────────────────────────────────────────────────

    def record_image_received(self, filename: str, metadata: dict, ground_quality: dict):
        """Call once per validated, saved image."""
        with self._lock:
            self._state["total_images_received"] += 1

            pass_number = metadata.get("pass_number", 0)
            if pass_number > self._state["total_passes"]:
                self._state["total_passes"] = pass_number

            cubesat_score = metadata.get("combined_score") or metadata.get("cubesat_quality_score")
            if cubesat_score is not None:
                scores = self._state["_cubesat_scores"]
                scores.append(float(cubesat_score))
                self._state["quality"]["avg_cubesat_score"] = round(
                    sum(scores) / len(scores), 3
                )

            if not ground_quality.get("passed", True):
                self._state["quality"]["ground_flagged"] += 1
                for note in ground_quality.get("notes", []):
                    reasons = self._state["quality"]["ground_flag_reasons"]
                    if note not in reasons:
                        reasons.append(note)

    def record_image_corrupted(self):
        """Call when a transfer fails MD5 or size check."""
        with self._lock:
            self._state["total_images_corrupted"] += 1

    def record_hazard_result(self, grid_cell: tuple, hazard_class: str):
        """Call after HazardClassifier.classify() for each image."""
        with self._lock:
            key = hazard_class.lower()
            if key in self._state["hazards"]:
                self._state["hazards"][key] += 1

            if grid_cell is not None:
                row, col = grid_cell
                self._state["_cells_covered"].add((row, col))

            self._state["coverage"]["cells_filled"] = len(
                self._state["_cells_covered"]
            )
            total = self._state["coverage"]["cells_total"]
            if total > 0:
                self._state["coverage"]["pct"] = round(
                    len(self._state["_cells_covered"]) / total * 100.0, 1
                )

    def record_mosaic_update(self, cells_surveyed: int, cells_total: int):
        """Call after MosaicGrid updates to sync dynamic coverage stats."""
        with self._lock:
            self._state["coverage"]["cells_total"] = cells_total
            self._state["coverage"]["cells_filled"] = cells_surveyed
            if cells_total > 0:
                self._state["coverage"]["pct"] = round(
                    cells_surveyed / cells_total * 100.0, 1
                )

    def record_change_result(self, change_summary: dict, change_events: list):
        """Call after ChangeDetector.detect() — only when events are found."""
        with self._lock:
            ch = self._state["changes"]
            ch["total_events"] += change_summary.get("total_events", 0)
            ch["total_changed_area_cm2"] = round(
                ch["total_changed_area_cm2"] + change_summary.get("total_changed_area_cm2", 0.0), 3
            )

            new_largest = change_summary.get("largest_change_cm2", 0.0)
            if new_largest > ch["largest_change_cm2"]:
                ch["largest_change_cm2"] = new_largest

            types = change_summary.get("types", {})
            ch["types"]["darkened"]   += types.get("darkened", 0)
            ch["types"]["brightened"] += types.get("brightened", 0)

            if change_summary.get("alignment_uncertain", False):
                ch["alignment_warnings"] += 1

            for event in change_events:
                cell = tuple(event.get("grid_cell", []))
                cells_list = ch["cells_with_changes"]
                if list(cell) not in cells_list:
                    cells_list.append(list(cell))

    def record_route_result(self, route_result: dict):
        """Call after RoutePlanner.plan() — overwrites route section (latest plan)."""
        with self._lock:
            default_start = list(config.ROUTE_START) if config.ROUTE_START else None
            default_end = list(config.ROUTE_END) if config.ROUTE_END else None
            self._state["route"] = {
                "start":               route_result.get("path", [[0, 0]])[0] if route_result.get("path") else default_start,
                "end":                 route_result.get("path", [])[-1] if route_result.get("path") else default_end,
                "path_length":         route_result.get("path_length", 0),
                "total_cost":          route_result.get("total_cost", 0.0),
                "shadow_exposure_pct": route_result.get("shadow_exposure_pct", 0.0),
                "status":              route_result.get("status", "unknown"),
            }

    def record_route_comparison(self, routes_dict: dict):
        """Call after RoutePlanner.plan_multiple_routes() — stores all 3 routes."""
        with self._lock:
            r = self._state["routes"]
            for name in ("fastest", "safest", "balanced"):
                if name in routes_dict:
                    r[name] = routes_dict[name]
            if r["selected"] is None:
                r["selected"] = "safest"

    def record_yolo_result(self, model_name: str, detections_per_cell: dict, fused_results: list):
        """Call after YOLODetector + fusion for each image."""
        with self._lock:
            ml = self._state["ml_detection"]
            ml["model"] = model_name

            # Flatten all detections for counts
            flat = []
            for dets in detections_per_cell.values():
                flat.extend(dets)

            ml["total_detections"] = len(flat)
            ml["craters_detected"] = sum(1 for d in flat if d["class"] == "crater")
            ml["boulders_detected"] = sum(1 for d in flat if d["class"] == "boulder")

            agreement_count = sum(1 for f in fused_results if f["agreement"])
            total_fused = len(fused_results)
            ml["cv_agreement_rate"] = round(
                agreement_count / total_fused, 3
            ) if total_fused else 1.0

            # Store per-cell detections (just class + confidence, not full bbox)
            ml["detections_per_cell"] = {
                k: [{"class": d["class"], "conf": d["confidence"]} for d in v]
                for k, v in detections_per_cell.items()
            }

    def record_downlink_bytes(self, n_bytes: int, duration_sec: float, success: bool):
        """Call from receiver/listener.py after each completed (or failed) transfer."""
        with self._lock:
            dl = self._state["downlink"]
            dl["total_bytes"] += n_bytes
            dl["total_time_sec"] = round(dl["total_time_sec"] + duration_sec, 2)
            if dl["total_time_sec"] > 0:
                dl["effective_rate_bps"] = round(
                    dl["total_bytes"] / dl["total_time_sec"], 1
                )
            if not success:
                dl["failed_transfers"] += 1

    def record_retransmit_request(self):
        with self._lock:
            self._state["downlink"]["retransmit_requests"] += 1

    def record_command(self, acked: bool):
        """Call from uplink/commander.py after each send_command()."""
        with self._lock:
            ul = self._state["uplink"]
            ul["commands_sent"] += 1
            if acked:
                ul["commands_acked"] += 1

    # ─────────────────────────────────────────────────────────────────────────
    # AI Advisor decision log
    # ─────────────────────────────────────────────────────────────────────────

    def record_advisor_decision(self, decision: dict):
        """Append an AI advisor decision to the decision log."""
        with self._lock:
            self._state["decision_log"].append(decision)

    def resolve_decision(self, decision_id: str, action: str, note: str = ""):
        """Resolve a pending decision: approve, override, or defer."""
        with self._lock:
            for entry in self._state["decision_log"]:
                if entry.get("id") == decision_id:
                    entry["status"] = action.upper()
                    entry["operator_action"] = action
                    entry["operator_note"] = note
                    entry["resolved_at"] = datetime.now(timezone.utc).isoformat()
                    return True
        return False

    def get_decision_log(self) -> list:
        """Return a copy of the decision log."""
        with self._lock:
            return list(self._state["decision_log"])

    def get_pending_decisions(self) -> list:
        """Return only PENDING_HUMAN_APPROVAL decisions."""
        with self._lock:
            return [
                d for d in self._state["decision_log"]
                if d.get("status") == "PENDING_HUMAN_APPROVAL"
            ]

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self):
        """Write mission_state.json atomically. Safe to call from any thread."""
        with self._lock:
            self._state["last_updated"] = datetime.now(timezone.utc).isoformat()
            # Build public snapshot (strip internal keys starting with _)
            snapshot = {k: v for k, v in self._state.items() if not k.startswith("_")}

        os.makedirs(os.path.dirname(config.MISSION_STATE_FILE), exist_ok=True)

        # Write to a temp file then rename (atomic on POSIX)
        dir_ = os.path.dirname(os.path.abspath(config.MISSION_STATE_FILE))
        try:
            with tempfile.NamedTemporaryFile(
                "w", dir=dir_, suffix=".tmp", delete=False
            ) as tf:
                json.dump(snapshot, tf, indent=2)
                tmp_path = tf.name
            os.replace(tmp_path, config.MISSION_STATE_FILE)
            logger.debug(f"mission_state.json saved")
        except Exception as e:
            logger.error(f"Failed to save mission_state.json: {e}")

    def _load(self):
        """Load existing mission_state.json on startup to resume accumulated state."""
        if not os.path.exists(config.MISSION_STATE_FILE):
            return
        try:
            with open(config.MISSION_STATE_FILE) as f:
                saved = json.load(f)
            # Merge saved data into state (fields that exist in our schema)
            for key in self._state:
                if key.startswith("_"):
                    continue
                if key in saved:
                    self._state[key] = saved[key]
            # Rebuild internal set from coverage data
            # Reconstruct _cells_covered from hazard counts and coverage info
            # (Best effort — cells_filled count is preserved even if exact set isn't)
            if self._state["coverage"]["cells_filled"] > 0 and not self._state["_cells_covered"]:
                self._state["_cells_covered"] = set()
            logger.info("MissionState: resumed from existing mission_state.json")
        except Exception as e:
            logger.warning(f"MissionState: could not load mission_state.json: {e} — starting fresh")

    def reset(self):
        """Reset all mission state to empty defaults."""
        with self._lock:
            self._state = self._empty_state()
        self.save()
        logger.info("MissionState: reset to empty defaults")

    def get_snapshot(self) -> dict:
        """Return a copy of the current state dict (for dashboard reads)."""
        with self._lock:
            snapshot = {k: v for k, v in self._state.items() if not k.startswith("_")}
            snapshot["last_updated"] = self._state["last_updated"]
        return snapshot

    # ─────────────────────────────────────────────────────────────────────────
    # State schema
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_state() -> dict:
        return {
            "last_updated": "",
            "total_passes": 0,
            "total_images_received": 0,
            "total_images_corrupted": 0,
            "quality": {
                "avg_cubesat_score": 0.0,
                "ground_flagged": 0,
                "ground_flag_reasons": [],
            },
            "coverage": {
                "cells_filled": 0,
                "cells_total": 0,  # dynamic — updated by MosaicGrid
                "pct": 0.0,
            },
            "hazards": {
                "safe": 0,
                "moderate": 0,
                "shadow": 0,
                "hazard": 0,
                "impassable": 0,
            },
            "changes": {
                "total_events": 0,
                "total_changed_area_cm2": 0.0,
                "largest_change_cm2": 0.0,
                "types": {"darkened": 0, "brightened": 0},
                "cells_with_changes": [],
                "alignment_warnings": 0,
            },
            "route": {
                "start": list(config.ROUTE_START) if config.ROUTE_START else None,
                "end": list(config.ROUTE_END) if config.ROUTE_END else None,
                "path_length": 0,
                "total_cost": 0.0,
                "shadow_exposure_pct": 0.0,
                "status": "not yet planned",
            },
            "routes": {
                "fastest": None,
                "safest": None,
                "balanced": None,
                "selected": None,
                "constrained": None,
            },
            "downlink": {
                "total_bytes": 0,
                "total_time_sec": 0.0,
                "effective_rate_bps": 0.0,
                "failed_transfers": 0,
                "retransmit_requests": 0,
            },
            "uplink": {
                "commands_sent": 0,
                "commands_acked": 0,
            },
            "ml_detection": {
                "model": "none",
                "total_detections": 0,
                "craters_detected": 0,
                "boulders_detected": 0,
                "cv_agreement_rate": 1.0,
                "detections_per_cell": {},
            },
            "decision_log": [],
            # Internal — not written to JSON
            "_cubesat_scores": [],
            "_cells_covered": set(),
        }
