# logic.py — MissionAnalyzer: pure-Python coded checks that produce structured flags
#
# No AI/LLM calls. This module scans pipeline state, telemetry, and detection
# data and returns a JSON-serializable flag_report. The LLM advisor layer
# consumes these flags to produce natural-language reasoning.
#
# Flag types:
#   SURVEY_GAP        — grid cell with < 2 observations
#   DETECTION_CONFLICT — YOLO low-confidence detection overlapping high-shadow area
#   SENSOR_DRIFT      — IMU variance exceeds threshold
#   THERMAL_ALERT     — Pi CPU temperature above safe operating limit

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

import config

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

SURVEY_MIN_OBSERVATIONS = 2          # cells with fewer obs → SURVEY_GAP
DETECTION_CONF_THRESHOLD = 0.70      # YOLO detections below this get scrutinised
SHADOW_OVERLAP_THRESHOLD = 0.40      # shadow fraction above this triggers conflict flag
IMU_YAW_VARIANCE_THRESHOLD = 5.0     # degrees² — yaw variance across images in a pass
IMU_PITCH_VARIANCE_THRESHOLD = 3.0   # degrees² — pitch variance (nadir should be stable)
IMU_ROLL_VARIANCE_THRESHOLD = 3.0    # degrees² — roll variance
THERMAL_WARNING_C = 75.0             # °C — warning threshold
THERMAL_CRITICAL_C = 82.0            # °C — critical threshold

# Fine grid cell size in cm (for physical context in flags)
_FINE_CELL_CM = config.SEG_GRID_CELL_PX / config.MOSAIC_PX_PER_CM


class MissionAnalyzer:
    """Pure-Python mission data analyzer. Produces structured flags from
    pipeline state — no AI/LLM calls."""

    def analyze(
        self,
        observation_count: np.ndarray,
        surveyed_mask: np.ndarray,
        fine_hazard_grid: np.ndarray,
        yolo_detections: dict[str, list[dict]],
        fused_results: list[dict],
        shadow_percentages: dict[str, float],
        image_metadata: list[dict],
        telemetry: dict,
        landing_candidates: list[dict] | None = None,
    ) -> dict:
        """Run all coded checks and return a flag_report.

        Args:
            observation_count: coarse grid (rows x cols) of observation counts
            surveyed_mask: coarse grid (rows x cols) bool — True if surveyed
            fine_hazard_grid: fine grid (rows x cols) uint8 labels
            yolo_detections: {"filename": [detection_dicts]} from pipeline
            fused_results: list of fusion dicts (from yolo_detector.fuse_classifications)
            shadow_percentages: {"filename": float} shadow % per image
            image_metadata: list of metadata dicts (one per image)
            telemetry: latest telemetry dict (flat/normalized)
            landing_candidates: optional list of landing candidate dicts

        Returns:
            {"flags": [...], "summary": {...}, "timestamp": "..."}
        """
        flags: list[dict] = []

        flags.extend(self._check_survey_gaps(observation_count, surveyed_mask))
        flags.extend(self._check_detection_conflicts(
            yolo_detections, fused_results, shadow_percentages
        ))
        flags.extend(self._check_sensor_drift(image_metadata))
        flags.extend(self._check_thermal(telemetry))

        # Optional: cross-check landing candidates against flags
        if landing_candidates:
            flags.extend(self._check_landing_risks(
                landing_candidates, observation_count, fine_hazard_grid
            ))

        # Sort: CRITICAL first, then WARNING, then INFO
        severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        flags.sort(key=lambda f: severity_order.get(f["severity"], 9))

        summary = self._build_summary(flags)

        return {
            "flags": flags,
            "summary": summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── SURVEY_GAP ────────────────────────────────────────────────────────

    def _check_survey_gaps(
        self, observation_count: np.ndarray, surveyed_mask: np.ndarray
    ) -> list[dict]:
        flags = []
        if observation_count.shape == (1, 1) and observation_count[0, 0] == 0:
            return flags  # no data yet

        rows, cols = observation_count.shape
        total_cells = rows * cols
        surveyed_count = int(np.sum(surveyed_mask)) if surveyed_mask.shape != (1, 1) else 0

        # Find cells that are surveyed but have < MIN observations
        low_obs_cells = []
        unsurveyed_cells = []

        for r in range(rows):
            for c in range(cols):
                obs = int(observation_count[r, c])
                is_surveyed = bool(surveyed_mask[r, c]) if surveyed_mask.shape == observation_count.shape else False

                if not is_surveyed:
                    unsurveyed_cells.append([r, c])
                elif obs < SURVEY_MIN_OBSERVATIONS:
                    low_obs_cells.append({
                        "cell": [r, c],
                        "observations": obs,
                        "position_cm": [
                            round(c * config.MOSAIC_GRID_CELL_PX / config.MOSAIC_PX_PER_CM, 1),
                            round(r * config.MOSAIC_GRID_CELL_PX / config.MOSAIC_PX_PER_CM, 1),
                        ],
                    })

        # Flag individual low-observation cells (cap at 10 for readability)
        if low_obs_cells:
            severity = "WARNING" if len(low_obs_cells) > 5 else "INFO"
            flags.append({
                "type": "SURVEY_GAP",
                "severity": severity,
                "context": {
                    "description": (
                        f"{len(low_obs_cells)} surveyed cell(s) have fewer than "
                        f"{SURVEY_MIN_OBSERVATIONS} observations"
                    ),
                    "cells": low_obs_cells[:10],
                    "total_low_obs_cells": len(low_obs_cells),
                    "min_required_observations": SURVEY_MIN_OBSERVATIONS,
                },
            })

        # Flag large unsurveyed regions
        unsurveyed_pct = round(100.0 * len(unsurveyed_cells) / max(1, total_cells), 1)
        if unsurveyed_pct > 40:
            flags.append({
                "type": "SURVEY_GAP",
                "severity": "CRITICAL" if unsurveyed_pct > 60 else "WARNING",
                "context": {
                    "description": f"{unsurveyed_pct}% of grid is unsurveyed",
                    "unsurveyed_cells": len(unsurveyed_cells),
                    "total_cells": total_cells,
                    "unsurveyed_pct": unsurveyed_pct,
                },
            })

        return flags

    # ── DETECTION_CONFLICT ────────────────────────────────────────────────

    def _check_detection_conflicts(
        self,
        yolo_detections: dict[str, list[dict]],
        fused_results: list[dict],
        shadow_percentages: dict[str, float],
    ) -> list[dict]:
        flags = []

        # Check 1: Low-confidence YOLO detections in high-shadow images
        for filename, dets in yolo_detections.items():
            shadow_pct = shadow_percentages.get(filename, 0.0)
            shadow_frac = shadow_pct / 100.0

            for det in dets:
                conf = det.get("confidence", 1.0)
                cls = det.get("class", "unknown")

                # Skip non-hazard classes
                if cls.lower() in ("sand", "plain", "plain_surface", "shadow"):
                    continue

                if conf < DETECTION_CONF_THRESHOLD and shadow_frac > SHADOW_OVERLAP_THRESHOLD:
                    center = det.get("center", [0, 0])
                    flags.append({
                        "type": "DETECTION_CONFLICT",
                        "severity": "WARNING",
                        "context": {
                            "description": (
                                f"Low-confidence {cls} detection ({conf:.0%}) in "
                                f"high-shadow image ({shadow_pct:.0f}% shadow)"
                            ),
                            "source_image": filename,
                            "detection_class": cls,
                            "detection_confidence": round(conf, 3),
                            "image_shadow_pct": round(shadow_pct, 1),
                            "detection_center_px": center,
                            "likely_cause": "shadow_artifact",
                        },
                    })

        # Check 2: CV/YOLO disagreements from fusion results
        for fused in fused_results:
            if not fused.get("agreement", True):
                yolo_dets = fused.get("yolo_detections", [])
                yolo_classes = [d.get("class", "?") for d in yolo_dets]
                fused_conf = fused.get("fused_confidence", 0)

                flags.append({
                    "type": "DETECTION_CONFLICT",
                    "severity": "WARNING" if fused_conf < 0.6 else "INFO",
                    "context": {
                        "description": (
                            f"CV/YOLO architecture disagreement at cell {fused.get('cell', '?')}"
                        ),
                        "cell": fused.get("cell"),
                        "classical_class": fused.get("classical_class", "?"),
                        "classical_confidence": fused.get("classical_confidence", 0),
                        "yolo_classes": yolo_classes,
                        "fused_classification": fused.get("fused_classification", "?"),
                        "fused_confidence": fused_conf,
                        "likely_cause": "architecture_blind_spot",
                    },
                })

        return flags

    # ── SENSOR_DRIFT ──────────────────────────────────────────────────────

    def _check_sensor_drift(self, image_metadata: list[dict]) -> list[dict]:
        flags = []
        if len(image_metadata) < 3:
            return flags  # need at least 3 images to compute meaningful variance

        # Group IMU readings by pass
        passes: dict[int, list[dict]] = {}
        for meta in image_metadata:
            pass_num = meta.get("pass_number", 1)
            imu = meta.get("imu", {})
            if imu:
                passes.setdefault(pass_num, []).append(imu)

        for pass_num, imu_readings in passes.items():
            if len(imu_readings) < 3:
                continue

            yaws = [r.get("yaw_deg") for r in imu_readings if r.get("yaw_deg") is not None]
            pitches = [r.get("pitch_deg") for r in imu_readings if r.get("pitch_deg") is not None]
            rolls = [r.get("roll_deg") for r in imu_readings if r.get("roll_deg") is not None]

            drift_details = []

            if len(yaws) >= 3:
                yaw_var = float(np.var(yaws))
                if yaw_var > IMU_YAW_VARIANCE_THRESHOLD:
                    drift_details.append({
                        "axis": "yaw",
                        "variance_deg2": round(yaw_var, 2),
                        "threshold_deg2": IMU_YAW_VARIANCE_THRESHOLD,
                        "readings": [round(y, 1) for y in yaws],
                    })

            if len(pitches) >= 3:
                pitch_var = float(np.var(pitches))
                if pitch_var > IMU_PITCH_VARIANCE_THRESHOLD:
                    drift_details.append({
                        "axis": "pitch",
                        "variance_deg2": round(pitch_var, 2),
                        "threshold_deg2": IMU_PITCH_VARIANCE_THRESHOLD,
                        "readings": [round(p, 1) for p in pitches],
                    })

            if len(rolls) >= 3:
                roll_var = float(np.var(rolls))
                if roll_var > IMU_ROLL_VARIANCE_THRESHOLD:
                    drift_details.append({
                        "axis": "roll",
                        "variance_deg2": round(roll_var, 2),
                        "threshold_deg2": IMU_ROLL_VARIANCE_THRESHOLD,
                        "readings": [round(r, 1) for r in rolls],
                    })

            if drift_details:
                max_var = max(d["variance_deg2"] for d in drift_details)
                severity = "CRITICAL" if max_var > 10.0 else "WARNING"
                flags.append({
                    "type": "SENSOR_DRIFT",
                    "severity": severity,
                    "context": {
                        "description": (
                            f"IMU variance exceeds threshold on pass {pass_num}: "
                            + ", ".join(f"{d['axis']} {d['variance_deg2']}°²" for d in drift_details)
                        ),
                        "pass_number": pass_num,
                        "num_images": len(imu_readings),
                        "drift_axes": drift_details,
                        "impact": "Mosaic stitching accuracy may be degraded. "
                                  "Image placement hints from IMU are unreliable.",
                    },
                })

        return flags

    # ── THERMAL_ALERT ─────────────────────────────────────────────────────

    def _check_thermal(self, telemetry: dict) -> list[dict]:
        flags = []
        cpu_temp = telemetry.get("cpu_temp_c")

        if cpu_temp is None:
            return flags

        try:
            temp = float(cpu_temp)
        except (TypeError, ValueError):
            return flags

        if temp >= THERMAL_CRITICAL_C:
            flags.append({
                "type": "THERMAL_ALERT",
                "severity": "CRITICAL",
                "context": {
                    "description": f"Pi CPU temperature {temp:.1f}°C exceeds critical threshold ({THERMAL_CRITICAL_C}°C)",
                    "cpu_temp_c": temp,
                    "threshold_c": THERMAL_CRITICAL_C,
                    "impact": "Risk of thermal throttling or shutdown. "
                              "Image capture timing and downlink speed will degrade.",
                    "recommended_action": "Enter safe mode or reduce capture rate.",
                },
            })
        elif temp >= THERMAL_WARNING_C:
            flags.append({
                "type": "THERMAL_ALERT",
                "severity": "WARNING",
                "context": {
                    "description": f"Pi CPU temperature {temp:.1f}°C approaching critical ({THERMAL_CRITICAL_C}°C)",
                    "cpu_temp_c": temp,
                    "threshold_c": THERMAL_WARNING_C,
                    "impact": "Performance may degrade if temperature continues to rise.",
                    "recommended_action": "Monitor closely. Consider reducing exposure time.",
                },
            })

        return flags

    # ── LANDING RISK CROSS-CHECK ──────────────────────────────────────────

    def _check_landing_risks(
        self,
        landing_candidates: list[dict],
        observation_count: np.ndarray,
        fine_hazard_grid: np.ndarray,
    ) -> list[dict]:
        """Cross-check landing candidates against survey data."""
        flags = []

        for cand in landing_candidates:
            rank = cand.get("rank", "?")
            grid_rc = cand.get("grid_rc", [0, 0])
            pos_cm = cand.get("position_cm", [0, 0])
            score = cand.get("score", 0)

            # Map fine grid cell to coarse grid cell for observation lookup
            fine_r, fine_c = grid_rc
            coarse_r = fine_r * config.SEG_GRID_CELL_PX // config.MOSAIC_GRID_CELL_PX
            coarse_c = fine_c * config.SEG_GRID_CELL_PX // config.MOSAIC_GRID_CELL_PX

            # Clamp to grid bounds
            coarse_r = min(coarse_r, observation_count.shape[0] - 1)
            coarse_c = min(coarse_c, observation_count.shape[1] - 1)

            obs = int(observation_count[coarse_r, coarse_c])

            if obs < SURVEY_MIN_OBSERVATIONS:
                flags.append({
                    "type": "SURVEY_GAP",
                    "severity": "CRITICAL",
                    "context": {
                        "description": (
                            f"Landing Zone #{rank} at ({pos_cm[0]}, {pos_cm[1]}) cm "
                            f"has only {obs} observation(s). Safe-Sample rule requires "
                            f">= {SURVEY_MIN_OBSERVATIONS}. Cannot certify."
                        ),
                        "landing_rank": rank,
                        "position_cm": pos_cm,
                        "observations": obs,
                        "required_observations": SURVEY_MIN_OBSERVATIONS,
                        "landing_score": score,
                        "certification": "DENIED",
                    },
                })

        return flags

    # ── SUMMARY ───────────────────────────────────────────────────────────

    def _build_summary(self, flags: list[dict]) -> dict:
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {"CRITICAL": 0, "WARNING": 0, "INFO": 0}

        for f in flags:
            by_type[f["type"]] = by_type.get(f["type"], 0) + 1
            by_severity[f["severity"]] = by_severity.get(f["severity"], 0) + 1

        if by_severity["CRITICAL"] > 0:
            status = "NO-GO"
        elif by_severity["WARNING"] > 0:
            status = "CONDITIONAL"
        else:
            status = "GO"

        return {
            "status": status,
            "total_flags": len(flags),
            "by_type": by_type,
            "by_severity": by_severity,
        }


# ── Test function ─────────────────────────────────────────────────────────────

def test_mission_analyzer():
    """Verify MissionAnalyzer produces correct flags from mock data."""
    analyzer = MissionAnalyzer()
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}")

    # ── Mock data ─────────────────────────────────────────────────────

    # Observation grid: 8x8 coarse grid, most cells have 3 obs, some have 1
    obs_count = np.full((8, 8), 3, dtype=np.int32)
    obs_count[2, 3] = 1  # low observation
    obs_count[5, 6] = 0  # unsurveyed
    obs_count[6, 7] = 1  # low observation

    surveyed = obs_count > 0

    # Fine hazard grid
    fine_hazard = np.full((40, 64), 1, dtype=np.uint8)  # all SAND
    fine_hazard[10:15, 20:25] = 4  # crater

    # YOLO detections — one low-confidence in high-shadow image
    yolo_detections = {
        "pass1_img02.jpg": [
            {"class": "crater", "confidence": 0.55, "center": [200, 150],
             "bbox": [180, 130, 220, 170]},
            {"class": "boulder", "confidence": 0.85, "center": [400, 300],
             "bbox": [380, 280, 420, 320]},
        ],
        "pass1_img05.jpg": [
            {"class": "crater", "confidence": 0.92, "center": [100, 100],
             "bbox": [80, 80, 120, 120]},
        ],
    }

    # Shadow percentages — img02 has high shadow
    shadow_pcts = {
        "pass1_img02.jpg": 55.0,
        "pass1_img05.jpg": 10.0,
    }

    # Fused results — one disagreement
    fused_results = [
        {
            "cell": [3, 7],
            "classical_class": "MODERATE",
            "classical_confidence": 0.65,
            "yolo_detections": [{"class": "crater", "confidence": 0.64}],
            "fused_classification": "HAZARD",
            "fused_confidence": 0.55,
            "agreement": False,
        },
        {
            "cell": [2, 4],
            "classical_class": "HAZARD",
            "classical_confidence": 0.88,
            "yolo_detections": [{"class": "crater", "confidence": 0.91}],
            "fused_classification": "HAZARD",
            "fused_confidence": 0.95,
            "agreement": True,
        },
    ]

    # IMU data — pitch drifting significantly
    image_metadata = [
        {"pass_number": 1, "imu": {"yaw_deg": 270.0, "pitch_deg": -88.0, "roll_deg": 0.5}},
        {"pass_number": 1, "imu": {"yaw_deg": 271.0, "pitch_deg": -85.0, "roll_deg": 0.3}},
        {"pass_number": 1, "imu": {"yaw_deg": 270.5, "pitch_deg": -82.0, "roll_deg": 0.8}},
        {"pass_number": 1, "imu": {"yaw_deg": 269.5, "pitch_deg": -78.0, "roll_deg": 0.4}},
        {"pass_number": 1, "imu": {"yaw_deg": 270.2, "pitch_deg": -88.5, "roll_deg": 0.6}},
    ]

    # Telemetry — hot CPU
    telemetry = {"cpu_temp_c": 78.5, "state": "IMAGING"}

    # Landing candidates — one in low-obs area
    landing_candidates = [
        {"rank": 1, "grid_rc": [20, 30], "position_cm": [75.0, 50.0], "score": 0.87,
         "breakdown": {}},
    ]

    # ── Run analysis ──────────────────────────────────────────────────

    print("=== MissionAnalyzer Test ===\n")

    report = analyzer.analyze(
        observation_count=obs_count,
        surveyed_mask=surveyed,
        fine_hazard_grid=fine_hazard,
        yolo_detections=yolo_detections,
        fused_results=fused_results,
        shadow_percentages=shadow_pcts,
        image_metadata=image_metadata,
        telemetry=telemetry,
        landing_candidates=landing_candidates,
    )

    flags = report["flags"]
    summary = report["summary"]

    # ── Verify flag generation ────────────────────────────────────────

    print("--- Flag Generation ---")
    check("Report has flags list", isinstance(flags, list))
    check("Report has summary", isinstance(summary, dict))
    check("Report has timestamp", "timestamp" in report)
    check("At least 1 flag generated", len(flags) >= 1)

    flag_types = [f["type"] for f in flags]
    print(f"\n  Generated {len(flags)} flags: {flag_types}\n")

    # SURVEY_GAP: cells with obs < 2
    print("--- SURVEY_GAP ---")
    survey_flags = [f for f in flags if f["type"] == "SURVEY_GAP"]
    check("SURVEY_GAP flag(s) generated", len(survey_flags) >= 1)
    gap_flag = next((f for f in survey_flags if "low_obs_cells" in str(f.get("context", {}).get("total_low_obs_cells", ""))), survey_flags[0] if survey_flags else None)
    if gap_flag:
        check("SURVEY_GAP has severity", "severity" in gap_flag)
        check("SURVEY_GAP has context", "context" in gap_flag)
        check("SURVEY_GAP context has description", "description" in gap_flag["context"])

    # DETECTION_CONFLICT: low-conf YOLO in high-shadow
    print("\n--- DETECTION_CONFLICT ---")
    conflict_flags = [f for f in flags if f["type"] == "DETECTION_CONFLICT"]
    check("DETECTION_CONFLICT flag(s) generated", len(conflict_flags) >= 1)

    # Should flag: crater at 0.55 conf in 55% shadow image
    shadow_conflict = [f for f in conflict_flags
                       if f.get("context", {}).get("likely_cause") == "shadow_artifact"]
    check("Shadow artifact conflict detected", len(shadow_conflict) >= 1)
    if shadow_conflict:
        ctx = shadow_conflict[0]["context"]
        check("Shadow conflict has detection_confidence", "detection_confidence" in ctx)
        check("Shadow conflict confidence < 0.70", ctx.get("detection_confidence", 1.0) < 0.70)
        check("Shadow conflict has image_shadow_pct", "image_shadow_pct" in ctx)

    # Should flag: CV/YOLO disagreement
    arch_conflict = [f for f in conflict_flags
                     if f.get("context", {}).get("likely_cause") == "architecture_blind_spot"]
    check("Architecture disagreement detected", len(arch_conflict) >= 1)
    if arch_conflict:
        ctx = arch_conflict[0]["context"]
        check("Arch conflict has classical_class", "classical_class" in ctx)
        check("Arch conflict has yolo_classes", "yolo_classes" in ctx)

    # Should NOT flag: high-confidence crater (0.92) in low-shadow image (10%)
    false_flags = [f for f in conflict_flags
                   if f.get("context", {}).get("source_image") == "pass1_img05.jpg"
                   and f.get("context", {}).get("likely_cause") == "shadow_artifact"]
    check("High-confidence detection NOT falsely flagged", len(false_flags) == 0)

    # Should NOT flag: boulder at 0.85 conf (above threshold)
    boulder_flags = [f for f in conflict_flags
                     if f.get("context", {}).get("detection_class") == "boulder"
                     and f.get("context", {}).get("likely_cause") == "shadow_artifact"]
    check("High-confidence boulder NOT flagged", len(boulder_flags) == 0)

    # SENSOR_DRIFT: pitch variance should exceed threshold
    print("\n--- SENSOR_DRIFT ---")
    drift_flags = [f for f in flags if f["type"] == "SENSOR_DRIFT"]
    check("SENSOR_DRIFT flag generated", len(drift_flags) >= 1)
    if drift_flags:
        ctx = drift_flags[0]["context"]
        check("Drift flag has pass_number", "pass_number" in ctx)
        check("Drift flag has drift_axes", "drift_axes" in ctx)
        axes = [d["axis"] for d in ctx.get("drift_axes", [])]
        check("Pitch drift detected", "pitch" in axes)
        # Yaw variance is ~0.28 — should NOT trigger (threshold 5.0)
        check("Yaw NOT falsely flagged as drift", "yaw" not in axes)

    # THERMAL_ALERT: 78.5°C > 75°C warning
    print("\n--- THERMAL_ALERT ---")
    thermal_flags = [f for f in flags if f["type"] == "THERMAL_ALERT"]
    check("THERMAL_ALERT flag generated", len(thermal_flags) >= 1)
    if thermal_flags:
        check("Thermal severity is WARNING (not CRITICAL at 78.5°C)",
              thermal_flags[0]["severity"] == "WARNING")
        ctx = thermal_flags[0]["context"]
        check("Thermal has cpu_temp_c", "cpu_temp_c" in ctx)
        check("Thermal temp is 78.5", ctx.get("cpu_temp_c") == 78.5)

    # ── Verify flag structure ─────────────────────────────────────────

    print("\n--- Flag Structure ---")
    for i, f in enumerate(flags):
        check(f"Flag {i} has type", "type" in f)
        check(f"Flag {i} has severity", f.get("severity") in ("INFO", "WARNING", "CRITICAL"))
        check(f"Flag {i} has context dict", isinstance(f.get("context"), dict))
        check(f"Flag {i} context has description", "description" in f.get("context", {}))

    # ── Verify summary ────────────────────────────────────────────────

    print("\n--- Summary ---")
    check("Summary has status", "status" in summary)
    check("Summary has total_flags", summary.get("total_flags") == len(flags))
    check("Summary has by_type", isinstance(summary.get("by_type"), dict))
    check("Summary has by_severity", isinstance(summary.get("by_severity"), dict))
    # With CRITICAL (landing zone in low-obs area) → should be NO-GO
    check("Summary status reflects CRITICAL flags",
          summary["status"] == "NO-GO" if summary["by_severity"].get("CRITICAL", 0) > 0
          else summary["status"] in ("CONDITIONAL", "GO"))

    # ── Verify sorting (CRITICAL first) ───────────────────────────────

    print("\n--- Sorting ---")
    severities = [f["severity"] for f in flags]
    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    numeric = [severity_order[s] for s in severities]
    check("Flags sorted CRITICAL → WARNING → INFO", numeric == sorted(numeric))

    # ── Verify JSON serializable ──────────────────────────────────────

    print("\n--- Serialization ---")
    import json
    try:
        json.dumps(report)
        check("Report is JSON-serializable", True)
    except (TypeError, ValueError) as e:
        check(f"Report is JSON-serializable (failed: {e})", False)

    # ── Edge case: no data ────────────────────────────────────────────

    print("\n--- Edge Case: Empty Data ---")
    empty_report = analyzer.analyze(
        observation_count=np.zeros((1, 1), dtype=np.int32),
        surveyed_mask=np.zeros((1, 1), dtype=bool),
        fine_hazard_grid=np.zeros((1, 1), dtype=np.uint8),
        yolo_detections={},
        fused_results=[],
        shadow_percentages={},
        image_metadata=[],
        telemetry={},
    )
    check("Empty data produces valid report", isinstance(empty_report["flags"], list))
    check("Empty data has timestamp", "timestamp" in empty_report)
    check("Empty data summary status is GO", empty_report["summary"]["status"] == "GO")

    # ── Edge case: critical thermal ───────────────────────────────────

    print("\n--- Edge Case: Critical Thermal ---")
    hot_report = analyzer.analyze(
        observation_count=np.zeros((1, 1), dtype=np.int32),
        surveyed_mask=np.zeros((1, 1), dtype=bool),
        fine_hazard_grid=np.zeros((1, 1), dtype=np.uint8),
        yolo_detections={},
        fused_results=[],
        shadow_percentages={},
        image_metadata=[],
        telemetry={"cpu_temp_c": 85.0},
    )
    hot_flags = [f for f in hot_report["flags"] if f["type"] == "THERMAL_ALERT"]
    check("Critical thermal flag at 85°C", len(hot_flags) == 1)
    check("Critical thermal severity is CRITICAL", hot_flags[0]["severity"] == "CRITICAL")

    # ── Results ───────────────────────────────────────────────────────

    total = passed + failed
    pct = round(100 * passed / total, 1) if total else 0
    print(f"\n===== RESULTS: {passed}/{total} passed ({pct}%) =====")
    if failed:
        print(f"{failed} test(s) FAILED")
    else:
        print("ALL TESTS PASSED")

    return failed == 0


if __name__ == "__main__":
    test_mission_analyzer()
