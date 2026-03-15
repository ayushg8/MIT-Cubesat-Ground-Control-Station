#!/usr/bin/env python3
from __future__ import annotations
"""
test_pipeline.py — Smoke test for the CV processing pipeline.

Run from ground_station/:
    python test_pipeline.py

Uses a real image from data/received_images/ if one exists.
Falls back to a synthetic JPEG with a visible dark ellipse (shadow) + noise.
"""

import os
import sys

# Must run from ground_station/ so relative paths in config work
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(_HERE)

import glob
import logging
logging.basicConfig(level=logging.WARNING)   # suppress INFO noise during test

import cv2
import numpy as np

import config
from processing.shadow_detector import ShadowDetector
from processing.hazard_classifier import HazardClassifier
from processing.route_planner import RoutePlanner

# ── Terminal colour helpers ────────────────────────────────────────────────
_GREEN = "\033[92m"
_RED   = "\033[91m"
_YELLOW= "\033[93m"
_RESET = "\033[0m"

_passes = []
_failures = []

def _pass(label: str):
    _passes.append(label)
    print(f"  {_GREEN}PASS{_RESET}  {label}")

def _fail(label: str, reason: str = ""):
    _failures.append(label)
    suffix = f" — {reason}" if reason else ""
    print(f"  {_RED}FAIL{_RESET}  {label}{suffix}")

def _info(msg: str):
    print(f"         {_YELLOW}{msg}{_RESET}")

# ── Required directories ───────────────────────────────────────────────────
def _setup_dirs():
    for d in [
        config.RECEIVED_DIR,
        os.path.join(config.PROCESSED_DIR, "shadow_masks"),
        os.path.join(config.PROCESSED_DIR, "hazard_maps"),
        os.path.join(config.PROCESSED_DIR, "routes"),
        os.path.join(config.PROCESSED_DIR, "change_maps"),
        os.path.join(config.PROCESSED_DIR, "mosaics"),
        config.TELEMETRY_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

# ── Test image ─────────────────────────────────────────────────────────────
def _get_test_image() -> str:
    """Return a real received image if one exists, otherwise create a synthetic one."""
    existing = sorted(glob.glob(os.path.join(config.RECEIVED_DIR, "*.jpg")))
    if existing:
        print(f"  Using real image: {os.path.basename(existing[0])}")
        return existing[0]

    print("  No real images — creating synthetic test image (640×480)")
    img = np.ones((480, 640, 3), dtype=np.uint8) * 160   # light grey surface

    # Noise — simulates real surface texture
    rng = np.random.default_rng(42)
    noise = rng.integers(-18, 18, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Large dark ellipse — simulates a shadow / crater
    cv2.ellipse(img, (310, 270), (95, 65), 0, 0, 360, (25, 25, 25), -1)

    # Smaller dark circle — second shadow region
    cv2.circle(img, (155, 385), 42, (18, 18, 18), -1)

    path = os.path.join(config.RECEIVED_DIR, "test_synthetic_pass1_img00.jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


# ── Step 1: Shadow detector ────────────────────────────────────────────────
def test_shadow_detector(image_path: str) -> dict | None:
    print("\n[1] Shadow Detector")
    try:
        result = ShadowDetector().run(image_path)

        if result is None:
            _fail("ShadowDetector.run() returned None")
            return None

        pct     = result["shadow_percentage"]
        regions = result["shadow_regions"]
        mask    = result["shadow_mask"]
        mpath   = result.get("shadow_mask_path", "")

        _info(f"shadow_percentage = {pct:.1f}%")
        _info(f"regions           = {len(regions)}")
        _info(f"mask_path         = {mpath}")
        shadow_count = len([r for r in regions if r.get("type") == "shadow"])
        object_count = len([r for r in regions if r.get("type") == "object"])
        _info(f"shadow regions    = {shadow_count}, object regions = {object_count}")

        checks = {
            "percentage in 0–100":       0.0 <= pct <= 100.0,
            "mask shape matches image":  mask is not None and len(mask.shape) == 2,
            "mask dtype uint8":          mask is not None and mask.dtype == np.uint8,
            "at least 1 shadow region":  len(regions) >= 1,
            "mask PNG saved":            bool(mpath) and os.path.exists(mpath),
        }

        all_ok = True
        for label, ok in checks.items():
            if ok:
                _pass(label)
            else:
                _fail(label)
                all_ok = False

        return result if all_ok else result   # return even on partial fail so next tests run

    except Exception as e:
        _fail("ShadowDetector raised exception", str(e))
        import traceback; traceback.print_exc()
        return None


# ── Step 2: Hazard classifier ──────────────────────────────────────────────
def test_hazard_classifier(image_path: str, shadow_result: dict | None) -> dict | None:
    print("\n[2] Hazard Classifier")
    try:
        shadow_mask = shadow_result["shadow_mask"] if shadow_result else None
        shadow_pct  = shadow_result["shadow_percentage"] if shadow_result else 0.0
        grid_cell   = (2, 3)

        result = HazardClassifier().classify(image_path, shadow_mask, shadow_pct, grid_cell)

        cls      = result["hazard_class"]
        cost     = result["cost"]
        conf     = result.get("confidence", 0.0)
        map_path = result.get("hazard_map_path", "")
        details  = result.get("details", {})

        _info(f"hazard_class          = {cls}")
        _info(f"cost                  = {cost}")
        _info(f"confidence            = {conf:.3f}")
        _info(f"lbp_variance          = {details.get('lbp_variance', '--')}")
        _info(f"edge_density          = {details.get('edge_density', '--')}")
        _info(f"contours              = {details.get('significant_contour_count', '--')}")
        _info(f"hazard_map_path       = {map_path}")

        valid_classes = {"SAFE", "MODERATE", "SHADOW", "HAZARD", "IMPASSABLE"}
        valid_costs   = {config.COST_SAFE, config.COST_MODERATE, config.COST_SHADOW,
                         config.COST_HAZARD, config.COST_IMPASSABLE}

        checks = {
            "hazard_class is valid string": cls in valid_classes,
            "cost matches class":           cost in valid_costs,
            "grid_cell echoed correctly":   result["grid_cell"] == grid_cell,
            "hazard_map PNG saved":         bool(map_path) and os.path.exists(map_path),
            "confidence in 0-1 range":      0.0 <= conf <= 1.0,
        }

        all_ok = True
        for label, ok in checks.items():
            if ok:
                _pass(label)
            else:
                _fail(label)
                all_ok = False

        return result

    except Exception as e:
        _fail("HazardClassifier raised exception", str(e))
        import traceback; traceback.print_exc()
        return None


# ── Step 3: Route planner ──────────────────────────────────────────────────
def test_route_planner() -> dict | None:
    print("\n[3] Route Planner — clear grid (path must exist)")
    try:
        planner = RoutePlanner()

        cost_grid   = np.full((config.GRID_ROWS, config.GRID_COLS),
                               config.COST_SAFE, dtype=np.int32)
        hazard_grid = [["SAFE"] * config.GRID_COLS for _ in range(config.GRID_ROWS)]
        start = config.ROUTE_START
        end   = config.ROUTE_END

        result = planner.plan(cost_grid, hazard_grid, start, end)

        path      = result["path"]
        status    = result["status"]
        map_path  = result.get("route_map_path", "")

        _info(f"status         = {status}")
        _info(f"path_length    = {result['path_length']}")
        _info(f"total_cost     = {result['total_cost']:.2f}")
        _info(f"shadow_exp_pct = {result['shadow_exposure_pct']}%")
        _info(f"start → end    = {path[0] if path else '?'} → {path[-1] if path else '?'}")
        _info(f"route_map      = {map_path}")

        checks = {
            "status == 'found'":             status == "found",
            "path starts at ROUTE_START":    path and path[0] == list(start),
            "path ends at ROUTE_END":        path and path[-1] == list(end),
            "path_length > 0":               result["path_length"] > 0,
            "total_cost > 0":                result["total_cost"] > 0,
            "route_map PNG saved":           bool(map_path) and os.path.exists(map_path),
        }

        all_ok = True
        for label, ok in checks.items():
            if ok:
                _pass(label)
            else:
                _fail(label)
                all_ok = False

        # Sub-test: fully blocked grid — no path
        print("\n[3b] Route Planner — blocked grid (must report 'no viable route')")
        blocked = np.full((config.GRID_ROWS, config.GRID_COLS),
                           config.COST_IMPASSABLE, dtype=np.int32)
        blocked[start[0], start[1]] = config.COST_SAFE   # unblock start only
        blocked[end[0],   end[1]]   = config.COST_SAFE   # unblock end only

        blocked_result = planner.plan(blocked, None, start, end)
        if blocked_result["status"] == "no viable route":
            _pass("correctly reports 'no viable route' on blocked grid")
        else:
            _fail("expected 'no viable route'",
                  f"got '{blocked_result['status']}'")

        return result if all_ok else result

    except Exception as e:
        _fail("RoutePlanner raised exception", str(e))
        import traceback; traceback.print_exc()
        return None


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  MuraltZ GCS — Pipeline Smoke Test")
    print("=" * 58)

    _setup_dirs()
    image_path = _get_test_image()
    print(f"  Image: {image_path}")

    shadow_result = test_shadow_detector(image_path)
    hazard_result = test_hazard_classifier(image_path, shadow_result)
    route_result  = test_route_planner()

    print("\n" + "=" * 58)
    total  = len(_passes) + len(_failures)
    passed = len(_passes)
    print(f"  {passed}/{total} checks passed")
    if _failures:
        print(f"  {_RED}Failed:{_RESET}")
        for f in _failures:
            print(f"    • {f}")
    else:
        print(f"  {_GREEN}All checks passed — pipeline is functional{_RESET}")
    print("=" * 58)
    return len(_failures) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
