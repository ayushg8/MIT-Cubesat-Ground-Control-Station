#!/usr/bin/env python3
"""Generate sample JSON data files so the dashboard can be tested without the Pi.

Creates realistic test data in data/processed/ matching the schema the dashboard
expects from the processing pipeline.

Usage:
    cd ground_station
    python3 generate_sample_outputs.py
"""

import json
import os
import random
import sys

# Add ground_station to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

PROCESSED = config.PROCESSED_DIR
os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(os.path.join(PROCESSED, "routes"), exist_ok=True)


def generate_cost_grid():
    """Generate a realistic 8x8 cost grid with mixed terrain."""
    random.seed(42)

    classifications = [
        ["SAFE",     "SAFE",     "MODERATE", "SAFE",       "SAFE",     "SHADOW",   "SAFE",       "SAFE"],
        ["SAFE",     "MODERATE", "MODERATE", "SAFE",       "SAFE",     "SHADOW",   "SHADOW",     "SAFE"],
        ["SAFE",     "SAFE",     "HAZARD",   "HAZARD",     "SAFE",     "SAFE",     "MODERATE",   "SAFE"],
        ["MODERATE", "SAFE",     "SAFE",     "IMPASSABLE", "SAFE",     "SAFE",     "SAFE",       "SAFE"],
        ["SAFE",     "SAFE",     "SAFE",     "SAFE",       "MODERATE", "SAFE",     "SAFE",       "MODERATE"],
        ["SHADOW",   "SAFE",     "SAFE",     "SAFE",       "SAFE",     "HAZARD",   "SAFE",       "SAFE"],
        ["SHADOW",   "SHADOW",   "SAFE",     "SAFE",       "SAFE",     "SAFE",     "MODERATE",   "SAFE"],
        ["SAFE",     "SAFE",     "SAFE",     "MODERATE",   "SAFE",     "SAFE",     "SAFE",       "SAFE"],
    ]

    cost_map = {
        "SAFE": config.COST_SAFE,
        "MODERATE": config.COST_MODERATE,
        "SHADOW": config.COST_SHADOW,
        "HAZARD": config.COST_HAZARD,
        "IMPASSABLE": config.COST_IMPASSABLE,
    }
    grid = [[cost_map[c] for c in row] for row in classifications]

    # Most cells surveyed, a few unsurveyed
    coverage = [[True] * 8 for _ in range(8)]
    coverage[0][7] = False
    coverage[7][0] = False
    coverage[4][6] = False

    pass_data = [
        [1, 1, 2, 1, 3, 2, 1, 0],
        [1, 2, 2, 1, 3, 2, 2, 3],
        [2, 1, 3, 3, 1, 2, 2, 1],
        [2, 1, 1, 3, 2, 1, 1, 2],
        [3, 2, 1, 1, 2, 3, 0, 2],
        [2, 1, 2, 1, 1, 3, 2, 1],
        [2, 2, 1, 3, 1, 1, 2, 3],
        [0, 1, 2, 2, 1, 1, 3, 1],
    ]

    # Confidence grid — higher for SAFE, lower for ambiguous
    conf_map = {
        "SAFE": lambda: round(random.uniform(0.82, 0.98), 3),
        "MODERATE": lambda: round(random.uniform(0.60, 0.85), 3),
        "SHADOW": lambda: round(random.uniform(0.70, 0.92), 3),
        "HAZARD": lambda: round(random.uniform(0.65, 0.88), 3),
        "IMPASSABLE": lambda: round(random.uniform(0.80, 0.95), 3),
    }
    confidences = [[conf_map[c]() for c in row] for row in classifications]

    data = {
        "grid": grid,
        "classifications": classifications,
        "coverage": coverage,
        "pass_data": pass_data,
        "change_cells": [[2, 3], [5, 5]],
        "confidences": confidences,
    }

    with open(os.path.join(PROCESSED, "cost_grid.json"), "w") as f:
        json.dump(data, f, indent=2)
    print("  cost_grid.json")


def generate_routes():
    """Generate 3 sample routes with realistic stats."""
    routes = [
        {
            "name": "Fastest",
            "path": [[0,0],[1,0],[2,0],[3,0],[4,0],[4,1],[4,2],[4,3],[4,4],[5,4],[5,3],[6,3],[6,4],[6,5],[7,5],[7,6],[7,7]],
            "stats": {
                "path_length_cells": 17,
                "distance_cm": 170.0,
                "max_shadow_exposure_pct": 11.8,
                "hazards_near_path": 2,
                "nearest_hazard_distance_cells": 1.41,
                "risk_level": "MODERATE",
                "total_cost": 28.5,
                "status": "found",
            },
            "color": "#00ff88",
        },
        {
            "name": "Safest",
            "path": [[0,0],[0,1],[1,1],[1,0],[2,0],[2,1],[3,1],[3,0],[4,0],[4,1],[4,2],[4,3],[4,4],[5,3],[5,2],[6,2],[6,3],[7,3],[7,4],[7,5],[7,6],[7,7]],
            "stats": {
                "path_length_cells": 22,
                "distance_cm": 220.0,
                "max_shadow_exposure_pct": 4.5,
                "hazards_near_path": 0,
                "nearest_hazard_distance_cells": 2.83,
                "risk_level": "LOW",
                "total_cost": 24.1,
                "status": "found",
            },
            "color": "#ffaa00",
        },
        {
            "name": "Balanced",
            "path": [[0,0],[1,0],[1,1],[2,1],[3,1],[4,1],[4,2],[4,3],[5,3],[5,4],[6,4],[6,5],[7,5],[7,6],[7,7]],
            "stats": {
                "path_length_cells": 15,
                "distance_cm": 150.0,
                "max_shadow_exposure_pct": 6.7,
                "hazards_near_path": 1,
                "nearest_hazard_distance_cells": 1.0,
                "risk_level": "MODERATE",
                "total_cost": 26.3,
                "status": "found",
            },
            "color": "#ff4444",
        },
    ]

    data = {
        "routes": routes,
        "start": [0, 0],
        "end": [7, 7],
        "selected": "safest",
        "constrained": None,
    }

    with open(os.path.join(PROCESSED, "routes.json"), "w") as f:
        json.dump(data, f, indent=2)
    print("  routes.json")


def generate_changes():
    """Generate sample change detection events with bounding boxes and image refs."""
    # Generate sample before/after images for the slider demo
    _generate_sample_change_images()

    data = {
        "events": [
            {
                "id": 1,
                "cell": [2, 3],
                "pass_before": 1,
                "pass_after": 3,
                "area_px": 320,
                "type": "darkened",
                "mean_diff": 67.3,
                "confidence": 0.89,
                "ssim_score": 0.847,
                "persistence": True,
                "bbox": [120, 80, 45, 38],
                "before_image": "sample_cell2_3_pass1.jpg",
                "after_image": "sample_cell2_3_pass3.jpg",
            },
            {
                "id": 2,
                "cell": [5, 5],
                "pass_before": 2,
                "pass_after": 3,
                "area_px": 185,
                "type": "brightened",
                "mean_diff": 42.1,
                "confidence": 0.76,
                "ssim_score": 0.912,
                "persistence": False,
                "bbox": [200, 150, 32, 28],
                "before_image": "sample_cell5_5_pass2.jpg",
                "after_image": "sample_cell5_5_pass3.jpg",
            },
        ],
        "summary": {
            "total_events": 2,
            "total_area": 505,
        },
    }

    with open(os.path.join(PROCESSED, "changes.json"), "w") as f:
        json.dump(data, f, indent=2)
    print("  changes.json")


def _generate_sample_change_images():
    """Create simple sample before/after images for change detection slider demo."""
    try:
        import numpy as np
        import cv2
    except ImportError:
        print("    (skipping sample images — opencv not available)")
        return

    img_dir = config.RECEIVED_DIR
    os.makedirs(img_dir, exist_ok=True)

    np.random.seed(42)

    # 320x240 sand-like texture
    def make_sand_image(seed_offset=0):
        np.random.seed(42 + seed_offset)
        base = np.full((240, 320, 3), (180, 170, 140), dtype=np.uint8)
        noise = np.random.randint(-20, 20, base.shape, dtype=np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Grid tape lines
        cv2.line(img, (0, 120), (320, 120), (60, 60, 60), 2)
        cv2.line(img, (160, 0), (160, 240), (60, 60, 60), 2)
        return img

    # Cell (2,3): before = plain sand, after = darkened region
    before1 = make_sand_image(0)
    after1 = before1.copy()
    cv2.rectangle(after1, (120, 80), (165, 118), (80, 70, 50), -1)  # darkened patch
    cv2.imwrite(os.path.join(img_dir, "sample_cell2_3_pass1.jpg"), before1)
    cv2.imwrite(os.path.join(img_dir, "sample_cell2_3_pass3.jpg"), after1)

    # Cell (5,5): before = plain sand, after = brightened region
    before2 = make_sand_image(10)
    after2 = before2.copy()
    cv2.rectangle(after2, (200, 150), (232, 178), (230, 225, 210), -1)  # brightened patch
    cv2.imwrite(os.path.join(img_dir, "sample_cell5_5_pass2.jpg"), before2)
    cv2.imwrite(os.path.join(img_dir, "sample_cell5_5_pass3.jpg"), after2)

    print("    sample change images (4 files)")


def generate_shadow_data():
    """Generate sample shadow detection data."""
    data = {
        "shadow_pct": 18.2,
        "regions": [
            {"id": 1, "area_px": 450, "width_px": 30, "height_px": 22, "centroid": [234.2, 156.8], "type": "shadow", "mean_boundary_gradient": 12.4},
            {"id": 2, "area_px": 280, "width_px": 20, "height_px": 18, "centroid": [89.5, 312.3], "type": "shadow", "mean_boundary_gradient": 18.7},
            {"id": 3, "area_px": 120, "width_px": 15, "height_px": 12, "centroid": [401.0, 78.6], "type": "object", "mean_boundary_gradient": 45.2},
        ],
    }

    with open(os.path.join(PROCESSED, "shadow_data.json"), "w") as f:
        json.dump(data, f, indent=2)
    print("  shadow_data.json")


def generate_mission_state():
    """Generate a realistic mission_state.json."""
    data = {
        "last_updated": "2026-03-15T19:30:00+00:00",
        "total_passes": 3,
        "total_images_received": 8,
        "total_images_corrupted": 0,
        "quality": {
            "avg_cubesat_score": 0.72,
            "ground_flagged": 1,
            "ground_flag_reasons": ["low_texture"],
        },
        "coverage": {
            "cells_filled": 61,
            "cells_total": 64,
            "pct": 95.3,
        },
        "hazards": {
            "safe": 42,
            "moderate": 10,
            "shadow": 6,
            "hazard": 3,
            "impassable": 1,
        },
        "changes": {
            "total_events": 2,
            "total_changed_area_cm2": 12.6,
            "largest_change_cm2": 8.0,
            "types": {"darkened": 1, "brightened": 1},
            "cells_with_changes": [[2, 3], [5, 5]],
            "alignment_warnings": 0,
        },
        "route": {
            "start": [0, 0],
            "end": [7, 7],
            "path_length": 22,
            "total_cost": 24.1,
            "shadow_exposure_pct": 4.5,
            "status": "found",
        },
        "routes": {
            "fastest": {
                "name": "Fastest",
                "path_length_cells": 17,
                "distance_cm": 170.0,
                "max_shadow_exposure_pct": 11.8,
                "hazards_near_path": 2,
                "risk_level": "MODERATE",
                "total_cost": 28.5,
                "status": "found",
                "color": "#00ff88",
            },
            "safest": {
                "name": "Safest",
                "path_length_cells": 22,
                "distance_cm": 220.0,
                "max_shadow_exposure_pct": 4.5,
                "hazards_near_path": 0,
                "risk_level": "LOW",
                "total_cost": 24.1,
                "status": "found",
                "color": "#ffaa00",
            },
            "balanced": {
                "name": "Balanced",
                "path_length_cells": 15,
                "distance_cm": 150.0,
                "max_shadow_exposure_pct": 6.7,
                "hazards_near_path": 1,
                "risk_level": "MODERATE",
                "total_cost": 26.3,
                "status": "found",
                "color": "#ff4444",
            },
            "selected": "safest",
            "constrained": None,
        },
        "downlink": {
            "total_bytes": 56000,
            "total_time_sec": 46.7,
            "effective_rate_bps": 1199.0,
            "failed_transfers": 0,
            "retransmit_requests": 0,
        },
        "uplink": {
            "commands_sent": 6,
            "commands_acked": 6,
        },
    }

    with open(config.MISSION_STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print("  mission_state.json")


if __name__ == "__main__":
    print("Generating sample output files...")
    generate_cost_grid()
    generate_routes()
    generate_changes()
    generate_shadow_data()
    generate_mission_state()
    print("Done. Restart the GCS server to see the data on the dashboard.")
