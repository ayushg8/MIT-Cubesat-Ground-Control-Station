#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from processing.mission_intelligence import build_task_queue, update_cell_state


def main():
    cell = update_cell_state(None, {
        "grid_cell": (1, 2),
        "pass_number": 1,
        "timestamp": "2026-03-20T00:00:00Z",
        "filename": "pass1_img00.jpg",
        "quality_score": 0.82,
        "hazard_class": "SHADOW",
        "hazard_confidence": 0.74,
        "shadow_percentage": 38.0,
        "has_change": False,
        "change_events": 0,
        "science_value": 0.88,
        "route_relevance": 0.0,
        "task_reason": "matches active GCS task",
    })

    assert cell["dominant_hazard"] == "SHADOW"
    assert cell["observation_count"] == 1
    assert cell["science_value"] >= 0.88

    snapshot = {
        "coverage": {"rows": 3, "cols": 3, "cells_filled": 1, "cells_total": 9, "pct": 11.1},
        "cell_states": {"1,2": cell},
        "routes": {"fastest": None, "safest": None, "balanced": None},
    }
    tasks = build_task_queue(snapshot, max_tasks=3)
    assert tasks
    assert tasks[0]["cmd"] in {"observe_cell", "revisit_cell"}
    print("mission_intelligence: PASS")


if __name__ == "__main__":
    main()
