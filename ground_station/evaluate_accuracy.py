#!/usr/bin/env python3
"""Evaluate detection accuracy against manually-created ground truth.

Usage:
    cd ground_station
    python evaluate_accuracy.py --ground-truth ground_truth.json

The ground_truth.json file describes what is actually on the surface:

    {
      "cells": {
        "0,0": "safe",
        "0,1": "safe",
        "0,3": "hazard_boulder",
        "1,2": "crater",
        "2,4": "shadow",
        "3,0": "moderate"
      },
      "changes": [
        {"cell": "2,3", "type": "boulder_added", "pass_before": 1, "pass_after": 3},
        {"cell": "4,5", "type": "crater_added", "pass_before": 1, "pass_after": 3}
      ]
    }

Ground truth labels are mapped to pipeline classes:
    safe              → SAFE
    moderate          → MODERATE
    shadow            → SHADOW
    hazard, hazard_boulder, hazard_crater, boulder, crater → HAZARD
    impassable        → IMPASSABLE

The report prints to stdout and saves to data/processed/accuracy_report.json.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ─── Label mapping ────────────────────────────────────────────────────────────

_LABEL_MAP = {
    "safe": "SAFE",
    "moderate": "MODERATE",
    "shadow": "SHADOW",
    "hazard": "HAZARD",
    "hazard_boulder": "HAZARD",
    "hazard_crater": "HAZARD",
    "boulder": "HAZARD",
    "crater": "HAZARD",
    "impassable": "IMPASSABLE",
}

_ALL_CLASSES = ["SAFE", "MODERATE", "SHADOW", "HAZARD", "IMPASSABLE"]
_HAZARDOUS = {"HAZARD", "IMPASSABLE"}


def _normalize(label: str) -> str:
    return _LABEL_MAP.get(label.lower().strip(), "SAFE")


# ─── Load pipeline data ──────────────────────────────────────────────────────

def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_cost_grid() -> dict | None:
    return _load_json(os.path.join(config.PROCESSED_DIR, "cost_grid.json"))


def _load_changes() -> dict | None:
    return _load_json(os.path.join(config.PROCESSED_DIR, "changes.json"))


def _load_routes() -> dict | None:
    return _load_json(os.path.join(config.PROCESSED_DIR, "routes.json"))


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_hazards(gt_cells: dict, cost_grid: dict) -> dict:
    """Compare ground truth cell labels to pipeline classifications."""
    classifications = cost_grid.get("classifications", [])
    rows = len(classifications)
    cols = len(classifications[0]) if rows else 0

    results = []
    # Per-class tracking for precision/recall
    tp = {c: 0 for c in _ALL_CLASSES}
    fp = {c: 0 for c in _ALL_CLASSES}
    fn = {c: 0 for c in _ALL_CLASSES}

    correct = 0
    total = 0
    misclassifications = []

    for cell_key, gt_label in gt_cells.items():
        parts = cell_key.split(",")
        if len(parts) != 2:
            continue
        r, c = int(parts[0].strip()), int(parts[1].strip())
        if r >= rows or c >= cols:
            continue

        expected = _normalize(gt_label)
        predicted = classifications[r][c]
        total += 1

        if predicted == expected:
            correct += 1
            tp[expected] += 1
        else:
            misclassifications.append({
                "cell": f"({r},{c})",
                "predicted": predicted,
                "actual": expected,
                "gt_label": gt_label,
            })
            fp[predicted] += 1
            fn[expected] += 1

        results.append({
            "cell": cell_key,
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
        })

    # Per-class precision and recall
    per_class = {}
    for c in _ALL_CLASSES:
        prec_denom = tp[c] + fp[c]
        rec_denom = tp[c] + fn[c]
        precision = tp[c] / prec_denom if prec_denom > 0 else None
        recall = tp[c] / rec_denom if rec_denom > 0 else None
        # Only include classes that appear in ground truth or predictions
        if prec_denom > 0 or rec_denom > 0:
            per_class[c] = {
                "precision": round(precision, 2) if precision is not None else None,
                "recall": round(recall, 2) if recall is not None else None,
                "tp": tp[c], "fp": fp[c], "fn": fn[c],
            }

    accuracy = correct / total if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "misclassifications": misclassifications,
        "per_class": per_class,
        "details": results,
    }


def evaluate_shadow(gt_cells: dict, cost_grid: dict) -> dict:
    """Evaluate shadow detection specifically — TP/FP rates for shadow class."""
    classifications = cost_grid.get("classifications", [])
    rows = len(classifications)
    cols = len(classifications[0]) if rows else 0

    gt_shadow = set()
    gt_not_shadow = set()

    for cell_key, gt_label in gt_cells.items():
        parts = cell_key.split(",")
        if len(parts) != 2:
            continue
        r, c = int(parts[0].strip()), int(parts[1].strip())
        if r >= rows or c >= cols:
            continue
        if _normalize(gt_label) == "SHADOW":
            gt_shadow.add((r, c))
        else:
            gt_not_shadow.add((r, c))

    tp = fp = fn = tn = 0

    for r, c in gt_shadow:
        if classifications[r][c] == "SHADOW":
            tp += 1
        else:
            fn += 1

    for r, c in gt_not_shadow:
        if classifications[r][c] == "SHADOW":
            fp += 1
        else:
            tn += 1

    total_pos = tp + fn
    total_neg = fp + tn

    return {
        "true_positive_rate": round(tp / total_pos, 2) if total_pos > 0 else None,
        "false_positive_rate": round(fp / total_neg, 2) if total_neg > 0 else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate_changes(gt_changes: list, detected: dict) -> dict:
    """Compare ground truth change events to pipeline-detected changes."""
    detected_events = detected.get("events", []) if detected else []

    # Build a set of detected cells
    detected_cells = set()
    ssim_scores = []
    for evt in detected_events:
        cell = evt.get("cell", [])
        if len(cell) == 2:
            detected_cells.add((cell[0], cell[1]))
        if "ssim_score" in evt:
            ssim_scores.append(evt["ssim_score"])

    # Check each ground truth change
    hits = []
    misses = []
    for gt_evt in gt_changes:
        cell_key = gt_evt.get("cell", "")
        parts = cell_key.split(",")
        if len(parts) != 2:
            continue
        r, c = int(parts[0].strip()), int(parts[1].strip())
        if (r, c) in detected_cells:
            hits.append({"cell": cell_key, "type": gt_evt.get("type", "")})
        else:
            misses.append({"cell": cell_key, "type": gt_evt.get("type", "")})

    # False positives: detected changes not in ground truth
    gt_change_cells = set()
    for gt_evt in gt_changes:
        parts = gt_evt.get("cell", "").split(",")
        if len(parts) == 2:
            gt_change_cells.add((int(parts[0].strip()), int(parts[1].strip())))

    false_positives = []
    for cell in detected_cells:
        if cell not in gt_change_cells:
            false_positives.append(f"({cell[0]},{cell[1]})")

    total_gt = len(gt_changes)
    detected_count = len(hits)

    # F1 score
    precision = detected_count / (detected_count + len(false_positives)) if (detected_count + len(false_positives)) > 0 else 0.0
    recall = detected_count / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total_ground_truth": total_gt,
        "detected": detected_count,
        "missed": misses,
        "false_positives": false_positives,
        "ssim_scores": [round(s, 3) for s in ssim_scores],
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
    }


def evaluate_routes(gt_cells: dict, routes_data: dict) -> dict:
    """Check if any planned route passes through a genuinely hazardous cell."""
    # Build ground truth hazard set
    hazard_cells = set()
    for cell_key, gt_label in gt_cells.items():
        parts = cell_key.split(",")
        if len(parts) != 2:
            continue
        r, c = int(parts[0].strip()), int(parts[1].strip())
        if _normalize(gt_label) in _HAZARDOUS:
            hazard_cells.add((r, c))

    results = {}
    if not routes_data:
        return results

    for route_name in ("fastest", "safest", "balanced"):
        route = routes_data.get(route_name)
        if not route:
            continue

        path = route.get("path", [])
        hazards_hit = []
        for step in path:
            if len(step) >= 2:
                cell = (step[0], step[1])
                if cell in hazard_cells:
                    hazards_hit.append(f"({cell[0]},{cell[1]})")

        results[route_name] = {
            "avoids_all_hazards": len(hazards_hit) == 0,
            "hazards_on_path": hazards_hit,
            "path_length": len(path),
        }

    return results


# ─── Report formatting ────────────────────────────────────────────────────────

def print_report(hazard_eval: dict, shadow_eval: dict, change_eval: dict, route_eval: dict):
    print()
    print("=" * 60)
    print("  DETECTION ACCURACY REPORT")
    print("=" * 60)

    # ── Hazard Classification ──
    print("\n  Hazard Classification:")
    print(f"    Total cells evaluated: {hazard_eval['total']}")
    pct = hazard_eval['accuracy'] * 100
    print(f"    Correct: {hazard_eval['correct']}/{hazard_eval['total']} ({pct:.1f}%)")

    if hazard_eval["misclassifications"]:
        print("    Misclassifications:")
        for m in hazard_eval["misclassifications"]:
            print(f"      Cell {m['cell']}: predicted {m['predicted']}, actual {m['actual']}")

    if hazard_eval["per_class"]:
        print("\n    Per-category:")
        for cls, stats in hazard_eval["per_class"].items():
            p = f"{stats['precision']:.2f}" if stats["precision"] is not None else "N/A"
            r = f"{stats['recall']:.2f}" if stats["recall"] is not None else "N/A"
            print(f"      {cls:12s} precision={p}  recall={r}")

    # ── Shadow Detection ──
    print("\n  Shadow Detection:")
    tpr = shadow_eval.get("true_positive_rate")
    fpr = shadow_eval.get("false_positive_rate")
    print(f"    True positive rate:  {tpr * 100:.0f}%" if tpr is not None else "    True positive rate:  N/A (no shadow cells in ground truth)")
    print(f"    False positive rate: {fpr * 100:.0f}%" if fpr is not None else "    False positive rate: N/A")

    # ── Change Detection ──
    print("\n  Change Detection:")
    total_gt = change_eval["total_ground_truth"]
    detected = change_eval["detected"]
    det_pct = detected / total_gt * 100 if total_gt > 0 else 0
    print(f"    Changes detected:    {detected}/{total_gt} ({det_pct:.0f}%)")
    print(f"    False positives:     {len(change_eval['false_positives'])}")
    if change_eval["false_positives"]:
        print(f"      Cells: {', '.join(change_eval['false_positives'])}")
    if change_eval["ssim_scores"]:
        print(f"    SSIM scores:         {change_eval['ssim_scores']}")
    if change_eval["missed"]:
        print(f"    Missed changes:")
        for m in change_eval["missed"]:
            print(f"      Cell {m['cell']}: {m['type']}")

    # ── Route Safety ──
    print("\n  Route Safety:")
    if route_eval:
        for name in ("fastest", "safest", "balanced"):
            if name in route_eval:
                r = route_eval[name]
                safe = r["avoids_all_hazards"]
                status = "TRUE" if safe else "FALSE"
                line = f"    {name.capitalize():10s} route avoids all hazards: {status}"
                if not safe:
                    line += f"  (hits: {', '.join(r['hazards_on_path'])})"
                print(line)
    else:
        print("    No route data available")

    # ── Summary ──
    print("\n  " + "-" * 40)
    print(f"  Overall Classification Accuracy: {pct:.1f}%")
    print(f"  Overall Change Detection F1:     {change_eval['f1']:.2f}")

    # Mission success: classification >= 80% AND change F1 >= 0.7 AND safest route safe
    safest_safe = route_eval.get("safest", {}).get("avoids_all_hazards", True) if route_eval else True
    mission_pass = pct >= 80.0 and change_eval["f1"] >= 0.7 and safest_safe
    if mission_pass:
        print("  Mission Success: PASS")
    else:
        reasons = []
        if pct < 80.0:
            reasons.append(f"classification accuracy {pct:.0f}% < 80%")
        if change_eval["f1"] < 0.7:
            reasons.append(f"change F1 {change_eval['f1']:.2f} < 0.70")
        if not safest_safe:
            reasons.append("safest route hits real hazards")
        print(f"  Mission Success: FAIL ({'; '.join(reasons)})")

    print()


def save_report(
    hazard_eval: dict,
    shadow_eval: dict,
    change_eval: dict,
    route_eval: dict,
):
    """Save structured report JSON for inclusion in mission exports."""
    report = {
        "hazard_classification": {
            "total": hazard_eval["total"],
            "correct": hazard_eval["correct"],
            "accuracy": hazard_eval["accuracy"],
            "misclassifications": hazard_eval["misclassifications"],
            "per_class": hazard_eval["per_class"],
        },
        "shadow_detection": shadow_eval,
        "change_detection": {
            "total_ground_truth": change_eval["total_ground_truth"],
            "detected": change_eval["detected"],
            "false_positives": change_eval["false_positives"],
            "precision": change_eval["precision"],
            "recall": change_eval["recall"],
            "f1": change_eval["f1"],
        },
        "route_safety": route_eval,
        "overall_accuracy": hazard_eval["accuracy"],
        "overall_change_f1": change_eval["f1"],
        "mission_pass": (
            hazard_eval["accuracy"] >= 0.8
            and change_eval["f1"] >= 0.7
            and route_eval.get("safest", {}).get("avoids_all_hazards", True)
        ),
    }

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(config.PROCESSED_DIR, "accuracy_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate detection accuracy against ground truth"
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Path to ground_truth.json"
    )
    args = parser.parse_args()

    # Load ground truth
    if not os.path.exists(args.ground_truth):
        print(f"Error: {args.ground_truth} not found")
        sys.exit(1)

    with open(args.ground_truth) as f:
        gt = json.load(f)

    gt_cells = gt.get("cells", {})
    gt_changes = gt.get("changes", [])

    if not gt_cells:
        print("Error: ground_truth.json has no 'cells' entries")
        sys.exit(1)

    print(f"Ground truth: {len(gt_cells)} cells, {len(gt_changes)} changes")

    # Load pipeline data
    cost_grid = _load_cost_grid()
    if cost_grid is None:
        print(f"Error: {os.path.join(config.PROCESSED_DIR, 'cost_grid.json')} not found")
        print("Run the pipeline or generate_sample_outputs.py first")
        sys.exit(1)

    changes = _load_changes()
    routes = _load_routes()

    # Evaluate
    hazard_eval = evaluate_hazards(gt_cells, cost_grid)
    shadow_eval = evaluate_shadow(gt_cells, cost_grid)
    change_eval = evaluate_changes(gt_changes, changes)
    route_eval = evaluate_routes(gt_cells, routes)

    # Report
    print_report(hazard_eval, shadow_eval, change_eval, route_eval)
    save_report(hazard_eval, shadow_eval, change_eval, route_eval)


if __name__ == "__main__":
    main()
