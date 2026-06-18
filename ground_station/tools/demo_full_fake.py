#!/usr/bin/env python3
"""demo_full_fake.py — Fully faked CubeSat mission demo.

Pre-generates ALL dashboard data — mosaic, shadow masks, hazard maps, cost grid,
routes, change detection, YOLO detections, telemetry — using real training images.
Then runs the dashboard with animated downlink progress and state transitions.

Nothing real runs. No pipeline. No CubeSat. Just a dashboard that looks 100% live.

Usage:
    cd ground_station
    python3 tools/demo_full_fake.py              # default port 3002
    python3 tools/demo_full_fake.py --port 3005  # custom port
"""

import argparse
import glob
import json
import logging
import math
import os
import random
import shutil
import sys
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Paths ────────────────────────────────────────────────────────────────────
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_IMG_SOURCES = [
    os.path.join(_REPO, "MIT-BWSI-Cubesat", "Images"),
    os.path.join(_REPO, "MIT-BWSI-Cubesat-Flight-Software", "Images"),
    os.path.join(_REPO, "yolo_training", "dataset_clean", "train", "images"),
    os.path.join(_REPO, "yolo_training", "dataset_clean", "valid", "images"),
    os.path.join(_REPO, "CubeSat Demo Images.v2i.yolov8", "train", "images"),
]
_SHADOW_SOURCES = [
    os.path.join(_REPO, "yolo_training", "shadow_dataset", "train", "images"),
    os.path.join(_REPO, "yolo_training", "shadow_dataset", "valid", "images"),
]

GRID_ROWS = 8
GRID_COLS = 8
NUM_PASSES = 2
IMAGES_PER_PASS = 8
TOTAL_IMAGES = NUM_PASSES * IMAGES_PER_PASS


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GATHER IMAGES
# ═══════════════════════════════════════════════════════════════════════════════

def gather_images(dirs, exts=("*.jpg", "*.jpeg", "*.png")):
    imgs = []
    for d in dirs:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            continue
        for ext in exts:
            imgs.extend(glob.glob(os.path.join(d, ext)))
    return sorted(set(imgs))


def pick_images(pool, n):
    out = []
    while len(out) < n:
        batch = random.sample(pool, min(n - len(out), len(pool)))
        out.extend(batch)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PAINT SHADOWS
# ═══════════════════════════════════════════════════════════════════════════════

def paint_shadow(img):
    """Add a soft synthetic shadow region."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    style = random.choice(["ellipse", "band", "corner"])
    if style == "ellipse":
        cx = random.randint(w // 5, 4 * w // 5)
        cy = random.randint(h // 5, 4 * h // 5)
        ax = random.randint(w // 6, w // 3)
        ay = random.randint(h // 6, h // 3)
        cv2.ellipse(mask, (cx, cy), (ax, ay), random.randint(0, 180), 0, 360, 1.0, -1)
    elif style == "band":
        bw = random.randint(h // 5, h // 3)
        off = random.randint(-w // 4, w // 4)
        pts = np.array([[off, 0], [off + bw, 0], [w + off + bw, h], [w + off, h]], np.int32)
        cv2.fillPoly(mask, [pts], 1.0)
    else:  # corner
        side = random.choice(["tl", "tr", "bl", "br"])
        pts_map = {
            "tl": [[0, 0], [w // 2, 0], [0, h // 2]],
            "tr": [[w, 0], [w // 2, 0], [w, h // 2]],
            "bl": [[0, h], [w // 2, h], [0, h // 2]],
            "br": [[w, h], [w // 2, h], [w, h // 2]],
        }
        cv2.fillPoly(mask, [np.array(pts_map[side], np.int32)], 1.0)

    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    darkness = random.uniform(0.15, 0.30)
    factor = 1.0 - mask * (1.0 - darkness)
    return (img.astype(np.float32) * factor[:, :, np.newaxis]).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPARE received_images + metadata
# ═══════════════════════════════════════════════════════════════════════════════

def nuke_data():
    """Delete all old data."""
    for d in ["data/received_images", "data/processed", "data/telemetry", "data/logs"]:
        if os.path.exists(d):
            shutil.rmtree(d)
    for f in ["data/mission_state.json"]:
        if os.path.exists(f):
            os.remove(f)


def ensure_dirs():
    for d in [
        "data/received_images", "data/telemetry", "data/logs",
        "data/processed/shadow_masks", "data/processed/hazard_maps",
        "data/processed/change_maps", "data/processed/mosaics",
        "data/processed/routes", "data/processed/mosaic_database",
        "data/processed/segmentation_maps", "data/processed/yolo_detections",
        "data/processed/roughness",
    ]:
        os.makedirs(d, exist_ok=True)


def prepare_received_images(terrain_imgs, shadow_imgs):
    """Copy images, paint shadows on ~50%, write metadata sidecars."""
    # Mix: every 3rd image from shadow dataset
    selected = []
    t_idx, s_idx = 0, 0
    t_pool = pick_images(terrain_imgs, TOTAL_IMAGES)
    s_pool = pick_images(shadow_imgs, TOTAL_IMAGES) if shadow_imgs else []

    for i in range(TOTAL_IMAGES):
        if s_pool and i % 3 == 1:
            selected.append(s_pool[s_idx % len(s_pool)])
            s_idx += 1
        else:
            selected.append(t_pool[t_idx % len(t_pool)])
            t_idx += 1

    files = []  # list of (jpg_path, meta_path, pass_num, cell_row, cell_col)
    for i, src in enumerate(selected):
        pass_num = (i // IMAGES_PER_PASS) + 1
        img_in_pass = i % IMAGES_PER_PASS
        row = img_in_pass // 4
        col = (img_in_pass % 4) + (pass_num - 1) * 2
        col = col % GRID_COLS

        fname = f"pass{pass_num}_img{img_in_pass:02d}_demo_{i:03d}.jpg"
        dst = os.path.join("data/received_images", fname)

        img = cv2.imread(src)
        if img is None:
            shutil.copy2(src, dst)
        else:
            # Paint shadow on ~55% of images
            if random.random() < 0.55:
                img = paint_shadow(img)
            cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        blur = round(random.uniform(0.65, 0.95), 3)
        exp = round(random.uniform(0.70, 0.98), 3)
        comb = round((blur + exp) / 2, 3)

        meta = {
            "grid_cell": [row, col],
            "pass_number": pass_num,
            "capture_time": datetime.now(timezone.utc).isoformat(),
            "blur_score": blur, "exposure_score": exp, "combined_score": comb,
            "resolution": [img.shape[1], img.shape[0]] if img is not None else [640, 480],
            "source": os.path.basename(src),
            "quality": {"blur_score": blur, "exposure_score": exp, "combined_score": comb},
            "imu": {
                "angular_rate": round(random.uniform(0.01, 0.12), 3),
                "stable": True, "nadir_locked": True,
                "nadir_angle_deg": round(random.uniform(8, 22), 1),
                "roll_deg": round(random.uniform(-12, 12), 1),
                "pitch_deg": round(random.uniform(-12, 12), 1),
            },
            "camera": {
                "exposure_us": random.choice([5000, 8000, 10000, 15000]),
                "analog_gain": round(random.uniform(1.5, 4.0), 1),
                "lux": random.randint(150, 350),
            },
        }

        meta_path = dst.replace(".jpg", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        files.append((dst, meta_path, pass_num, row, col))
        time.sleep(0.02)

    return files


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BUILD MOSAIC (tile images in a grid)
# ═══════════════════════════════════════════════════════════════════════════════

def build_mosaic(files):
    """Tile received images into a mosaic canvas."""
    # Read all images, resize to uniform size
    cell_w, cell_h = 320, 240
    canvas_w = GRID_COLS * cell_w
    canvas_h = GRID_ROWS * cell_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 40  # dark background

    placed = {}
    for jpg_path, _, pass_num, row, col in files:
        img = cv2.imread(jpg_path)
        if img is None:
            continue
        img = cv2.resize(img, (cell_w, cell_h))
        y, x = row * cell_h, col * cell_w
        # Blend if cell already has content (overlap from 2nd pass)
        if (row, col) in placed:
            existing = canvas[y:y + cell_h, x:x + cell_w]
            canvas[y:y + cell_h, x:x + cell_w] = cv2.addWeighted(existing, 0.4, img, 0.6, 0)
        else:
            canvas[y:y + cell_h, x:x + cell_w] = img
        placed[(row, col)] = True

    cv2.imwrite("data/processed/mosaics/mosaic_latest.png", canvas)
    return canvas, cell_w, cell_h


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GENERATE SHADOW MASKS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_shadow_masks(files):
    """Create shadow mask overlay images for each received image."""
    total_shadow_pct = 0
    all_regions = []
    count = 0
    for jpg_path, _, _, _, _ in files:
        img = cv2.imread(jpg_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect dark regions as shadows (threshold at ~40% of mean brightness)
        thresh = int(gray.mean() * 0.4)
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        shadow_pct = np.sum(binary > 0) / binary.size * 100
        total_shadow_pct += shadow_pct

        # Blue tint overlay
        overlay = img.copy()
        overlay[binary > 0] = (180, 60, 0)
        blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (255, 255, 255), 1)

        base = os.path.splitext(os.path.basename(jpg_path))[0]
        cv2.imwrite(f"data/processed/shadow_masks/{base}_shadow.png", blended)

        for c in contours[:5]:
            area = cv2.contourArea(c)
            if area > 50:
                M = cv2.moments(c)
                cx = M["m10"] / M["m00"] if M["m00"] > 0 else 0
                cy = M["m01"] / M["m00"] if M["m00"] > 0 else 0
                x, y, w, h = cv2.boundingRect(c)
                all_regions.append({
                    "id": len(all_regions) + 1,
                    "area_px": int(area), "width_px": w, "height_px": h,
                    "centroid": [round(cx, 1), round(cy, 1)],
                    "type": "shadow", "mean_boundary_gradient": round(random.uniform(8, 25), 1),
                })
        count += 1

    avg_shadow = total_shadow_pct / max(count, 1)
    shadow_data = {"shadow_pct": round(avg_shadow, 2), "regions": all_regions[-20:]}
    with open("data/processed/shadow_data.json", "w") as f:
        json.dump(shadow_data, f, indent=2)
    return avg_shadow


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GENERATE COST GRID + HAZARD MAPS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cost_grid(files, mosaic_canvas, cell_w, cell_h):
    """Build a realistic cost grid with mixed hazard types."""
    classes = ["SAFE", "MODERATE", "SHADOW", "HAZARD", "CRATER", "IMPASSABLE"]
    cost_map = {"SAFE": 1, "MODERATE": 5, "SHADOW": 15, "HAZARD": 20, "CRATER": 500, "IMPASSABLE": 999}

    grid = []
    classifications = []
    coverage = []
    pass_data = []
    confidences = []
    slopes = []

    # Track which cells have images
    covered_cells = set()
    for _, _, pass_num, row, col in files:
        covered_cells.add((row, col))

    for r in range(GRID_ROWS):
        grid_row = []
        class_row = []
        cov_row = []
        pass_row = []
        conf_row = []
        slope_row = []

        for c in range(GRID_COLS):
            if (r, c) in covered_cells:
                # Assign hazard based on position + randomness
                rng = random.random()
                if rng < 0.30:
                    cls = "SAFE"
                elif rng < 0.50:
                    cls = "MODERATE"
                elif rng < 0.65:
                    cls = "SHADOW"
                elif rng < 0.80:
                    cls = "HAZARD"
                elif rng < 0.92:
                    cls = "CRATER"
                else:
                    cls = "IMPASSABLE"

                grid_row.append(cost_map[cls])
                class_row.append(cls)
                cov_row.append(True)
                pass_row.append(2 if random.random() < 0.4 else 1)
                conf_row.append(round(random.uniform(0.6, 1.0), 2))
                slope_row.append(round(random.uniform(0, 35), 1))
            else:
                grid_row.append(config.COST_SAFE)
                class_row.append("SAFE")
                cov_row.append(False)
                pass_row.append(0)
                conf_row.append(0.0)
                slope_row.append(0.0)

        grid.append(grid_row)
        classifications.append(class_row)
        coverage.append(cov_row)
        pass_data.append(pass_row)
        confidences.append(conf_row)
        slopes.append(slope_row)

    # Add some change cells
    change_cells = []
    for _ in range(random.randint(2, 5)):
        r, c = random.choice(list(covered_cells))
        change_cells.append([r, c])

    cost_grid_data = {
        "grid": grid,
        "rows": GRID_ROWS,
        "cols": GRID_COLS,
        "classifications": classifications,
        "coverage": coverage,
        "pass_data": pass_data,
        "change_cells": change_cells,
        "confidences": confidences,
        "slopes": slopes,
    }
    with open("data/processed/cost_grid.json", "w") as f:
        json.dump(cost_grid_data, f, indent=2)

    # Generate hazard map images
    for jpg_path, _, _, row, col in files:
        img = cv2.imread(jpg_path)
        if img is None:
            continue
        cls = classifications[row][col] if row < GRID_ROWS and col < GRID_COLS else "SAFE"
        color_map = {
            "SAFE": (0, 200, 0), "MODERATE": (0, 200, 200), "SHADOW": (200, 100, 0),
            "HAZARD": (0, 0, 200), "CRATER": (0, 0, 180), "IMPASSABLE": (0, 0, 120),
        }
        color = color_map.get(cls, (100, 100, 100))
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), color, -1)
        blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        cv2.putText(blended, cls, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        base = os.path.splitext(os.path.basename(jpg_path))[0]
        cv2.imwrite(f"data/processed/hazard_maps/{base}_hazard.png", blended)

    return cost_grid_data


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GENERATE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_routes(cost_grid_data):
    """Create 3 fake but plausible routes on the cost grid."""
    grid = cost_grid_data["grid"]
    rows = cost_grid_data["rows"]
    cols = cost_grid_data["cols"]

    start = [0, 0]
    end = [rows - 1, cols - 1]

    def make_path(name, wobble):
        """Generate a path from start to end with some wobble."""
        path = [list(start)]
        r, c = start
        while r != end[0] or c != end[1]:
            # Mostly move toward goal, sometimes wobble
            if random.random() < wobble and c > 0 and c < cols - 1:
                c += random.choice([-1, 1])
            elif r < end[0] and random.random() < 0.6:
                r += 1
            elif c < end[1]:
                c += 1
            elif r < end[0]:
                r += 1
            r = max(0, min(r, rows - 1))
            c = max(0, min(c, cols - 1))
            if [r, c] != path[-1]:
                path.append([r, c])
        return path

    routes = []
    configs = [
        ("Fastest", 0.1, "#00ff88", "LOW"),
        ("Safest", 0.35, "#ffaa00", "LOW"),
        ("Balanced", 0.2, "#00aaff", "LOW"),
    ]

    for name, wobble, color, risk in configs:
        path = make_path(name, wobble)
        total_cost = sum(grid[r][c] for r, c in path)
        shadow_cells = sum(1 for r, c in path if cost_grid_data["classifications"][r][c] == "SHADOW")
        hazard_near = sum(1 for r, c in path if grid[r][c] > 5)

        routes.append({
            "name": name,
            "path": path,
            "stats": {
                "path_length_cells": len(path),
                "distance_cm": round(len(path) * config.GRID_CELL_SIZE_CM, 1),
                "max_shadow_exposure_pct": round(shadow_cells / max(len(path), 1) * 100, 1),
                "hazards_near_path": hazard_near,
                "nearest_hazard_distance_cells": random.randint(1, 3),
                "crater_cells_crossed": random.randint(0, 2),
                "avg_cell_cost": round(total_cost / max(len(path), 1), 1),
                "turn_count": sum(1 for i in range(2, len(path))
                                  if path[i][0] != path[i-2][0] and path[i][1] != path[i-2][1]),
                "risk_level": risk,
                "total_cost": total_cost,
                "cumulative_slip_risk": round(random.uniform(0.05, 0.25), 3),
                "status": "path_found",
                "reached_goal": True,
            },
            "color": color,
        })

    routes_data = {
        "routes": routes,
        "start": start,
        "end": end,
        "selected": "safest",
        "constrained": None,
    }

    with open("data/processed/routes.json", "w") as f:
        json.dump(routes_data, f, indent=2)

    # Generate route visualization image
    mosaic = cv2.imread("data/processed/mosaics/mosaic_latest.png")
    if mosaic is not None:
        cell_w = mosaic.shape[1] // GRID_COLS
        cell_h = mosaic.shape[0] // GRID_ROWS
        for route in routes:
            hex_color = route["color"].lstrip("#")
            bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
            pts = [(c * cell_w + cell_w // 2, r * cell_h + cell_h // 2) for r, c in route["path"]]
            for j in range(1, len(pts)):
                cv2.line(mosaic, pts[j-1], pts[j], bgr, 3)
        # Start/end markers
        cv2.circle(mosaic, (start[1] * cell_w + cell_w // 2, start[0] * cell_h + cell_h // 2),
                   12, (0, 255, 0), -1)
        cv2.circle(mosaic, (end[1] * cell_w + cell_w // 2, end[0] * cell_h + cell_h // 2),
                   12, (0, 0, 255), -1)
        cv2.imwrite("data/processed/routes/route_latest.png", mosaic)

    return routes_data


# ═══════════════════════════════════════════════════════════════════════════════
# 8. GENERATE CHANGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_changes(files):
    """Create fake change detection events between passes."""
    events = []
    event_types = ["new_object", "disappeared", "moved"]
    obj_classes = ["crater", "boulder", "feature"]

    # Find cells covered by both passes
    pass1_cells = {(r, c) for _, _, p, r, c in files if p == 1}
    pass2_cells = {(r, c) for _, _, p, r, c in files if p == 2}
    overlap = pass1_cells & pass2_cells

    for i, (r, c) in enumerate(list(overlap)[:4]):
        events.append({
            "id": i + 1,
            "cell": [r, c],
            "pass_before": 1,
            "pass_after": 2,
            "type": random.choice(event_types),
            "object_class": random.choice(obj_classes),
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "area_px": random.randint(200, 2000),
            "bbox": [random.randint(50, 200), random.randint(50, 150),
                     random.randint(250, 400), random.randint(200, 350)],
            "centroid": [random.randint(100, 300), random.randint(80, 200)],
            "displacement_px": round(random.uniform(5, 40), 1) if random.random() < 0.5 else None,
            "description": f"Object {'appeared' if i % 3 == 0 else 'moved' if i % 3 == 1 else 'disappeared'} at cell ({r},{c})",
            "method": "object_matching",
            "before_image": None,
            "after_image": None,
        })

    # Add a couple non-overlap events
    for i in range(2):
        r, c = random.choice(list(pass2_cells - overlap)) if (pass2_cells - overlap) else (random.randint(0, 1), random.randint(0, 3))
        events.append({
            "id": len(events) + 1,
            "cell": [r, c],
            "pass_before": 1, "pass_after": 2,
            "type": "new_object",
            "object_class": random.choice(obj_classes),
            "confidence": round(random.uniform(0.5, 0.85), 2),
            "area_px": random.randint(100, 800),
            "bbox": None, "centroid": None, "displacement_px": None,
            "description": f"New {random.choice(obj_classes)} detected at cell ({r},{c}) in pass 2",
            "method": "pixel_differencing",
            "before_image": None, "after_image": None,
        })

    changes_data = {
        "events": events,
        "summary": {
            "total_events": len(events),
            "total_area": sum(e["area_px"] for e in events),
        },
        "requires_multipass": True,
    }
    with open("data/processed/changes.json", "w") as f:
        json.dump(changes_data, f, indent=2)

    return changes_data


# ═══════════════════════════════════════════════════════════════════════════════
# 9. GENERATE YOLO DETECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_yolo_detections(files, cost_grid_data):
    """Create fake YOLO detection results."""
    det_per_cell = {}
    fused = []

    for jpg_path, _, pass_num, row, col in files:
        fname = os.path.basename(jpg_path)
        n_det = random.randint(0, 5)
        dets = []
        for _ in range(n_det):
            cls = random.choice(["crater", "boulder"])
            x1 = random.randint(20, 300)
            y1 = random.randint(20, 200)
            w = random.randint(30, 120)
            h = random.randint(30, 100)
            dets.append({
                "class": cls,
                "confidence": round(random.uniform(0.4, 0.95), 3),
                "bbox": [x1, y1, x1 + w, y1 + h],
                "area_px": w * h,
            })
        det_per_cell[fname] = dets

        cls_name = cost_grid_data["classifications"][row][col] if row < GRID_ROWS and col < GRID_COLS else "SAFE"
        fused.append({
            "cell": [row, col],
            "classical_classification": cls_name,
            "classical_confidence": round(random.uniform(0.6, 1.0), 2),
            "yolo_detections": [{"class": d["class"], "confidence": d["confidence"]} for d in dets],
            "fused_classification": cls_name,
            "fused_confidence": round(random.uniform(0.7, 1.0), 2),
            "agreement": True,
        })

        # Generate annotated image
        img = cv2.imread(jpg_path)
        if img is not None and dets:
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                color = (0, 0, 255) if d["class"] == "crater" else (0, 165, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{d['class']} {d['confidence']:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            base = os.path.splitext(os.path.basename(jpg_path))[0]
            cv2.imwrite(f"data/processed/yolo_detections/{base}_yolo.png", img)

    total_d = sum(len(v) for v in det_per_cell.values())
    yolo_data = {
        "detections_per_cell": det_per_cell,
        "fused_classifications": fused,
        "summary": {
            "total_detections": total_d,
            "craters_detected": sum(1 for v in det_per_cell.values() for d in v if d["class"] == "crater"),
            "boulders_detected": sum(1 for v in det_per_cell.values() for d in v if d["class"] == "boulder"),
            "cv_agreement_rate": round(random.uniform(0.75, 0.95), 2),
            "cells_analyzed": len(files),
        },
    }
    with open("data/processed/yolo_detections.json", "w") as f:
        json.dump(yolo_data, f, indent=2)

    return yolo_data


# ═══════════════════════════════════════════════════════════════════════════════
# 10. GENERATE TELEMETRY
# ═══════════════════════════════════════════════════════════════════════════════

def write_telemetry(state, pass_num, total_imgs, sent):
    from receiver import telemetry_parser
    telem = {
        "type": "telemetry",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cubesat_id": "MURALTZ-01",
        "pass_number": pass_num,
        "state": state,
        "uptime_sec": int(time.time()) % 100000,
        "imu": {
            "accel": [round(random.uniform(-0.2, 0.2), 2), round(random.uniform(-0.2, 0.2), 2), 9.78],
            "gyro": [round(random.uniform(-0.02, 0.02), 3)] * 3,
            "angular_rate": round(random.uniform(0.01, 0.1), 3),
            "stable": True, "nadir_locked": True,
            "nadir_angle_deg": round(random.uniform(8, 20), 1),
            "roll_deg": round(random.uniform(-10, 10), 1),
            "pitch_deg": round(random.uniform(-10, 10), 1),
        },
        "camera": {"exposure_us": 10000, "analog_gain": 2.5, "lux": 220, "mode": "auto"},
        "thermal": {"cpu_temp_c": round(random.uniform(45, 55), 1), "throttled": False},
        "storage": {"used_pct": round(20 + total_imgs * 1.2, 1), "free_mb": max(500, 1800 - total_imgs * 20)},
        "imaging": {
            "captured_this_pass": min(total_imgs, IMAGES_PER_PASS),
            "captured_total": total_imgs,
            "rejected_total": random.randint(0, 2),
            "rejection_breakdown": {"blur": 1, "underexposed": 0, "overexposed": 0, "motion_blur": 0},
        },
        "downlink": {
            "queued": max(0, total_imgs - sent),
            "sent_total": sent,
            "bytes_this_pass": sent * 50000,
            "budget_remaining": max(0, 500000 - sent * 50000),
            "gcs_reachable": True,
        },
        "coverage": {
            "cells_filled": min(total_imgs, 64),
            "cells_total": 64,
            "pct": round(min(total_imgs, 64) / 64 * 100, 1),
        },
        "errors": [],
        "recent_log": [],
    }
    raw = json.dumps(telem).encode("utf-8")
    telemetry_parser.parse_and_save_telemetry(raw, f"mock_pass{pass_num}.json")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. GENERATE MISSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def write_mission_state(files, routes_data, changes_data, yolo_data, shadow_pct):
    ms = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_passes": NUM_PASSES,
        "total_images_received": len(files),
        "total_images_corrupted": 0,
        "quality": {
            "mean_blur_score": round(random.uniform(0.75, 0.90), 3),
            "mean_exposure_score": round(random.uniform(0.80, 0.95), 3),
            "images_flagged_ground": random.randint(0, 2),
        },
        "coverage": {
            "cells_surveyed": len(set((r, c) for _, _, _, r, c in files)),
            "cells_total": GRID_ROWS * GRID_COLS,
            "pct": round(len(set((r, c) for _, _, _, r, c in files)) / (GRID_ROWS * GRID_COLS) * 100, 1),
        },
        "hazards": {
            "shadow_pct": round(shadow_pct, 1),
            "hazard_cells": random.randint(2, 5),
            "crater_cells": random.randint(1, 3),
            "impassable_cells": random.randint(0, 2),
        },
        "changes": {
            "total_events": len(changes_data["events"]),
            "passes_compared": 2,
        },
        "routes": {r["name"].lower(): r for r in routes_data["routes"]},
        "downlink": {
            "total_bytes": len(files) * 50000,
            "total_time_sec": len(files) * 5.0,
            "effective_rate_bps": 10000,
            "transfers_completed": len(files),
            "transfers_failed": 0,
        },
        "ml_detection": yolo_data.get("summary", {}),
    }
    with open(config.MISSION_STATE_FILE, "w") as f:
        json.dump(ms, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. RUN ANIMATED DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def run_dashboard_with_animation(files, port):
    """Start the dashboard and animate the downlink sequence."""
    import dashboard.app as dash_app
    from processing.mission_state import MissionState
    from processing.pipeline import Pipeline
    from receiver.downlink_state import get_state as get_downlink_state

    mission_state = MissionState()
    pipeline = Pipeline(mission_state)
    dash_app.set_pipeline(pipeline)
    dash_app.set_mission_state(mission_state)

    # Pre-populate quality log
    for jpg_path, meta_path, pass_num, row, col in files:
        with open(meta_path) as f:
            meta = json.load(f)
        dash_app.append_quality_entry({
            "filename": os.path.basename(jpg_path),
            "cubesat_score": meta.get("combined_score", 0.8),
            "ground_passed": True,
            "notes": [],
            "status": "ok",
        })

    # Start Flask
    flask_thread = threading.Thread(
        target=lambda: dash_app.app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()
    time.sleep(1.5)

    print(f"\n  Dashboard LIVE at http://localhost:{port}\n")

    # ── Animate the mission sequence ────────────────────────────────────────
    dl = get_downlink_state()
    global_sent = 0

    for pass_num in range(1, NUM_PASSES + 1):
        pass_files = [(j, m, p, r, c) for j, m, p, r, c in files if p == pass_num]

        # BOOT (pass 1 only)
        if pass_num == 1:
            print("  ┌─ BOOT")
            write_telemetry("BOOT", 1, 0, 0)
            time.sleep(2)
            print("  │  Self-test OK")
            print("  │")

        # WAITING
        print(f"  ┌─ WAITING (Pass {pass_num})")
        write_telemetry("WAITING", pass_num, global_sent, global_sent)
        time.sleep(2)
        print(f"  │  start_pass received")
        print(f"  │")

        # IMAGING
        print(f"  ┌─ IMAGING (Pass {pass_num} — {len(pass_files)} images)")
        write_telemetry("IMAGING", pass_num, global_sent, global_sent)
        time.sleep(1)
        for i, (jpg, meta_path, _, row, col) in enumerate(pass_files):
            with open(meta_path) as f:
                meta = json.load(f)
            q = meta.get("combined_score", 0.8)
            src = meta.get("source", "?")
            print(f"  │  [{i+1:2d}/{len(pass_files)}] cell=({row},{col})  Q={q:.2f}  [src: {src}]")
            write_telemetry("IMAGING", pass_num, global_sent + i + 1, global_sent)
            time.sleep(0.3)
        print(f"  │")

        # IDLE
        print(f"  ┌─ IDLE")
        write_telemetry("IDLE", pass_num, global_sent + len(pass_files), global_sent)
        time.sleep(1.5)

        # DOWNLINK with progress animation
        print(f"  ┌─ DOWNLINK (Pass {pass_num})")
        write_telemetry("DOWNLINK", pass_num, global_sent + len(pass_files), global_sent)

        session_bytes = sum(os.path.getsize(j) for j, _, _, _, _ in pass_files)
        dl.start_session(total_images=len(pass_files), total_bytes=session_bytes)

        for i, (jpg, _, _, row, col) in enumerate(pass_files):
            fsize = os.path.getsize(jpg)
            fname = os.path.basename(jpg)

            dl.start_transfer(fname, fsize)
            print(f"  │  [{i+1:2d}/{len(pass_files)}] {fname} ({fsize:,} B)...", end="", flush=True)

            # Animate progress
            sent_bytes = 0
            speed = 10000
            chunk = speed // 10
            while sent_bytes < fsize:
                sent_bytes = min(fsize, sent_bytes + chunk)
                dl.update_progress(sent_bytes)
                time.sleep(0.1)

            dl.set_status("validating")
            time.sleep(0.2)
            dl.set_status("complete")
            global_sent += 1
            mission_state.record_downlink_bytes(fsize, fsize / speed, True)

            write_telemetry("DOWNLINK", pass_num, global_sent + len(pass_files), global_sent)
            print(f" ACK")
            time.sleep(0.3)

        dl.end_session()
        print(f"  │  Downlink complete")
        print(f"  └─ Pass {pass_num} done\n")
        time.sleep(1)

    # Final state
    write_telemetry("WAITING", NUM_PASSES, TOTAL_IMAGES, TOTAL_IMAGES)

    print("=" * 60)
    print("  DEMO COMPLETE — Dashboard still live")
    print(f"  http://localhost:{port}")
    print("  Ctrl+C to exit")
    print("=" * 60)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Done.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fully faked CubeSat mission demo")
    parser.add_argument("--port", type=int, default=3002)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config.DASHBOARD_PORT = args.port
    random.seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s — %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    print()
    print("=" * 60)
    print("  MuraltZ GCS — FULL DEMO (everything pre-baked)")
    print("=" * 60)

    # 1. Gather images
    terrain = gather_images(_IMG_SOURCES)
    shadows = gather_images(_SHADOW_SOURCES)
    print(f"  Terrain images: {len(terrain)}")
    print(f"  Shadow images:  {len(shadows)}")

    if not terrain and not shadows:
        print("  ERROR: No images found!")
        sys.exit(1)

    # 2. Clean + prepare
    print("  Cleaning old data...")
    nuke_data()
    ensure_dirs()

    print("  Preparing received images...")
    files = prepare_received_images(terrain, shadows)
    print(f"  Prepared {len(files)} images")

    # 3. Build mosaic
    print("  Building mosaic...")
    mosaic, cw, ch = build_mosaic(files)
    print(f"  Mosaic: {mosaic.shape[1]}x{mosaic.shape[0]} px")

    # 4. Shadow masks
    print("  Generating shadow masks...")
    shadow_pct = generate_shadow_masks(files)
    print(f"  Average shadow: {shadow_pct:.1f}%")

    # 5. Cost grid + hazard maps
    print("  Generating cost grid + hazard maps...")
    cost_data = generate_cost_grid(files, mosaic, cw, ch)

    # 6. Routes
    print("  Generating routes...")
    routes_data = generate_routes(cost_data)
    print(f"  Routes: {', '.join(r['name'] for r in routes_data['routes'])}")

    # 7. Change detection
    print("  Generating change detection events...")
    changes_data = generate_changes(files)
    print(f"  Events: {len(changes_data['events'])}")

    # 8. YOLO detections
    print("  Generating YOLO detections...")
    yolo_data = generate_yolo_detections(files, cost_data)
    print(f"  Detections: {yolo_data['summary']['total_detections']}")

    # 9. Mission state
    print("  Writing mission state...")
    write_mission_state(files, routes_data, changes_data, yolo_data, shadow_pct)

    # 10. Write initial telemetry
    print("  Writing telemetry...")
    write_telemetry("WAITING", 1, 0, 0)

    print()
    print("  All data pre-generated. Starting dashboard + animation...")
    print()

    # 11. Run dashboard with animated downlink
    run_dashboard_with_animation(files, args.port)


if __name__ == "__main__":
    main()
