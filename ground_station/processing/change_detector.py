from __future__ import annotations
# processing/change_detector.py — Change detection between passes (CORE SCIENCE)
#
# Compares the same grid cell across two different passes to find real physical
# changes using Structural Similarity Index (SSIM) instead of raw pixel
# differencing. SSIM compares structure (edges, textures, patterns) not raw
# brightness — a flashlight shift changes brightness but not structure, so
# SSIM ignores it. A real terrain change (rock moved, new crater) alters
# structure and SSIM detects it.
#
# Alignment method:
#   Template matching on a corner patch for translation-only correction.
#   If confidence < 0.7 → flag "alignment_uncertain".
#
# Additional filtering:
#   - Aspect ratio > 5:1 → shadow edge shift → discard
#   - Area < CHANGE_MIN_AREA_PX → noise → discard
#
# Persistence check (if 3+ images of same cell):
#   - If change appears in both pass1→2 AND pass1→3 comparisons → persistent (real)
#   - If only in one → transient (might be lighting artifact)

import json
import logging
import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

import config

logger = logging.getLogger(__name__)

# Template patch: a square cropped from the corner of the image used as an
# anchor for translation-only alignment between passes.
_TEMPLATE_CROP = (10, 10, 80, 80)   # (x, y, w, h) — top-left corner region
_ALIGN_CONFIDENCE_THRESHOLD = 0.7

# Shape filtering
_MAX_ASPECT_RATIO = 5.0  # discard very elongated regions (shadow edges)

# Contour approximation for change region outlines
_CONTOUR_APPROX = cv2.CHAIN_APPROX_SIMPLE


class ChangeDetector:

    def detect(
        self,
        prev_image_path: str,
        new_image_path: str,
        grid_cell: tuple,
        pass_before: int,
        pass_after: int,
        all_cell_entries: list | None = None,
        yolo_before: list | None = None,
        yolo_after: list | None = None,
    ) -> dict | None:
        """
        Compare two real images of the same grid cell from different passes.

        Args:
            prev_image_path: Path to the earlier image (pass_before).
            new_image_path:  Path to the later image (pass_after).
            grid_cell:       (row, col) tuple.
            pass_before:     Pass number of the earlier image.
            pass_after:      Pass number of the later image.
            all_cell_entries: Optional list of all image index entries for this cell
                              (for persistence checking). Each entry has "pass" and "path".
            yolo_before: Optional YOLO detections for prev image (list of dicts with bbox, class).
            yolo_after:  Optional YOLO detections for new image.

        Returns dict with change_map_path, change_events, change_summary,
        or None if either image cannot be read.
        """
        if pass_before >= pass_after:
            logger.info(
                "ChangeDetector: skipping non-sequential comparison for cell %s "
                "(pass %s vs %s)",
                grid_cell,
                pass_before,
                pass_after,
            )
            return None

        prev_gray = _load_gray(prev_image_path)
        new_gray  = _load_gray(new_image_path)

        if prev_gray is None or new_gray is None:
            logger.error(
                f"ChangeDetector: cannot read images for cell {grid_cell} "
                f"(pass {pass_before} vs {pass_after})"
            )
            return None

        # Match sizes — should be identical, but guard against any resize
        if prev_gray.shape != new_gray.shape:
            new_gray = cv2.resize(new_gray, (prev_gray.shape[1], prev_gray.shape[0]))

        # ── Alignment ──
        aligned_new, alignment_confidence = _align_via_template(prev_gray, new_gray)
        alignment_uncertain = alignment_confidence < _ALIGN_CONFIDENCE_THRESHOLD

        if alignment_uncertain:
            logger.warning(
                f"ChangeDetector: cell {grid_cell} pass {pass_before}→{pass_after}: "
                f"alignment confidence {alignment_confidence:.2f} — results may be unreliable"
            )

        # ── SSIM comparison ──
        ssim_score, diff_map = ssim(prev_gray, aligned_new, full=True)

        # Convert SSIM diff map to change mask
        # diff_map values are 0.0–1.0 (1.0 = identical), invert so changes are bright
        change_map_raw = ((1.0 - diff_map) * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(
            change_map_raw, config.CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY
        )

        # ── Contours + filtering ──
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, _CONTOUR_APPROX)

        change_events = []
        event_id = 1

        for cnt in contours:
            area_px = int(cv2.contourArea(cnt))
            if area_px < config.CHANGE_MIN_AREA_PX:
                continue

            # Aspect ratio filter — discard shadow edge shifts
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bh == 0:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > _MAX_ASPECT_RATIO:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Mask for this contour — to compute mean difference and brightness direction
            contour_mask = np.zeros_like(prev_gray, dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)

            mean_diff = float(change_map_raw[contour_mask > 0].mean())

            # Brightness direction: compare new vs prev pixel values within contour
            prev_mean = float(prev_gray[contour_mask > 0].mean())
            new_mean  = float(aligned_new[contour_mask > 0].mean())
            change_type = "darkened" if new_mean < prev_mean else "brightened"

            px_per_cm = max(float(config.MOSAIC_PX_PER_CM), 1e-6)
            area_cm2 = round(area_px / (px_per_cm * px_per_cm), 3)
            description = _describe(change_type, area_cm2)

            change_events.append({
                "id": event_id,
                "grid_cell": list(grid_cell),
                "pass_before": pass_before,
                "pass_after": pass_after,
                "area_px": area_px,
                "area_cm2": area_cm2,
                "centroid": [cx, cy],
                "bbox": [bx, by, bw, bh],
                "type": change_type,
                "method": "pixel_diff",
                "mean_difference": round(mean_diff, 1),
                "ssim_score": round(ssim_score, 4),
                "alignment_confidence": round(alignment_confidence, 3),
                "persistence": False,  # updated below if applicable
                "description": description,
            })
            event_id += 1

        # ── YOLO-assisted classification ──
        if change_events and (yolo_before or yolo_after):
            _classify_changes_with_yolo(change_events, yolo_before or [], yolo_after or [])

        if yolo_before or yolo_after:
            event_id = _append_yolo_object_events(
                change_events,
                yolo_before or [],
                yolo_after or [],
                grid_cell,
                pass_before,
                pass_after,
                event_id,
            )

        # ── Persistence check ──
        if change_events and all_cell_entries and len(all_cell_entries) >= 3:
            _check_persistence(
                change_events, grid_cell, pass_before, pass_after, all_cell_entries
            )

        # ── Summary ──
        total_area_cm2 = sum(e["area_cm2"] for e in change_events if e["area_cm2"] is not None)
        largest_cm2 = max((e["area_cm2"] for e in change_events if e["area_cm2"] is not None), default=0.0)

        change_summary = {
            "total_events": len(change_events),
            "total_changed_area_px": sum(e["area_px"] for e in change_events),
            "total_changed_area_cm2": round(total_area_cm2, 3),
            "largest_change_cm2": round(largest_cm2, 3),
            "types": {
                "darkened":   sum(1 for e in change_events if e["type"] == "darkened"),
                "brightened": sum(1 for e in change_events if e["type"] == "brightened"),
            },
            "ssim_score": round(ssim_score, 4),
            "alignment_uncertain": alignment_uncertain,
            "alignment_confidence": round(alignment_confidence, 3),
        }

        logger.info(
            f"ChangeDetector: cell {grid_cell} pass {pass_before}→{pass_after}: "
            f"SSIM={ssim_score:.3f}, {len(change_events)} events, "
            f"alignment_conf={alignment_confidence:.2f}"
        )

        # ── Save output image ──
        new_bgr = cv2.imread(new_image_path)
        map_path = _save_change_map(
            new_bgr, prev_gray, aligned_new, contours, change_events,
            grid_cell, pass_before, pass_after, alignment_uncertain,
            new_image_path, ssim_score
        )

        # Save JSON data
        _save_changes_json(change_events, change_summary, grid_cell,
                           prev_image_path, new_image_path)

        return {
            "change_map_path": map_path,
            "change_events": change_events,
            "change_summary": change_summary,
        }


    def detect_mosaic(self, prev_entry, new_entry, stitcher) -> dict | None:
        """
        Compare two images that overlap in the mosaic using their homographies
        for alignment instead of template matching.

        Args:
            prev_entry: MosaicEntry for the earlier image
            new_entry:  MosaicEntry for the later image
            stitcher:   MosaicStitcher instance (for canvas access)

        Returns dict with change_events, change_summary, or None.
        """
        prev_gray = _load_gray(prev_entry.image_path)
        new_gray = _load_gray(new_entry.image_path)
        if prev_gray is None or new_gray is None:
            return None

        # Compute the overlapping region in mosaic space
        px, py, pw, ph = prev_entry.bbox
        nx, ny, nw, nh = new_entry.bbox

        ox_min = max(px, nx)
        oy_min = max(py, ny)
        ox_max = min(px + pw, nx + nw)
        oy_max = min(py + ph, ny + nh)

        if ox_max <= ox_min or oy_max <= oy_min:
            return None  # No overlap

        ow = ox_max - ox_min
        oh = oy_max - oy_min

        if ow < 20 or oh < 20:
            return None  # Overlap too small

        # Warp both images to the overlap region in mosaic space
        import numpy as np

        # Translate so overlap region starts at (0,0)
        T = np.eye(3, dtype=np.float64)
        T[0, 2] = -ox_min
        T[1, 2] = -oy_min

        H_prev = T @ prev_entry.homography
        H_new = T @ new_entry.homography

        prev_img = cv2.imread(prev_entry.image_path)
        new_img = cv2.imread(new_entry.image_path)
        if prev_img is None or new_img is None:
            return None

        prev_warped = cv2.warpPerspective(prev_img, H_prev, (ow, oh))
        new_warped = cv2.warpPerspective(new_img, H_new, (ow, oh))

        prev_warped_gray = cv2.cvtColor(prev_warped, cv2.COLOR_BGR2GRAY)
        new_warped_gray = cv2.cvtColor(new_warped, cv2.COLOR_BGR2GRAY)

        # SSIM on the overlap
        ssim_score, diff_map = ssim(prev_warped_gray, new_warped_gray, full=True)

        change_map_raw = ((1.0 - diff_map) * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(
            change_map_raw, config.CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, _CONTOUR_APPROX)

        change_events = []
        event_id = 1
        for cnt in contours:
            area_px = int(cv2.contourArea(cnt))
            if area_px < config.CHANGE_MIN_AREA_PX:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bh == 0:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > _MAX_ASPECT_RATIO:
                continue

            # Translate bbox back to mosaic coordinates
            mosaic_bx = bx + ox_min
            mosaic_by = by + oy_min

            change_events.append({
                "id": event_id,
                "mosaic_bbox": [mosaic_bx, mosaic_by, bw, bh],
                "area_px": area_px,
                "type": "detected",
                "ssim_score": round(ssim_score, 4),
                "persistence": False,
            })
            event_id += 1

        change_summary = {
            "total_events": len(change_events),
            "total_changed_area_cm2": 0.0,
            "largest_change_cm2": 0.0,
            "types": {"darkened": 0, "brightened": 0},
            "ssim_score": round(ssim_score, 4),
            "alignment_uncertain": False,
            "alignment_confidence": 1.0,
        }

        return {
            "change_events": change_events,
            "change_summary": change_summary,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_gray(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        logger.error(f"ChangeDetector: cannot read '{path}'")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _align_via_template(reference: np.ndarray, to_align: np.ndarray) -> tuple:
    """
    Align to_align onto reference using a small corner patch as anchor.
    Returns (aligned_image, confidence).
    """
    x, y, tw, th = _TEMPLATE_CROP
    template = reference[y:y + th, x:x + tw]

    if template.size == 0:
        return to_align, 0.0

    result = cv2.matchTemplate(to_align, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    dx = max_loc[0] - x
    dy = max_loc[1] - y

    if dx == 0 and dy == 0:
        return to_align, float(max_val)

    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    h, w = to_align.shape
    aligned = cv2.warpAffine(to_align, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    return aligned, float(max_val)


def _check_persistence(
    change_events: list,
    grid_cell: tuple,
    pass_before: int,
    pass_after: int,
    all_cell_entries: list,
):
    """
    Check if detected changes persist across multiple comparisons.

    If we have 3+ images (e.g. pass 1, 2, 3), and we just compared pass 2→3,
    also compare pass 1→3. If a change region overlaps in both comparisons,
    it's persistent (real terrain change). If only in one, it's transient.
    """
    # Find the earliest entry that isn't the current pass_before
    sorted_entries = sorted(all_cell_entries, key=lambda e: e["pass"])
    earliest = None
    for entry in sorted_entries:
        if entry["pass"] < pass_before:
            earliest = entry
            break

    if earliest is None:
        return  # only 2 images, can't do persistence check

    # Load and align the earliest image against the newest
    earliest_gray = _load_gray(earliest["path"])
    if earliest_gray is None:
        return

    # Load the new image again
    new_gray = _load_gray(all_cell_entries[-1]["path"] if all_cell_entries else "")
    # Find the entry matching pass_after
    new_entry = None
    for entry in all_cell_entries:
        if entry["pass"] == pass_after:
            new_entry = entry
            break
    if new_entry is None:
        return

    new_gray = _load_gray(new_entry["path"])
    if new_gray is None:
        return

    if earliest_gray.shape != new_gray.shape:
        new_gray = cv2.resize(new_gray, (earliest_gray.shape[1], earliest_gray.shape[0]))

    aligned_new, conf = _align_via_template(earliest_gray, new_gray)
    if conf < _ALIGN_CONFIDENCE_THRESHOLD:
        return  # can't trust this comparison

    # Run SSIM on earliest vs newest
    try:
        _, diff_map_alt = ssim(earliest_gray, aligned_new, full=True)
    except Exception:
        return

    change_map_alt = ((1.0 - diff_map_alt) * 255).astype(np.uint8)
    _, binary_alt = cv2.threshold(change_map_alt, config.CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)

    # For each change event, check if the same region shows change in the alt comparison
    for event in change_events:
        bx, by, bw, bh = event["bbox"]
        # Check if the alt binary mask has significant change pixels in this bbox
        roi = binary_alt[by:by + bh, bx:bx + bw]
        if roi.size == 0:
            continue
        overlap_pct = float(np.count_nonzero(roi)) / roi.size * 100.0
        # If >30% of the bbox also shows change in alt comparison → persistent
        event["persistence"] = overlap_pct > 30.0

    persistent_count = sum(1 for e in change_events if e["persistence"])
    logger.info(
        f"ChangeDetector: persistence check cell {grid_cell}: "
        f"{persistent_count}/{len(change_events)} events persistent "
        f"(earliest pass {earliest['pass']} vs pass {pass_after})"
    )


def _bbox_iou(a, b):
    """Intersection-over-union for two [x, y, w, h] bboxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _yolo_bbox_to_xywh(det):
    """Convert YOLO [x1, y1, x2, y2] bbox to [x, y, w, h]."""
    x1, y1, x2, y2 = det["bbox"]
    return [x1, y1, x2 - x1, y2 - y1]


def _classify_changes_with_yolo(change_events, yolo_before, yolo_after):
    """
    For each SSIM change region, check if it overlaps a YOLO detection
    that is NEW in the after image (not present in before). If so, classify
    the change with the YOLO object class instead of generic darkened/brightened.
    """
    IOU_THRESH = 0.15  # low threshold — change bbox and YOLO bbox won't align perfectly

    before_boxes = [_yolo_bbox_to_xywh(d) for d in yolo_before]

    for event in change_events:
        evt_bbox = event["bbox"]  # [x, y, w, h]
        best_match = None
        best_conf = 0.0

        for det in yolo_after:
            det_xywh = _yolo_bbox_to_xywh(det)
            iou = _bbox_iou(evt_bbox, det_xywh)
            if iou < IOU_THRESH:
                continue

            # Check this detection is NEW — not present in before image
            is_new = True
            for bb in before_boxes:
                if _bbox_iou(det_xywh, bb) > 0.3:
                    is_new = False
                    break

            if is_new and det["confidence"] > best_conf:
                best_match = det
                best_conf = det["confidence"]

        if best_match:
            event["yolo_class"] = best_match["class"]
            event["yolo_confidence"] = best_match["confidence"]
            event["yolo_original_class"] = best_match.get("original_class", "")
            event["description"] = _describe(
                event["type"], event.get("area_cm2"), best_match["class"], best_match["confidence"]
            )
        else:
            # Check if a before-only detection disappeared at this location
            for det in yolo_before:
                det_xywh = _yolo_bbox_to_xywh(det)
                iou = _bbox_iou(evt_bbox, det_xywh)
                if iou < IOU_THRESH:
                    continue
                # Verify it's gone in the after image
                still_there = False
                for aft in yolo_after:
                    if _bbox_iou(det_xywh, _yolo_bbox_to_xywh(aft)) > 0.3:
                        still_there = True
                        break
                if not still_there:
                    event["yolo_class"] = f"{det['class']}_removed"
                    event["yolo_confidence"] = det["confidence"]
                    event["description"] = _describe(
                        "removed", event.get("area_cm2"),
                        det["class"], det["confidence"]
                    )
                    break

    classified = sum(1 for e in change_events if "yolo_class" in e)
    if classified:
        logger.info(f"YOLO-assisted change classification: {classified}/{len(change_events)} events classified")


def _append_yolo_object_events(
    change_events,
    yolo_before,
    yolo_after,
    grid_cell,
    pass_before,
    pass_after,
    event_id,
):
    """Add semantic moved/new/removed events from matched detector boxes."""
    unmatched_after = set(range(len(yolo_after)))
    px_per_cm = max(float(config.MOSAIC_PX_PER_CM), 1e-6)

    for before in yolo_before:
        before_box = _yolo_bbox_to_xywh(before)
        before_class = before.get("class")
        bx = before_box[0] + before_box[2] / 2
        by = before_box[1] + before_box[3] / 2

        candidates = []
        for index in unmatched_after:
            after = yolo_after[index]
            if after.get("class") != before_class:
                continue
            after_box = _yolo_bbox_to_xywh(after)
            ax = after_box[0] + after_box[2] / 2
            ay = after_box[1] + after_box[3] / 2
            candidates.append(((ax - bx) ** 2 + (ay - by) ** 2, index, after_box))

        if not candidates:
            change_events.append(_object_event(
                event_id, grid_cell, pass_before, pass_after, before_box,
                "removed_object", before, px_per_cm,
            ))
            event_id += 1
            continue

        distance_sq, match_index, after_box = min(candidates)
        unmatched_after.remove(match_index)
        displacement = distance_sq ** 0.5
        movement_threshold = max(3.0, 0.08 * max(before_box[2], before_box[3]))
        if displacement >= movement_threshold:
            change_events.append(_object_event(
                event_id, grid_cell, pass_before, pass_after, after_box,
                "moved", yolo_after[match_index], px_per_cm,
                displacement_px=round(displacement, 2),
            ))
            event_id += 1

    for index in sorted(unmatched_after):
        detection = yolo_after[index]
        change_events.append(_object_event(
            event_id, grid_cell, pass_before, pass_after,
            _yolo_bbox_to_xywh(detection), "new_object", detection, px_per_cm,
        ))
        event_id += 1

    return event_id


def _object_event(
    event_id,
    grid_cell,
    pass_before,
    pass_after,
    bbox,
    change_type,
    detection,
    px_per_cm,
    displacement_px=None,
):
    x, y, width, height = bbox
    area_px = max(0, width) * max(0, height)
    result = {
        "id": event_id,
        "grid_cell": list(grid_cell),
        "pass_before": pass_before,
        "pass_after": pass_after,
        "area_px": int(area_px),
        "area_cm2": round(area_px / (px_per_cm * px_per_cm), 3),
        "centroid": [int(x + width / 2), int(y + height / 2)],
        "bbox": [int(x), int(y), int(width), int(height)],
        "type": change_type,
        "method": "object_tracking",
        "yolo_class": detection.get("class", "object"),
        "yolo_confidence": detection.get("confidence", 0.0),
        "persistence": False,
        "description": (
            f"{change_type.replace('_', ' ').title()}: "
            f"{detection.get('class', 'object')}"
        ),
    }
    if displacement_px is not None:
        result["displacement_px"] = displacement_px
    return result


def _describe(change_type: str, area_cm2: float | None,
              yolo_class: str | None = None, yolo_conf: float | None = None) -> str:
    area_str = f"{area_cm2:.1f} cm²" if area_cm2 is not None else "unknown area"

    if yolo_class and yolo_conf is not None:
        if change_type == "removed":
            return (f"{yolo_class.upper()} disappeared ({area_str}) — "
                    f"YOLO confidence {yolo_conf:.0%}")
        return (f"New {yolo_class.upper()} detected ({area_str}) — "
                f"YOLO confidence {yolo_conf:.0%}")

    if change_type == "darkened":
        return f"New dark region ({area_str}) — possible new crater, boulder, or shadow shift"
    elif change_type == "removed":
        return f"Object removed ({area_str})"
    else:
        return f"New bright region ({area_str}) — possible removed obstacle or lighting change"


def _save_change_map(
    new_bgr: np.ndarray,
    prev_gray: np.ndarray,
    aligned_new_gray: np.ndarray,
    contours,
    change_events: list,
    grid_cell: tuple,
    pass_before: int,
    pass_after: int,
    alignment_uncertain: bool,
    new_image_path: str,
    ssim_score: float = 0.0,
) -> str:
    os.makedirs(os.path.join(config.PROCESSED_DIR, "change_maps"), exist_ok=True)
    basename = os.path.splitext(os.path.basename(new_image_path))[0]
    out_path = os.path.join(config.PROCESSED_DIR, "change_maps",
                            basename + f"_change_p{pass_before}vs{pass_after}.png")

    if alignment_uncertain:
        # Side-by-side: prev (grey) | new (colour) with "ALIGNMENT UNCERTAIN" header
        prev_bgr  = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
        if new_bgr is None:
            new_bgr = cv2.cvtColor(aligned_new_gray, cv2.COLOR_GRAY2BGR)

        h = max(prev_bgr.shape[0], new_bgr.shape[0])
        pw, nw = prev_bgr.shape[1], new_bgr.shape[1]

        canvas_w = pw + nw + 4
        canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)
        canvas[:prev_bgr.shape[0], :pw] = prev_bgr
        canvas[:new_bgr.shape[0], pw + 4:pw + 4 + nw] = new_bgr

        banner_h = 30
        banner = np.full((banner_h, canvas_w, 3), (0, 100, 200), dtype=np.uint8)
        label = f"Cell {grid_cell}  pass {pass_before} | pass {pass_after}  [ALIGNMENT UNCERTAIN]"
        cv2.putText(banner, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        output = np.vstack([banner, canvas])

    else:
        if new_bgr is None:
            new_bgr = cv2.cvtColor(aligned_new_gray, cv2.COLOR_GRAY2BGR)
        output = new_bgr.copy()

        for event in change_events:
            cnt_idx = event["id"] - 1
            if cnt_idx < len(contours):
                cv2.drawContours(output, contours, cnt_idx, (0, 0, 220), 2)

            cx, cy = event["centroid"]
            area_str = f"{event['area_cm2']:.1f}cm²" if event["area_cm2"] else f"{event['area_px']}px"
            persist_tag = " [P]" if event.get("persistence") else ""
            label = f"#{event['id']} {event['type']} {area_str}{persist_tag}"
            cv2.putText(output, label, (max(0, cx - 40), max(12, cy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1, cv2.LINE_AA)

        banner_h = 28
        banner = np.full((banner_h, output.shape[1], 3), (30, 30, 30), dtype=np.uint8)
        n_events = len(change_events)
        label = f"Cell {grid_cell}  p{pass_before}→{pass_after}  SSIM={ssim_score:.3f}  {n_events} change(s)"
        cv2.putText(banner, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        output = np.vstack([banner, output])

    cv2.imwrite(out_path, output)
    logger.debug(f"Change map saved: {out_path}")
    return out_path


def _save_changes_json(change_events: list, change_summary: dict, grid_cell: tuple,
                       prev_image_path: str = None, new_image_path: str = None):
    """Save change detection results as JSON and update cost_grid.json change_cells."""
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(config.PROCESSED_DIR, "changes.json")

    # Load existing changes or start fresh
    existing = {"events": [], "summary": {"total_events": 0, "total_area": 0}}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
        except Exception:
            pass

    before_file = os.path.basename(prev_image_path) if prev_image_path else None
    after_file = os.path.basename(new_image_path) if new_image_path else None

    # Add new events with globally incrementing IDs
    max_id = max((e.get("id", 0) for e in existing.get("events", [])), default=0)
    for evt in change_events:
        max_id += 1
        existing["events"].append({
            "id": max_id,
            "cell": list(grid_cell),
            "pass_before": evt.get("pass_before", 0),
            "pass_after": evt.get("pass_after", 0),
            "area_px": evt.get("area_px", 0),
            "type": evt.get("type", "unknown"),
            "mean_diff": round(evt.get("mean_difference", 0), 1),
            "confidence": evt.get("alignment_confidence", 0),
            "ssim_score": evt.get("ssim_score", 0),
            "persistence": evt.get("persistence", False),
            "bbox": evt.get("bbox"),
            "before_image": before_file,
            "after_image": after_file,
        })

    # Update summary
    existing["summary"] = {
        "total_events": len(existing["events"]),
        "total_area": sum(e.get("area_px", 0) for e in existing["events"]),
    }

    try:
        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save changes.json: {e}")

    # Update change_cells in cost_grid.json
    cost_grid_path = os.path.join(config.PROCESSED_DIR, "cost_grid.json")
    if os.path.exists(cost_grid_path):
        try:
            with open(cost_grid_path) as f:
                cg = json.load(f)
            cell_list = cg.get("change_cells", [])
            if list(grid_cell) not in cell_list:
                cell_list.append(list(grid_cell))
                cg["change_cells"] = cell_list
                with open(cost_grid_path, "w") as f:
                    json.dump(cg, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update cost_grid.json change_cells: {e}")
