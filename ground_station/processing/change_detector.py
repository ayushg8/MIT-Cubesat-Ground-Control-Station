from __future__ import annotations
# processing/change_detector.py — Change detection between passes (CORE SCIENCE)
#
# Compares the same grid cell across two different passes to find real physical
# changes: moved rocks, new craters pressed into sand, lighting shifts.
#
# DOES NOT USE ORB. ORB feature matching fails on textureless sand — it finds
# almost no keypoints and produces a noisy, unreliable affine transform.
#
# Alignment method:
#   Both images cover the same taped grid cell at the same camera height, so
#   they are already approximately aligned. A small patch of a grid tape
#   intersection (high contrast, distinct on sand) is used as a template anchor
#   for fine translation-only correction via cv2.matchTemplate.
#
#   If template match confidence < 0.7 → flag "alignment_uncertain":
#     - still compute the diff (may be useful)
#     - save images side-by-side instead of diff overlay
#     - log a warning for the dashboard and mission_state.json

import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Template patch: a square cropped from the corner of the image where the grid
# tape intersection is most likely to appear. Adjust crop if camera framing changes.
_TEMPLATE_CROP = (10, 10, 80, 80)   # (x, y, w, h) — top-left corner region
_ALIGN_CONFIDENCE_THRESHOLD = 0.7

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
    ) -> dict | None:
        """
        Compare two real images of the same grid cell from different passes.

        Args:
            prev_image_path: Path to the earlier image (pass_before).
            new_image_path:  Path to the later image (pass_after).
            grid_cell:       (row, col) tuple.
            pass_before:     Pass number of the earlier image.
            pass_after:      Pass number of the later image.

        Returns dict with change_map_path, change_events, change_summary,
        or None if either image cannot be read.
        """
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

        # ── Absolute difference ──
        diff = cv2.absdiff(prev_gray, aligned_new)

        # ── Threshold + contours ──
        _, thresh = cv2.threshold(diff, config.CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, _CONTOUR_APPROX)

        change_events = []
        event_id = 1

        for cnt in contours:
            area_px = int(cv2.contourArea(cnt))
            if area_px < config.CHANGE_MIN_AREA_PX:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Mask for this contour — to compute mean difference and brightness direction
            contour_mask = np.zeros_like(diff, dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)

            mean_diff = float(diff[contour_mask > 0].mean())

            # Brightness direction: compare new vs prev pixel values within contour
            prev_mean = float(prev_gray[contour_mask > 0].mean())
            new_mean  = float(aligned_new[contour_mask > 0].mean())
            change_type = "darkened" if new_mean < prev_mean else "brightened"

            area_cm2 = None
            if config.GSD_CM_PER_PIXEL > 0.0:
                area_cm2 = round(area_px * (config.GSD_CM_PER_PIXEL ** 2), 3)

            description = _describe(change_type, area_cm2)

            change_events.append({
                "id": event_id,
                "grid_cell": list(grid_cell),
                "pass_before": pass_before,
                "pass_after": pass_after,
                "area_px": area_px,
                "area_cm2": area_cm2,
                "centroid": [cx, cy],
                "type": change_type,
                "mean_difference": round(mean_diff, 1),
                "alignment_confidence": round(alignment_confidence, 3),
                "description": description,
            })
            event_id += 1

        # ── Summary ──
        total_area_cm2 = sum(e["area_cm2"] for e in change_events if e["area_cm2"] is not None)
        largest_cm2 = max((e["area_cm2"] for e in change_events if e["area_cm2"] is not None), default=0.0)

        change_summary = {
            "total_events": len(change_events),
            "total_changed_area_cm2": round(total_area_cm2, 3),
            "largest_change_cm2": round(largest_cm2, 3),
            "types": {
                "darkened":   sum(1 for e in change_events if e["type"] == "darkened"),
                "brightened": sum(1 for e in change_events if e["type"] == "brightened"),
            },
            "alignment_uncertain": alignment_uncertain,
            "alignment_confidence": round(alignment_confidence, 3),
        }

        logger.info(
            f"ChangeDetector: cell {grid_cell} pass {pass_before}→{pass_after}: "
            f"{len(change_events)} events, "
            f"total_area={total_area_cm2:.2f} cm², "
            f"alignment_conf={alignment_confidence:.2f}"
        )

        # ── Save output image ──
        new_bgr = cv2.imread(new_image_path)
        map_path = _save_change_map(
            new_bgr, prev_gray, aligned_new, contours, change_events,
            grid_cell, pass_before, pass_after, alignment_uncertain, new_image_path
        )

        return {
            "change_map_path": map_path,
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
    Align to_align onto reference using a small grid tape corner patch as anchor.
    Returns (aligned_image, confidence).

    Uses translation-only correction — no rotation or scale change, since camera
    height and angle are consistent between passes.
    """
    x, y, tw, th = _TEMPLATE_CROP
    template = reference[y:y + th, x:x + tw]

    if template.size == 0:
        return to_align, 0.0

    result = cv2.matchTemplate(to_align, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Offset: how far the template landed from its expected position
    dx = max_loc[0] - x
    dy = max_loc[1] - y

    if dx == 0 and dy == 0:
        return to_align, float(max_val)

    # Apply translation via warp affine
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    h, w = to_align.shape
    aligned = cv2.warpAffine(to_align, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    return aligned, float(max_val)


def _describe(change_type: str, area_cm2: float | None) -> str:
    area_str = f"{area_cm2:.1f} cm²" if area_cm2 is not None else "unknown area"
    if change_type == "darkened":
        return f"New dark region ({area_str}) — possible new crater, boulder, or shadow shift"
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

        # Header banner
        banner_h = 30
        banner = np.full((banner_h, canvas_w, 3), (0, 100, 200), dtype=np.uint8)
        label = f"Cell {grid_cell}  pass {pass_before} | pass {pass_after}  [ALIGNMENT UNCERTAIN]"
        cv2.putText(banner, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        output = np.vstack([banner, canvas])

    else:
        # Diff overlay: new image with red outlines + labels on changed regions
        if new_bgr is None:
            new_bgr = cv2.cvtColor(aligned_new_gray, cv2.COLOR_GRAY2BGR)
        output = new_bgr.copy()

        for event in change_events:
            # Draw red contour outline (use bounding rect for labelling position)
            cnt_idx = event["id"] - 1
            if cnt_idx < len(contours):
                cv2.drawContours(output, contours, cnt_idx, (0, 0, 220), 2)

            cx, cy = event["centroid"]
            area_str = f"{event['area_cm2']:.1f}cm²" if event["area_cm2"] else f"{event['area_px']}px"
            label = f"#{event['id']} {event['type']} {area_str}"
            cv2.putText(output, label, (max(0, cx - 40), max(12, cy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1, cv2.LINE_AA)

        # Header banner
        banner_h = 28
        banner = np.full((banner_h, output.shape[1], 3), (30, 30, 30), dtype=np.uint8)
        n_events = len(change_events)
        label = f"Cell {grid_cell}  pass {pass_before}→{pass_after}  {n_events} change(s) detected"
        cv2.putText(banner, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        output = np.vstack([banner, output])

    cv2.imwrite(out_path, output)
    logger.debug(f"Change map saved: {out_path}")
    return out_path
