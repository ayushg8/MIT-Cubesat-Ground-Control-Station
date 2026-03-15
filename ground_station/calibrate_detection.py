#!/usr/bin/env python3
"""Calibration tool — tune shadow + hazard detection against your real surface.

Usage:
    cd ground_station
    python calibrate_detection.py --images test_photos/

Loads every image in the directory, runs shadow detection and hazard
classification on each one, and asks you whether the result looks correct.
At the end it prints accuracy stats and suggests threshold adjustments.

Goal: iterate until both accuracies are > 85%.

If cv2.imshow is unavailable (headless / SSH), comparison images are saved
to a temp directory and paths are printed so you can open them manually.
"""

import argparse
import glob
import logging
import os
import sys
import tempfile

import cv2
import numpy as np

# Add ground_station to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from processing.shadow_detector import ShadowDetector
from processing.hazard_classifier import HazardClassifier

logging.basicConfig(level=logging.WARNING)

# ─── Display helpers ──────────────────────────────────────────────────────────

def _can_show():
    """Check if we can use cv2.imshow (fails on headless / SSH)."""
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except Exception:
        return False


_USE_IMSHOW = None
_TEMP_DIR = None


def _show(title: str, img: np.ndarray):
    """Display an image — via imshow if available, otherwise save to temp."""
    global _USE_IMSHOW, _TEMP_DIR

    if _USE_IMSHOW is None:
        _USE_IMSHOW = _can_show()
        if not _USE_IMSHOW:
            _TEMP_DIR = tempfile.mkdtemp(prefix="gcs_calibrate_")
            print(f"  (headless mode — saving images to {_TEMP_DIR})\n")

    if _USE_IMSHOW:
        cv2.imshow(title, img)
        cv2.waitKey(1)
    else:
        safe_title = title.replace(" ", "_").replace("/", "_")
        path = os.path.join(_TEMP_DIR, f"{safe_title}.png")
        cv2.imwrite(path, img)
        print(f"  → saved: {path}")


def _close_all():
    if _USE_IMSHOW:
        cv2.destroyAllWindows()


# ─── Prompt helper ────────────────────────────────────────────────────────────

def _ask(prompt: str) -> str:
    """Ask y/n/q. Returns 'y', 'n', or 'q'."""
    while True:
        try:
            ans = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "q"
        if ans in ("y", "n", "q"):
            return ans
        print("  Please enter y, n, or q.")


# ─── Main calibration loop ───────────────────────────────────────────────────

def run(image_dir: str):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(image_dir, pat)))
        files.extend(glob.glob(os.path.join(image_dir, pat.upper())))
    files = sorted(set(files))

    if not files:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    print(f"\nFound {len(files)} image(s) in {image_dir}")
    print("For each image you'll be asked if shadow detection and hazard")
    print("classification look correct.  y = correct, n = wrong, q = quit\n")
    print("Current thresholds:")
    print(f"  Shadow:  blockSize={51}  C={10}  gradient_threshold={30}")
    print(f"  Hazard:  LBP_VARIANCE_MODERATE={config.LBP_VARIANCE_MODERATE}"
          f"  LBP_VARIANCE_HIGH={config.LBP_VARIANCE_HIGH}")
    print(f"           EDGE_DENSITY_MODERATE={config.EDGE_DENSITY_MODERATE}"
          f"  EDGE_DENSITY_HIGH={config.EDGE_DENSITY_HIGH}")
    print(f"           CHANGE_THRESHOLD={config.CHANGE_THRESHOLD}")
    print("-" * 70)

    sd = ShadowDetector()
    hc = HazardClassifier()

    shadow_correct = 0
    shadow_total = 0
    shadow_issues = []  # track what went wrong

    hazard_correct = 0
    hazard_total = 0
    hazard_issues = []

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        print(f"\n[{i + 1}/{len(files)}] {fname}")
        print("=" * 50)

        img = cv2.imread(fpath)
        if img is None:
            print(f"  Cannot read — skipping")
            continue

        # ── Shadow detection ──────────────────────────────────────────
        print("\n  --- Shadow Detection ---")
        sr = sd.run(fpath)

        if sr is None:
            print("  Shadow detector returned None — skipping")
        else:
            shadow_regions = sr["shadow_regions"]
            shadow_pct = sr["shadow_percentage"]
            shadow_mask = sr["shadow_mask"]
            n_shadow = len([r for r in shadow_regions if r["type"] == "shadow"])
            n_object = len([r for r in shadow_regions if r["type"] == "object"])

            print(f"  Shadow regions: {n_shadow}  |  Dark objects: {n_object}")
            print(f"  Shadow percentage: {shadow_pct:.1f}%")

            for r in shadow_regions[:5]:
                print(f"    [{r['type']:7s}] area={r['area_px']:5d}px  "
                      f"gradient={r['mean_boundary_gradient']:.1f}")

            # Build side-by-side comparison
            mask_bgr = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)
            # Tint shadow pixels blue on a copy of the original
            overlay = img.copy()
            overlay[shadow_mask > 0] = (
                overlay[shadow_mask > 0].astype(np.float32) * 0.4 +
                np.array([180, 60, 0], dtype=np.float32) * 0.6
            ).astype(np.uint8)
            comparison = np.hstack([img, overlay, mask_bgr])

            # Add labels
            h = comparison.shape[0]
            cv2.putText(comparison, "Original", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison, "Shadow Overlay", (img.shape[1] + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison, "Mask", (img.shape[1] * 2 + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            _show(f"shadow_{i+1}_{fname}", comparison)

            ans = _ask("  Does shadow detection look correct? (y/n/q): ")
            if ans == "q":
                break
            shadow_total += 1
            if ans == "y":
                shadow_correct += 1
            else:
                issue = input("  What's wrong? (too_many / too_few / wrong_area / other): ").strip().lower()
                shadow_issues.append(issue or "unspecified")

        # ── Hazard classification ─────────────────────────────────────
        print("\n  --- Hazard Classification ---")
        shadow_mask_for_hc = sr["shadow_mask"] if sr else np.zeros(img.shape[:2], dtype=np.uint8)
        shadow_pct_for_hc = sr["shadow_percentage"] if sr else 0.0

        hr = hc.classify(fpath, shadow_mask_for_hc, shadow_pct_for_hc, (0, 0))

        hazard_class = hr["hazard_class"]
        confidence = hr["confidence"]
        details = hr["details"]

        print(f"  Classification: {hazard_class} (confidence {confidence:.2f})")
        print(f"    LBP variance:  {details.get('lbp_variance', '--')}")
        print(f"    Edge density:  {details.get('edge_density', '--')}")
        print(f"    Brightness:    mean={details.get('mean_brightness', '--')} "
              f"std={details.get('std_brightness', '--')}")
        print(f"    Shadow:        {details.get('shadow_pct', '--')}%")
        print(f"    Contours:      {details.get('significant_contour_count', '--')} "
              f"(coverage {details.get('contour_coverage_pct', '--')}%)")

        # Show hazard map if saved
        hmap_path = hr.get("hazard_map_path")
        if hmap_path and os.path.exists(hmap_path):
            hmap = cv2.imread(hmap_path)
            if hmap is not None:
                _show(f"hazard_{i+1}_{fname}", hmap)
        else:
            # Build a quick overlay
            label_img = img.copy()
            color = {
                "SAFE": (0, 200, 0), "MODERATE": (0, 200, 200),
                "SHADOW": (180, 60, 0), "HAZARD": (0, 0, 220),
                "IMPASSABLE": (0, 0, 120),
            }.get(hazard_class, (128, 128, 128))
            cv2.putText(label_img, f"{hazard_class} ({confidence:.0%})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            _show(f"hazard_{i+1}_{fname}", label_img)

        ans = _ask("  Is this classification correct? (y/n/q): ")
        if ans == "q":
            break
        hazard_total += 1
        if ans == "y":
            hazard_correct += 1
        else:
            expected = input("  What should it be? (safe/moderate/shadow/hazard/impassable): ").strip().lower()
            hazard_issues.append({
                "got": hazard_class,
                "expected": expected or "unknown",
                "lbp": details.get("lbp_variance", 0),
                "edge": details.get("edge_density", 0),
                "shadow": details.get("shadow_pct", 0),
            })

    _close_all()

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    if shadow_total > 0:
        s_pct = shadow_correct / shadow_total * 100
        print(f"\n  Shadow detection:      {shadow_correct}/{shadow_total} correct ({s_pct:.0f}%)")
        if s_pct >= 85:
            print("    ✓ Looks good!")
        else:
            print("    ✗ Accuracy below 85% — consider adjusting:")
            _suggest_shadow_fixes(shadow_issues)
    else:
        print("\n  Shadow detection: no images evaluated")

    if hazard_total > 0:
        h_pct = hazard_correct / hazard_total * 100
        print(f"\n  Hazard classification: {hazard_correct}/{hazard_total} correct ({h_pct:.0f}%)")
        if h_pct >= 85:
            print("    ✓ Looks good!")
        else:
            print("    ✗ Accuracy below 85% — consider adjusting:")
            _suggest_hazard_fixes(hazard_issues)
    else:
        print("\n  Hazard classification: no images evaluated")

    print()


# ─── Suggestion logic ─────────────────────────────────────────────────────────

def _suggest_shadow_fixes(issues: list):
    too_many = issues.count("too_many")
    too_few = issues.count("too_few")
    wrong_area = issues.count("wrong_area")

    if too_many > too_few:
        print(f"    → Too many shadow detections ({too_many}x).")
        print(f"      Try raising C parameter (current: 10) → try 15 or 20")
        print(f"      Or raise blockSize (current: 51) → try 71 or 91")
        print(f"      Or raise gradient threshold (current: 30) → try 40")
        print(f"      Edit: processing/shadow_detector.py constants at top")
    elif too_few > too_many:
        print(f"    → Too few shadow detections ({too_few}x).")
        print(f"      Try lowering C parameter (current: 10) → try 5 or 3")
        print(f"      Or lower blockSize (current: 51) → try 31")
        print(f"      Or lower _MIN_REGION_AREA_PX (current: 100) → try 50")
        print(f"      Edit: processing/shadow_detector.py constants at top")
    if wrong_area > 0:
        print(f"    → Wrong areas detected ({wrong_area}x).")
        print(f"      Morphological kernels may need tuning:")
        print(f"      _OPEN_KERNEL_SIZE (current: 5) — raise to remove more noise")
        print(f"      _CLOSE_KERNEL_SIZE (current: 9) — lower to reduce gap-filling")


def _suggest_hazard_fixes(issues: list):
    if not issues:
        return

    # Count patterns
    safe_as_hazard = sum(1 for i in issues if isinstance(i, dict)
                         and i["got"] in ("HAZARD", "IMPASSABLE", "MODERATE")
                         and i["expected"] == "safe")
    hazard_as_safe = sum(1 for i in issues if isinstance(i, dict)
                         and i["got"] == "SAFE"
                         and i["expected"] in ("hazard", "moderate", "impassable"))
    shadow_wrong = sum(1 for i in issues if isinstance(i, dict)
                       and (i["got"] == "SHADOW" or i["expected"] == "shadow"))

    if safe_as_hazard > 0:
        print(f"    → False hazards: safe terrain classified as hazardous ({safe_as_hazard}x).")
        print(f"      Too sensitive — raise thresholds:")
        print(f"      LBP_VARIANCE_MODERATE (current: {config.LBP_VARIANCE_MODERATE}) → try {config.LBP_VARIANCE_MODERATE + 100}")
        print(f"      EDGE_DENSITY_MODERATE (current: {config.EDGE_DENSITY_MODERATE}) → try {config.EDGE_DENSITY_MODERATE + 0.02}")
        print(f"      Edit: config.py")

        # Show feature values from misclassified images
        lbp_vals = [i["lbp"] for i in issues if isinstance(i, dict) and i["expected"] == "safe"]
        edge_vals = [i["edge"] for i in issues if isinstance(i, dict) and i["expected"] == "safe"]
        if lbp_vals:
            print(f"      Misclassified LBP values: {[round(v, 1) for v in lbp_vals]}")
            print(f"      → Set LBP_VARIANCE_MODERATE above {max(lbp_vals):.0f}")
        if edge_vals:
            print(f"      Misclassified edge density values: {[round(v, 4) for v in edge_vals]}")
            print(f"      → Set EDGE_DENSITY_MODERATE above {max(edge_vals):.4f}")

    if hazard_as_safe > 0:
        print(f"    → Missed hazards: hazardous terrain classified as safe ({hazard_as_safe}x).")
        print(f"      Not sensitive enough — lower thresholds:")
        print(f"      LBP_VARIANCE_MODERATE (current: {config.LBP_VARIANCE_MODERATE}) → try {max(50, config.LBP_VARIANCE_MODERATE - 100)}")
        print(f"      EDGE_DENSITY_MODERATE (current: {config.EDGE_DENSITY_MODERATE}) → try {max(0.02, config.EDGE_DENSITY_MODERATE - 0.02)}")
        print(f"      Edit: config.py")

        lbp_vals = [i["lbp"] for i in issues if isinstance(i, dict) and i["expected"] in ("hazard", "moderate")]
        edge_vals = [i["edge"] for i in issues if isinstance(i, dict) and i["expected"] in ("hazard", "moderate")]
        if lbp_vals:
            print(f"      Misclassified LBP values: {[round(v, 1) for v in lbp_vals]}")
            print(f"      → Set LBP_VARIANCE_MODERATE below {min(lbp_vals):.0f}")
        if edge_vals:
            print(f"      Misclassified edge density values: {[round(v, 4) for v in edge_vals]}")
            print(f"      → Set EDGE_DENSITY_MODERATE below {min(edge_vals):.4f}")

    if shadow_wrong > 0:
        print(f"    → Shadow classification issues ({shadow_wrong}x).")
        print(f"      Shadow threshold in hazard_classifier is 40%.")
        shadow_vals = [i["shadow"] for i in issues if isinstance(i, dict)
                       and (i["got"] == "SHADOW" or i["expected"] == "shadow")]
        if shadow_vals:
            print(f"      Misclassified shadow percentages: {[round(v, 1) for v in shadow_vals]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate shadow + hazard detection against real surface photos"
    )
    parser.add_argument(
        "--images", required=True,
        help="Directory containing test photos (JPG/PNG)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.images):
        print(f"Error: {args.images} is not a directory")
        sys.exit(1)

    run(args.images)
