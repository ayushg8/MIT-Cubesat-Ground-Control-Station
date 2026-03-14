# receiver/quality_check.py — Ground-side image quality verification
#
# These checks are DIFFERENT from the CubeSat's quality checks.
#
#   CubeSat checks (already done before downlink):
#       - Blur: Laplacian variance < threshold → retake
#       - Exposure: mean brightness out of range → retake
#       - Motion blur: IMU angular rate too high during capture → retake
#
#   Ground checks (done here, on received images):
#       - Texture sufficiency: does the image have enough local variation for
#         the hazard classifier to work? A sharp image of flat sand passes the
#         CubeSat blur check but has no texture features.
#       - Contrast range: does the histogram span enough grey levels for shadow
#         detection? Flat lighting kills contrast even on a sharp image.
#       - Color validity: is more than 90% of the image one uniform tone?
#         Catches: camera at table edge, operator hand in frame, lens obstructed.
#
# Flagged images are still processed — they already transferred and may contain
# partial useful data. The flag appears in logs and mission_state.json.

import logging

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Patch size for local variance (texture) check
_PATCH_SIZE = 8

# Narrow band width for color-validity check (pixel values within ±BAND of mode)
_COLOR_BAND_HALF_WIDTH = 15


def run_ground_quality_check(image_path: str) -> dict:
    """
    Run all three ground-side quality checks on a saved image file.

    Returns a dict:
        {
            "passed": bool,          # True only if ALL checks pass
            "score": float,          # 0.0–1.0 (fraction of checks passed)
            "notes": [str],          # Human-readable flag reasons
            "texture_variance": float,
            "contrast_range": int,
            "single_color_pct": float,
        }
    """
    result = {
        "passed": True,
        "score": 1.0,
        "notes": [],
        "texture_variance": 0.0,
        "contrast_range": 0,
        "single_color_pct": 0.0,
    }

    img = cv2.imread(image_path)
    if img is None:
        result["passed"] = False
        result["score"] = 0.0
        result["notes"].append("could_not_read_image")
        logger.error(f"Could not read image for quality check: {image_path}")
        return result

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    checks_passed = 0
    total_checks = 3

    # --- Check 1: Texture sufficiency ---
    texture_variance = _compute_patch_variance(gray, _PATCH_SIZE)
    result["texture_variance"] = float(texture_variance)
    if texture_variance >= config.GROUND_MIN_TEXTURE_VARIANCE:
        checks_passed += 1
    else:
        result["passed"] = False
        result["notes"].append(
            f"low_texture (avg patch variance {texture_variance:.1f} < {config.GROUND_MIN_TEXTURE_VARIANCE})"
        )
        logger.warning(
            f"Quality flag — low_texture: avg patch variance {texture_variance:.1f} "
            f"(threshold {config.GROUND_MIN_TEXTURE_VARIANCE}) in {image_path}"
        )

    # --- Check 2: Contrast range ---
    contrast_range = _compute_contrast_range(gray)
    result["contrast_range"] = int(contrast_range)
    if contrast_range >= config.GROUND_MIN_CONTRAST_RANGE:
        checks_passed += 1
    else:
        result["passed"] = False
        result["notes"].append(
            f"low_contrast (histogram span {contrast_range} < {config.GROUND_MIN_CONTRAST_RANGE})"
        )
        logger.warning(
            f"Quality flag — low_contrast: histogram span {contrast_range} "
            f"(threshold {config.GROUND_MIN_CONTRAST_RANGE}) in {image_path}"
        )

    # --- Check 3: Color validity ---
    single_color_pct = _compute_single_color_pct(gray, _COLOR_BAND_HALF_WIDTH)
    result["single_color_pct"] = float(single_color_pct)
    if single_color_pct <= config.GROUND_MAX_SINGLE_COLOR_PCT:
        checks_passed += 1
    else:
        result["passed"] = False
        result["notes"].append(
            f"color_invalid ({single_color_pct:.1f}% pixels in narrow band > {config.GROUND_MAX_SINGLE_COLOR_PCT}%)"
        )
        logger.warning(
            f"Quality flag — color_invalid: {single_color_pct:.1f}% of pixels in a narrow "
            f"brightness band (threshold {config.GROUND_MAX_SINGLE_COLOR_PCT}%) in {image_path}"
        )

    result["score"] = checks_passed / total_checks
    return result


def _compute_patch_variance(gray: np.ndarray, patch_size: int) -> float:
    """
    Divide the image into non-overlapping patches of patch_size×patch_size.
    Compute the variance within each patch. Return the mean patch variance.

    Low mean variance → featureless/textureless image → hazard classifier will struggle.
    """
    h, w = gray.shape
    variances = []
    for row in range(0, h - patch_size + 1, patch_size):
        for col in range(0, w - patch_size + 1, patch_size):
            patch = gray[row:row + patch_size, col:col + patch_size]
            variances.append(float(np.var(patch)))
    if not variances:
        return 0.0
    return float(np.mean(variances))


def _compute_contrast_range(gray: np.ndarray) -> int:
    """
    Compute the range of the grayscale histogram: max_pixel_value - min_pixel_value
    where min/max are the lowest and highest grey levels that actually appear.

    Low range → flat/low-contrast image → shadow detection will struggle.
    """
    min_val = int(gray.min())
    max_val = int(gray.max())
    return max_val - min_val


def _compute_single_color_pct(gray: np.ndarray, half_width: int) -> float:
    """
    Find the mode pixel brightness. Count pixels within [mode - half_width,
    mode + half_width]. Return as a percentage of total pixels.

    High percentage → nearly the whole image is one tone → unusable for terrain analysis.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    mode = int(np.argmax(hist))
    lo = max(0, mode - half_width)
    hi = min(255, mode + half_width)
    band_count = int(hist[lo:hi + 1].sum())
    total = gray.size
    return (band_count / total) * 100.0
