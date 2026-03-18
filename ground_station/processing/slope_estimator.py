from __future__ import annotations
# processing/slope_estimator.py — Shadow-based terrain slope estimation
#
# Uses shadow geometry to estimate terrain slope. Given a shadow mask and
# sun position (elevation + azimuth), computes slope at shadow boundaries.
#
# Algorithm per shadow region:
#   1. Shadow length along sun direction → object_height = shadow_length * tan(sun_elevation)
#   2. Slope at boundary = arctan(object_height / shadow_width_perpendicular)
#   3. Map shadow boundaries to grid cells with slope values

import logging
import math

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class SlopeEstimator:
    """Estimate terrain slope from shadow geometry and sun position."""

    def __init__(self):
        self._sun_elevation_rad = math.radians(config.SUN_ELEVATION_DEG)
        self._sun_azimuth_rad = math.radians(config.SUN_AZIMUTH_DEG)
        # Sun direction vector (unit, in image pixel space: +x=right, +y=down)
        az = self._sun_azimuth_rad
        self._sun_dir = np.array([math.sin(az), math.cos(az)], dtype=np.float64)

    def estimate(self, shadow_mask: np.ndarray) -> dict:
        """
        Estimate slopes from a binary shadow mask.

        Args:
            shadow_mask: uint8 array (255=shadow, 0=lit)

        Returns:
            dict with:
                slope_map: np.ndarray (float32) of slope in degrees, same shape as shadow_mask
                regions: list of {slope_deg, direction, centroid}
        """
        if shadow_mask is None:
            return {"slope_map": None, "regions": []}

        h, w = shadow_mask.shape[:2]
        slope_map = np.zeros((h, w), dtype=np.float32)
        regions = []

        # Find shadow contours
        binary = (shadow_mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # skip tiny shadows
                continue

            # Fit rotated rectangle to get shadow extent
            rect = cv2.minAreaRect(contour)
            center = rect[0]
            size = rect[1]  # (width, height)
            angle_deg = rect[2]

            if size[0] == 0 or size[1] == 0:
                continue

            # Shadow length = extent along sun direction
            # Shadow width = extent perpendicular to sun direction
            rect_angle_rad = math.radians(angle_deg)
            rect_dir = np.array([math.cos(rect_angle_rad), math.sin(rect_angle_rad)])

            # Project rectangle dims onto sun direction
            dot = abs(np.dot(rect_dir, self._sun_dir))
            cross = math.sqrt(1.0 - dot * dot)

            shadow_length = size[0] * dot + size[1] * cross
            shadow_width = size[0] * cross + size[1] * dot

            if shadow_length < 1:
                continue

            # Object height from shadow length and sun elevation
            object_height = shadow_length * math.tan(self._sun_elevation_rad)

            # Slope at shadow boundary
            if shadow_width > 0:
                slope_rad = math.atan2(object_height, shadow_width)
            else:
                slope_rad = math.pi / 2

            slope_deg = math.degrees(slope_rad)
            slope_deg = min(90.0, max(0.0, slope_deg))

            # Paint slope values along shadow boundary (dilated contour edge)
            boundary_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(boundary_mask, [contour], 0, 255, thickness=3)

            # Extend slope into adjacent cells (the area around the shadow)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            extended = cv2.dilate(boundary_mask, kernel, iterations=1)

            slope_map[extended > 0] = np.maximum(
                slope_map[extended > 0], slope_deg
            )

            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = center

            regions.append({
                "slope_deg": round(slope_deg, 1),
                "direction": round(math.degrees(self._sun_azimuth_rad), 1),
                "centroid": [round(cx, 1), round(cy, 1)],
                "shadow_length_px": round(shadow_length, 1),
                "object_height_px": round(object_height, 1),
            })

        logger.info(
            f"SlopeEstimator: {len(regions)} slope region(s), "
            f"max slope={max((r['slope_deg'] for r in regions), default=0):.1f}°"
        )

        return {
            "slope_map": slope_map,
            "regions": regions,
        }
