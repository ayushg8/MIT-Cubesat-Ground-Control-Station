#!/usr/bin/env python3
from __future__ import annotations
"""
test_change_detector.py — regression checks for multi-pass change detection.

Run from ground_station/:
    python test_change_detector.py
"""

import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(_HERE)

import cv2
import numpy as np

import config
from processing.change_detector import ChangeDetector


def _textured_surface(height: int = 256, width: int = 256) -> np.ndarray:
    rng = np.random.default_rng(42)
    base = np.full((height, width, 3), 150, dtype=np.uint8)
    noise = rng.integers(-25, 26, size=(height, width, 3), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.circle(img, (80, 80), 24, (70, 70, 70), -1)
    cv2.circle(img, (170, 140), 18, (185, 185, 185), -1)
    cv2.line(img, (25, 220), (230, 190), (105, 105, 105), 2)
    return img


class ChangeDetectorRegressionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_processed = config.PROCESSED_DIR
        config.PROCESSED_DIR = self._tmp.name
        self.detector = ChangeDetector()

    def tearDown(self):
        config.PROCESSED_DIR = self._old_processed
        self._tmp.cleanup()

    def _write(self, name: str, img: np.ndarray) -> str:
        path = os.path.join(self._tmp.name, name)
        ok = cv2.imwrite(path, img)
        self.assertTrue(ok, f"failed to write {name}")
        return path

    def test_object_and_residual_events_are_reported(self):
        before = _textured_surface()
        after = before.copy()
        cv2.circle(after, (80, 80), 26, (150, 150, 150), -1)    # erase old object footprint
        cv2.circle(after, (98, 90), 24, (70, 70, 70), -1)       # moved object
        cv2.circle(after, (212, 54), 14, (25, 25, 25), -1)      # new object
        cv2.rectangle(after, (104, 180), (138, 208), (245, 245, 245), -1)  # residual bright change

        before_path = self._write("before.jpg", before)
        after_path = self._write("after.jpg", after)

        yolo_before = [
            {"class": "boulder", "confidence": 0.93, "bbox": [56, 56, 104, 104]},
        ]
        yolo_after = [
            {"class": "boulder", "confidence": 0.92, "bbox": [74, 66, 122, 114]},
            {"class": "crater", "confidence": 0.88, "bbox": [198, 40, 226, 68]},
        ]

        result = self.detector.detect(
            before_path,
            after_path,
            grid_cell=(2, 3),
            pass_before=1,
            pass_after=2,
            yolo_before=yolo_before,
            yolo_after=yolo_after,
        )

        self.assertIsNotNone(result)
        events = result["change_events"]
        types = {evt["type"] for evt in events}
        self.assertIn("moved", types)
        self.assertIn("new_object", types)

        summary = result["change_summary"]
        self.assertGreater(summary["total_events"], 0)
        self.assertGreater(summary["total_changed_area_px"], 0)
        self.assertGreater(summary["total_changed_area_cm2"], 0)
        self.assertTrue(os.path.exists(result["change_map_path"]))
        self.assertTrue(os.path.exists(os.path.join(self._tmp.name, "changes.json")))

    def test_residual_pixel_change_without_yolo_is_reported(self):
        before = _textured_surface()
        after = before.copy()
        cv2.rectangle(after, (120, 96), (174, 156), (20, 20, 20), -1)

        before_path = self._write("residual_before.jpg", before)
        after_path = self._write("residual_after.jpg", after)

        result = self.detector.detect(
            before_path,
            after_path,
            grid_cell=(4, 5),
            pass_before=1,
            pass_after=2,
            yolo_before=[],
            yolo_after=[],
        )

        self.assertIsNotNone(result)
        types = {evt["type"] for evt in result["change_events"]}
        self.assertTrue(types & {"brightened", "darkened"})
        methods = {evt.get("method") for evt in result["change_events"]}
        self.assertIn("pixel_diff", methods)

    def test_same_pass_is_skipped(self):
        img = _textured_surface()
        before_path = self._write("same_before.jpg", img)
        after_path = self._write("same_after.jpg", img)
        result = self.detector.detect(
            before_path,
            after_path,
            grid_cell=(0, 0),
            pass_before=2,
            pass_after=2,
            yolo_before=[],
            yolo_after=[],
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ChangeDetectorRegressionTest)
    runner = unittest.TextTestRunner(verbosity=2)
    raise SystemExit(0 if runner.run(suite).wasSuccessful() else 1)
