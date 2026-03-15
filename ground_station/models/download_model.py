#!/usr/bin/env python3
"""Download or locate the best available YOLO model for lunar surface detection.

Dual-detection architecture:
  TIER 1 (preferred): Lunar-trained YOLOv8 weights from Roboflow (5,600+ real
         LROC images of craters, boulders, and plain surface).
         Download from Roboflow Universe and save as models/lunar_detector.pt
  TIER 2 (fallback):  YOLOv8n pre-trained on COCO. Common round/concave objects
         are mapped to lunar categories. Lower accuracy but works out of the box.

Usage:
    python models/download_model.py          # downloads YOLOv8n fallback
    python models/download_model.py --check  # prints which model is available
"""

import argparse
import os
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
LUNAR_MODEL_PATH = os.path.join(MODEL_DIR, "lunar_detector.pt")
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")


def get_model():
    """Return the best available YOLO model instance."""
    from ultralytics import YOLO

    if os.path.exists(LUNAR_MODEL_PATH):
        print("Using lunar-trained YOLO model")
        return YOLO(LUNAR_MODEL_PATH)

    print("Lunar model not found. Using COCO-pretrained YOLOv8n as fallback.")
    print("To use the lunar model, download weights from Roboflow and save to:")
    print(f"  {LUNAR_MODEL_PATH}")
    model = YOLO("yolov8n.pt")
    return model


def is_lunar_model_available() -> bool:
    return os.path.exists(LUNAR_MODEL_PATH)


def download_fallback():
    """Download YOLOv8n (COCO) as a fallback model."""
    from ultralytics import YOLO

    print("Downloading YOLOv8n (COCO pre-trained) as fallback...")
    model = YOLO("yolov8n.pt")
    print(f"Model ready: {model.model_name}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YOLO model for lunar detection")
    parser.add_argument("--check", action="store_true", help="Check which model is available")
    args = parser.parse_args()

    if args.check:
        if os.path.exists(LUNAR_MODEL_PATH):
            print(f"Lunar model: FOUND ({LUNAR_MODEL_PATH})")
        else:
            print(f"Lunar model: NOT FOUND (expected at {LUNAR_MODEL_PATH})")
        if os.path.exists(COCO_MODEL_PATH):
            print(f"COCO fallback: FOUND ({COCO_MODEL_PATH})")
        else:
            print("COCO fallback: NOT DOWNLOADED")
        sys.exit(0)

    download_fallback()
