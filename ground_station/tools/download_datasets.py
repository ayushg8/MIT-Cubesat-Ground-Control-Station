#!/usr/bin/env python3
"""Download terrain datasets from Roboflow for YOLO training."""

import os
from roboflow import Roboflow

OUT_DIR = "models/terrain_dataset"
os.makedirs(OUT_DIR, exist_ok=True)

# Dataset 1: Crater & Boulder (5578 images, ground-level + orbital mix)
# Has: crater, boulder classes — closest to our demo
print("Downloading Crater & Boulder dataset (5578 images)...")
rf = Roboflow(api_key="YOUR_API_KEY")  # Will use public download
try:
    project = rf.workspace("space-visionaries").project("crater-boulder-iyqxa")
    dataset = project.version(1).download("yolov8", location=os.path.join(OUT_DIR, "crater_boulder"))
    print(f"  Done: {dataset.location}")
except Exception as e:
    print(f"  Skipped (need API key): {e}")
    print("  Trying public URL download instead...")

# Dataset 2: Terrain Recognition (1767 images, 16 terrain classes)
# Has: Rocky Terrain, Sandy Terrain, etc.
print("\nDownloading Terrain Recognition dataset (1767 images)...")
try:
    project = rf.workspace("yolo-v-models").project("terrain-recognition-model")
    dataset = project.version(1).download("yolov8", location=os.path.join(OUT_DIR, "terrain"))
    print(f"  Done: {dataset.location}")
except Exception as e:
    print(f"  Skipped: {e}")

# Dataset 3: Rocks detection (416 images)
print("\nDownloading Rocks dataset (416 images)...")
try:
    project = rf.workspace("chris-conrad-le7vi").project("rocks-s0gnn")
    dataset = project.version(2).download("yolov8", location=os.path.join(OUT_DIR, "rocks"))
    print(f"  Done: {dataset.location}")
except Exception as e:
    print(f"  Skipped: {e}")

print("\nDone! Check", OUT_DIR)
