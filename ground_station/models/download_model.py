#!/usr/bin/env python3
"""Download or locate the best available YOLO model for lunar surface detection.

Dual-detection architecture:
  TIER 1 (preferred): Crater & boulder YOLOv8 from Roboflow Universe.
         Trained on real lunar/planetary crater and boulder imagery.
         Classes: crater, boulder (mapped from dataset classes)
  TIER 2 (fallback):  YOLOv8n pre-trained on COCO. Common round/concave objects
         are mapped to lunar categories. Lower accuracy but works out of the box.

Usage:
    python models/download_model.py                          # download with stored key
    python models/download_model.py --api-key YOUR_KEY       # download with key
    python models/download_model.py --check                  # check which model is available

To get a free Roboflow API key:
    1. Go to https://app.roboflow.com/ and sign in with Google
    2. Click your profile icon → Settings → API Key
    3. Run: python models/download_model.py --api-key YOUR_KEY
"""

import argparse
import os
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
LUNAR_MODEL_PATH = os.path.join(MODEL_DIR, "lunar_detector.pt")
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")
API_KEY_FILE = os.path.join(MODEL_DIR, ".roboflow_key")

# Roboflow dataset options (crater + boulder detection)
# Try these in order — first one that works wins
ROBOFLOW_MODELS = [
    # Space Visionaries: 5,578 images, crater + boulder
    {"workspace": "sapce", "project": "crater-boulder", "version": 1},
    # ISRO: 712 images, crater + rille
    {"workspace": "isro-w900k", "project": "craters-boulders", "version": 1},
    # Projects CVR: crater + boulder
    {"workspace": "projects-cvr", "project": "crater-and-boulder", "version": 1},
    # Nandini Jaiswal: 27 images
    {"workspace": "nandini-jaiswal-rleyg", "project": "crater-and-boulder-detection", "version": 1},
]


def _get_api_key(cli_key: str = None) -> str:
    """Get Roboflow API key from CLI arg, file, or env var."""
    if cli_key:
        # Save for future use
        with open(API_KEY_FILE, "w") as f:
            f.write(cli_key.strip())
        return cli_key.strip()

    if os.environ.get("ROBOFLOW_API_KEY"):
        return os.environ["ROBOFLOW_API_KEY"]

    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE) as f:
            return f.read().strip()

    return None


def download_lunar_model(api_key: str) -> bool:
    """Download a crater/boulder YOLOv8 model from Roboflow Universe."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed. Run: pip install roboflow")
        return False

    rf = Roboflow(api_key=api_key)

    for model_info in ROBOFLOW_MODELS:
        try:
            print(f"\nTrying: {model_info['workspace']}/{model_info['project']} v{model_info['version']}...")
            project = rf.workspace(model_info["workspace"]).project(model_info["project"])
            version = project.version(model_info["version"])

            # Download in YOLOv8 format
            dataset = version.download("yolov8", location=os.path.join(MODEL_DIR, "roboflow_dataset"))

            # Check if there's a pre-trained model we can download
            model = version.model
            if model:
                print(f"Found pre-trained model for {model_info['project']}")
                # Test inference to confirm it works
                test_result = model.predict(
                    os.path.join(os.path.dirname(MODEL_DIR), "data", "received", "sample_cell2_3_pass3.jpg"),
                    confidence=20
                ).json()
                print(f"Test inference returned {len(test_result.get('predictions', []))} predictions")

            # Now train/export a YOLOv8 model on this dataset
            print(f"\nDataset downloaded to {dataset.location}")
            print("Training YOLOv8 on the dataset...")

            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")  # Start from COCO pretrained
            results = model.train(
                data=os.path.join(dataset.location, "data.yaml"),
                epochs=25,
                imgsz=640,
                batch=8,
                name="lunar_crater_boulder",
                project=os.path.join(MODEL_DIR, "training"),
                exist_ok=True,
            )

            # Copy best weights to lunar_detector.pt
            best_weights = os.path.join(MODEL_DIR, "training", "lunar_crater_boulder", "weights", "best.pt")
            if os.path.exists(best_weights):
                import shutil
                shutil.copy2(best_weights, LUNAR_MODEL_PATH)
                print(f"\nLunar model saved to: {LUNAR_MODEL_PATH}")
                return True
            else:
                print("Training completed but best.pt not found")
                return False

        except Exception as e:
            print(f"  Failed: {e}")
            continue

    print("\nAll Roboflow model sources failed.")
    return False


def get_model():
    """Return the best available YOLO model instance."""
    from ultralytics import YOLO

    if os.path.exists(LUNAR_MODEL_PATH):
        print("Using lunar-trained YOLO model (crater/boulder detection)")
        return YOLO(LUNAR_MODEL_PATH)

    print("Lunar model not found. Using COCO-pretrained YOLOv8n as fallback.")
    print("To get the lunar model, run:")
    print(f"  python models/download_model.py --api-key YOUR_ROBOFLOW_KEY")
    print(f"  Get a free key at: https://app.roboflow.com/")
    model = YOLO("yolov8n.pt")
    return model


def is_lunar_model_available() -> bool:
    return os.path.exists(LUNAR_MODEL_PATH)


def download_fallback():
    """Download YOLOv8n (COCO) as a fallback model."""
    from ultralytics import YOLO

    print("Downloading YOLOv8n (COCO pre-trained) as fallback...")
    model = YOLO("yolov8n.pt")
    print("Model ready")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YOLO model for lunar detection")
    parser.add_argument("--check", action="store_true", help="Check which model is available")
    parser.add_argument("--api-key", type=str, help="Roboflow API key (free at app.roboflow.com)")
    parser.add_argument("--fallback-only", action="store_true", help="Only download COCO fallback")
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

    if args.fallback_only:
        download_fallback()
        sys.exit(0)

    api_key = _get_api_key(args.api_key)
    if not api_key:
        print("No Roboflow API key provided.")
        print()
        print("To get a free key:")
        print("  1. Go to https://app.roboflow.com/ and sign in with Google")
        print("  2. Click your profile icon → Settings → API Key")
        print("  3. Run: python models/download_model.py --api-key YOUR_KEY")
        print()
        print("Downloading COCO fallback instead...")
        download_fallback()
        sys.exit(0)

    success = download_lunar_model(api_key)
    if not success:
        print("\nFalling back to COCO model...")
        download_fallback()
