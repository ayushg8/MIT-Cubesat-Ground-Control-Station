#!/usr/bin/env python3
"""Train a YOLO model for sandbox/desert terrain detection.

Combines:
1. Existing Roboflow crater dataset (1493 orbital images)
2. Real Pi camera images with pseudo-labels from current detectors
3. Augmented variants for robustness

Target classes:
  0: crater
  1: boulder
  2: sand (safe terrain)
  3: shadow

Usage:
    python3 tools/train_terrain_yolo.py
    python3 tools/train_terrain_yolo.py --epochs 50
"""

import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np


def create_dataset_structure(base_dir):
    """Create YOLO dataset directory structure."""
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, "labels"), exist_ok=True)


def copy_roboflow_data(base_dir):
    """Copy existing Roboflow crater data, remapping classes."""
    rf_dir = "models/roboflow_dataset"
    copied = 0

    for split, rf_split in [("train", "train"), ("valid", "valid")]:
        img_dir = os.path.join(rf_dir, rf_split, "images")
        lbl_dir = os.path.join(rf_dir, rf_split, "labels")

        if not os.path.exists(img_dir):
            continue

        for img_file in os.listdir(img_dir):
            if not img_file.endswith((".jpg", ".png")):
                continue

            # Copy image
            src_img = os.path.join(img_dir, img_file)
            dst_img = os.path.join(base_dir, split, "images", f"rf_{img_file}")
            shutil.copy2(src_img, dst_img)

            # Copy and remap label (class 0=crater stays 0, class 1=rille → 0 crater)
            lbl_file = os.path.splitext(img_file)[0] + ".txt"
            src_lbl = os.path.join(lbl_dir, lbl_file)
            dst_lbl = os.path.join(base_dir, split, "labels", f"rf_{lbl_file}")

            if os.path.exists(src_lbl):
                with open(src_lbl) as f:
                    lines = f.readlines()
                with open(dst_lbl, "w") as f:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Remap: 0 (crater) → 0, 1 (rille) → 0 (treat as crater)
                            parts[0] = "0"
                            f.write(" ".join(parts) + "\n")
            copied += 1

    print(f"  Copied {copied} Roboflow images (craters)")
    return copied


def generate_pi_labels(base_dir):
    """Generate pseudo-labels for Pi camera images using classical CV features."""
    img_dir = "data/received_images"
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    if not imgs:
        print("  No Pi camera images found")
        return 0

    # Use a subset for training
    random.shuffle(imgs)
    selected = imgs[:min(150, len(imgs))]
    labeled = 0

    for img_path in selected:
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        basename = os.path.splitext(os.path.basename(img_path))[0]

        labels = []

        # Detect dark circular regions (potential craters)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=50, param2=30, minRadius=15, maxRadius=min(h, w) // 3
        )
        if circles is not None:
            for circle in circles[0]:
                cx, cy, r = circle
                # Check if region is darker than surroundings (crater-like)
                mask = np.zeros_like(gray)
                cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
                inner_mean = gray[mask > 0].mean() if np.any(mask > 0) else 128
                # Create annular mask for surroundings
                outer_mask = np.zeros_like(gray)
                cv2.circle(outer_mask, (int(cx), int(cy)), int(r * 1.5), 255, -1)
                outer_mask[mask > 0] = 0
                outer_mean = gray[outer_mask > 0].mean() if np.any(outer_mask > 0) else 128

                if inner_mean < outer_mean - 10:  # Darker inside = crater-like
                    # YOLO format: class cx cy w h (normalized)
                    nx = cx / w
                    ny = cy / h
                    nw = (2 * r) / w
                    nh = (2 * r) / h
                    # Clamp to image bounds
                    nx = max(nw/2, min(1 - nw/2, nx))
                    ny = max(nh/2, min(1 - nh/2, ny))
                    if nw < 0.5 and nh < 0.5:  # Skip if too large
                        labels.append(f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

        # Detect bright irregular blobs (potential boulders)
        _, bright_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200 or area > (h * w * 0.3):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Check circularity — boulders are roughly round
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.3:  # Reasonably round
                nx = (bx + bw / 2) / w
                ny = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h
                if nw < 0.4 and nh < 0.4:
                    labels.append(f"1 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

        # Detect shadow regions
        _, shadow_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        shadow_contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in shadow_contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > (h * w * 0.5):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            nx = (bx + bw / 2) / w
            ny = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            if nw < 0.5 and nh < 0.5:
                labels.append(f"3 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

        # If no specific detections, label the whole image as sand (safe terrain)
        # Add a large centered sand box covering the image
        if not labels:
            labels.append(f"2 0.5 0.5 0.8 0.8")

        # Save
        split = "train" if random.random() < 0.85 else "valid"
        dst_img = os.path.join(base_dir, split, "images", f"pi_{basename}.jpg")
        dst_lbl = os.path.join(base_dir, split, "labels", f"pi_{basename}.txt")

        shutil.copy2(img_path, dst_img)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(labels) + "\n")
        labeled += 1

    print(f"  Labeled {labeled} Pi camera images")
    return labeled


def augment_images(base_dir, count=200):
    """Create augmented variants of existing training images."""
    train_dir = os.path.join(base_dir, "train", "images")
    label_dir = os.path.join(base_dir, "train", "labels")

    images = [f for f in os.listdir(train_dir) if f.endswith((".jpg", ".png"))]
    if not images:
        return 0

    augmented = 0
    selected = random.sample(images, min(count, len(images)))

    for img_file in selected:
        img_path = os.path.join(train_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        basename = os.path.splitext(img_file)[0]
        lbl_file = basename + ".txt"
        lbl_path = os.path.join(label_dir, lbl_file)

        # Random augmentation
        aug_type = random.choice(["brightness", "flip", "rotate", "noise"])

        if aug_type == "brightness":
            factor = random.uniform(0.6, 1.4)
            aug_img = np.clip(img * factor, 0, 255).astype(np.uint8)
        elif aug_type == "flip":
            aug_img = cv2.flip(img, 1)  # horizontal flip
        elif aug_type == "rotate":
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug_img = cv2.warpAffine(img, M, (w, h))
        else:  # noise
            noise = np.random.normal(0, 15, img.shape).astype(np.int16)
            aug_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        aug_name = f"aug_{aug_type}_{basename}"
        dst_img = os.path.join(train_dir, aug_name + ".jpg")
        dst_lbl = os.path.join(label_dir, aug_name + ".txt")

        cv2.imwrite(dst_img, aug_img)

        # Copy labels (flip needs bbox adjustment for horizontal flip)
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                lines = f.readlines()
            with open(dst_lbl, "w") as f:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5 and aug_type == "flip":
                        # Mirror x coordinate: new_cx = 1 - cx
                        parts[1] = f"{1.0 - float(parts[1]):.6f}"
                    f.write(" ".join(parts) + "\n")
        augmented += 1

    print(f"  Created {augmented} augmented images")
    return augmented


def write_data_yaml(base_dir):
    """Write the dataset YAML config for YOLOv8."""
    yaml_path = os.path.join(base_dir, "data.yaml")
    abs_base = os.path.abspath(base_dir)
    with open(yaml_path, "w") as f:
        f.write(f"""names:
- crater
- boulder
- sand
- shadow
nc: 4
train: {abs_base}/train/images
val: {abs_base}/valid/images
""")
    print(f"  Dataset config: {yaml_path}")
    return yaml_path


def train_yolo(data_yaml, epochs=30, imgsz=640):
    """Train YOLOv8 on the combined dataset."""
    from ultralytics import YOLO

    # Start from YOLOv8n pretrained on COCO
    model = YOLO("yolov8n.pt")

    print(f"\nTraining YOLOv8n for {epochs} epochs on {imgsz}px images...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device="cpu",
        workers=2,
        patience=10,
        save=True,
        project="models",
        name="terrain_yolo",
        exist_ok=True,
        verbose=True,
    )

    # Copy best model to the standard location
    best_path = os.path.join("models", "terrain_yolo", "weights", "best.pt")
    if os.path.exists(best_path):
        dst = "models/terrain_detector.pt"
        shutil.copy2(best_path, dst)
        size_mb = os.path.getsize(dst) / 1024 / 1024
        print(f"\nModel saved: {dst} ({size_mb:.1f} MB)")
        print("Update config.py or yolo_detector.py to use 'terrain_detector.pt'")
    else:
        print("\nWarning: best.pt not found, check training output")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--skip-train", action="store_true", help="Only prepare dataset")
    args = parser.parse_args()

    base_dir = "models/terrain_combined"
    print(f"Preparing combined dataset in {base_dir}/\n")

    # Clean and create
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    create_dataset_structure(base_dir)

    # Step 1: Copy Roboflow data
    print("Step 1: Roboflow crater dataset")
    n_rf = copy_roboflow_data(base_dir)

    # Step 2: Generate Pi camera pseudo-labels
    print("\nStep 2: Pi camera images (pseudo-labels)")
    n_pi = generate_pi_labels(base_dir)

    # Step 3: Augment
    print("\nStep 3: Data augmentation")
    n_aug = augment_images(base_dir, count=300)

    # Step 4: Write config
    print("\nStep 4: Dataset config")
    data_yaml = write_data_yaml(base_dir)

    # Count final dataset
    train_imgs = len(os.listdir(os.path.join(base_dir, "train", "images")))
    valid_imgs = len(os.listdir(os.path.join(base_dir, "valid", "images")))
    print(f"\nFinal dataset: {train_imgs} train + {valid_imgs} valid images")
    print(f"  Sources: {n_rf} Roboflow + {n_pi} Pi camera + {n_aug} augmented")

    if args.skip_train:
        print("\nSkipping training (--skip-train). Run training manually:")
        print(f"  yolo train data={data_yaml} model=yolov8n.pt epochs={args.epochs}")
        return

    # Step 5: Train
    train_yolo(data_yaml, epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
