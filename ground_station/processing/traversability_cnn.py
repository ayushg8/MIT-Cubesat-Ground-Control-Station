from __future__ import annotations
# processing/traversability_cnn.py — Learned traversability prediction
#
# MobileNetV2 backbone (ImageNet pretrained) with a regression head that
# outputs a per-patch traversability score 0.0 (impassable) → 1.0 (safe).
#
# The model is trained from existing pipeline classifications (bootstrapped
# labels). When CNN_ENABLED=True in config, inference runs after segmentation
# and the predictions are blended with classical costs.

import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

_model = None
_device = None


def _get_device():
    """Get torch device (lazy import to avoid torch at module level)."""
    global _device
    if _device is not None:
        return _device
    import torch
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
    return _device


def _build_model():
    """Build MobileNetV2 with traversability regression head."""
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # Replace classifier: 1280 → 1 (sigmoid for 0-1 score)
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 1),
        nn.Sigmoid(),
    )
    return backbone


def load_model() -> bool:
    """Load trained model from CNN_MODEL_PATH. Returns True if successful."""
    global _model
    if _model is not None:
        return True

    model_path = config.CNN_MODEL_PATH
    if not os.path.exists(model_path):
        logger.warning(f"TraversabilityCNN: model not found at {model_path}")
        return False

    try:
        import torch
        device = _get_device()
        _model = _build_model()
        _model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        _model.to(device)
        _model.eval()
        logger.info(f"TraversabilityCNN: loaded model from {model_path} on {device}")
        return True
    except Exception as e:
        logger.error(f"TraversabilityCNN: failed to load model: {e}")
        _model = None
        return False


def infer_grid(mosaic_crop: np.ndarray) -> np.ndarray:
    """
    Run traversability inference on a mosaic crop.

    Args:
        mosaic_crop: BGR image (h, w, 3)

    Returns:
        np.ndarray (grid_h, grid_w) float32 traversability scores 0.0-1.0
    """
    import torch
    from torchvision import transforms

    if _model is None:
        if not load_model():
            return None

    device = _get_device()
    patch_size = config.CNN_PATCH_SIZE
    batch_size = config.CNN_BATCH_SIZE
    h, w = mosaic_crop.shape[:2]

    # Calculate grid dimensions
    grid_h = max(1, h // patch_size)
    grid_w = max(1, w // patch_size)

    # Extract patches
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    patches = []
    for r in range(grid_h):
        for c in range(grid_w):
            y0 = r * patch_size
            x0 = c * patch_size
            y1 = min(h, y0 + patch_size)
            x1 = min(w, x0 + patch_size)
            patch = mosaic_crop[y0:y1, x0:x1]
            if patch.size == 0:
                patches.append(None)
                continue
            # Convert BGR to RGB
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patches.append(transform(patch_rgb))

    # Batch inference
    trav_grid = np.full((grid_h, grid_w), 0.5, dtype=np.float32)

    valid_patches = [(i, p) for i, p in enumerate(patches) if p is not None]
    if not valid_patches:
        return trav_grid

    with torch.no_grad():
        for batch_start in range(0, len(valid_patches), batch_size):
            batch_items = valid_patches[batch_start:batch_start + batch_size]
            batch_tensor = torch.stack([p for _, p in batch_items]).to(device)
            scores = _model(batch_tensor).squeeze(-1).cpu().numpy()

            for j, (idx, _) in enumerate(batch_items):
                r = idx // grid_w
                c = idx % grid_w
                trav_grid[r, c] = float(scores[j])

    return trav_grid


def generate_training_data(mosaic: np.ndarray, fine_grid: np.ndarray,
                           cost_map: dict) -> tuple:
    """
    Generate training data from existing classifications.

    Args:
        mosaic: BGR mosaic image
        fine_grid: uint8 label grid from PixelSegmenter
        cost_map: label → cost mapping

    Returns:
        (patches, labels): list of BGR patches and float traversability labels
    """
    patch_size = config.CNN_PATCH_SIZE
    h, w = mosaic.shape[:2]

    # Label mapping: cost → traversability score
    cost_to_trav = {
        config.COST_SAFE: 1.0,
        config.COST_MODERATE: 0.6,
        config.COST_SHADOW: 0.3,
        config.COST_HAZARD: 0.1,
        config.COST_IMPASSABLE: 0.0,
    }

    patches = []
    labels = []

    grid_h = fine_grid.shape[0]
    grid_w = fine_grid.shape[1]
    fine_px = config.SEG_GRID_CELL_PX

    for r in range(0, h - patch_size + 1, patch_size // 2):
        for c in range(0, w - patch_size + 1, patch_size // 2):
            patch = mosaic[r:r + patch_size, c:c + patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            # Get dominant label from fine grid
            fr0 = max(0, r // fine_px)
            fc0 = max(0, c // fine_px)
            fr1 = min(grid_h, (r + patch_size) // fine_px)
            fc1 = min(grid_w, (c + patch_size) // fine_px)

            if fr1 <= fr0 or fc1 <= fc0:
                continue

            region = fine_grid[fr0:fr1, fc0:fc1]
            if region.size == 0:
                continue

            unique, counts = np.unique(region, return_counts=True)
            dominant_label = int(unique[np.argmax(counts)])
            cost = cost_map.get(dominant_label, config.COST_SAFE)
            trav = cost_to_trav.get(cost, 0.5)

            patches.append(patch)
            labels.append(trav)

    logger.info(f"TraversabilityCNN: generated {len(patches)} training patches")
    return patches, labels


def train(patches: list, labels: list, epochs: int = 10,
          save_path: str | None = None):
    """Train the model on patches and labels."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import transforms

    global _model
    device = _get_device()

    if not patches:
        logger.warning("TraversabilityCNN: no training data")
        return

    # Prepare data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensors = []
    for patch in patches:
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        tensors.append(transform(patch_rgb))

    X = torch.stack(tensors)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=config.CNN_BATCH_SIZE, shuffle=True)

    model = _build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"TraversabilityCNN: epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}")

    model.eval()
    _model = model

    if save_path is None:
        save_path = config.CNN_MODEL_PATH

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"TraversabilityCNN: model saved to {save_path}")


if __name__ == "__main__":
    """Training script: load existing mosaic data and train traversability model."""
    import sys
    logging.basicConfig(level=logging.INFO)

    mosaic_path = os.path.join(config.PROCESSED_DIR, "mosaics", "mosaic_latest.png")
    if not os.path.exists(mosaic_path):
        print(f"No mosaic found at {mosaic_path}")
        sys.exit(1)

    mosaic = cv2.imread(mosaic_path)
    if mosaic is None:
        print(f"Failed to read mosaic: {mosaic_path}")
        sys.exit(1)

    # Try to load fine grid from pipeline data
    from processing.mosaic_grid import MosaicGrid
    grid = MosaicGrid()
    h, w = mosaic.shape[:2]
    grid.update_from_mosaic(w, h)

    # Use default labels (all safe) if no segmentation data
    fine_grid = grid.get_fine_hazard_grid()
    patches, labels = generate_training_data(mosaic, fine_grid, config.SEG_COST_MAP)

    if not patches:
        print("No training patches generated")
        sys.exit(1)

    print(f"Training on {len(patches)} patches...")
    train(patches, labels, epochs=10, save_path=config.CNN_MODEL_PATH)
    print("Done!")
