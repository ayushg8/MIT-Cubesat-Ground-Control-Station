from __future__ import annotations
# processing/cell_identifier.py — Ground-side image-based grid cell identification
#
# Determines which grid cell an image belongs to WITHOUT relying on CubeSat
# metadata. The ground station is the smart side — the CubeSat just sends images.
#
# Identification pipeline (per image):
#   1. CNN embedding (ResNet18) — fast cosine similarity screening against known cells
#   2. SIFT keypoint matching — precise confirmation against top candidates
#   3. Delaunay triangulation — structural verification of matched keypoints
#   4. Spatial stitching — if no match, find overlap with neighbors to place spatially
#   5. New cell assignment — if completely new area, assign next available cell
#
# Persistence: cell database saved to data/processed/cell_database/

import json
import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

_DB_DIR = os.path.join(config.PROCESSED_DIR, "cell_database")
_DB_INDEX_FILE = os.path.join(_DB_DIR, "cells.json")

# Matching thresholds
_SIFT_RATIO_THRESH = 0.75       # Lowe's ratio test
_REVISIT_MIN_MATCHES = 15       # Minimum good SIFT matches to confirm revisit
_CNN_REVISIT_THRESH = 0.95      # CNN similarity above this = strong revisit signal
_SPATIAL_MIN_MATCHES = 20       # Minimum matches for spatial neighbor detection
_SPATIAL_MAX_MATCHES = 80       # Above this → probably same cell, not neighbor
_SPATIAL_INLIER_RATIO = 0.5     # Minimum RANSAC inlier ratio for valid homography
_CNN_SIMILARITY_THRESH = 0.80   # Cosine similarity threshold for CNN screening
_DELAUNAY_ANGLE_TOL = 15.0      # Degrees tolerance for triangle angle comparison
_DELAUNAY_CONSISTENCY = 0.5     # Fraction of triangles that must be consistent
_SPATIAL_SHIFT_FRAC = 0.3       # Min shift as fraction of image size to count as neighbor


class CellIdentifier:
    """
    Identifies which grid cell an image belongs to using image content alone.
    Maintains a database of cell fingerprints that persists across restarts.
    """

    def __init__(self):
        self._sift = cv2.SIFT_create(nfeatures=500)
        self._bf = cv2.BFMatcher(cv2.NORM_L2)

        # CNN model (lazy-loaded)
        self._cnn_model = None
        self._cnn_transform = None
        self._cnn_loaded = False

        # Cell database: { "R,C": CellData dict }
        self._cells: dict[str, dict] = {}

        # Track next cell to assign when no spatial info available
        self._next_id = 0

        os.makedirs(_DB_DIR, exist_ok=True)
        self._load_database()

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def identify(self, image_path: str) -> dict:
        """
        Identify which grid cell an image belongs to.

        Returns:
        {
            "cell": (row, col),
            "confidence": float 0-1,
            "is_revisit": bool,
            "pass_number": int,
            "method": str,  # "cnn+sift+delaunay" | "sift" | "spatial" | "new"
        }
        """
        gray = self._load_gray(image_path)
        if gray is None:
            logger.error(f"CellIdentifier: cannot read '{image_path}'")
            return self._fallback_result()

        # Extract features
        kp, desc = self._sift.detectAndCompute(gray, None)
        if desc is None or len(kp) < 5:
            logger.warning(f"CellIdentifier: too few SIFT features in '{os.path.basename(image_path)}'")
            return self._fallback_result()

        kp_pts = np.array([k.pt for k in kp], dtype=np.float32)
        embedding = self._get_cnn_embedding(gray)

        # First image ever → cell (0, 0)
        if not self._cells:
            cell = (0, 0)
            self._register_cell(cell, desc, kp_pts, embedding, image_path)
            logger.info(f"CellIdentifier: first image → cell {cell}")
            return {
                "cell": cell, "confidence": 1.0, "is_revisit": False,
                "pass_number": 1, "method": "new",
            }

        # ── Step 1: CNN screening ──
        candidates = self._cnn_screen(embedding, top_k=5)
        top_cnn_sim = candidates[0][1] if candidates else 0.0

        # ── Step 2: SIFT matching against candidates ──
        best_cell_key = None
        best_good_matches = []
        best_score = 0
        best_cnn_sim = 0.0

        for cand_key, cand_sim in candidates:
            cand = self._cells[cand_key]
            good = self._sift_match(desc, cand["descriptors"])
            score = len(good)
            if score > best_score:
                best_score = score
                best_cell_key = cand_key
                best_good_matches = good
                best_cnn_sim = cand_sim

        # ── Step 3: Check if it's a revisit ──
        # Use a combined signal: CNN similarity + SIFT matches
        # High CNN (>0.95) lowers the SIFT threshold needed
        sift_threshold = _REVISIT_MIN_MATCHES
        if best_cnn_sim >= _CNN_REVISIT_THRESH:
            sift_threshold = max(8, _REVISIT_MIN_MATCHES // 2)

        if best_cell_key and best_score >= sift_threshold:
            # Delaunay verification
            cand = self._cells[best_cell_key]
            verified = self._delaunay_verify(kp_pts, best_good_matches, desc, cand)

            if verified:
                cell = tuple(int(x) for x in best_cell_key.split(","))
                pass_num = self._record_revisit(best_cell_key, image_path)
                conf = min(1.0, 0.6 + best_score / 100.0 + best_cnn_sim * 0.1)
                logger.info(
                    f"CellIdentifier: REVISIT cell {cell} "
                    f"(sift={best_score}, cnn={best_cnn_sim:.3f}, delaunay=OK, pass={pass_num})"
                )
                return {
                    "cell": cell, "confidence": round(conf, 3),
                    "is_revisit": True, "pass_number": pass_num,
                    "method": "cnn+sift+delaunay",
                }

            # Delaunay failed but CNN+SIFT both strong — still likely a revisit
            if best_cnn_sim >= _CNN_REVISIT_THRESH or best_score >= _REVISIT_MIN_MATCHES * 1.5:
                cell = tuple(int(x) for x in best_cell_key.split(","))
                pass_num = self._record_revisit(best_cell_key, image_path)
                conf = min(0.9, 0.5 + best_score / 150.0 + best_cnn_sim * 0.1)
                logger.info(
                    f"CellIdentifier: REVISIT cell {cell} "
                    f"(sift={best_score}, cnn={best_cnn_sim:.3f}, delaunay=FAIL)"
                )
                return {
                    "cell": cell, "confidence": round(conf, 3),
                    "is_revisit": True, "pass_number": pass_num,
                    "method": "cnn+sift",
                }

        # ── Step 4: Try spatial stitching ──
        spatial = self._try_spatial_assignment(kp, desc, gray)
        if spatial is not None:
            cell = self._clamp_cell(spatial)
            cell_key = f"{cell[0]},{cell[1]}"
            if cell_key in self._cells:
                # Already exists — it's a revisit we missed
                pass_num = self._record_revisit(cell_key, image_path)
                logger.info(f"CellIdentifier: spatial match → revisit cell {cell}")
                return {
                    "cell": cell, "confidence": 0.7, "is_revisit": True,
                    "pass_number": pass_num, "method": "spatial",
                }
            else:
                self._register_cell(cell, desc, kp_pts, embedding, image_path)
                logger.info(f"CellIdentifier: spatial → NEW cell {cell}")
                return {
                    "cell": cell, "confidence": 0.7, "is_revisit": False,
                    "pass_number": 1, "method": "spatial",
                }

        # ── Step 5: Completely new, unrelated area ──
        cell = self._next_available_cell()
        self._register_cell(cell, desc, kp_pts, embedding, image_path)
        logger.info(f"CellIdentifier: no match → NEW cell {cell}")
        return {
            "cell": cell, "confidence": 0.5, "is_revisit": False,
            "pass_number": 1, "method": "new",
        }

    def get_cell_count(self) -> int:
        """Number of unique cells identified so far."""
        return len(self._cells)

    def get_cell_map(self) -> dict:
        """Return cell database summary for dashboard."""
        summary = {}
        for key, data in self._cells.items():
            summary[key] = {
                "image_count": len(data["image_paths"]),
                "visit_count": data["visit_count"],
                "first_image": data["image_paths"][0] if data["image_paths"] else None,
            }
        return summary

    def reset(self):
        """Clear the cell database."""
        self._cells.clear()
        self._next_id = 0
        self._save_database()
        logger.info("CellIdentifier: database reset")

    # ─────────────────────────────────────────────────────────────────────
    # CNN Embedding
    # ─────────────────────────────────────────────────────────────────────

    def _load_cnn(self):
        """Lazy-load ResNet18 for feature extraction."""
        if self._cnn_loaded:
            return
        self._cnn_loaded = True

        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            # ResNet18 with classification head removed → 512-dim embedding
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            model.eval()
            self._cnn_model = model

            self._cnn_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            logger.info("CellIdentifier: ResNet18 loaded for CNN embeddings")

        except Exception as e:
            logger.warning(f"CellIdentifier: CNN unavailable ({e}) — using SIFT only")
            self._cnn_model = None

    def _get_cnn_embedding(self, gray: np.ndarray) -> np.ndarray | None:
        """Extract a 512-dim CNN embedding from a grayscale image."""
        self._load_cnn()
        if self._cnn_model is None:
            return None

        try:
            import torch

            # Convert grayscale → 3-channel for ResNet
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            tensor = self._cnn_transform(rgb).unsqueeze(0)

            with torch.no_grad():
                embedding = self._cnn_model(tensor).squeeze().numpy()

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.warning(f"CellIdentifier: CNN embedding failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────
    # CNN Screening
    # ─────────────────────────────────────────────────────────────────────

    def _cnn_screen(self, embedding: np.ndarray | None, top_k: int = 5) -> list:
        """
        Fast screening: find top-k most similar cells by CNN cosine similarity.
        Returns list of (cell_key, similarity) sorted by similarity descending.
        Falls back to returning all cells if CNN is unavailable.
        """
        if embedding is None:
            # No CNN — return all cells as candidates
            return [(k, 0.0) for k in self._cells]

        scored = []
        for key, data in self._cells.items():
            cell_emb = data.get("embedding")
            if cell_emb is not None:
                sim = float(np.dot(embedding, cell_emb))
                scored.append((key, sim))
            else:
                scored.append((key, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ─────────────────────────────────────────────────────────────────────
    # SIFT Matching
    # ─────────────────────────────────────────────────────────────────────

    def _sift_match(self, desc_new: np.ndarray, desc_ref: np.ndarray) -> list:
        """
        Match SIFT descriptors using Lowe's ratio test.
        Returns list of good cv2.DMatch objects.
        """
        if desc_new is None or desc_ref is None:
            return []
        if len(desc_new) < 2 or len(desc_ref) < 2:
            return []

        try:
            matches = self._bf.knnMatch(desc_new, desc_ref, k=2)
        except cv2.error:
            return []

        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < _SIFT_RATIO_THRESH * n.distance:
                    good.append(m)
        return good

    # ─────────────────────────────────────────────────────────────────────
    # Delaunay Triangulation Verification
    # ─────────────────────────────────────────────────────────────────────

    def _delaunay_verify(
        self,
        new_kp_pts: np.ndarray,
        good_matches: list,
        new_desc: np.ndarray,
        candidate: dict,
    ) -> bool:
        """
        Verify a match using Delaunay triangulation.
        Checks that the triangle structure of matched keypoints is preserved
        between the two images (invariant to rotation/scale).
        """
        from scipy.spatial import Delaunay

        if len(good_matches) < 10:
            return False

        ref_kp_pts = candidate["keypoints"]

        # Get matched point coordinates
        src_pts = np.array([new_kp_pts[m.queryIdx] for m in good_matches])
        dst_pts = np.array([ref_kp_pts[m.trainIdx] for m in good_matches])

        # Build Delaunay on source points
        try:
            tri = Delaunay(src_pts)
        except Exception:
            return False

        if len(tri.simplices) == 0:
            return False

        # Check angle consistency for each triangle
        consistent = 0
        total = len(tri.simplices)

        for simplex in tri.simplices:
            src_tri = src_pts[simplex]
            dst_tri = dst_pts[simplex]

            src_angles = _triangle_angles(src_tri)
            dst_angles = _triangle_angles(dst_tri)

            if src_angles is None or dst_angles is None:
                continue

            # Compare sorted angles
            angle_diff = np.abs(np.sort(src_angles) - np.sort(dst_angles)).max()
            if angle_diff < _DELAUNAY_ANGLE_TOL:
                consistent += 1

        ratio = consistent / total if total > 0 else 0
        return ratio >= _DELAUNAY_CONSISTENCY

    # ─────────────────────────────────────────────────────────────────────
    # Spatial Stitching
    # ─────────────────────────────────────────────────────────────────────

    def _try_spatial_assignment(self, new_kp, new_desc, new_gray) -> tuple | None:
        """
        Try to find partial overlap with existing cells and determine
        spatial position via homography.

        Returns (row, col) of the new cell, or None if no spatial
        relationship can be determined.
        """
        img_h, img_w = new_gray.shape

        new_kp_pts = np.array([k.pt for k in new_kp], dtype=np.float32)

        for cell_key, cell_data in self._cells.items():
            good = self._sift_match(new_desc, cell_data["descriptors"])

            # Need enough matches for homography but not so many it's the same cell
            if len(good) < _SPATIAL_MIN_MATCHES:
                continue
            if len(good) > _SPATIAL_MAX_MATCHES:
                continue  # Probably same cell — handled in revisit check

            ref_kp_pts = cell_data["keypoints"]

            src_pts = np.float32([new_kp_pts[m.queryIdx] for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([ref_kp_pts[m.trainIdx] for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                continue

            inliers = int(mask.sum()) if mask is not None else 0
            inlier_ratio = inliers / len(good) if len(good) > 0 else 0

            if inliers < 8 or inlier_ratio < _SPATIAL_INLIER_RATIO:
                continue

            # Reject if homography has significant rotation/skew/scale
            # (neighbors should be pure translation)
            det = abs(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])
            if det < 0.7 or det > 1.3:
                continue  # Significant scale change — not a neighbor

            # Translation component of homography
            dx = H[0, 2]
            dy = H[1, 2]

            existing_cell = tuple(int(x) for x in cell_key.split(","))
            er, ec = existing_cell

            # Determine neighbor position based on dominant shift direction
            if abs(dx) > abs(dy):
                if dx > img_w * _SPATIAL_SHIFT_FRAC:
                    new_cell = (er, ec - 1)
                elif dx < -img_w * _SPATIAL_SHIFT_FRAC:
                    new_cell = (er, ec + 1)
                else:
                    continue
            else:
                if dy > img_h * _SPATIAL_SHIFT_FRAC:
                    new_cell = (er - 1, ec)
                elif dy < -img_h * _SPATIAL_SHIFT_FRAC:
                    new_cell = (er + 1, ec)
                else:
                    continue

            logger.info(
                f"CellIdentifier: spatial stitch — "
                f"overlap with cell {existing_cell}, shift=({dx:.0f},{dy:.0f}) "
                f"→ neighbor {new_cell}"
            )
            return new_cell

        return None

    # ─────────────────────────────────────────────────────────────────────
    # Cell Database Management
    # ─────────────────────────────────────────────────────────────────────

    def _register_cell(
        self,
        cell: tuple,
        descriptors: np.ndarray,
        keypoints: np.ndarray,
        embedding: np.ndarray | None,
        image_path: str,
    ):
        """Register a new cell in the database."""
        key = f"{cell[0]},{cell[1]}"
        self._cells[key] = {
            "descriptors": descriptors,
            "keypoints": keypoints,
            "embedding": embedding,
            "image_paths": [image_path],
            "visit_count": 1,
        }
        # Update next_id if needed
        idx = cell[0] * config.GRID_COLS + cell[1]
        if idx >= self._next_id:
            self._next_id = idx + 1
        self._save_database()

    def _record_revisit(self, cell_key: str, image_path: str) -> int:
        """Record a revisit to an existing cell. Returns the new visit count."""
        data = self._cells[cell_key]
        if image_path not in data["image_paths"]:
            data["image_paths"].append(image_path)
        data["visit_count"] += 1
        self._save_database()
        return data["visit_count"]

    def _next_available_cell(self) -> tuple:
        """Assign the next available grid cell (row-major order)."""
        max_cells = config.GRID_ROWS * config.GRID_COLS

        while self._next_id < max_cells:
            r = self._next_id // config.GRID_COLS
            c = self._next_id % config.GRID_COLS
            key = f"{r},{c}"
            if key not in self._cells:
                return (r, c)
            self._next_id += 1

        # Grid is full — reuse last cell (shouldn't happen in practice)
        logger.warning("CellIdentifier: grid full — reusing last cell")
        return (config.GRID_ROWS - 1, config.GRID_COLS - 1)

    def _clamp_cell(self, cell: tuple) -> tuple:
        """Clamp cell coordinates to valid grid range."""
        r = max(0, min(config.GRID_ROWS - 1, cell[0]))
        c = max(0, min(config.GRID_COLS - 1, cell[1]))
        return (r, c)

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def _save_database(self):
        """Save cell database to disk."""
        os.makedirs(_DB_DIR, exist_ok=True)

        index = {}
        for key, data in self._cells.items():
            # Save numpy arrays separately
            desc_path = os.path.join(_DB_DIR, f"cell_{key.replace(',','_')}_sift.npy")
            kp_path = os.path.join(_DB_DIR, f"cell_{key.replace(',','_')}_kp.npy")

            np.save(desc_path, data["descriptors"])
            np.save(kp_path, data["keypoints"])

            emb_path = None
            if data["embedding"] is not None:
                emb_path = os.path.join(_DB_DIR, f"cell_{key.replace(',','_')}_emb.npy")
                np.save(emb_path, data["embedding"])

            index[key] = {
                "descriptors_file": os.path.basename(desc_path),
                "keypoints_file": os.path.basename(kp_path),
                "embedding_file": os.path.basename(emb_path) if emb_path else None,
                "image_paths": data["image_paths"],
                "visit_count": data["visit_count"],
            }

        index["_meta"] = {"next_id": self._next_id}

        with open(_DB_INDEX_FILE, "w") as f:
            json.dump(index, f, indent=2)

    def _load_database(self):
        """Load cell database from disk."""
        if not os.path.exists(_DB_INDEX_FILE):
            return

        try:
            with open(_DB_INDEX_FILE) as f:
                index = json.load(f)
        except Exception as e:
            logger.warning(f"CellIdentifier: could not load database: {e}")
            return

        meta = index.pop("_meta", {})
        self._next_id = meta.get("next_id", 0)

        for key, info in index.items():
            desc_path = os.path.join(_DB_DIR, info["descriptors_file"])
            kp_path = os.path.join(_DB_DIR, info["keypoints_file"])

            if not os.path.exists(desc_path) or not os.path.exists(kp_path):
                logger.warning(f"CellIdentifier: missing data files for cell {key}")
                continue

            descriptors = np.load(desc_path)
            keypoints = np.load(kp_path)

            embedding = None
            if info.get("embedding_file"):
                emb_path = os.path.join(_DB_DIR, info["embedding_file"])
                if os.path.exists(emb_path):
                    embedding = np.load(emb_path)

            self._cells[key] = {
                "descriptors": descriptors,
                "keypoints": keypoints,
                "embedding": embedding,
                "image_paths": info.get("image_paths", []),
                "visit_count": info.get("visit_count", 1),
            }

        logger.info(f"CellIdentifier: loaded {len(self._cells)} cells from database")

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_gray(image_path: str) -> np.ndarray | None:
        img = cv2.imread(image_path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _fallback_result(self) -> dict:
        """Return a default result when identification fails."""
        cell = self._next_available_cell()
        return {
            "cell": cell, "confidence": 0.0, "is_revisit": False,
            "pass_number": 1, "method": "fallback",
        }


# ─────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────

def _triangle_angles(pts: np.ndarray) -> np.ndarray | None:
    """
    Compute the three interior angles (in degrees) of a triangle
    defined by 3 points.  Returns sorted array of 3 angles, or None
    if the triangle is degenerate.
    """
    a = pts[1] - pts[0]
    b = pts[2] - pts[0]
    c = pts[2] - pts[1]

    la = np.linalg.norm(a)
    lb = np.linalg.norm(b)
    lc = np.linalg.norm(c)

    if la < 1e-6 or lb < 1e-6 or lc < 1e-6:
        return None

    # Angles via dot product
    cos_A = np.clip(np.dot(a, b) / (la * lb), -1, 1)
    cos_B = np.clip(np.dot(-a, c) / (la * lc), -1, 1)

    A = np.degrees(np.arccos(cos_A))
    B = np.degrees(np.arccos(cos_B))
    C = 180.0 - A - B

    return np.array([A, B, C])
