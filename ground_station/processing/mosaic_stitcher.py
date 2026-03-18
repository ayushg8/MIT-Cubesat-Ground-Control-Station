from __future__ import annotations
# processing/mosaic_stitcher.py — AI-powered mosaic stitching with SuperPoint + LightGlue
#
# Uses learned feature detection (SuperPoint) and learned matching (LightGlue)
# instead of classical SIFT. This handles low-texture surfaces like sand/regolith
# where SIFT fails to find distinctive features.
#
# Pipeline per image:
#   1. SuperPoint extracts learned keypoints + descriptors
#   2. LightGlue matches against all existing entries (attention-based)
#   3. MAGSAC++ estimates homography (adaptive, no manual threshold)
#   4. Exposure compensation normalizes brightness
#   5. Multi-band blending (Laplacian pyramid) eliminates seams
#   6. Bundle adjustment (every N images) globally refines all poses
#
# Public API is identical to the previous SIFT-based implementation.

import json
import logging
import os

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

import config

logger = logging.getLogger(__name__)

_DB_DIR = os.path.join(config.PROCESSED_DIR, "mosaic_database")
_DB_INDEX_FILE = os.path.join(_DB_DIR, "mosaic_index.json")
_CANVAS_FILE = os.path.join(_DB_DIR, "mosaic_canvas.png")


class MosaicEntry:
    """Metadata for one registered image in the mosaic."""
    __slots__ = ("filename", "image_path", "homography", "bbox",
                 "keypoints", "descriptors", "_feats_cache")

    def __init__(self, filename, image_path, homography, bbox,
                 keypoints=None, descriptors=None):
        self.filename = filename
        self.image_path = image_path
        self.homography = homography  # 3x3 warp from image space -> mosaic space
        self.bbox = bbox              # (x, y, w, h) in mosaic space
        self.keypoints = keypoints    # Nx2 float32 numpy array
        self.descriptors = descriptors  # NxD float32 numpy array
        self._feats_cache = None      # torch feats dict (not persisted)


class MosaicStitcher:
    """
    Incrementally stitches images into a growing mosaic canvas.
    Uses SuperPoint + LightGlue for feature matching, MAGSAC++ for
    homography estimation, multi-band blending, and bundle adjustment.
    Thread-safe: caller (Pipeline) holds a lock before calling register_image().
    """

    def __init__(self):
        self._device = torch.device("cpu")

        # SuperPoint feature extractor
        self._extractor = SuperPoint(
            max_num_keypoints=config.MOSAIC_MAX_KEYPOINTS
        ).eval().to(self._device)

        # LightGlue learned matcher
        self._matcher = LightGlue(
            features="superpoint"
        ).eval().to(self._device)

        self._canvas: np.ndarray | None = None
        self._entries: list[MosaicEntry] = []
        self._exposure_gains: list[float] = []

        # Origin offset: when canvas grows left/up, all coords shift
        self._origin_x = 0
        self._origin_y = 0

        # Sequential placement fallback
        self._seq_x = 0
        self._seq_y = 0

        os.makedirs(_DB_DIR, exist_ok=True)
        self._load()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def register_image(self, image_path: str, metadata: dict | None = None) -> dict:
        """Register a new image into the mosaic. Returns placement info dict."""
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"MosaicStitcher: cannot read '{image_path}'")
            return self._error_result()

        h, w = img.shape[:2]
        basename = os.path.basename(image_path)

        # Extract SuperPoint features
        feats, kp_np, desc_np = self._extract_features(img)

        # IMU hint
        imu = (metadata or {}).get("imu")

        # -- First image: place at center of initial canvas --
        if self._canvas is None:
            canvas_size = config.MOSAIC_INITIAL_CANVAS_PX
            self._canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            cx = max(0, (canvas_size - w) // 2)
            cy = max(0, (canvas_size - h) // 2)

            pw = min(w, canvas_size - cx)
            ph = min(h, canvas_size - cy)
            self._canvas[cy:cy + ph, cx:cx + pw] = img[:ph, :pw]

            H = np.eye(3, dtype=np.float64)
            H[0, 2] = cx
            H[1, 2] = cy

            entry = MosaicEntry(
                filename=basename, image_path=image_path,
                homography=H, bbox=(cx, cy, pw, ph),
                keypoints=kp_np, descriptors=desc_np,
            )
            entry._feats_cache = feats
            self._entries.append(entry)
            self._exposure_gains.append(1.0)

            self._seq_x = cx + pw
            self._seq_y = cy

            self._save_canvas()
            self._save_index()
            logger.info(f"MosaicStitcher: first image '{basename}' placed at ({cx},{cy})")

            return {
                "mosaic_bbox": (cx, cy, pw, ph),
                "entry_index": 0,
                "method": "first",
                "canvas_size": (self._canvas.shape[1], self._canvas.shape[0]),
                "image_count": 1,
            }

        # -- Try matching against existing entries --
        best_entry = None
        best_H = None
        best_inliers = 0
        best_method = "sequential"

        if feats is not None and kp_np is not None and len(kp_np) >= 5:
            # Limit candidates to most recent entries for performance
            # (most likely to overlap with new image)
            _MAX_MATCH_CANDIDATES = 30
            candidates = self._entries[-_MAX_MATCH_CANDIDATES:]
            for entry in candidates:
                ref_feats = self._get_entry_feats(entry)
                if ref_feats is None:
                    continue

                match_result = self._match_features(feats, ref_feats)
                if match_result is None:
                    continue

                pts_new, pts_ref, num_matches = match_result

                # MAGSAC++ homography estimation
                H_to_entry, mask = cv2.findHomography(
                    pts_new, pts_ref, cv2.USAC_MAGSAC,
                    5.0, maxIters=2000, confidence=0.999,
                )
                if H_to_entry is None:
                    continue

                inliers = int(mask.sum()) if mask is not None else 0
                if inliers < config.MOSAIC_MIN_SIFT_INLIERS:
                    continue

                if inliers > best_inliers:
                    H_to_mosaic = entry.homography @ H_to_entry
                    best_entry = entry
                    best_H = H_to_mosaic
                    best_inliers = inliers
                    best_method = "superpoint+lightglue"

        if best_H is not None:
            # Compute bounding box in mosaic space
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners, best_H)
            wc = warped_corners.reshape(-1, 2)

            mx_min = int(np.floor(wc[:, 0].min()))
            my_min = int(np.floor(wc[:, 1].min()))
            mx_max = int(np.ceil(wc[:, 0].max()))
            my_max = int(np.ceil(wc[:, 1].max()))

            self._ensure_canvas_size(mx_min, my_min, mx_max, my_max)

            # Exposure compensation
            img_corrected = self._exposure_compensate(img, best_H, best_entry)
            gain = np.mean(img_corrected.astype(np.float32)) / max(np.mean(img.astype(np.float32)), 1.0)

            # Multi-band blend
            self._warp_and_blend_multiband(img_corrected, best_H)

            entry = MosaicEntry(
                filename=basename, image_path=image_path,
                homography=best_H.copy(),
                bbox=(mx_min, my_min, mx_max - mx_min, my_max - my_min),
                keypoints=kp_np, descriptors=desc_np,
            )
            entry._feats_cache = feats
            self._entries.append(entry)
            self._exposure_gains.append(float(np.clip(gain, *config.MOSAIC_EXPOSURE_GAIN_RANGE)))

            # Bundle adjustment periodically
            if (len(self._entries) >= 3 and
                    len(self._entries) % config.MOSAIC_BUNDLE_ADJUST_INTERVAL == 0):
                self._bundle_adjust()

            self._save_canvas()
            self._save_index()

            logger.info(
                f"MosaicStitcher: '{basename}' stitched via {best_method} "
                f"({best_inliers} inliers, matched '{best_entry.filename}')"
            )

            return {
                "mosaic_bbox": (mx_min, my_min, mx_max - mx_min, my_max - my_min),
                "entry_index": len(self._entries) - 1,
                "method": best_method,
                "canvas_size": (self._canvas.shape[1], self._canvas.shape[0]),
                "image_count": len(self._entries),
            }

        # -- Fallback: sequential placement with assumed overlap --
        logger.warning(
            f"MosaicStitcher: '{basename}' no match — placing sequentially"
        )

        ch, cw = self._canvas.shape[:2]
        overlap_px = int(w * 0.30)
        px = max(0, self._seq_x - overlap_px)
        py = self._seq_y
        if px + w > cw + config.MOSAIC_CANVAS_PAD_PX * 2:
            px = 0
            py = py + int(h * 0.70)

        self._ensure_canvas_size(px, py, px + w, py + h)

        H = np.eye(3, dtype=np.float64)
        H[0, 2] = px
        H[1, 2] = py

        ch, cw = self._canvas.shape[:2]
        pw = min(w, cw - px)
        ph = min(h, ch - py)
        if pw > 0 and ph > 0:
            self._canvas[py:py + ph, px:px + pw] = img[:ph, :pw]

        entry = MosaicEntry(
            filename=basename, image_path=image_path,
            homography=H, bbox=(px, py, pw, ph),
            keypoints=kp_np, descriptors=desc_np,
        )
        entry._feats_cache = feats
        self._entries.append(entry)
        self._exposure_gains.append(1.0)

        self._seq_x = px + pw + 5
        self._seq_y = py

        self._save_canvas()
        self._save_index()

        return {
            "mosaic_bbox": (px, py, pw, ph),
            "entry_index": len(self._entries) - 1,
            "method": "sequential",
            "canvas_size": (self._canvas.shape[1], self._canvas.shape[0]),
            "image_count": len(self._entries),
        }

    def get_canvas(self) -> np.ndarray | None:
        return self._canvas

    def get_canvas_size(self) -> tuple[int, int]:
        if self._canvas is None:
            return (0, 0)
        return (self._canvas.shape[1], self._canvas.shape[0])

    def get_entries(self) -> list[MosaicEntry]:
        return self._entries

    def get_image_count(self) -> int:
        return len(self._entries)

    def get_overlapping_entries(self, bbox: tuple) -> list[MosaicEntry]:
        x, y, w, h = bbox
        results = []
        for entry in self._entries:
            ex, ey, ew, eh = entry.bbox
            if (x < ex + ew and x + w > ex and y < ey + eh and y + h > ey):
                results.append(entry)
        return results

    def get_entry_by_filename(self, filename: str) -> MosaicEntry | None:
        for entry in self._entries:
            if entry.filename == filename:
                return entry
        return None

    # -----------------------------------------------------------------
    # Feature extraction (SuperPoint)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _extract_features(self, img_bgr: np.ndarray):
        """Extract SuperPoint features from a BGR image.
        Returns (feats_dict, keypoints_np, descriptors_np) or (None, None, None).
        """
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_f = img_rgb.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(self._device)

            feats = self._extractor.extract(img_tensor)

            kp = feats["keypoints"][0].cpu().numpy()   # Nx2
            desc = feats["descriptors"][0].cpu().numpy()  # NxD

            return feats, kp.astype(np.float32), desc.astype(np.float32)
        except Exception as e:
            logger.warning(f"SuperPoint extraction failed: {e}")
            return None, None, None

    # -----------------------------------------------------------------
    # Feature matching (LightGlue)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _match_features(self, feats_new: dict, feats_ref: dict):
        """Match features using LightGlue.
        Returns (pts_new, pts_ref, num_matches) or None.
        """
        try:
            result = self._matcher({"image0": feats_new, "image1": feats_ref})
            result = rbd(result)

            matches = result["matches"].cpu().numpy()  # Kx2
            if len(matches) < config.MOSAIC_MIN_SIFT_INLIERS:
                return None

            kp0 = feats_new["keypoints"][0].cpu().numpy()
            kp1 = feats_ref["keypoints"][0].cpu().numpy()

            pts0 = kp0[matches[:, 0]].reshape(-1, 1, 2).astype(np.float32)
            pts1 = kp1[matches[:, 1]].reshape(-1, 1, 2).astype(np.float32)

            return pts0, pts1, len(matches)
        except Exception as e:
            logger.warning(f"LightGlue matching failed: {type(e).__name__}: {e}")
            return None

    def _get_entry_feats(self, entry: MosaicEntry):
        """Get or rebuild the torch feats dict for an entry."""
        if entry._feats_cache is not None:
            return entry._feats_cache

        # Rebuild from stored numpy arrays
        if entry.keypoints is None or entry.descriptors is None:
            return None
        if len(entry.keypoints) < 5:
            return None

        try:
            kp_t = torch.from_numpy(entry.keypoints).unsqueeze(0).to(self._device)
            desc_t = torch.from_numpy(entry.descriptors).unsqueeze(0).to(self._device)

            # Read image to get dimensions for image_size
            if os.path.exists(entry.image_path):
                img = cv2.imread(entry.image_path)
                if img is not None:
                    h, w = img.shape[:2]
                    feats = {
                        "keypoints": kp_t,
                        "descriptors": desc_t,
                        "image_size": torch.tensor([[h, w]], device=self._device),
                    }
                    entry._feats_cache = feats
                    return feats
        except Exception as e:
            logger.warning(f"Failed to rebuild feats for {entry.filename}: {e}")
        return None

    # -----------------------------------------------------------------
    # Exposure compensation
    # -----------------------------------------------------------------

    def _exposure_compensate(self, img: np.ndarray, H: np.ndarray,
                             ref_entry: MosaicEntry) -> np.ndarray:
        """Normalize brightness of img to match the reference entry's region."""
        if self._canvas is None:
            return img

        try:
            ch, cw = self._canvas.shape[:2]
            h, w = img.shape[:2]

            # Warp new image to mosaic space to find overlap
            warped = cv2.warpPerspective(img, H, (cw, ch))
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            canvas_gray = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)

            # Overlap mask
            overlap = (warped_gray > 0) & (canvas_gray > 0)
            if overlap.sum() < 100:
                return img

            mean_new = float(warped_gray[overlap].mean())
            mean_ref = float(canvas_gray[overlap].mean())

            if mean_new < 1.0:
                return img

            gain = mean_ref / mean_new
            lo, hi = config.MOSAIC_EXPOSURE_GAIN_RANGE
            gain = float(np.clip(gain, lo, hi))

            return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.warning(f"Exposure compensation failed: {e}")
            return img

    # -----------------------------------------------------------------
    # Multi-band blending (Laplacian pyramid)
    # -----------------------------------------------------------------

    def _warp_and_blend_multiband(self, img: np.ndarray, H: np.ndarray):
        """Warp image into canvas using multi-band Laplacian pyramid blending."""
        if self._canvas is None:
            return

        ch, cw = self._canvas.shape[:2]
        n_levels = config.MOSAIC_BLEND_LEVELS

        # Warp the image
        warped = cv2.warpPerspective(img, H, (cw, ch))

        # Create masks
        h, w = img.shape[:2]
        weight_mask = np.ones((h, w), dtype=np.float32)
        # Feather edges for smooth blending
        feather = min(h, w) // 8
        if feather > 2:
            weight_mask = cv2.GaussianBlur(weight_mask, (0, 0), sigmaX=feather)
        warped_mask = cv2.warpPerspective(weight_mask, H, (cw, ch))

        warped_binary = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
        canvas_binary = (cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
        overlap = warped_binary * canvas_binary

        # If no overlap, just paste directly
        if overlap.sum() < 10:
            non_overlap = warped_binary - overlap
            for c in range(3):
                self._canvas[:, :, c] = np.where(
                    non_overlap > 0.5, warped[:, :, c], self._canvas[:, :, c]
                )
            return

        # Build blend mask: warped_mask normalized in overlap region
        total_mask = warped_mask + (1.0 - warped_mask) * canvas_binary
        blend_weight = np.where(total_mask > 0.01, warped_mask / total_mask, 0.0)

        # Build Laplacian pyramids
        canvas_f = self._canvas.astype(np.float32)
        warped_f = warped.astype(np.float32)

        lp_canvas = self._laplacian_pyramid(canvas_f, n_levels)
        lp_warped = self._laplacian_pyramid(warped_f, n_levels)
        gp_mask = self._gaussian_pyramid(blend_weight, n_levels)

        # Blend at each level
        lp_blended = []
        for lc, lw, gm in zip(lp_canvas, lp_warped, gp_mask):
            gm3 = np.stack([gm] * 3, axis=-1) if lc.ndim == 3 else gm
            blended = lw * gm3 + lc * (1.0 - gm3)
            lp_blended.append(blended)

        # Reconstruct
        result = self._reconstruct_laplacian(lp_blended)
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Apply: in warped or canvas regions, use blended result
        combined_mask = np.maximum(warped_binary, canvas_binary)
        for c in range(3):
            self._canvas[:, :, c] = np.where(
                combined_mask > 0.5, result[:, :, c], self._canvas[:, :, c]
            )

    def _laplacian_pyramid(self, img: np.ndarray, levels: int) -> list:
        """Build a Laplacian pyramid."""
        gp = [img]
        for _ in range(levels):
            down = cv2.pyrDown(gp[-1])
            gp.append(down)

        lp = []
        for i in range(levels):
            up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lap = gp[i] - up
            lp.append(lap)
        lp.append(gp[-1])  # lowest resolution
        return lp

    def _gaussian_pyramid(self, mask: np.ndarray, levels: int) -> list:
        """Build a Gaussian pyramid of a single-channel mask."""
        gp = [mask]
        for _ in range(levels):
            gp.append(cv2.pyrDown(gp[-1]))
        return gp

    def _reconstruct_laplacian(self, lp: list) -> np.ndarray:
        """Reconstruct an image from its Laplacian pyramid."""
        img = lp[-1]
        for i in range(len(lp) - 2, -1, -1):
            up = cv2.pyrUp(img, dstsize=(lp[i].shape[1], lp[i].shape[0]))
            img = up + lp[i]
        return img

    # -----------------------------------------------------------------
    # Bundle adjustment
    # -----------------------------------------------------------------

    def _bundle_adjust(self):
        """Globally refine all homographies by minimizing reprojection error."""
        n = len(self._entries)
        if n < 3:
            return

        try:
            from scipy.optimize import least_squares
        except ImportError:
            logger.warning("scipy not available — skipping bundle adjustment")
            return

        # Find overlapping pairs and their matches
        pairs = []  # (i, j, pts_i, pts_j)
        for i in range(n):
            fi = self._get_entry_feats(self._entries[i])
            if fi is None:
                continue
            for j in range(i + 1, n):
                # Check bbox overlap
                ei, ej = self._entries[i], self._entries[j]
                bx1, by1, bw1, bh1 = ei.bbox
                bx2, by2, bw2, bh2 = ej.bbox
                if not (bx1 < bx2 + bw2 and bx1 + bw1 > bx2 and
                        by1 < by2 + bh2 and by1 + bh1 > by2):
                    continue

                fj = self._get_entry_feats(self._entries[j])
                if fj is None:
                    continue

                match_result = self._match_features(fi, fj)
                if match_result is None:
                    continue

                pts_i, pts_j, _ = match_result
                # Limit points for performance
                max_pts = 50
                if len(pts_i) > max_pts:
                    idx = np.random.choice(len(pts_i), max_pts, replace=False)
                    pts_i = pts_i[idx]
                    pts_j = pts_j[idx]
                pairs.append((i, j, pts_i.reshape(-1, 2), pts_j.reshape(-1, 2)))

        if len(pairs) < 2:
            return

        logger.info(f"Bundle adjustment: {n} images, {len(pairs)} overlapping pairs")

        # Parameterize: H_0 fixed (reference), H_1..H_{n-1} as 8 params each
        # H = [[p0 p1 p2], [p3 p4 p5], [p6 p7 1.0]]
        def h_to_params(H):
            return H.flatten()[:8]

        def params_to_h(p):
            return np.array([[p[0], p[1], p[2]],
                             [p[3], p[4], p[5]],
                             [p[6], p[7], 1.0]], dtype=np.float64)

        x0 = []
        for i in range(1, n):
            x0.extend(h_to_params(self._entries[i].homography))
        x0 = np.array(x0, dtype=np.float64)

        H0 = self._entries[0].homography.copy()

        def residuals(x):
            Hs = [H0]
            for i in range(1, n):
                offset = (i - 1) * 8
                Hs.append(params_to_h(x[offset:offset + 8]))

            res = []
            for i, j, pts_i, pts_j in pairs:
                # Project pts_i through H_i and pts_j through H_j
                ones_i = np.ones((len(pts_i), 1), dtype=np.float64)
                ones_j = np.ones((len(pts_j), 1), dtype=np.float64)

                pi = np.hstack([pts_i, ones_i])  # Nx3
                pj = np.hstack([pts_j, ones_j])  # Nx3

                proj_i = (Hs[i] @ pi.T).T  # Nx3
                proj_j = (Hs[j] @ pj.T).T  # Nx3

                # Normalize by w
                proj_i = proj_i[:, :2] / (proj_i[:, 2:3] + 1e-10)
                proj_j = proj_j[:, :2] / (proj_j[:, 2:3] + 1e-10)

                diff = (proj_i - proj_j).flatten()
                res.extend(diff)
            return np.array(res, dtype=np.float64)

        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=200)

            if result.success or result.cost < np.sum(residuals(x0) ** 2):
                # Update homographies
                for i in range(1, n):
                    offset = (i - 1) * 8
                    self._entries[i].homography = params_to_h(result.x[offset:offset + 8])

                # Update bounding boxes
                for entry in self._entries:
                    img = cv2.imread(entry.image_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                        wc = cv2.perspectiveTransform(corners, entry.homography).reshape(-1, 2)
                        entry.bbox = (
                            int(np.floor(wc[:, 0].min())),
                            int(np.floor(wc[:, 1].min())),
                            int(np.ceil(wc[:, 0].max())) - int(np.floor(wc[:, 0].min())),
                            int(np.ceil(wc[:, 1].max())) - int(np.floor(wc[:, 1].min())),
                        )

                # Re-render canvas
                self._rerender_canvas()
                logger.info(f"Bundle adjustment converged (cost={result.cost:.2f})")
            else:
                logger.warning("Bundle adjustment did not improve — keeping original poses")

        except Exception as e:
            logger.warning(f"Bundle adjustment failed: {e}")

    def _rerender_canvas(self):
        """Re-render the entire canvas from scratch using current homographies."""
        if not self._entries:
            return

        # Find canvas bounds
        all_min_x, all_min_y = float('inf'), float('inf')
        all_max_x, all_max_y = float('-inf'), float('-inf')

        for entry in self._entries:
            bx, by, bw, bh = entry.bbox
            all_min_x = min(all_min_x, bx)
            all_min_y = min(all_min_y, by)
            all_max_x = max(all_max_x, bx + bw)
            all_max_y = max(all_max_y, by + bh)

        cw = min(int(all_max_x - all_min_x) + 200, config.MOSAIC_MAX_CANVAS_PX)
        ch = min(int(all_max_y - all_min_y) + 200, config.MOSAIC_MAX_CANVAS_PX)

        # Offset if bounds went negative
        offset_x = -int(all_min_x) + 100 if all_min_x < 0 else 0
        offset_y = -int(all_min_y) + 100 if all_min_y < 0 else 0

        if offset_x > 0 or offset_y > 0:
            offset_H = np.eye(3, dtype=np.float64)
            offset_H[0, 2] = offset_x
            offset_H[1, 2] = offset_y
            for entry in self._entries:
                entry.homography = offset_H @ entry.homography
                bx, by, bw, bh = entry.bbox
                entry.bbox = (bx + offset_x, by + offset_y, bw, bh)
            self._origin_x += offset_x
            self._origin_y += offset_y

        new_canvas = np.zeros((ch, cw, 3), dtype=np.uint8)

        # Warp and blend each image
        for i, entry in enumerate(self._entries):
            img = cv2.imread(entry.image_path)
            if img is None:
                continue

            gain = self._exposure_gains[i] if i < len(self._exposure_gains) else 1.0
            if abs(gain - 1.0) > 0.01:
                img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

            # Temporarily swap canvas for blending
            old_canvas = self._canvas
            self._canvas = new_canvas
            self._warp_and_blend_multiband(img, entry.homography)
            new_canvas = self._canvas
            self._canvas = old_canvas

        self._canvas = new_canvas

    # -----------------------------------------------------------------
    # Canvas management
    # -----------------------------------------------------------------

    def _ensure_canvas_size(self, x_min: int, y_min: int, x_max: int, y_max: int):
        if self._canvas is None:
            return

        ch, cw = self._canvas.shape[:2]
        pad = config.MOSAIC_CANVAS_PAD_PX
        max_dim = config.MOSAIC_MAX_CANVAS_PX

        need_left = max(0, -x_min + pad)
        need_top = max(0, -y_min + pad)
        need_right = max(0, x_max - cw + pad)
        need_bottom = max(0, y_max - ch + pad)

        if need_left == 0 and need_top == 0 and need_right == 0 and need_bottom == 0:
            return

        new_w = min(max_dim, cw + need_left + need_right)
        new_h = min(max_dim, ch + need_top + need_bottom)

        actual_left = min(need_left, new_w - cw)
        actual_top = min(need_top, new_h - ch)

        if actual_left == 0 and actual_top == 0 and new_w == cw and new_h == ch:
            return

        new_canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        dst_x = actual_left
        dst_y = actual_top
        copy_w = min(cw, new_w - dst_x)
        copy_h = min(ch, new_h - dst_y)
        new_canvas[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = self._canvas[:copy_h, :copy_w]
        self._canvas = new_canvas

        if actual_left > 0 or actual_top > 0:
            offset = np.eye(3, dtype=np.float64)
            offset[0, 2] = actual_left
            offset[1, 2] = actual_top

            for entry in self._entries:
                entry.homography = offset @ entry.homography
                bx, by, bw, bh = entry.bbox
                entry.bbox = (bx + actual_left, by + actual_top, bw, bh)

            self._origin_x += actual_left
            self._origin_y += actual_top
            self._seq_x += actual_left
            self._seq_y += actual_top

        logger.info(
            f"MosaicStitcher: canvas grew from {cw}x{ch} to {new_w}x{new_h} "
            f"(offset +{actual_left},+{actual_top})"
        )

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def _save_canvas(self):
        if self._canvas is not None:
            os.makedirs(_DB_DIR, exist_ok=True)
            cv2.imwrite(_CANVAS_FILE, self._canvas)
            mosaic_dir = os.path.join(config.PROCESSED_DIR, "mosaics")
            os.makedirs(mosaic_dir, exist_ok=True)
            cv2.imwrite(os.path.join(mosaic_dir, "mosaic_latest.png"), self._canvas)

    def _save_index(self):
        os.makedirs(_DB_DIR, exist_ok=True)

        index = {
            "origin_x": self._origin_x,
            "origin_y": self._origin_y,
            "seq_x": self._seq_x,
            "seq_y": self._seq_y,
            "feature_type": "superpoint",
            "exposure_gains": self._exposure_gains,
            "entries": [],
        }

        for i, entry in enumerate(self._entries):
            desc_file = f"entry_{i}_desc.npy"
            kp_file = f"entry_{i}_kp.npy"
            h_file = f"entry_{i}_H.npy"

            if entry.descriptors is not None:
                np.save(os.path.join(_DB_DIR, desc_file), entry.descriptors)
            if entry.keypoints is not None:
                np.save(os.path.join(_DB_DIR, kp_file), entry.keypoints)
            np.save(os.path.join(_DB_DIR, h_file), entry.homography)

            index["entries"].append({
                "filename": entry.filename,
                "image_path": entry.image_path,
                "bbox": list(entry.bbox),
                "desc_file": desc_file,
                "kp_file": kp_file,
                "h_file": h_file,
            })

        with open(_DB_INDEX_FILE, "w") as f:
            json.dump(index, f, indent=2)

    def _load(self):
        if not os.path.exists(_DB_INDEX_FILE):
            return

        if os.path.exists(_CANVAS_FILE):
            self._canvas = cv2.imread(_CANVAS_FILE)

        try:
            with open(_DB_INDEX_FILE) as f:
                index = json.load(f)
        except Exception as e:
            logger.warning(f"MosaicStitcher: could not load index: {e}")
            return

        self._origin_x = index.get("origin_x", 0)
        self._origin_y = index.get("origin_y", 0)
        self._seq_x = index.get("seq_x", 0)
        self._seq_y = index.get("seq_y", 0)
        self._exposure_gains = index.get("exposure_gains", [])

        feat_type = index.get("feature_type", "sift")
        if feat_type != "superpoint":
            logger.warning(
                f"MosaicStitcher: old feature type '{feat_type}' — "
                "features will be re-extracted on next match"
            )

        for edata in index.get("entries", []):
            desc = None
            kp = None
            H = np.eye(3, dtype=np.float64)

            desc_path = os.path.join(_DB_DIR, edata["desc_file"])
            kp_path = os.path.join(_DB_DIR, edata["kp_file"])
            h_path = os.path.join(_DB_DIR, edata["h_file"])

            if os.path.exists(desc_path):
                desc = np.load(desc_path)
            if os.path.exists(kp_path):
                kp = np.load(kp_path)
            if os.path.exists(h_path):
                H = np.load(h_path)

            entry = MosaicEntry(
                filename=edata["filename"],
                image_path=edata["image_path"],
                homography=H,
                bbox=tuple(edata["bbox"]),
                keypoints=kp,
                descriptors=desc,
            )
            self._entries.append(entry)

        # Pad exposure gains if needed
        while len(self._exposure_gains) < len(self._entries):
            self._exposure_gains.append(1.0)

        logger.info(f"MosaicStitcher: loaded {len(self._entries)} entries from database")

    def _error_result(self) -> dict:
        return {
            "mosaic_bbox": (0, 0, 0, 0),
            "entry_index": -1,
            "method": "error",
            "canvas_size": self.get_canvas_size(),
            "image_count": len(self._entries),
        }
