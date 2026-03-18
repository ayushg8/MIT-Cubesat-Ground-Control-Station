# Assumptions, Limitations, and Error Sources

This document catalogs every assumption, known limitation, and potential source of error in the MuraltZ Ground Control Station software. Understanding these is critical for interpreting results and planning improvements.

---

## 1. The Grid Problem (Most Critical)

### What we do
The terrain is divided into an 8×8 virtual grid. Each incoming image is assigned to a cell using image feature matching (CNN + SIFT + Delaunay triangulation). The entire system — hazard classification, route planning, change detection, coverage tracking — operates on this grid.

### The fundamental problem
**We don't know the CubeSat's orbit, trajectory, or pointing direction.** The CubeSat just sends images. The GCS has to figure out where each image "is" relative to previous images using only pixel content.

### How cell identification currently works
1. First image ever → assigned to cell (0,0) arbitrarily
2. Next image → CNN embedding compared against all known cells
3. If similar enough → SIFT keypoint matching confirms if it's the same cell (revisit)
4. If not a revisit → spatial stitching tries to find overlap with known neighbors and compute a homography to estimate relative position
5. If no overlap found → assigned to the next available cell in sequence

### Specific error sources

| Error | Impact | Likelihood |
|-------|--------|------------|
| **No absolute reference frame** — cell (0,0) is wherever the first image happened to point. The grid has no real-world anchor. | Route "Landing → Target" has no physical meaning unless the operator manually sets them. | Certain — by design |
| **Cumulative drift** — spatial stitching estimates each new cell's position relative to its neighbors. Errors accumulate. Cell (7,7) could be significantly misplaced relative to (0,0). | Route distances in cm are unreliable. Grid cell size assumption (10cm) may not hold across the grid. | High on larger grids |
| **Featureless terrain** — lunar regolith (and sand testbeds) have low texture. SIFT struggles on uniform surfaces. CNN embeddings become ambiguous. | Cells get misidentified. Two different locations assigned to the same cell. Same location split across two cells. | High on sand/regolith |
| **Grid discretization** — each image is forced into exactly one cell. If the CubeSat camera FOV doesn't align with the grid, an image might span 2-4 cells. | Hazard classification applies to the wrong area. Change detection compares images that don't fully overlap. | Moderate to high |
| **Unknown camera movement pattern** — we assume the CubeSat surveys roughly one cell at a time. If it skips cells, rotates, or images at an angle, the spatial stitching fails. | Large gaps in coverage, wrong neighbor assignments. | Depends on CubeSat behavior |
| **Revisit false positives** — if two physically different areas look similar (e.g., two patches of flat sand), the system thinks it's the same cell. | Change detection triggers on non-changes. Coverage count is inflated. | Moderate on uniform terrain |
| **Revisit false negatives** — if the same area looks different due to lighting/angle change, the system thinks it's a new cell. | Same area gets two cells. Coverage is undercounted. Change detection misses it. | Moderate |
| **No rotation handling** — spatial stitching uses homography which can handle rotation, but the grid assignment assumes axis-aligned images. A rotated image gets assigned to the wrong cell. | Misplacement by 1-2 cells. | Low to moderate |

### Better approaches (ranked by feasibility)

#### A. IMU-Based Dead Reckoning (Best near-term fix)
**How:** The CubeSat already has an IMU. Send attitude + angular velocity with each image as telemetry metadata. Use this to estimate how far the camera has moved between frames.

**Pros:** No extra hardware. Relative positioning between consecutive images is much more reliable than pure image matching. Works on featureless terrain.

**Cons:** Drift over time (IMU integration error). Needs calibration. Doesn't give absolute position.

**Effort:** Moderate — CubeSat sends extra telemetry fields, GCS uses them to seed cell assignment before image matching confirms.

#### B. Overlap-Based Continuous Mosaic (Better accuracy)
**How:** Instead of forcing images into a pre-defined grid, build a continuous mosaic using feature matching and bundle adjustment. Register every image relative to every other overlapping image simultaneously. Then overlay a grid on the final mosaic for route planning.

**Pros:** No grid discretization error. Each image's position is optimized globally, not just relative to one neighbor. Bundle adjustment reduces cumulative drift.

**Cons:** Computationally expensive. Needs sufficient overlap between images (~30%+). Still struggles on featureless terrain.

**Effort:** High — requires replacing cell_identifier.py with a proper mosaic/SLAM pipeline (OpenCV stitcher, ORB-SLAM2, or similar).

#### C. Fiducial Markers on Testbed (Best for lab demos)
**How:** Place AprilTags or ArUco markers at known positions on the sand testbed. The GCS detects markers in each image to get absolute position.

**Pros:** Extremely accurate. Trivial to implement (OpenCV has ArUco built in). Gives absolute positioning.

**Cons:** Only works on the testbed, not on the Moon. Adds artificial elements to the terrain.

**Effort:** Low — add ArUco detection, print and place ~16 markers on testbed edges.

#### D. Visual Odometry (Most robust long-term)
**How:** Track features between consecutive frames to estimate 6-DOF camera motion. Build a trajectory, then map images onto a coordinate system.

**Pros:** Works without orbit knowledge. Handles arbitrary camera motion. Can be combined with IMU (VIO) for even better results.

**Cons:** Needs consecutive frames (not just snapshots). Computationally intensive. Still drifts without loop closure.

**Effort:** High — significant new subsystem.

#### E. Known Orbit Model (If orbit is predictable)
**How:** If the CubeSat's path over the testbed follows a repeatable pattern (e.g., linear pass, fixed altitude), encode that pattern. Use image timestamps + orbit model to predict which cell each image should map to, then verify with image matching.

**Pros:** Very accurate if the orbit model is good. Image matching becomes verification, not primary positioning.

**Cons:** Requires knowing/controlling the CubeSat trajectory. Breaks if the path changes.

**Effort:** Low to moderate — depends on how predictable the CubeSat's motion is.

### Recommendation
For the BWSI prototype: **Option A (IMU metadata) + Option C (ArUco markers for testbed)**. This gives you reliable cell identification with minimal added complexity. The image-based system becomes a fallback/verification layer rather than the primary positioning method.

---

## 2. Change Detection Assumptions

| Assumption | Reality | Impact |
|-----------|---------|--------|
| Images of the same cell are taken from approximately the same angle and distance | Camera angle and height vary between passes | SIFT alignment may fail or produce artifacts. SSIM comparison becomes unreliable. |
| Lighting conditions are consistent between passes | Lighting changes with time, CubeSat angle, shadows | Brightness changes get flagged as "terrain changes" (false positives) |
| SSIM threshold (0.85) is appropriate for all terrain types | Optimal threshold varies by texture complexity | Too sensitive on uniform terrain (noise → false changes), too lenient on complex terrain (real changes missed) |
| Image alignment (homography) preserves all information | Homography warping introduces interpolation artifacts at edges | Edge regions of aligned images show spurious differences |
| Minimum 2 passes of the same cell needed | Depends on cell identification being correct | If cell ID is wrong, we compare images of different locations |

---

## 3. Shadow Detection Assumptions

| Assumption | Reality | Impact |
|-----------|---------|--------|
| Shadows are darker than the surrounding terrain by a consistent amount | On lunar regolith, the entire surface is dark; shadow contrast varies with sun angle | Adaptive thresholding may set wrong boundary, missing faint shadows or flagging dark rocks |
| Shadow regions are contiguous | A single shadow can be fragmented by terrain features | Region counting may overcount |
| Dark objects (rocks, boulders) are distinguishable from shadows by shape | In low-res images, a dark rock and a shadow blob look identical | Misclassification between shadow and obstacle |
| The sun angle is unknown | We could compute expected shadow direction from CubeSat telemetry | Without sun angle, we can't validate shadow direction or estimate obstacle height |

---

## 4. Hazard Classification Assumptions

| Assumption | Reality | Impact |
|-----------|---------|--------|
| One classification per grid cell | A cell might contain both safe and hazardous terrain | Classification represents the "average" hazard level, not worst-case within the cell |
| Cost values (1, 5, 15, 20, 999) are hand-tuned and fixed | Optimal costs depend on rover capabilities, mission objectives | Routes may be suboptimal if cost ratios don't match actual traversability |
| Classification is based on single-image analysis | A cell's hazard level could change between passes (shadow movement, terrain disturbance) | Stale classification if not re-evaluated after new images |
| YOLO detections and classical CV are fused with equal weight | ML model confidence varies by object type and image quality | Fusion may over- or under-weight one detector |
| YOLO model trained on Roboflow lunar dataset generalizes to our testbed | Training data may not match testbed terrain | False positives/negatives on novel terrain features |

---

## 5. Route Planning Assumptions

| Assumption | Reality | Impact |
|-----------|---------|--------|
| 8-connected grid movement (horizontal, vertical, diagonal) | A real rover has continuous motion with turning radius constraints | Planned routes may not be physically followable |
| All cells are the same physical size (GRID_CELL_SIZE_CM) | Camera height variations and grid drift mean cells vary in size | Distance estimates in cm are approximate |
| Unsurveyed cells have cost=10 (moderate, traversable) | Unsurveyed areas could be impassable | Routes through unsurveyed areas are speculative and potentially dangerous |
| The grid is flat (2D) | Real terrain has elevation changes | A "safe" cell on a steep slope is actually dangerous |
| Route is computed once and stays valid | Terrain changes, new images may reveal new hazards | Route becomes stale; the change→route impact warning helps but doesn't auto-replan |
| Landing and Target points are set manually | Operator must know where the rover starts and where it should go | If grid positioning is wrong, the "landing" cell may not correspond to the actual rover location |

---

## 6. Communication Assumptions

| Assumption | Reality | Impact |
|-----------|---------|--------|
| CubeSat throttles to 1200 B/s | Actual throughput depends on WiFi conditions | Transfer time estimates may be wrong |
| TCP ensures reliable delivery | TCP retransmits on packet loss, but connection can drop entirely | Large images may fail mid-transfer; MD5 check catches corruption but not loss |
| CubeSat pushes images; GCS never pulls | If the CubeSat misses a push, the GCS has no way to request it except via the retransmit command | Images can be silently lost if the CubeSat doesn't retry |
| Commands are acknowledged | ACK could be lost even if command was executed | GCS may think command failed when it actually succeeded |
| WiFi connection is stable for the duration of a pass | WiFi is inherently unreliable, especially at distance | Dropped connections during imaging passes |

---

## 7. Image Quality Assumptions

| Assumption | Reality | Impact |
|-----------|---------|--------|
| CubeSat-side quality score (blur, exposure, motion) is trustworthy | CubeSat checks are basic and threshold-based | Poor images may pass CubeSat checks |
| Ground-side checks (texture variance, contrast, color) catch what CubeSat misses | Ground checks are also threshold-based | Some unsuitable images still enter the pipeline |
| A failed quality check means the image is discarded | Some failed images might still contain useful partial information | Wasted bandwidth on images that are then discarded |

---

## 8. System-Level Assumptions

| Assumption | Reality | Impact |
|-----------|---------|--------|
| The GCS runs on a single laptop | Processing is single-threaded for CV pipeline | Slow processing on CPU — pipeline may fall behind if images arrive faster than processing time |
| Mission state persists across restarts | State is saved to JSON files in data/ | If the GCS crashes mid-write, state files could be corrupted |
| All timestamps are from the GCS clock | CubeSat and GCS clocks may not be synchronized | "Before" and "After" timestamps in change detection are GCS-receive-time, not capture-time |
| The operator is watching the dashboard | Many events (changes, quality failures, route impacts) only appear as UI updates | Critical events could be missed if the operator is away |

---

## Summary: What To Fix First

**Priority 1 — Grid Cell Identification Accuracy**
This is the foundation everything else depends on. If cell assignment is wrong, hazard maps are wrong, routes are wrong, change detection compares wrong images. Add IMU metadata from CubeSat + ArUco markers for testbed demos.

**Priority 2 — Change Detection Robustness**
Lighting-induced false positives are the most common failure mode. Add illumination normalization before SSIM comparison.

**Priority 3 — Unsurveyed Cell Handling in Routes**
Routes through unsurveyed cells assume moderate cost. Add a user-configurable "unsurveyed penalty" or block unsurveyed cells entirely.

**Priority 4 — Real-Time Replanning**
When change detection finds a new hazard on an active route, the system warns but doesn't auto-replan. Add automatic route recalculation.
