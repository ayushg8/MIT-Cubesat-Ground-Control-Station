# config.py — Ground Station Configuration
# All physical setup values marked 0.0 MUST be measured from the real physical
# setup and filled in before demo day. See docs/ARCHITECTURE.md section 15.

# === NETWORK ===
LISTEN_PORT = 5000
COMMAND_PORT = 5001
LISTEN_HOST = "0.0.0.0"
CUBESAT_IP = "127.0.0.1"       # Fill in: CubeSat's real IP on the shared network

# === STORAGE ===
RECEIVED_DIR = "data/received_images"
PROCESSED_DIR = "data/processed"
TELEMETRY_DIR = "data/telemetry"
MISSION_STATE_FILE = "data/mission_state.json"

# === GROUND QUALITY CHECK ===
# These checks are DIFFERENT from the CubeSat's checks.
# CubeSat checks: blur (Laplacian variance), exposure (mean brightness), motion blur (IMU).
# Ground checks: texture sufficiency, contrast range, color validity.
# Purpose: catch images the CubeSat passed but that will break the CV pipeline.
# TCP guarantees delivery — no point re-checking blur/exposure here.
GROUND_MIN_TEXTURE_VARIANCE = 20   # Avg local variance across 8x8 patches must exceed this
GROUND_MIN_CONTRAST_RANGE = 50     # Grayscale histogram must span at least this many levels
GROUND_MAX_SINGLE_COLOR_PCT = 90   # If >90% of pixels are in a narrow band, image is invalid

# === HAZARD COSTS ===
COST_SAFE = 1
COST_MODERATE = 5
COST_SHADOW = 15
COST_HAZARD = 20
COST_CRATER = 500                  # Near-impassable — only traverse if no other path exists
COST_IMPASSABLE = 999

# === HAZARD CLASSIFICATION THRESHOLDS ===
LBP_VARIANCE_HIGH = 5.0        # LBP variance above this → rocky/hazardous texture
LBP_VARIANCE_MODERATE = 3.0    # LBP variance above this → moderate texture
EDGE_DENSITY_HIGH = 0.02       # Canny edge density above this → rough terrain
EDGE_DENSITY_MODERATE = 0.005  # Canny edge density above this → moderate terrain

# === CHANGE DETECTION ===
CHANGE_THRESHOLD = 30              # Pixel difference (0-255) to count as changed
CHANGE_MIN_AREA_PX = 50            # Min contiguous changed pixels to report as an event

# === ROUTE PLANNING ===
GRID_ROWS = 8                      # DEPRECATED — kept for legacy references
GRID_COLS = 8                      # DEPRECATED — kept for legacy references
ROUTE_START = None                 # Set via dashboard click on mosaic
ROUTE_END = None                   # Set via dashboard click on mosaic
GRID_CELL_SIZE_CM = 10.0           # Physical size of each grid cell (cm)

# === PIXEL SEGMENTATION ===
SEG_GRID_CELL_PX = 20              # Fine grid cell size in pixels (high-res route planning)
SEG_SAFETY_DILATION_PX = 3         # Dilate hazard masks by this many pixels as safety margin
SEG_MIN_CONTOUR_AREA_PCT = 5.0     # Min contour area as % of bbox to keep (noise filter)
SEG_FALLBACK_ELLIPSE_PCT = 60.0    # If no good contour, fill this % ellipse within bbox
SEG_ENABLED = True                 # Feature flag — False reverts to coarse grid routing
SEG_COST_MAP = {                   # Semantic label → traversal cost for A* routing
    0: COST_SAFE,                  # UNSURVEYED — assume passable
    1: COST_SAFE,                  # SAND — safe
    2: COST_SAFE,                  # PLAIN_SURFACE — safe
    3: COST_SHADOW,                # SHADOW — uncertain, high cost
    4: COST_CRATER,                  # CRATER — near-impassable, last resort only
    5: COST_IMPASSABLE,            # BOULDER — impassable
}

# === MOSAIC ===
MOSAIC_GRID_CELL_PX = 80           # Each dynamic grid cell = this many mosaic pixels
MOSAIC_INITIAL_CANVAS_PX = 640     # Initial canvas size (square)
MOSAIC_PX_PER_CM = 8.0             # Mosaic pixels per centimetre (calibration)
MOSAIC_MIN_SIFT_INLIERS = 6       # Minimum RANSAC inliers for a valid match (low for sandy terrain)
MOSAIC_CANVAS_PAD_PX = 100         # Padding added when canvas grows
MOSAIC_MAX_CANVAS_PX = 4096        # Memory cap for canvas dimensions
MOSAIC_MAX_KEYPOINTS = 1024        # SuperPoint max keypoints per image
MOSAIC_BUNDLE_ADJUST_INTERVAL = 3  # Run bundle adjustment every N images
MOSAIC_BLEND_LEVELS = 4            # Laplacian pyramid levels for multi-band blend
MOSAIC_EXPOSURE_GAIN_RANGE = (0.5, 2.0)  # Clamp exposure gain to this range

# === DASHBOARD ===
DASHBOARD_PORT = 3000
DASHBOARD_REFRESH_SEC = 2

# === UNCERTAINTY-AWARE COSTS ===
UNCERTAINTY_WEIGHT = 3.0            # cost multiplier for low confidence
UNSURVEYED_COST = 15                # base cost for unknown terrain (= SHADOW level)
CONFIDENCE_OBS_BOOST = 0.1          # confidence boost per additional observation
UNCERTAINTY_ENABLED = True

# === SHADOW-BASED SLOPE ESTIMATION ===
SUN_ELEVATION_DEG = 30.0            # degrees above horizon
SUN_AZIMUTH_DEG = 180.0             # degrees from N, clockwise
SLOPE_GENTLE_DEG = 15.0             # no penalty
SLOPE_MODERATE_DEG = 30.0           # 2x cost
SLOPE_STEEP_DEG = 45.0              # 5x cost, above = impassable
SLOPE_ENABLED = True

# === LANDING SITE RECOMMENDER ===
LANDING_MIN_RADIUS_CM = 5.0          # Minimum safe zone radius around landing point
LANDING_CANDIDATE_STRIDE = 2         # Sample every Nth fine grid cell (performance tuning)
LANDING_TOP_K = 3                    # Return top K candidates
LANDING_WEIGHTS = {                  # Scoring weights (sum ≈ 1.0)
    "hazard_clearance": 0.25,
    "zone_size": 0.15,
    "confidence": 0.10,
    "flatness": 0.15,
    "route_viability": 0.15,
    "smoothness": 0.20,              # Terrain roughness — prefer smooth landing zones
}
LANDING_MIN_CLEARANCE_CM = 3.0       # Hard reject if nearest hazard < this

# === TRAVERSABILITY CNN ===
CNN_MODEL_PATH = "models/traversability_model.pt"
CNN_PATCH_SIZE = 64
CNN_BLEND_WEIGHT = 0.4              # final = alpha*CNN + (1-alpha)*classical
CNN_ENABLED = True                  # model trained on existing data
CNN_BATCH_SIZE = 32

# === LLM (optional — mission Q&A via Ollama only; no advisor pipeline) ===
LLM_MODEL = "llama3.2"
LLM_TIMEOUT_SEC = 60

# === MISSION INTELLIGENCE ===
MISSION_MAX_TASKS = 5
MISSION_FEED_LIMIT = 25
MISSION_WEIGHT_COVERAGE = 0.30
MISSION_WEIGHT_UNCERTAINTY = 0.25
MISSION_WEIGHT_CHANGE = 0.20
MISSION_WEIGHT_ROUTE = 0.15
MISSION_WEIGHT_HAZARD = 0.10
