# config.py — Ground Station Configuration
# All physical setup values marked 0.0 MUST be measured from the real physical
# setup and filled in before demo day. See docs/ARCHITECTURE.md section 15.

# === NETWORK ===
LISTEN_PORT = 5000
COMMAND_PORT = 5001
LISTEN_HOST = "0.0.0.0"
CUBESAT_IP = "192.168.1.229"       # Fill in: CubeSat's real IP on the shared network

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
COST_IMPASSABLE = 999

# === HAZARD CLASSIFICATION THRESHOLDS ===
LBP_VARIANCE_HIGH = 500        # LBP variance above this → rocky/hazardous texture
LBP_VARIANCE_MODERATE = 200    # LBP variance above this → moderate texture
EDGE_DENSITY_HIGH = 0.15       # Canny edge density above this → rough terrain
EDGE_DENSITY_MODERATE = 0.08   # Canny edge density above this → moderate terrain

# === CHANGE DETECTION ===
CHANGE_THRESHOLD = 30              # Pixel difference (0-255) to count as changed
CHANGE_MIN_AREA_PX = 50            # Min contiguous changed pixels to report as an event

# === ROUTE PLANNING ===
GRID_ROWS = 8                      # DEPRECATED — kept for legacy references
GRID_COLS = 8                      # DEPRECATED — kept for legacy references
ROUTE_START = None                 # Set via dashboard click on mosaic
ROUTE_END = None                   # Set via dashboard click on mosaic
GRID_CELL_SIZE_CM = 10.0           # Physical size of each grid cell (cm)

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

# === LLM (optional) ===
LLM_MODEL = "llama3.2"
