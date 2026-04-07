"""
Pipeline 3 Configuration
All paths, parameters, and constants for the localization pipeline.
"""

from pathlib import Path
import math

# ═══════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALL_IN_ONE_ROOT = Path(r"C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline")

# ═══════════════════════════════════════════════════════════════════
# INPUT DATA PATHS
# ═══════════════════════════════════════════════════════════════════

# Reference TMS tileset (Vejle, Denmark - zoom 16)
REFERENCE_TILES_DIR = ALL_IN_ONE_ROOT / "REFERENCE_MAP_VEJLE_20260321_162024" / "aerial"
REFERENCE_PRED_DIR = ALL_IN_ONE_ROOT / "REFERENCE_MAP_VEJLE_20260321_162024" / "prediction"
REFERENCE_METADATA_CSV = ALL_IN_ONE_ROOT / "reference_tiles_metadata.csv"

# Query frames
QUERY_FRAMES_DIR = ALL_IN_ONE_ROOT / "Logs_Run_20260321_162024" / "images_20260321_162024"

# IMU log
IMU_CSV_PATH = ALL_IN_ONE_ROOT / "Logs_Run_20260321_162024" / "imu_gps_log_20260321_162024.csv"

# Semantic model
SEMANTIC_MODEL_PATH = ALL_IN_ONE_ROOT / "SemanticTerrainSegmentationModel" / "best.pth"

# ═══════════════════════════════════════════════════════════════════
# OUTPUT PATHS
# ═══════════════════════════════════════════════════════════════════

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
METATILE_OUTPUT_DIR = OUTPUT_DIR / "metatiles"
SEMANTIC_OUTPUT_DIR = OUTPUT_DIR / "semantic"
LOG_OUTPUT_DIR = OUTPUT_DIR / "logs"
TRAJECTORY_OUTPUT_DIR = OUTPUT_DIR / "trajectories"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

# ═══════════════════════════════════════════════════════════════════
# DEVICE
# ═══════════════════════════════════════════════════════════════════

DEVICE = "cuda"

# ═══════════════════════════════════════════════════════════════════
# TMS / TILE PARAMETERS
# ═══════════════════════════════════════════════════════════════════

TMS_ZOOM_LEVEL = 16
TMS_TILE_SIZE_PX = 512  # pixels per tile
EARTH_RADIUS_METERS = 6371000.0

# Tile coordinate ranges for the reference map
TILE_X_MIN = 34494
TILE_X_MAX = 34508
TILE_Y_MIN = 45025
TILE_Y_MAX = 45042

# Approximate tile size in meters at latitude ~55.7°N, zoom 16
# Formula: (circumference * cos(lat)) / 2^zoom
_LAT_RAD = math.radians(55.7)
TILE_SIZE_METERS = (2 * math.pi * EARTH_RADIUS_METERS * math.cos(_LAT_RAD)) / (2 ** TMS_ZOOM_LEVEL)

# ═══════════════════════════════════════════════════════════════════
# SEMANTIC MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════

SEMANTIC_INPUT_SIZE = 512
SEMANTIC_NUM_CLASSES = 6
SEMANTIC_CLASSES = {
    0: "waterbodies",
    1: "forest_trees",
    2: "land",
    3: "railway",
    4: "roads",
    5: "buildings",
}
COLOR_MAP = {
    0: (4, 4, 255),
    1: (0, 167, 2),
    2: (243, 255, 150),
    3: (193, 105, 53),
    4: (255, 0, 231),
    5: (150, 150, 150),
}

# ═══════════════════════════════════════════════════════════════════
# FEATURE MATCHER (SuperPoint + LightGlue)
# ═══════════════════════════════════════════════════════════════════

MAX_NUM_KEYPOINTS = 4096
MIN_MATCHES_FOR_HOMOGRAPHY = 4
RANSAC_REPROJ_THRESH = 8.0

# ═══════════════════════════════════════════════════════════════════
# PIPELINE 1: BEST-FIRST SEARCH
# ═══════════════════════════════════════════════════════════════════

IMU_SEARCH_RADIUS_METERS = 500.0
MAX_SEARCH_ITERATIONS = 200
TOP_K_CANDIDATES = 5

# ═══════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

# Query frames: 1920x1079 -> resize to 512x288 -> pad to 512x512
QUERY_RESIZE_WIDTH = 512
QUERY_RESIZE_HEIGHT = 288
QUERY_PAD_TOP = 112
QUERY_PAD_BOTTOM = 112

# ═══════════════════════════════════════════════════════════════════
# TEMPORAL PARTICLE FILTER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

NUM_PARTICLES = 100
PROCESS_NOISE_POSITION_M = 5.0
PROCESS_NOISE_HEADING_DEG = 2.0
MEASUREMENT_NOISE_POSITION_M = 500.0   # Large: domain shift makes tile matches unreliable
MEASUREMENT_NOISE_HEADING_DEG = 15.0
RESAMPLE_THRESHOLD = 0.5

TEMPORAL_SEARCH_MAX_ITERATIONS = 50
TEMPORAL_MIN_SEARCH_RADIUS = 0.3       # tiles (~100m)
TEMPORAL_MIN_ROTATION_RANGE = 10.0     # degrees

DIVERGENCE_POSITION_THRESHOLD_M = 500.0
DIVERGENCE_WEIGHT_THRESHOLD = 0.01

PARTICLE_INIT_SPREAD_HIGH_CONF = {"position_meters": 50, "heading_degrees": 10}
PARTICLE_INIT_SPREAD_MED_CONF = {"position_meters": 100, "heading_degrees": 20}
PARTICLE_INIT_SPREAD_LOW_CONF = {"position_meters": 200, "heading_degrees": 30}

# ═══════════════════════════════════════════════════════════════════
# TWO-PASS META-TILE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

FIRST_PASS_SEARCH_RADIUS_M = 500.0
SECOND_PASS_NEIGHBOURS = 8
METATILE_TOP_K = 3
METATILE_MATCH_THRESHOLD = 25  # initial hypothesis — tune from logs

# ═══════════════════════════════════════════════════════════════════
# SEMANTIC CONFIRMATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CENTROID_MATCH_DISTANCE_THRESHOLD_PX = 50
SEMANTIC_CONFIRM_MIN_PAIRS = 3

# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

EVALUATION_THRESHOLDS = [10, 25, 50, 100, 250, 500]  # meters


def ensure_output_dirs():
    """Create all output directories."""
    for d in [OUTPUT_DIR, CACHE_DIR, METATILE_OUTPUT_DIR, SEMANTIC_OUTPUT_DIR,
              LOG_OUTPUT_DIR, TRAJECTORY_OUTPUT_DIR, VISUALIZATION_DIR]:
        d.mkdir(parents=True, exist_ok=True)
