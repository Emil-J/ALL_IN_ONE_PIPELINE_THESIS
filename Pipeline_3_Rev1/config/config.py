"""
Pipeline 3 Configuration
All paths, parameters, and constants for the localization pipeline.
"""

from pathlib import Path
import math
import os

# ═══════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_env_root = os.environ.get("PIPELINE3_DATA_ROOT")
ALL_IN_ONE_ROOT = Path(_env_root) if _env_root else PROJECT_ROOT.parent

# ═══════════════════════════════════════════════════════════════════
# INPUT DATA PATHS
# ═══════════════════════════════════════════════════════════════════

# Reference TMS tileset (Copenhagen, Denmark - zoom 16)
REFERENCE_TILES_DIR = ALL_IN_ONE_ROOT / "REFERENCE_MAP_CPH" / "aerial"
REFERENCE_PRED_DIR = ALL_IN_ONE_ROOT / "REFERENCE_MAP_CPH" / "prediction"
REFERENCE_METADATA_CSV = ALL_IN_ONE_ROOT / "reference_tiles_metadata.csv"

# Query frames
QUERY_FRAMES_DIR = ALL_IN_ONE_ROOT / "Logs_Run_20260321_162024" / "images_20260321_162024"

# IMU log
IMU_CSV_PATH = ALL_IN_ONE_ROOT / "Logs_Run_20260321_162024" / "imu_gps_log_20260321_162024.csv"

# Semantic model
SEMANTIC_MODEL_PATH = ALL_IN_ONE_ROOT / "SemanticTerrainSegmentationModel" / "best.pth"

# Precomputed SuperPoint reference features (HDF5 store from Dataset_Preprocessing)
REFERENCE_FEATURES_PATH = ALL_IN_ONE_ROOT / "REFERENCE_MAP_CPH" /  "reference_features.h5"

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

# Tile coordinate ranges for the reference map (matches tiles on disk)
TILE_X_MIN = 34994
TILE_X_MAX = 35090
TILE_Y_MIN = 44976
TILE_Y_MAX = 45063

# Approximate tile size in meters at latitude ~55.6°N, zoom 16
# Formula: (circumference * cos(lat)) / 2^zoom
_LAT_RAD = math.radians(55.6)
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

MAX_NUM_KEYPOINTS = 2048
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
# PHASE B1: VISUAL MEASUREMENT / ROTATION
# ═══════════════════════════════════════════════════════════════════

MAX_ROTATED_DIMENSION = 1280  # resize rotated query to cap long edge (perf)
QUALITY_GATE_CSHAPE = 0.3     # min shape confidence to trust visual measurement
QUALITY_GATE_INLIERS = 20     # min inlier count to trust visual measurement
NEAR_NADIR_THRESHOLD_RAD = 0.087  # ~5 degrees — use nadir_corrected when pitch/roll below this

# ═══════════════════════════════════════════════════════════════════
# ONLINE EKF VISUAL POSITION UPDATE
# ═══════════════════════════════════════════════════════════════════

VISUAL_POSITION_NOISE_M = 50.0     # R_pos std-dev: how much we trust visual measurements
POSITION_PROCESS_NOISE_M = 5.0     # Q_pos std-dev: how fast position uncertainty grows per √s
INITIAL_POSITION_VARIANCE_M = 200.0  # P[8:10] init: starting position uncertainty

# ═══════════════════════════════════════════════════════════════════
# SEMANTIC CONFIRMATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CENTROID_MATCH_DISTANCE_THRESHOLD_PX = 50
SEMANTIC_CONFIRM_MIN_PAIRS = 3

# Semantic histogram pre-filter (Phase 4)
# Run SemanticTileScorer on all first-pass candidates and keep only the
# top-K before feeding to SuperPoint+LightGlue.  Disabled by default
# until timing is verified on the target machine.
SEMANTIC_PREFILTER_ENABLED = True
SEMANTIC_PREFILTER_TOP_K = 10

# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

EVALUATION_THRESHOLDS = [10, 25, 50, 100, 250, 500]  # meters

# ═══════════════════════════════════════════════════════════════════
# DEPLOYMENT FLAGS
# ═══════════════════════════════════════════════════════════════════

# Set True in notebook/debug only — saves PNG meta-tile every frame (~30-50ms overhead)
DEBUG_SAVE_METATILES = False

# Set True in notebook/analysis only — appends every frame result to history list
# (grows unbounded in long runs; not needed for pure runtime)
ACCUMULATE_HISTORY = False

# ═══════════════════════════════════════════════════════════════════
# OPTIONAL DATA CAPTURE FLAGS  (all independent — mix and match)
# ═══════════════════════════════════════════════════════════════════

# Save query frame JPEG per frame to:
#   outputs/runs/<run_id>/flight_data/frame_NNNN.jpg
# Use when: you want to visually review what the drone saw.
# Timing impact: ~10-20 ms/frame (JPEG encode + write).
SAVE_QUERY_FRAMES = False

# Save raw IMU row JSON per frame to:
#   outputs/runs/<run_id>/flight_data/frame_NNNN_imu.json
# Use when: you want the raw sensor data for offline re-processing.
# Timing impact: <1 ms/frame.
SAVE_IMU_ROWS = False

# Save PX4 GPS_INPUT CSV (MAVLink MSG 232) to:
#   outputs/runs/<run_id>/px4_gps_input.csv
# Also saves per-frame analysis extras (n_eff, particle_spread, homo offsets) to:
#   outputs/runs/<run_id>/analysis_extras.csv
# Use when: you need PX4 integration output or deeper analysis beyond results.csv.
# Timing impact: <1 ms/frame.
SAVE_ANALYSIS_DATA = False

# Save per-component inference timing to:
#   outputs/runs/<run_id>/timing_data.csv
# Use when: you want a timing breakdown per pipeline component.
# Timing impact: <0.1 ms/frame (time.perf_counter calls only).
SAVE_TIMING_DATA = False

# Save full per-frame pipeline trace to:
#   outputs/runs/<run_id>/pipeline_data/frame_NNNN/
# Saves per frame: query.jpg, query_rotated.jpg, semantic_mask.png,
#   reference_tile.png, matches.png, imu.json, trace.json
# trace.json contains: EKF state before/after, PF state, tile candidate table,
#   homography quality, semantic conf, gate decision — every step A-to-Z.
# Use when: you need step-by-step data for analysis or research figures.
# Implies: SAVE_QUERY_FRAMES + SAVE_IMU_ROWS behaviour (saved to pipeline_data/).
# Timing impact: ~80-150 ms/frame (image encoding + multiple disk writes).
# Intended for short runs or selected-frame analysis — not full 970-frame runs.
SAVE_PIPELINE_TRACE = True

# ═══════════════════════════════════════════════════════════════════
# RUNTIME OUTPUT PATHS
# ═══════════════════════════════════════════════════════════════════

RUNS_OUTPUT_DIR = OUTPUT_DIR / "runs"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "analysis"


def ensure_output_dirs():
    """Create all output directories."""
    for d in [OUTPUT_DIR, CACHE_DIR, METATILE_OUTPUT_DIR, SEMANTIC_OUTPUT_DIR,
              LOG_OUTPUT_DIR, TRAJECTORY_OUTPUT_DIR, VISUALIZATION_DIR,
              RUNS_OUTPUT_DIR, ANALYSIS_OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
