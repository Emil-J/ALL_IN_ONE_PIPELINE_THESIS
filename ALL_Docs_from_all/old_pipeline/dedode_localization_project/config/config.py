"""
Configuration for GPS-Denied Local Navigation System
All absolute paths point to existing data in All_In_One_Pipeline
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════

# Project root (this dedode_localization_project folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Parent directory containing all datasets
ALL_IN_ONE_ROOT = Path(r"C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline")

# ═══════════════════════════════════════════════════════════════════
# INPUT DATA PATHS (EXISTING DATASETS)
# ═══════════════════════════════════════════════════════════════════

# Query frames (drone flight route at 1200ft AMSL, 15fps)
QUERY_FRAMES_DIR = ALL_IN_ONE_ROOT / "REFERENCE MAP CROPPED" / "aerial"
QUERY_PRED_DIR = ALL_IN_ONE_ROOT / "REFERENCE MAP CROPPED" / "prediction"  # Optional precomputed cache

# Reference TMS tileset (Vejle, Denmark - zoom 16)
REFERENCE_TMS_DIR = ALL_IN_ONE_ROOT / "REFERENCE_MAP_VEJLE" / "aerial"
REFERENCE_PRED_DIR = ALL_IN_ONE_ROOT / "REFERENCE_MAP_VEJLE" / "prediction"  # Optional precomputed cache

# Reference database - use CSV metadata (H5 file doesn't have metadata columns)
REFERENCE_DB_PATH = ALL_IN_ONE_ROOT / "reference_tiles_metadata.csv"

# IMU pipeline directory (contains ekf_ins.py, dead_reckoning.py, etc.)
IMU_PIPELINE_DIR = ALL_IN_ONE_ROOT / "IMU_Pipeline_Final"

# ── IMU Prior (simulated) ──
# ModifiedGPS.kml contains GPS coordinates with ~100m offset from ground truth,
# simulating what an IMU dead-reckoning output would look like.
IMU_PRIOR_KML_PATH = ALL_IN_ONE_ROOT / "ModifiedGPS.kml"

# ── Ground Truth ──
# GroundTruthGPS.kml contains the actual GPS coordinates from the flight path.
GROUND_TRUTH_KML_PATH = ALL_IN_ONE_ROOT / "GroundTruthGPS.kml"

# Auto-converted CSV outputs (generated from KML on first run)
GROUND_TRUTH_CSV_PATH = None  # Set to CSV path if you have a pre-made CSV
GROUND_TRUTH_AUTO_CSV = ALL_IN_ONE_ROOT / "GroundTruthGPS_ground_truth.csv"
IMU_PRIOR_AUTO_CSV = ALL_IN_ONE_ROOT / "ModifiedGPS_imu_prior.csv"

# ═══════════════════════════════════════════════════════════════════
# SEMANTIC SEGMENTATION MODEL
# ═══════════════════════════════════════════════════════════════════

SEMANTIC_MODEL_PATH = ALL_IN_ONE_ROOT / "SemanticTerrainSegmentationModel" / "best.pth"
SEMANTIC_INPUT_SIZE = 512  # Model trained on 256x256
SEMANTIC_CLASSES = {
    0: "waterbodies",
    1: "forest_trees",
    2: "land",
    3: "railway",
    4: "roads",
    5: "buildings"
}
# Classes stable across domains (avoid forest/water due to domain shift)
SEMANTIC_FILTER_CLASSES = [0, 1, 2, 3, 4, 5]  # land, roads, buildings
SEMANTIC_MIN_AREA = 100  # Minimum pixel area for landmark extraction

# Color map for visualization
COLOR_MAP = {
    0: (4, 4, 255),        # waterbodies - blue
    1: (0, 167, 2),        # forest_trees - green
    2: (243, 255, 150),    # land - yellow
    3: (193, 105, 53),     # railway - brown
    4: (255, 0, 231),      # roads - magenta
    5: (150, 150, 150)     # buildings - gray
}

# ═══════════════════════════════════════════════════════════════════
# OUTPUT PATHS
# ═══════════════════════════════════════════════════════════════════

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
SEMANTIC_CACHE_DIR = CACHE_DIR / "semantic"
TILES_INDEX_CACHE_DIR = CACHE_DIR / "tiles_index"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
MATCHES_DIR = OUTPUT_DIR / "matches"
METRICS_DIR = OUTPUT_DIR / "metrics"
LOGS_DIR = OUTPUT_DIR / "logs"

# ═══════════════════════════════════════════════════════════════════
# PROCESSING PARAMETERS
# ═══════════════════════════════════════════════════════════════════

# Debug mode (process limited frames for testing)
DEBUG_MODE = True
DEBUG_QUERY_COUNT = 10  # Process only first N frames in debug mode

# Device
DEVICE = "cuda"  # or "cpu"

# ═══════════════════════════════════════════════════════════════════
# LOCALIZATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════

# IMU parameters
IMU_SEARCH_RADIUS_METERS = 350.0  # Local search radius around IMU prior
MAX_CANDIDATE_TILES = 9  # Maximum tiles to consider per frame

# Image processing
IMAGE_SIZE = 560  # Resize images to this size (DeDoDe recommendation)
NUM_KEYPOINTS = 5000  # Number of keypoints for DeDoDe

# Matching parameters
TOP_K_MATCHES = 5  # Return top K candidates
MIN_MATCHES_FOR_HOMOGRAPHY = 4  # Minimum matches required for homography
RANSAC_REPROJ_THRESH = 8.0  # RANSAC reprojection threshold (pixels)

# ═══════════════════════════════════════════════════════════════════
# DEDODE PARAMETERS
# ═══════════════════════════════════════════════════════════════════

USE_KORNIA_DEDODE = False  # kornia API changed - using standalone DeDoDe instead

# DeDoDe model weights
DEDODE_DETECTOR_WEIGHTS = "L-upright"  # Options: "L-upright", "L-C4"
DEDODE_DESCRIPTOR_WEIGHTS = "B-upright"  # Options: "B-upright", "B-C4"

# Matching
DEDODE_MATCH_THRESHOLD = 0.2  # Mutual nearest neighbor threshold

# ═══════════════════════════════════════════════════════════════════
# SEMANTIC MATCHING PARAMETERS
# ═══════════════════════════════════════════════════════════════════

USE_SEMANTICS = True  # Enable semantic consistency scoring

# Semantic score components (weights for individual metrics)
SEMANTIC_SCORE_WEIGHTS = {
    "iou": 1.0,           # Intersection over Union
    "boundary": 0.5       # Boundary overlap
}

# Weight of semantic score in final combined score
SEMANTIC_WEIGHT_IN_FINAL_SCORE = 3.0

# ═══════════════════════════════════════════════════════════════════
# SCORING WEIGHTS
# ═══════════════════════════════════════════════════════════════════

# Combined scoring weights for candidate ranking
SCORING_WEIGHTS = {
    "num_inliers": 1.0,           # Number of RANSAC inliers (higher is better)
    "num_matches": 0.2,           # Total number of matches
    "inlier_ratio": 10.0,         # Inlier ratio (inliers/matches)
    "median_confidence": 2.0,     # Median match confidence
    "reprojection_error": -0.5    # Median reprojection error (lower is better)
}

# ═══════════════════════════════════════════════════════════════════
# REFERENCE TMS PARAMETERS
# ═══════════════════════════════════════════════════════════════════

# TileMetaData parameters
TMS_ZOOM_LEVEL = 16
TMS_TILE_SIZE = 512  # Standard TMS tile size in pixels

# Earth radius for coordinate conversions
EARTH_RADIUS_METERS = 6371000.0

# ═══════════════════════════════════════════════════════════════════
# EVALUATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════

# Distance thresholds for success rate computation
EVALUATION_THRESHOLDS = [10, 25, 50, 100, 250, 500]  # meters

# ═══════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_config():
    """Validate that all required paths exist"""
    required_paths = {
        "QUERY_FRAMES_DIR": QUERY_FRAMES_DIR,
        "REFERENCE_TMS_DIR": REFERENCE_TMS_DIR,
        "REFERENCE_DB_PATH": REFERENCE_DB_PATH,
        "IMU_PRIOR_KML_PATH": IMU_PRIOR_KML_PATH,
        "GROUND_TRUTH_KML_PATH": GROUND_TRUTH_KML_PATH,
        "SEMANTIC_MODEL_PATH": SEMANTIC_MODEL_PATH,
    }
    
    missing = []
    for name, path in required_paths.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
    
    if missing:
        raise FileNotFoundError(
            f"Missing required paths:\n" + "\n".join(f"  - {m}" for m in missing)
        )
    
    # Ensure output directories exist
    for out_dir in [OUTPUT_DIR, CACHE_DIR, SEMANTIC_CACHE_DIR, TILES_INDEX_CACHE_DIR,
                    VISUALIZATION_DIR, MATCHES_DIR, METRICS_DIR, LOGS_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    return True


# Auto-validate on import (can be disabled if needed)
if __name__ != "__main__":
    try:
        validate_config()
    except FileNotFoundError as e:
        print(f"⚠️  Configuration validation warning: {e}")
        print("   Some paths may not exist yet. This is okay if you're setting up.")
