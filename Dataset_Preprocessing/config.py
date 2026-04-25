"""
Dataset Preprocessing — Configuration.

Paths, parameters, and constants for offline reference-map preprocessing.
All paths are relative to the ALL_IN_ONE_ROOT workspace root.
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════

ALL_IN_ONE_ROOT = Path(r"C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline")
PREPROCESSING_ROOT = ALL_IN_ONE_ROOT / "Dataset_Preprocessing"

# ═══════════════════════════════════════════════════════════════════
# INPUT PATHS
# ═══════════════════════════════════════════════════════════════════

# Reference TMS tileset — aerial imagery
REFERENCE_AERIAL_DIR = ALL_IN_ONE_ROOT / "REFERENCE_MAP_CPH" / "aerial"

# Semantic model
SEMANTIC_MODEL_PATH = ALL_IN_ONE_ROOT / "SemanticTerrainSegmentationModel" / "best.pth"

# ═══════════════════════════════════════════════════════════════════
# OUTPUT PATHS
# ═══════════════════════════════════════════════════════════════════

# Semantic prediction tiles — matches existing structure used by pipeline
REFERENCE_PREDICTION_DIR = ALL_IN_ONE_ROOT / "REFERENCE_MAP_CPH" / "prediction"

# SuperPoint precomputed features — HDF5 store
REFERENCE_FEATURES_PATH = ALL_IN_ONE_ROOT / "REFERENCE_MAP_CPH" / "reference_features.h5"

# ═══════════════════════════════════════════════════════════════════
# TMS PARAMETERS
# ═══════════════════════════════════════════════════════════════════

TMS_ZOOM_LEVEL = 16
TMS_TILE_SIZE_PX = 512

# ═══════════════════════════════════════════════════════════════════
# SUPERPOINT PARAMETERS (must match runtime pipeline config exactly)
# ═══════════════════════════════════════════════════════════════════

MAX_NUM_KEYPOINTS = 2048
DEVICE = "cuda"

# Descriptor storage dtype.  "float32" is safe (native SuperPoint output).
# "float16" halves storage but must be validated against matching accuracy.
DESCRIPTOR_DTYPE = "float32"

# ═══════════════════════════════════════════════════════════════════
# SEMANTIC MODEL PARAMETERS (must match pipeline)
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
