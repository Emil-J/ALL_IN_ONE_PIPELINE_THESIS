# Dataset Preprocessing

Offline preprocessing pipeline for the reference tile map used by **Pipeline 3 Rev1**
(GPS-free drone localization). Extracts and stores SuperPoint features and semantic
segmentation predictions for all reference TMS tiles, so the online pipeline can skip
expensive per-tile feature extraction at runtime.

## Quick Start

```bash
# Activate the project venv
.final_Pipeline_venv\Scripts\activate        # Windows
# source .final_Pipeline_venv/bin/activate   # Linux

# Run ALL preprocessing (semantic + SuperPoint)
python -m Dataset_Preprocessing.preprocess_reference --all

# Run only SuperPoint feature extraction
python -m Dataset_Preprocessing.preprocess_reference --superpoint

# Run only semantic segmentation
python -m Dataset_Preprocessing.preprocess_reference --semantic

# Force re-processing (overwrite existing)
python -m Dataset_Preprocessing.preprocess_reference --all --force
```

## What It Does

### 1. Semantic Preprocessing (`semantic_preprocessor.py`)
- Loads the UNet++ EfficientNet-B3 model (`SemanticTerrainSegmentationModel/best.pth`)
- Runs inference on every aerial tile in the TMS directory
- Saves RGB-encoded class prediction masks to `prediction/` (same TMS structure)
- **Incremental**: skips tiles whose prediction already exists on disk

### 2. SuperPoint Feature Extraction (`superpoint_preprocessor.py`)
- Runs SuperPoint keypoint + descriptor extraction (max 2048 keypoints per tile)
- Stores all features in a single HDF5 file (`reference_features.h5`)
- **Incremental**: skips tiles already in the HDF5 store
- Output format per tile: `keypoints (N,2)`, `descriptors (N,256)`, `scores (N,)`, `image_size (2,)`

### 3. Feature Store (`feature_store.py`)
- **`FeatureStoreWriter`**: HDF5 writer with gzip compression, incremental support, metadata
- **`FeatureStoreLoader`**: Read-only loader used by the online pipeline — opens file once,
  returns PyTorch tensors formatted for LightGlue input `{keypoints: (1,N,2), descriptors: (1,N,D), keypoint_scores: (1,N), image_size: (1,2)}`
- **`validate_feature_store()`**: Integrity checker that samples random tiles

## Pipeline Integration

The online pipeline (`Pipeline_3_Rev1/`) automatically detects and uses the precomputed
feature store when available:

1. **Notebook Cell 3** loads `FeatureStoreLoader` if `config.REFERENCE_FEATURES_PATH` exists
2. **`TemporalSearcher`** passes `feature_store` to `MetaTileBuilder` and `BestFirstSearcher`
3. **`MetaTileBuilder.first_pass()`** and **`second_pass()`** call
   `matcher.match_both_precomputed(query_feats, ref_feats)` instead of loading the tile image
4. **Fallback**: If a tile is not in the store, the pipeline falls back to runtime extraction

### Performance Impact
- **Without precomputed features**: ~33 SuperPoint extractions per frame (~2.0s/frame on GPU)
- **With precomputed features**: 0 reference extractions per frame — only 1 query extraction
- Expected speedup: ~40-60% faster per frame

## File Layout

```
Dataset_Preprocessing/
├── __init__.py                  # Package marker
├── config.py                    # All paths, parameters, constants
├── feature_store.py             # HDF5 writer + loader + validator
├── superpoint_preprocessor.py   # SuperPoint extraction CLI
├── semantic_preprocessor.py     # Semantic segmentation CLI
├── preprocess_reference.py      # Unified driver (--all/--semantic/--superpoint)
├── reference_features.h5        # [Generated] HDF5 feature store
└── README.md                    # This file
```

## Configuration

All parameters are in `config.py` and mirror the pipeline's `config/config.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `MAX_NUM_KEYPOINTS` | 2048 | Must match pipeline config exactly |
| `TMS_ZOOM_LEVEL` | 16 | TMS zoom level |
| `DEVICE` | `"cuda"` | Auto-falls back to CPU if CUDA unavailable |
| `DESCRIPTOR_DTYPE` | `"float32"` | Can use `"float16"` to halve storage |

## HDF5 Store Structure

```
reference_features.h5
├── metadata/
│   ├── creation_date
│   ├── device, max_keypoints, zoom_level, ...
│   └── tile_index (JSON: {"tx_ty": [tx, ty], ...})
├── 34482_45003/
│   ├── keypoints     (N, 2)  float32
│   ├── descriptors   (N, 256) float32
│   ├── scores        (N,)    float32
│   └── image_size    (2,)    int32
├── 34482_45004/
│   └── ...
└── ...  (one group per tile)
```

## Requirements

Same as the main pipeline — no additional dependencies:
- `torch`, `lightglue` (for SuperPoint)
- `segmentation_models_pytorch`, `albumentations` (for semantic model)
- `h5py`, `numpy`, `Pillow`
