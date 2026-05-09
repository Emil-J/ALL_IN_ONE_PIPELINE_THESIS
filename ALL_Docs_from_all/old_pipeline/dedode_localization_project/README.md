# GPS-Denied Local Navigation System

**Master Thesis Project - Complete Implementation**

A complete, practical, and well-structured local-area GPS-denied navigation prototype combining:
- **IMU/INS estimation** (Error-State EKF from existing pipeline)
- **DeDoDe v2 visual matching** (geometric feature-based correction)
- **Semantic terrain segmentation** (additional consistency cue for robust matching)

---

## 📋 Project Overview

This system implements **frame-by-frame streaming localization** for a drone operating without GPS in a local area (Vejle, Denmark). 

### System Flow (Per Frame)
1. **IMU Prior** → Get coarse position estimate from INS
2. **Candidate Search** → Find nearby reference tiles within search radius
3. **DeDoDe Matching** → Detect and match visual features geometrically
4. **Semantic Matching** → Compute semantic consistency IoU scores
5. **Combined Scoring** → Rank candidates by geometric + semantic scores
6. **Position Estimate** → Select best match and refine position
7. **Evaluation** → Compare to ground truth and compute errors

---

## 📂 Project Structure

```
dedode_localization_project/
│
├── main_localization_pipeline.ipynb    # Main notebook - run this!
│
├── config/
│   ├── __init__.py
│   └── config.py                       # All configuration and paths
│
├── src/
│   ├── __init__.py
│   ├── io_utils.py                     # File I/O, CSV, JSON, H5 loading
│   ├── image_utils.py                  # Image processing utilities
│   ├── tms_utils.py                    # TMS tile coordinate conversions
│   ├── imu_adapter.py                  # IMU pipeline → streaming wrapper
│   ├── semantic_adapter.py             # Semantic segmentation model loader
│   ├── semantic_matching_utils.py      # IoU, boundary overlap scoring
│   ├── dedode_adapter.py               # DeDoDe v2 wrapper
│   ├── matching_utils.py               # Combined scoring logic
│   ├── localization_utils.py           # Main localization pipeline
│   ├── evaluation_utils.py             # Metrics, success rates, errors
│   └── visualization_utils.py          # Plotting and visualization
│
├── outputs/
│   ├── cache/                          # Cached semantic masks, tile index
│   ├── visualizations/                 # Generated plots
│   ├── matches/                        # Match details (optional)
│   ├── metrics/                        # CSV and JSON evaluation results
│   └── logs/                           # Processing logs
│
└── scripts/
    └── (utility scripts if needed)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install numpy pandas torch torchvision matplotlib tqdm jupyter
pip install opencv-python Pillow scikit-image h5py

# Semantic segmentation
pip install segmentation-models-pytorch albumentations

# DeDoDe (via kornia - recommended)
pip install kornia kornia-rs

# Alternative: standalone DeDoDe (if kornia fails)
pip install git+https://github.com/Parskatt/DeDoDe.git
```

Or use the included requirements file:
```bash
pip install -r requirements.txt

and for torch cuda version this:

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

```

### 2. Verify Paths

Open `config/config.py` and verify all paths point to your data:
- `QUERY_FRAMES_DIR` → Query video frames
- `REFERENCE_TMS_DIR` → Reference tile map
- `REFERENCE_DB_PATH` → H5 database
- `IMU_PIPELINE_DIR` → Existing IMU pipeline
- `SEMANTIC_MODEL_PATH` → Semantic segmentation model weights
- `GROUND_TRUTH_KML_PATH` → KML file with ground truth coordinates (optional)

**Ground Truth Setup (Optional)**:
If you have a KML file with ground truth GPS coordinates:
1. Set `GROUND_TRUTH_KML_PATH` to your KML file path in config.py
2. The notebook will automatically convert it to CSV on first run
3. Ground truth will be used for evaluation metrics

**Manual KML Conversion** (if needed):
```bash
python scripts/convert_kml_to_csv.py coords_fixed_100m_east.kml
# Output: coords_fixed_100m_east_ground_truth.csv
```

The KML file should contain `<Placemark>` entries with:
- `<name>`: Frame filename (e.g., "Capture00022760.jpeg")
- `<coordinates>`: longitude,latitude,altitude

### 3. Run the Notebook

```bash
jupyter notebook main_localization_pipeline.ipynb
```

Run all cells sequentially. The notebook is fully documented.

---

## 📊 Expected Outputs

After running the complete pipeline, you will have:

### Metrics
- `outputs/metrics/per_frame_metrics.csv` - Per-frame localization results with ground truth errors (if GT provided)
- `outputs/metrics/summary.json` - Aggregate statistics and success rates

### Visualizations
- `trajectory_map.png` - IMU vs corrected vs ground truth trajectories
- `error_timeline.png` - Frame-by-frame error evolution
- `error_distribution.png` - Error histograms
- `per_frame/frame_XXXX.png` - Individual frame results (if enabled)

### Data
- `reference_index.csv` - Reference tile database
- `imu_stream_outputs.csv` - IMU estimates for all frames
- `localization_results_full.json` - Complete results with all metadata

---

## ⚙️ Configuration

Key parameters in `config/config.py`:

### Debug Mode
```python
DEBUG_MODE = True              # Limit to first N frames for testing
DEBUG_QUERY_COUNT = 10         # Number of frames to process in debug mode
```

### Localization Parameters
```python
IMU_SEARCH_RADIUS_METERS = 250.0   # Search radius around IMU prior
MAX_CANDIDATE_TILES = 100          # Max tiles to evaluate per frame
TOP_K_MATCHES = 5                  # Return top K matches
```

### DeDoDe Parameters
```python
USE_KORNIA_DEDODE = True           # Use kornia.feature.DeDoDe
NUM_KEYPOINTS = 5000               # Max keypoints per image
IMAGE_SIZE = 560                   # Resize images to this size
RANSAC_REPROJ_THRESH = 4.0         # RANSAC inlier threshold (pixels)
```

### Semantic Parameters
```python
USE_SEMANTICS = True               # Enable semantic matching
SEMANTIC_FILTER_CLASSES = [2,4,5] # land, roads, buildings
SEMANTIC_WEIGHT_IN_FINAL_SCORE = 3.0  # Weight in combined score
```

### Scoring Weights
```python
SCORING_WEIGHTS = {
    "num_inliers": 1.0,            # RANSAC inliers
    "num_matches": 0.2,            # Total matches
    "inlier_ratio": 10.0,          # Inlier/match ratio
    "median_confidence": 2.0,      # Match confidence
    "reprojection_error": -0.5     # Reprojection error (negative weight)
}
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. DeDoDe Import Errors
```bash
pip install kornia kornia-rs --upgrade
```
If still failing, set `USE_KORNIA_DEDODE = False` in config.

#### 2. Semantic Model Not Found
```bash
pip install segmentation-models-pytorch albumentations
```
Verify `SEMANTIC_MODEL_PATH` points to `best.pth`.

#### 3. H5 Database Schema Mismatch
The `load_h5_reference_database()` function tries multiple common field names. If it fails, inspect your H5 file:
```python
import h5py
with h5py.File("reference_database_vejle.h5", 'r') as f:
    print(list(f.keys()))
```
Then update `io_utils.py` to match your schema.

#### 4. IMU Pipeline Integration Issues
The IMU adapter wraps existing code. If it fails:
- Check that `ekf_ins.py` or `dead_reckoning.py` exist
- Verify they return a DataFrame with position estimates
- May need to adapt `IMUEstimatorStream._run_batch_processing()` to your API

#### 5. CUDA Out of Memory
Reduce:
- `NUM_KEYPOINTS` (try 2000)
- `IMAGE_SIZE` (try 384)
- `MAX_CANDIDATE_TILES` (try 50)
Or set `DEVICE = "cpu"`.

#### 6. Low Success Rate
- Increase `IMU_SEARCH_RADIUS_METERS` (try 500m)
- Reduce `MIN_MATCHES_FOR_HOMOGRAPHY` (try 4)
- Increase `RANSAC_REPROJ_THRESH` (try 8.0)
- Check for domain shift between query and reference imagery

---

## 📖 Architecture Details

### Streaming Design
The system processes one frame at a time (frame-by-frame) to simulate real-time inference. This differs from batch processing where all frames are processed upfront.

**IMU Adapter**: Wraps the existing batch-mode IMU pipeline into a streaming `step()` interface. Internally, it still runs the full batch estimation but provides frame-by-frame access.

**Semantic Caching**: Predictions are cached to disk to avoid recomputation on repeated runs.

**Modular Scoring**: DeDoDe geometric scores and semantic consistency scores are computed independently and combined using configurable weights.

### Reused Components
- **IMU Pipeline**: 100% reused from `IMU_Pipeline_Final/` (ekf_ins.py)
- **Semantic Model**: Reused from `SemanticTerrainSegmentationModel/best.pth`
- **Landmark Extraction**: Reused logic from existing localization notebook
- **Database**: Reused `reference_database_vejle.h5` without modification

### New Components
- **DeDoDe Integration**: New visual matching module
- **Combined Scoring**: Unified geometric + semantic scoring
- **Streaming Pipeline**: Frame-by-frame orchestration
- **Comprehensive Evaluation**: Detailed metrics and visualizations

---

## 🎯 Performance Expectations

### Baseline (Semantic Only - from previous tests)
- Mean error: ~2600m
- Median error: ~2500m
- Within 50m: 0.0%
- **Issue**: Domain shift between simulator and TMS imagery

### Expected (With This Pipeline)
- **Target**: Significant improvement over semantic-only approach
- **Goal**: <100m error for >80% of frames
- **Factors**:
  - IMU prior quality
  - Visual feature richness in reference tiles
  - Semantic consistency between domains

### Evaluation Metrics Provided
- Per-frame errors (IMU, corrected, improvement)
- Success rate at multiple thresholds (10m, 25m, 50m, 100m, 250m, 500m)
- Percentile analysis (p25, p50, p75, p90, p95, p99)
- Comparison: IMU vs corrected (improvement/degradation rates)
- Failure analysis (reasons and counts)

---

## 🔬 Next Steps and Extensions

1. **Add Ground Truth**: Load GPS coordinates from flight logs for accurate evaluation
2. **Parameter Tuning**: Grid search over scoring weights and thresholds
3. **Temporal Filtering**: Add Kalman filter or particle filter over time
4. **Position Refinement**: Improve homography-based sub-tile localization
5. **Domain Adaptation**: Fine-tune semantic model on TMS tiles
6. **Test-Time Augmentation**: Try multiple rotations per frame
7. **Multi-Frame Fusion**: Use temporal consistency across frames
8. **Real-Time Optimization**: Profile and optimize bottlenecks

---

## 📚 References

- **DeDoDe**: Edstedt et al., "DeDoDe: Detect, Don't Describe" (CVPR 2024)
- **IMU/INS**: Kok et al., "Using Inertial Sensors for Position and Orientation Estimation" (2017)
- **Semantic Segmentation**: UNet++ with EfficientNet-B3 encoder
- **TMS**: Tile Map Service standard (zoom level 16)

---

## 📝 License

This is a master thesis project. Adapt and extend as needed for your research.

---

## 👤 Author

Emil J. - Master Thesis 2026  
Topic: GPS-Free Drone Localization using Computer Vision

---

## ✅ Validation

The notebook includes a complete validation section (Section 15) that checks:
- Configuration validity
- Component initialization
- Processing completion
- Output generation
- Common failure points and solutions

Run the validation cell to ensure everything is working correctly.

---

**Questions?** Check the troubleshooting section in the notebook (Section 16) or review the inline code documentation.
