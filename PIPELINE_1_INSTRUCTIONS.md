# PIPELINE 1: BEST-FIRST SEMANTIC-GUIDED TILE SEARCH
## Complete Implementation Instructions for Claude Opus 4.6

---

## MISSION OVERVIEW

You are building a GPS-denied drone localization system using best-first search (A*-inspired algorithm). Given a query frame from MSFS and an IMU position estimate (potentially 50-500m off), you must find the matching reference tile in a TMS database at zoom level 16.

**Core Strategy**: Use semantic fingerprints as cheap heuristics to prioritize expensive geometric matching (SuperPoint+SuperGlue). Expand intelligently from IMU prior, test most promising tiles first, stop when confident.

**Implementation Approach**: Build incrementally. After each module, validate thoroughly. Test each function. Reason through correctness. Only proceed when current step verified.

---

## WORKSPACE CONTEXT

### Files Already Present in Workspace

The user will provide the following in the workspace:

1. **Semantic Segmentation Model**
   - Model architecture and weights
   - Information about classes, input/output format
   
2. **IMU Estimator Pipeline**
   - EKF implementation
   - How to read IMU estimates (position, heading, covariance)
   
3. **Feature Matching Models**
   - SuperPoint + SuperGlue OR DeDoDe implementation
   - How to run matching
   
4. **Reference Tile Dataset**
   - TMS tiles at zoom 16
   - Directory structure
   - Tile naming convention

### USER INPUT SECTION - CRITICAL INFORMATION

**USER: Please provide the following information about your workspace:**

```
it is very important to mention that some of the columns in imu gps log are being used as ground truth so it is very important to stick with what is being used in EKF_ins because what is being used there is what is the actual sensor values not simulator ground truth values.

=== SEMANTIC MODEL INFORMATION ===
Model file path: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\SemanticTerrainSegmentationModel\best.pth
Model architecture (UNet++, SegFormer, etc.): UNet++ with EfficientNet-B3 + scSE attention
Number of classes path: 6
Class names and indices path: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\SemanticTerrainSegmentationModel\legend.txt
Input image size: 512x512 pixels
Output format (logits, probabilities, class indices): mask of aerial image 512x512 pixels
How to load and run inference: I reconsctruct architecture in memory & pour the saved weights into the architecture. Everything about model is available here: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\SemanticTerrainSegmentationModel


=== IMU ESTIMATOR INFORMATION ===
Below questions are hard to answer, the code will need to be analyzed in detail first and be created based on my requirement.
My requirement is frame by frame and gps log by gps log corresponding to each frame. we get the frame we get the log, we then use it in this manner as real-time. The logs and the frames are currently all available, but we need to measure the inference time of running the algorithms all in one.
Frames: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\Logs_Run_20260321_162024\images_20260321_162024
IMU_GPS_LOG from MSFS2020: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\Logs_Run_20260321_162024\imu_gps_log_20260321_162024.csv
All algorithms for estimator: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\MSFS2020_IMU_Pipeline
ekf_ins is EKF estimator we want to use, data logger is the python code for obtainin all logs and frames, this will need to be modified so it takes a frame by frame and imu gps log by imu gps log from the Logs_Run_20260321_162024 folder then.
IMU log file format (.csv columns):
How to read position estimate (lat, lon):
How to read heading estimate:
How to read position covariance (uncertainty):
How to read heading uncertainty:
Example code to load IMU data:


=== FEATURE MATCHER INFORMATION ===
Using SuperPoint+SuperGlue or DeDoDe:
Model weights location:
How to initialize matcher:
Input image format (grayscale/RGB, size):
Output format (keypoints, descriptors, matches):
Example code to run matching:


=== REFERENCE TILES INFORMATION ===
Tiles directory path:
Directory structure (e.g., zoom/x/y.png):
Tile image format (.png, .jpg):
Tile size in pixels:
Zoom level:
Geographic coverage (lat/lon bounds):
Total number of tiles:


=== QUERY FRAMES INFORMATION ===
Query frames directory: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\Logs_Run_20260321_162024\images_20260321_162024
Frame naming convention: Logs_Run_20260321_162024\images_20260321_162024\frame_0.523.jpg frame_timestamp.jpg is the convention. To find the match between the frame and the imu gps log, we look at the timestamp. C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\Logs_Run_20260321_162024\imu_gps_log_20260321_162024.csv

timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,latitude,longitude,altitude,pitch,bank,heading,pressure_altitude,barometer_pressure,heading_magnetic,magnetic_compass,airspeed_indicated,airspeed_true,ground_velocity,vertical_speed,ap_master,ap_alt_hold,ap_airspeed_hold,ap_nav_hold,throttle_pos
0.021384716033935547,-1.1473736921700188,-0.023356076732570407,0.25925249965650027,0.0020434188333624476,-0.0188590901522448,-0.0037643509571449814,55.637686341348946,9.646494670656745,1659.7195401058282,-0.0126539422389413,0.07452804837460124,0.6524763376261318,505.8960416015625,953.9335327148438,0.5816409013908155,39.14772415161133,63.059431791442876,64.60575063748169,65.24435005045738,0.019751603704690895,1.0,1.0,0.0,1.0,100.0
0.5229570865631104,-1.1338921640089075,-0.5145437703052318,0.24165344165731062,-0.005916015863394788,-0.020710927700973533,-0.005620399138723439,55.637920141618146,9.64679610280539,1659.7600723829016,-0.012473129651327792,0.07384684445012761,0.6495645937250715,505.90512011718755,953.9325561523438,0.578757240343933,38.952308654785156,63.114502010589604,64.65472941766357,65.28380096365738,0.015539164379239051,1.0,1.0,0.0,1.0,100.0
Check why the first timestamp doesn't have an image, there must be a reason and explanation in the code. I am not sure why, but I assume it is because of the intial GPS location being taken from gorund truth and then the second one is estimated. algorithm works in this manner we know intital GPS.
Image size (width x height):1920x1079 furthermore if model takes 512x512, cropping will take out too much detail from image, so we migth have to resize the images and fill in background without image content with a color black so the model doesn't get photometrically wrong images.
Frame rate (fps):Not sure you have timestamps for each frame, i assume 2fps
Associated ground truth file (if any): For each frame as I said we have C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\Logs_Run_20260321_162024\imu_gps_log_20260321_162024.csv
inside we have columns latitude longitude which contain GPS coordinates for each taken frame, the thing is frames have rounded up or rounded down numbers(not quite sure, you have to check) but csv has whole timestamp with all decimals, so keep that in mind as a potential problem since the third number digit after the decimal separator might not always be the same.


=== ADDITIONAL CONTEXT ===
Any special preprocessing needed: Read all the documentation from all of the pipelines here: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\ALL_Docs_from_all
Furthermore if in doubt about the model training process look here into the ipynb used to obtain the model: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\SemanticTerrainSegmentationModel\Semantic_Model_QGIS_8_Class_Rev6.ipynb
If more curious about how DeDoDe vs Superoint Superglue comparison was done, and to see what you need to do to make your code work look for clues in this ipynb: C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\feature_matching_comparison_v2.ipynb
If in need of my personal notes from start of project to now here is the .txt file you can read easily, at the start of the .txt is start of my thesis then it goes down chronologically over a longer time period:C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\ALL_Docs_from_all\notes from me.txt
Known issues or limitations:
Performance requirements:
```

**After the user fills this out, proceed with implementation using their specific details.**

---

## PROJECT STRUCTURE

Create the following structure in the workspace:

```
best_first_localization/
├── config/
│   ├── __init__.py
│   └── config.py                  # All configuration parameters
├── src/
│   ├── __init__.py
│   ├── tile_utils.py              # TMS coordinate conversions, tile loading
│   ├── semantic_fingerprint.py    # Semantic fingerprint extraction
│   ├── heuristic.py               # Priority calculation for best-first
│   ├── geometric_matcher.py       # SuperPoint+SuperGlue wrapper
│   ├── best_first_search.py       # Core search algorithm
│   └── position_estimator.py      # Homography → GPS conversion
├── preprocessing/
│   ├── __init__.py
│   └── build_semantic_cache.ipynb # Generate fingerprints for all tiles
├── notebooks/
│   └── test_pipeline.ipynb        # End-to-end testing
├── tests/
│   ├── __init__.py
│   ├── test_tile_utils.py
│   ├── test_semantic_fingerprint.py
│   ├── test_heuristic.py
│   └── test_best_first_search.py
├── outputs/
│   ├── visualizations/
│   ├── results/
│   └── logs/
├── requirements.txt
└── README.md
```

**CRITICAL**: After creating structure, verify every directory exists. Run: `tree best_first_localization` or equivalent.

---

## ENVIRONMENT SETUP

### Step 1: Virtual Environment

```bash
# Create virtual environment with Python 3.14.2
python3.14 -m venv venv

# Activate
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Verify
python --version  # Must be 3.14.2
```

### Step 2: Install PyTorch with CUDA 12.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### Step 3: Install Dependencies

Create `requirements.txt`:
```
numpy>=2.0.0
opencv-python>=4.9.0
Pillow>=10.2.0
scikit-image>=0.22.0
scipy>=1.12.0
pandas>=2.2.0
tqdm>=4.66.0
pyyaml>=6.0.1
h5py>=3.10.0
matplotlib>=3.8.0
seaborn>=0.13.0
kornia>=0.7.1
pytest>=8.0.0
jupyter>=1.0.0
ipykernel>=6.29.0
```

Install: `pip install -r requirements.txt`

**Validation**: Run `pip list` and verify all packages present. Test imports in Python.

---

## MODULE IMPLEMENTATION ORDER

Implement in this exact order. After each module, validate completely before proceeding.

---

## MODULE 1: config/config.py

**Purpose**: Central configuration. All paths, parameters, constants.

**Implementation Requirements**:

1. **Import Section**
   - pathlib.Path for all paths
   - torch for device configuration
   - typing for type hints

2. **Path Configuration**
   - PROJECT_ROOT: Auto-detect from __file__
   - REFERENCE_TILES_DIR: Use user's provided path
   - QUERY_FRAMES_DIR: Use user's provided path
   - SEMANTIC_CACHE_DIR: Where to store cached fingerprints (auto-created)
   - OUTPUTS_DIR: For results, visualizations, logs (auto-created)
   - Model paths: Use user-provided paths for semantic model, feature matcher

3. **Device Configuration**
   - DEVICE: Auto-detect CUDA availability
   - TORCH_DTYPE: torch.float32

4. **TMS Configuration**
   - ZOOM_LEVEL: 16 (use user's value if different)
   - TILE_SIZE_PIXELS: Use user's provided tile size
   - TILE_SIZE_METERS: ~350.0 at zoom 16, latitude ~55°N
   - SUPERTILE_GRID_SIZE: 5 (for 5×5 supertiles)

5. **Semantic Configuration**
   - Use user-provided class count and names
   - SEMANTIC_ACTIVE_CLASSES: Exclude "unknown" and any merged classes

6. **Fingerprint Configuration**
   - GLOBAL_HISTOGRAM_BINS: 10 bins per class
   - SPATIAL_GRID_SIZE: 50×50 spatial grid
   - EDGE_ORIENTATION_BINS: 8 orientation bins

7. **Best-First Search Configuration**
   - MAX_SEARCH_RADIUS_TILES: 2.0 (~700m)
   - MAX_ITERATIONS: 200
   - CONFIDENCE_THRESHOLD: 150.0

8. **Heuristic Weights**
   - HEURISTIC_WEIGHT_SPATIAL: 0.3
   - HEURISTIC_WEIGHT_SEMANTIC: 500.0
   - HEURISTIC_WEIGHT_HEADING: 100.0

9. **Rotation Search Configuration**
   - MIN_ROTATION_RANGE_DEG: 30.0
   - ROTATION_SIGMA_MULTIPLIER: 3.0
   - ROTATION_STEP_DEG: 10.0

10. **Geometric Matching Configuration**
    - Use user's feature matcher config
    - MIN_MATCHES_FOR_HOMOGRAPHY: 8
    - RANSAC_REPROJ_THRESHOLD: 4.0
    - RANSAC_CONFIDENCE: 0.995
    - RANSAC_MAX_ITERS: 2000

11. **Scoring Weights**
    ```python
    SCORING_WEIGHTS = {
        "num_inliers": 1.0,
        "inlier_ratio": 50.0,
        "mean_confidence": 20.0,
        "reproj_error": -0.5,
    }
    ```

12. **Helper Functions**
    - `get_rotation_range(heading_sigma_deg)`: Calculate search range
    - `create_output_directories()`: Auto-create all output dirs

13. **Auto-Initialization**
    - Call `create_output_directories()` on module import

**Validation After Implementation**:
```python
from config import config
print(f"Project root: {config.PROJECT_ROOT}")
print(f"Device: {config.DEVICE}")
print(f"Rotation range for 5° uncertainty: {config.get_rotation_range(5.0)}")
assert config.ZOOM_LEVEL == 16
assert config.SUPERTILE_GRID_SIZE == 5
assert config.CONFIDENCE_THRESHOLD > 0
print("✓ Config validation passed")
```

---

## MODULE 2: src/tile_utils.py

**Purpose**: TMS coordinate conversions, tile loading, supertile stitching.

**Required Functions**:

### 2.1: lat_lon_to_tile(lat, lon, zoom)
- Convert GPS coordinates to TMS tile coordinates
- Use Web Mercator projection
- Formula: 
  - `n = 2^zoom`
  - `tile_x = int((lon + 180) / 360 * n)`
  - `tile_y = int((1 - asinh(tan(lat_rad)) / π) / 2 * n)`
- Return: (tile_x, tile_y)
- Validate: bounds check (0 ≤ x,y < n)

### 2.2: tile_to_lat_lon(tile_x, tile_y, zoom)
- Convert tile coordinates to GPS (tile center)
- Inverse of above
- Return: (lat, lon)

### 2.3: euclidean_tile_distance(tile1, tile2)
- Compute Euclidean distance in tile units
- tile1, tile2 are (x, y) tuples
- Return: sqrt((x1-x2)² + (y1-y2)²)

### 2.4: get_tile_path(tile_x, tile_y, zoom)
- Construct filesystem path for tile
- Use user's directory structure from input section
- Return: Path object

### 2.5: load_tile(tile_x, tile_y, zoom)
- Load single tile image
- Handle missing files gracefully (return None, log warning)
- Convert BGR→RGB if using cv2.imread
- Validate tile size matches config
- Resize if mismatch
- Return: np.ndarray (H, W, 3) uint8 or None

### 2.6: build_supertile(center_x, center_y, grid_size, zoom)
- Stitch N×N tile mosaic centered on (center_x, center_y)
- For 5×5: load tiles from (center_x-2, center_y-2) to (center_x+2, center_y+2)
- Initialize black mosaic: (grid_size × tile_size, grid_size × tile_size, 3)
- Place each tile at correct position
- Handle missing tiles: leave black, log warning
- If >20% tiles missing: return None
- Return: np.ndarray mosaic or None

### 2.7: rotate_image(img, angle_deg)
- Rotate image around center by angle_deg
- Use cv2.getRotationMatrix2D
- Positive angle = counter-clockwise
- Border: constant black (0,0,0)
- Return: rotated image (same size as input)

**Validation After Implementation**:
```python
from src import tile_utils
from config import config

# Test coordinate conversion
lat, lon = 55.7, 9.5  # Vejle area
tx, ty = tile_utils.lat_lon_to_tile(lat, lon, 16)
lat2, lon2 = tile_utils.tile_to_lat_lon(tx, ty, 16)
assert abs(lat - lat2) < 0.01
assert abs(lon - lon2) < 0.01
print(f"✓ Coordinate conversion: {lat},{lon} → {tx},{ty} → {lat2},{lon2}")

# Test distance
dist = tile_utils.euclidean_tile_distance((100, 100), (103, 104))
assert abs(dist - 5.0) < 0.01
print(f"✓ Distance calculation: {dist}")

# Test tile loading (use a known existing tile)
tile = tile_utils.load_tile(tx, ty, 16)
if tile is not None:
    assert tile.shape == (config.TILE_SIZE_PIXELS, config.TILE_SIZE_PIXELS, 3)
    print(f"✓ Tile loading: {tile.shape}")
else:
    print("⚠ Tile not found (check reference data path)")

# Test supertile
supertile = tile_utils.build_supertile(tx, ty, 3, 16)
if supertile is not None:
    expected = 3 * config.TILE_SIZE_PIXELS
    assert supertile.shape == (expected, expected, 3)
    print(f"✓ Supertile stitching: {supertile.shape}")

# Test rotation
rotated = tile_utils.rotate_image(np.ones((512, 512, 3), dtype=np.uint8) * 128, 45)
assert rotated.shape == (512, 512, 3)
print("✓ Image rotation")

print("✓ All tile_utils validations passed")
```

---

## MODULE 3: src/semantic_fingerprint.py

**Purpose**: Extract semantic fingerprints from segmented masks.

**Implementation Requirements**:

### 3.1: Load Semantic Model
- Function: `load_semantic_model(model_path, device)`
- Use user's provided model loading code
- Set to eval mode
- Move to device
- Return: model

### 3.2: segment_image(image, model, device)
- Input: RGB image (H, W, 3) numpy array
- Preprocess according to user's model requirements
- Run inference
- Get predicted class mask (H, W) with class indices
- Return: mask as np.ndarray (H, W) dtype=uint8

### 3.3: compute_global_histogram(mask, num_classes, bins_per_class)
- For each active class:
  - Extract binary mask (mask == class_idx)
  - Compute histogram over pixel values (0-255 range, bins_per_class bins)
- Concatenate all class histograms
- Normalize to sum=1
- Return: 1D array of length (num_classes × bins_per_class)

### 3.4: compute_spatial_histogram(mask, num_classes, grid_size)
- Divide mask into grid_size × grid_size cells
- For each cell:
  - Count pixels of each class
  - Normalize by cell area
- Result: (grid_size, grid_size, num_classes) array
- Flatten to 1D
- Optional: Apply PCA to compress (25000 → 500 dimensions)
- Return: 1D array

### 3.5: compute_edge_orientation_histogram(mask, num_bins)
- Convert mask to edges using Canny or gradient
- Compute gradient orientation at each edge pixel
- Bin orientations into num_bins (e.g., 8 bins for 0-360°)
- Normalize histogram
- Return: 1D array of length num_bins

### 3.6: extract_fingerprint(image, model, device)
- High-level function combining all above
- Segment image
- Compute global histogram
- Compute spatial histogram
- Compute edge histogram
- Concatenate all features
- Return: complete fingerprint vector

### 3.7: save_fingerprint(fingerprint, tile_x, tile_y, zoom)
- Save fingerprint to cache directory
- Format: SEMANTIC_CACHE_DIR / f"{zoom}_{tile_x}_{tile_y}.npy"
- Use np.save

### 3.8: load_fingerprint(tile_x, tile_y, zoom)
- Load cached fingerprint
- Return None if not found
- Return: fingerprint array

### 3.9: cosine_similarity(fp1, fp2)
- Compute cosine similarity between two fingerprints
- Formula: dot(fp1, fp2) / (norm(fp1) × norm(fp2))
- Return: similarity score [0, 1]

**Validation After Implementation**:
```python
from src import semantic_fingerprint
from config import config
import numpy as np

# Load model
model = semantic_fingerprint.load_semantic_model(config.SEMANTIC_MODEL_PATH, config.DEVICE)
print(f"✓ Model loaded on {config.DEVICE}")

# Test on dummy image
dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
mask = semantic_fingerprint.segment_image(dummy_img, model, config.DEVICE)
assert mask.shape == (512, 512)
assert mask.dtype == np.uint8
print(f"✓ Segmentation: {mask.shape}, unique classes: {np.unique(mask)}")

# Test fingerprint extraction
fp = semantic_fingerprint.extract_fingerprint(dummy_img, model, config.DEVICE)
assert len(fp.shape) == 1
assert fp.shape[0] > 100  # Should be ~600 dimensional
print(f"✓ Fingerprint extraction: {fp.shape}")

# Test save/load
semantic_fingerprint.save_fingerprint(fp, 1000, 2000, 16)
fp_loaded = semantic_fingerprint.load_fingerprint(1000, 2000, 16)
assert np.allclose(fp, fp_loaded)
print("✓ Fingerprint save/load")

# Test similarity
fp2 = np.random.rand(fp.shape[0])
fp2 = fp2 / np.linalg.norm(fp2)
sim = semantic_fingerprint.cosine_similarity(fp, fp2)
assert 0 <= sim <= 1
print(f"✓ Cosine similarity: {sim}")

print("✓ All semantic_fingerprint validations passed")
```

---

## MODULE 4: src/heuristic.py

**Purpose**: Compute priority scores for best-first search.

**Implementation Requirements**:

### 4.1: compute_spatial_component(candidate_tile, imu_tile)
- candidate_tile, imu_tile: (x, y) tuples
- Compute Euclidean distance in tile units
- Convert to meters: distance_tiles × TILE_SIZE_METERS
- Return: spatial_distance_meters

### 4.2: compute_semantic_component(query_fingerprint, ref_fingerprint)
- Compute cosine similarity
- Invert: semantic_score = 1.0 - similarity
  - Because lower priority = better in min-heap
  - High similarity → low score → high priority
- Return: semantic_score

### 4.3: compute_heading_component(imu_heading, ref_dominant_orientation)
- ref_dominant_orientation: Optional, can be None
- If None: return 0.0
- Compute angular difference (handle wraparound at 360°)
- Normalize to [0, 1]: diff / 180.0
- Return: heading_score

### 4.4: compute_priority(candidate_tile, query_fingerprint, imu_tile, imu_heading, ref_fingerprint, ref_orientation=None)
- Compute all three components
- Weighted sum:
  ```
  priority = (
      spatial_component × HEURISTIC_WEIGHT_SPATIAL +
      semantic_component × HEURISTIC_WEIGHT_SEMANTIC +
      heading_component × HEURISTIC_WEIGHT_HEADING
  )
  ```
- Lower priority = more promising = test sooner
- Return: priority (float)

**Validation After Implementation**:
```python
from src import heuristic
from config import config
import numpy as np

# Test spatial
imu_tile = (1000, 2000)
candidate = (1003, 2004)
spatial = heuristic.compute_spatial_component(candidate, imu_tile)
expected_dist = 5.0 * config.TILE_SIZE_METERS
assert abs(spatial - expected_dist) < 1.0
print(f"✓ Spatial component: {spatial}m")

# Test semantic
fp1 = np.random.rand(100)
fp1 = fp1 / np.linalg.norm(fp1)
fp2 = fp1.copy()  # Identical
semantic = heuristic.compute_semantic_component(fp1, fp2)
assert abs(semantic - 0.0) < 0.01  # 1 - 1.0 = 0
print(f"✓ Semantic component (identical): {semantic}")

fp3 = np.random.rand(100)
fp3 = fp3 / np.linalg.norm(fp3)
semantic2 = heuristic.compute_semantic_component(fp1, fp3)
assert semantic2 > 0  # Not identical
print(f"✓ Semantic component (different): {semantic2}")

# Test heading
heading = heuristic.compute_heading_component(90, 95)
assert 0 <= heading <= 1
print(f"✓ Heading component: {heading}")

# Test priority
priority = heuristic.compute_priority(
    candidate, fp1, imu_tile, 90, fp2, 95
)
assert priority > 0
print(f"✓ Priority: {priority}")

print("✓ All heuristic validations passed")
```

---

## MODULE 5: src/geometric_matcher.py

**Purpose**: Wrapper for SuperPoint+SuperGlue or DeDoDe matching.

**Implementation Requirements**:

### 5.1: Initialize Matcher
- Function: `initialize_matcher(device)`
- Use user's provided initialization code
- Load SuperPoint+SuperGlue or DeDoDe
- Set to eval mode
- Return: matcher object(s)

### 5.2: detect_and_match(image1, image2, matcher, device)
- image1, image2: RGB numpy arrays
- Convert to format expected by matcher (grayscale, tensor, etc.)
- Detect keypoints in both images
- Match descriptors
- Return: dict with:
  - `keypoints1`: Nx2 array
  - `keypoints2`: Nx2 array
  - `matches`: Mx2 array of indices
  - `confidences`: M array of match confidences

### 5.3: compute_homography(kp1, kp2, matches, method, threshold, confidence, max_iters)
- Extract matched keypoint coordinates
- If len(matches) < MIN_MATCHES_FOR_HOMOGRAPHY: return None
- Run cv2.findHomography with RANSAC
- Return: dict with:
  - `H`: 3×3 homography matrix or None
  - `inlier_mask`: boolean array
  - `num_inliers`: int
  - `reproj_errors`: array of reprojection errors for inliers

### 5.4: compute_geometric_score(match_result, homography_result)
- Extract metrics:
  - num_matches = len(matches)
  - num_inliers = homography_result['num_inliers']
  - inlier_ratio = num_inliers / num_matches
  - mean_confidence = mean(confidences[inlier_mask])
  - reproj_error = mean(homography_result['reproj_errors'])
- Compute score:
  ```
  score = (
      num_inliers × SCORING_WEIGHTS['num_inliers'] +
      inlier_ratio × SCORING_WEIGHTS['inlier_ratio'] +
      mean_confidence × SCORING_WEIGHTS['mean_confidence'] +
      reproj_error × SCORING_WEIGHTS['reproj_error']
  )
  ```
- Return: score (float)

### 5.5: match_and_score(query_image, reference_image, matcher, device)
- High-level function
- Run detect_and_match
- Run compute_homography
- If homography valid: compute score
- Return: dict with all results and score

**Validation After Implementation**:
```python
from src import geometric_matcher
from config import config
import numpy as np

# Initialize matcher
matcher = geometric_matcher.initialize_matcher(config.DEVICE)
print("✓ Matcher initialized")

# Test with dummy images
img1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
img2 = img1.copy()  # Identical should give perfect match

match_result = geometric_matcher.detect_and_match(img1, img2, matcher, config.DEVICE)
assert 'keypoints1' in match_result
assert 'matches' in match_result
print(f"✓ Matching: {len(match_result['matches'])} matches")

# Test homography
if len(match_result['matches']) >= 8:
    H_result = geometric_matcher.compute_homography(
        match_result['keypoints1'],
        match_result['keypoints2'],
        match_result['matches'],
        'RANSAC', 4.0, 0.995, 2000
    )
    if H_result['H'] is not None:
        print(f"✓ Homography: {H_result['num_inliers']} inliers")
    
    # Test scoring
    score = geometric_matcher.compute_geometric_score(match_result, H_result)
    print(f"✓ Geometric score: {score}")

print("✓ All geometric_matcher validations passed")
```

---

## MODULE 6: src/position_estimator.py

**Purpose**: Convert homography to GPS position estimate.

**Implementation Requirements**:

### 6.1: homography_to_position(H, query_center, supertile_center_tile, grid_size, zoom)
- query_center: (x, y) in query image pixel coordinates (typically image center)
- Apply homography: ref_pixel = H @ query_center
- ref_pixel is in supertile pixel coordinates
- Convert to tile coordinates:
  - supertile covers grid_size × grid_size tiles
  - center tile is at grid_size//2 offset
  - tile_offset_x = (ref_pixel_x / TILE_SIZE_PIXELS) - grid_size//2
  - tile_offset_y = (ref_pixel_y / TILE_SIZE_PIXELS) - grid_size//2
  - estimated_tile_x = supertile_center_tile[0] + tile_offset_x
  - estimated_tile_y = supertile_center_tile[1] + tile_offset_y
- Convert tile coordinates to lat/lon
- Return: (lat, lon)

### 6.2: estimate_heading_from_homography(H)
- Decompose homography to extract rotation component
- Use cv2.decomposeHomographyMat or manual decomposition
- Extract rotation angle in degrees
- Return: heading (0-360°)

**Validation After Implementation**:
```python
from src import position_estimator
from src import tile_utils
import numpy as np

# Test identity homography (no transformation)
H = np.eye(3)
query_center = (1280, 720)  # Typical image center
supertile_center = (1000, 2000)
grid_size = 5

lat, lon = position_estimator.homography_to_position(
    H, query_center, supertile_center, grid_size, 16
)
# Should return center tile lat/lon
expected_lat, expected_lon = tile_utils.tile_to_lat_lon(1000, 2000, 16)
assert abs(lat - expected_lat) < 0.01
assert abs(lon - expected_lon) < 0.01
print(f"✓ Position estimation: {lat}, {lon}")

# Test heading
heading = position_estimator.estimate_heading_from_homography(H)
assert 0 <= heading < 360
print(f"✓ Heading estimation: {heading}°")

print("✓ All position_estimator validations passed")
```

---

## MODULE 7: src/best_first_search.py

**Purpose**: Core best-first search algorithm.

**Implementation Requirements**:

### 7.1: Class BestFirstSearcher

**Initialization**:
- `__init__(query_frame, imu_lat, imu_lon, imu_heading, imu_pos_sigma, imu_heading_sigma)`
- Store all parameters
- Load semantic model
- Initialize feature matcher
- Segment query frame, extract fingerprint
- Convert IMU lat/lon to tile coordinates
- Compute rotation search range using config.get_rotation_range(imu_heading_sigma)
- Initialize priority queue (heapq)
- Initialize closed set (set of tested tiles)
- Initialize best match tracker

**Main Search Method**:
- `search() -> dict`
- While queue not empty AND iterations < MAX_ITERATIONS:
  1. Pop tile with lowest priority
  2. Skip if in closed set
  3. Add to closed set
  4. Test tile (call self._test_tile)
  5. Update best match if better
  6. Check early termination (score > CONFIDENCE_THRESHOLD)
  7. Expand neighbors (call self._expand_neighbors)
- Return best match dict

**Helper Methods**:

### 7.2: _test_tile(tile_x, tile_y)
- Build 5×5 supertile
- If None: return None
- For each rotation angle in search range:
  - Rotate supertile
  - Match with query
  - Compute homography
  - If valid: compute score
  - Track best rotation
- Return: best result for this tile

### 7.3: _expand_neighbors(tile_x, tile_y)
- For 8-connected neighbors: (-1,-1), (-1,0), ..., (1,1)
- Skip if in closed set
- Skip if distance from IMU > MAX_SEARCH_RADIUS_TILES
- Load neighbor fingerprint
- Compute priority
- Add to queue

### 7.4: _visualize_search(save_path)
- Optional: visualize tiles tested, priorities, final match
- Save to outputs/visualizations/

**Validation After Implementation**:
```python
from src import best_first_search
import numpy as np

# Load a real query frame
query_img = ... # Load from query frames dir
imu_lat, imu_lon, imu_heading = ... # From IMU log

# Run search
searcher = best_first_search.BestFirstSearcher(
    query_frame=query_img,
    imu_lat=imu_lat,
    imu_lon=imu_lon,
    imu_heading=imu_heading,
    imu_pos_sigma=100.0,  # meters
    imu_heading_sigma=5.0  # degrees
)

result = searcher.search()

if result is not None:
    print(f"✓ Best match found:")
    print(f"  Tile: {result['tile']}")
    print(f"  Score: {result['score']}")
    print(f"  Position: {result['position']}")
    print(f"  Tiles tested: {result['tiles_tested']}")
    print(f"  Time: {result['search_time']}")
else:
    print("⚠ No match found")

print("✓ Best-first search validation complete")
```

---

## PREPROCESSING NOTEBOOK: preprocessing/build_semantic_cache.ipynb

**Purpose**: Pre-compute semantic fingerprints for all reference tiles.

**Cells**:

1. **Setup**: Import modules, load config
2. **Load Model**: Load semantic segmentation model
3. **Get Tile List**: Scan reference tiles directory, list all tiles
4. **Progress Bar**: Setup tqdm for progress tracking
5. **Processing Loop**:
   - For each tile:
     - Check if fingerprint already cached (skip if exists)
     - Load tile image
     - Extract fingerprint
     - Save to cache
6. **Validation**: Load random cached fingerprints, verify shape
7. **Statistics**: Report total tiles processed, cache size

**Critical**: Run this notebook BEFORE running main pipeline.

---

## TESTING NOTEBOOK: notebooks/test_pipeline.ipynb

**Purpose**: End-to-end testing of complete pipeline.

**Cells**:

1. **Setup**: Imports, config loading
2. **Load Test Data**: Load query frame, IMU data, ground truth
3. **Run Search**: Execute BestFirstSearcher
4. **Visualize Results**: Show matched tile, homography overlay
5. **Compute Error**: Compare estimated position to ground truth
6. **Performance Analysis**: Timing breakdown, tiles tested
7. **Multiple Frames**: Test on 10 frames, aggregate statistics

---

## UNIT TESTS

Create pytest tests for each module in `tests/` directory.

**Example test_tile_utils.py**:
```python
import pytest
from src import tile_utils

def test_coordinate_conversion():
    lat, lon = 55.7, 9.5
    tx, ty = tile_utils.lat_lon_to_tile(lat, lon, 16)
    lat2, lon2 = tile_utils.tile_to_lat_lon(tx, ty, 16)
    assert abs(lat - lat2) < 0.01
    assert abs(lon - lon2) < 0.01

def test_distance():
    dist = tile_utils.euclidean_tile_distance((0, 0), (3, 4))
    assert abs(dist - 5.0) < 0.01

# Add more tests...
```

Run: `pytest tests/ -v`

---

## EXECUTION CHECKLIST

After implementing all modules:

1. ✓ All imports work
2. ✓ Config loads correctly
3. ✓ Tile loading works (test on existing tiles)
4. ✓ Semantic model loads and runs
5. ✓ Feature matcher initializes
6. ✓ Fingerprint extraction works
7. ✓ Heuristic computation works
8. ✓ Homography computation works
9. ✓ Run preprocessing notebook (build cache)
10. ✓ Run test notebook (end-to-end)
11. ✓ Run pytest suite
12. ✓ Analyze results, tune parameters if needed

---

## CRITICAL REMINDERS

1. **Validate After Each Step**: Don't proceed until current module passes all tests
2. **Use Existing Code**: Integrate user's provided semantic model, IMU reader, feature matcher
3. **Handle Errors Gracefully**: Missing tiles, failed matches, invalid homographies
4. **Log Everything**: Use Python logging for debugging
5. **Save Intermediate Results**: For debugging and analysis
6. **Type Hints**: Use typing annotations everywhere
7. **Docstrings**: Document every function with inputs, outputs, behavior
8. **Code Review**: After each module, re-read code, check for logical errors
9. **Test Incrementally**: Don't write all code then test - test as you go

---

## EXPECTED OUTPUT

After successful implementation, running on a query frame should produce:

```
Best-First Search Results:
  IMU Prior: (55.7123, 9.5456)
  Estimated Position: (55.7089, 9.5423)
  Best Tile: (34495, 20494)
  Rotation: 185°
  Match Score: 234.5
  Inliers: 456 / 523 matches
  Tiles Tested: 23
  Iterations: 31
  Search Time: 4.2s
  Position Error: 45.6m (if ground truth available)
```

---

## DEBUGGING TIPS

If search fails:
- Check semantic cache exists
- Verify tile loading works
- Test matcher on identical images (should get high score)
- Visualize heuristic priorities
- Check rotation range (too narrow?)
- Verify RANSAC parameters
- Inspect supertile stitching (missing tiles?)

If search is slow:
- Profile code (cProfile)
- Check semantic cache hits
- Verify GPU usage for model inference
- Consider reducing SUPERTILE_GRID_SIZE to 3×3

If poor accuracy:
- Tune CONFIDENCE_THRESHOLD
- Adjust heuristic weights
- Check domain shift (MSFS vs real tiles)
- Verify semantic model quality
- Test with more rotations (smaller step size)

---

END OF PIPELINE 1 INSTRUCTIONS
