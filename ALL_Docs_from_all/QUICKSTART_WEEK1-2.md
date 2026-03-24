# 🚀 QUICK START: GPS-Free Localization (Week 1-2)

**Status**: ✅ Ready to run immediately with your REFERENCE_MAP_VEJLE dataset!

---

## What Was Created

I've implemented the complete Week 1-2 pipeline based on the MDPI paper. Here's what you have now:

### 📦 Core Modules (`localization/` package)

1. **`centroid_extractor.py`** - Extracts landmark centroids using `cv2.moments()` (MDPI paper method)
2. **`feature_vector.py`** - Computes distance ratios + angle differences (Equations 1-4)
3. **`database_builder.py`** - Builds searchable reference database from TMS tiles
4. **`utils.py`** - TMS coordinate conversion, visualization tools

### 📓 Notebooks & Scripts

1. **`Build_Reference_Database.ipynb`** ⭐ - Main notebook (run this first!)
2. **`test_localization.py`** - Verify installation
3. **`LOCALIZATION_ROADMAP.md`** - Full 10-week implementation plan
4. **`localization/README.md`** - Detailed documentation

---

## How to Run (3 Steps)

### Step 1: Install Dependencies

```powershell
# Install localization requirements
pip install h5py opencv-python
```

*Note: Your existing `requirements.txt` already has numpy, matplotlib, pillow, etc.*

### Step 2: Verify Installation

```powershell
cd "C:\Users\emilj\Documents\Thesis\TRAINING"
python test_localization.py
```

**Expected output**:
```
✓✓✓ ALL TESTS PASSED! ✓✓✓
```

### Step 3: Build Reference Database

```powershell
jupyter notebook Build_Reference_Database.ipynb
```

**This notebook will**:
1. Load your 300 tiles from `REFERENCE_MAP_VEJLE/prediction/`
2. Extract centroids using `cv2.moments()` (MDPI paper method)
3. Compute topological feature vectors (distance ratios + angles)
4. Build searchable database: `localization_output/reference_database_vejle.h5`
5. Test query and show top matches

**Runtime**: ~2-5 minutes for 300 tiles

---

## What You'll Get

### Outputs in `localization_output/`:

1. **`reference_database_vejle.h5`** (~10-50 MB)
   - Searchable reference map database
   - ~287 entries (tiles with sufficient landmarks)
   - 350-dimensional feature vectors per tile

2. **Visualizations** (PNG files):
   - `sample_tile_comparison.png` - Aerial vs segmentation
   - `sample_landmarks.png` - Extracted centroids overlay
   - `sample_triplet.png` - Example landmark triplet with distances
   - `query_results.png` - Query and top matches

### Database Statistics (Expected):
```
Total entries: ~287
Total landmarks: ~5,000-10,000
Avg landmarks per tile: ~25-35
Feature dimension: 350 (50 triplets × 7 features)
```

---

## Understanding the Output

### Feature Vector Structure

Each tile is represented by a **350-dimensional vector**:

```
50 triplets × 7 features = 350 dimensions

Each triplet contributes:
  [d_12/d_13, d_23/d_13, θ_213, θ_312, class_1, class_2, class_3]
   |          |          |       |       |        |        |
   Distance   Distance   Angle   Angle   Semantic labels
   ratios     ratios     (norm)  (norm)  (buildings, roads, etc.)
```

**Why this works**:
- **Distance ratios**: Scale-invariant (works at different altitudes)
- **Angles**: Rotation-invariant (works at different headings)
- **Semantic labels**: Discriminative (reduces false matches)

### Query Example

```python
# Load database
db = ReferenceDatabase("localization_output/reference_database_vejle.h5")

# Query with drone image features
matches = db.query_by_features(drone_features, top_k=10)

# Best match gives you position!
best_match = matches[0]
position = (best_match['metadata']['lat'], best_match['metadata']['lon'])
```

---

## Troubleshooting

### Issue: "Module not found"
```powershell
# Make sure you're in the TRAINING directory
cd "C:\Users\emilj\Documents\Thesis\TRAINING"
python test_localization.py
```

### Issue: "No tiles found"
Check that paths match your system:
```python
PREDICTION_DIR = Path(r"C:\Users\emilj\Documents\Thesis\TRAINING\REFERENCE_MAP_VEJLE\prediction")
```

### Issue: "Not enough landmarks"
Lower the `MIN_LANDMARK_AREA` parameter:
```python
MIN_LANDMARK_AREA = 50  # Default was 100
```

---

## Next Steps After Week 1-2

Once you have the reference database built:

### Week 3-4: Matching Pipeline
- Implement real-time query processing
- Test with different drone image crops
- Benchmark accuracy and speed

### Week 5-6: Rotation/Scale Invariance
- Multi-scale database
- Rotation augmentation
- Robustness testing

### Week 7-8: Jetson Optimization
- TensorRT model conversion
- FP16 quantization
- Target: 10-20 FPS

See `LOCALIZATION_ROADMAP.md` for detailed timeline.

---

## 📊 Validation Checklist

After running the notebook, verify:

- [ ] Database file created: `reference_database_vejle.h5`
- [ ] Database has >200 entries
- [ ] Sample visualizations look correct
- [ ] Query test returns matches with distance ≈0 for self-match
- [ ] Top matches visually similar to query

---

## 🎯 Week 1-2 Success Criteria

✅ **You're done when**:
1. `test_localization.py` passes all tests
2. Notebook runs without errors
3. Database file exists and is >5 MB
4. Query returns sensible matches
5. Visualizations show landmarks correctly

---

## Need Help?

Check these files:
1. `localization/README.md` - Detailed module documentation
2. `LOCALIZATION_ROADMAP.md` - Full project plan
3. `Build_Reference_Database.ipynb` - Example usage

---

**Ready to start? Run this**: 
```powershell
python test_localization.py
```

Then open: `Build_Reference_Database.ipynb`

**Good luck! 🚀**
