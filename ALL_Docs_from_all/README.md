# IMU Estimator - MSFS Connection

GPS-free drone localization using IMU data from Microsoft Flight Simulator 2020.

## Setup

1. **Install Python dependencies:**
   ```powershell
   .\.venv10032026\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Configure MSFS 2020:**
   - Install MSFS 2020
   - Launch the simulator
   - **No manual setup needed** - the scripts handle everything automatically!

## Quick Start - Run Everything

```powershell
python run_pipeline.py
```

This master script runs the entire pipeline:
1. **Data Collection** - Auto-spawns at Vejle, engages autopilot, logs data
2. **Dead Reckoning** - Integrates IMU to estimate position
3. **Evaluation** - Compares estimates vs ground truth, generates plots

### Pipeline Configuration

Edit the config at the top of [run_pipeline.py](run_pipeline.py):

```python
RUN_DATA_LOGGER    = True   # Set False to skip data collection
RUN_DEAD_RECKONING = True   # Set False to skip dead reckoning
RUN_EVALUATE       = True   # Set False to skip evaluation

# If skipping data logger, specify existing file:
EXISTING_LOG_FILE  = r"logs\imu_gps_log_20260311_143022.csv"
```

## Manual Mode - Run Scripts Individually

### Step 1: Collect Flight Data
```powershell
python data_logger.py
```

**Auto-setup (via SimConnect):**
- Teleports Cessna 172 Skyhawk G1000 to Vejle Rev3 start (55.76777°N, 9.40529°E, 1000ft)
- Sets airspeed to 80 knots, initial heading 180° (south)
- Engages autopilot (master + altitude hold + airspeed hold + GPS navigation)
- **Follows Vejle_Rev3_Cessna waypoint route automatically**

**Logging:**
- Logs IMU (accelerometer + gyroscope) at 50Hz
- Logs GPS (lat/lon/alt) at 50Hz
- Captures nadir camera frames at 5fps
- **Press 'q' to stop** (when running in pipeline) or Ctrl+C (standalone)

**Outputs:** 
- `logs/imu_gps_log_YYYYMMDD_HHMMSS.csv` - IMU + GPS data
- `logs/images_YYYYMMDD_HHMMSS/frame_*.jpg` - Camera frames

### Step 2: Run Dead Reckoning
```powershell
python dead_reckoning.py
```

Or specify input/output:
```powershell
python dead_reckoning.py --input logs\imu_gps_log_20260311_143022.csv --output logs\my_results.csv
```

**Process:**
- Integrates IMU data to estimate position
- Uses gyroscope for orientation tracking
- Converts acceleration from body to NED frame

**Outputs:** `logs/dead_reckoning_YYYYMMDD_HHMMSS.csv`

### Step 3: Evaluate Performance
```powershell
python evaluate.py
```

Or specify files:
```powershell
python evaluate.py --estimated logs\dead_reckoning_*.csv --truth logs\imu_gps_log_*.csv
```

**Analysis:**
- Compares estimated trajectory vs ground truth GPS
- Calculates Haversine distance errors
- Generates plots: error over time, trajectory comparison, error distribution
- Prints statistics: mean, median, max error, error at 30s/60s/120s

**Outputs:** 
- `logs/evaluation_plot_*.png` - Visualizations
- `logs/evaluation_errors_*.csv` - Error data

## File Structure
```
IMU_Estimator_MSFS_Connection/
├── run_pipeline.py         # Master script - runs everything
├── data_logger.py          # Script 1: Data collection from MSFS
├── dead_reckoning.py       # Script 2: IMU integration
├── evaluate.py             # Script 3: Performance evaluation
├── requirements.txt        # Python dependencies
├── .venv10032026/          # Virtual environment
└── logs/                   # Output directory (auto-created)
    ├── imu_gps_log_*.csv           # Raw IMU + GPS data
    ├── images_*/                   # Captured nadir frames per run
    ├── dead_reckoning_*.csv        # Estimated positions
    ├── evaluation_errors_*.csv     # Error analysis
    └── evaluation_plot_*.png       # Visualizations
```

## Expected Behavior

The dead reckoning approach will accumulate **drift** over time due to:
- Integration of noisy accelerometer data
- Gyroscope drift affecting orientation
- No external position corrections

This drift is expected and demonstrates the **need for vision-based corrections** in Phase 3.

## Notes

- **Auto-setup eliminates manual flight:** No need to configure cameras.cfg or fly manually
- **Reproducible flights:** Same spawn point, heading, and speed every time
- **Timestamped runs:** Each run's data is kept separate by timestamp
- **Flexible pipeline:** Enable/disable stages as needed for debugging
