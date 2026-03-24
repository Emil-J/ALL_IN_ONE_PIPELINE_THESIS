"""
Master Pipeline Script - GPS-Free Drone Localization
Runs all scripts in sequence: data collection → dead reckoning → evaluation
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# ─── PIPELINE CONFIG ───────────────────────────────
RUN_DATA_LOGGER    = False   # Set False to skip data collection
RUN_DEAD_RECKONING = True    # Set False to skip sensor fusion
RUN_EVALUATE       = True    # Set False to skip evaluation

# Sensor fusion algorithm selection:
# "ekf" = Error-State EKF (recommended, based on Kok et al. 2017)
# "simple" = Simple quaternion integration (baseline, high drift)
ALGORITHM = "ekf"

# If skipping data logger, point to existing log file:
EXISTING_LOG_FILE  = r"logs\imu_gps_log_20260312_163530.csv"  # Update with your file
# ───────────────────────────────────────────────────

def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_script(script_name, args=None):
    """Run a Python script and capture its output"""
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 70)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {script_name} failed with exit code {result.returncode}")
        return None
    
    print(f"\n✓ {script_name} completed successfully")
    return result.returncode

def find_latest_file(directory, pattern):
    """Find the most recent file matching a pattern"""
    if not os.path.exists(directory):
        return None
    
    files = [f for f in os.listdir(directory) if pattern in f and f.endswith('.csv')]
    if not files:
        return None
    
    files.sort(reverse=True)
    return os.path.join(directory, files[0])

def main():
    print_banner("GPS-FREE DRONE LOCALIZATION PIPELINE")
    print(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nPipeline Configuration:")
    print(f"  • Data Logger:    {'ENABLED' if RUN_DATA_LOGGER else 'DISABLED'}")
    print(f"  • Sensor Fusion:  {'ENABLED' if RUN_DEAD_RECKONING else 'DISABLED'}")
    print(f"  • Algorithm:      {ALGORITHM.upper()} ({'Error-State EKF' if ALGORITHM == 'ekf' else 'Simple Integration'})")
    print(f"  • Evaluation:     {'ENABLED' if RUN_EVALUATE else 'DISABLED'}")
    
    if not RUN_DATA_LOGGER:
        print(f"  • Using existing log: {EXISTING_LOG_FILE}")
    
    # Track file paths
    log_file = None
    dead_reckoning_file = None
    
    # ─── STEP 1: DATA COLLECTION ───
    if RUN_DATA_LOGGER:
        print_banner("STEP 1/3: DATA COLLECTION")
        print("This will:")
        print("  • Teleport aircraft to Vejle, Denmark")
        print("  • Engage autopilot (heading + altitude hold)")
        print("  • Log IMU + GPS data at 50Hz")
        print("  • Capture nadir camera frames at 5fps")
        print("\nMake sure Microsoft Flight Simulator is running!")
        print("Press 'q' in the data_logger window to stop logging.")
        print("The pipeline will continue automatically after data collection.\n")
        input("Press Enter to start data collection...")
        
        try:
            result = run_script("data_logger.py")
        except KeyboardInterrupt:
            print("\n\nData collection stopped by user (Ctrl+C)")
            result = 0  # Treat as success if Ctrl+C during data logger
        
        # Find the generated log file (even if Ctrl+C was pressed)
        log_file = find_latest_file("logs", "imu_gps_log_")
        if not log_file:
            print("\n❌ Pipeline aborted: Could not find generated log file")
            print("   Data collection may not have completed. Try running data_logger.py manually.")
            return
        
        print(f"\n✓ Log file created: {os.path.basename(log_file)}")
        print("✓ Data collection complete - continuing to dead reckoning...")
        time.sleep(2)
    else:
        print_banner("STEP 1/3: DATA COLLECTION (SKIPPED)")
        log_file = EXISTING_LOG_FILE
        if not os.path.exists(log_file):
            print(f"\n❌ Pipeline aborted: Existing log file not found: {log_file}")
            return
        print(f"Using existing log: {os.path.basename(log_file)}")
    
    # ─── STEP 2: DEAD RECKONING ───
    if RUN_DEAD_RECKONING:
        if ALGORITHM == "ekf":
            print_banner("STEP 2/3: SENSOR FUSION (Error-State EKF)")
            print("This will:")
            print("  • Fuse IMU + magnetometer + barometer data")
            print("  • Error-State Extended Kalman Filter (Kok et al. 2017)")
            print("  • Estimate position, velocity, and orientation")
            print("  • Estimate gyroscope bias online")
            print(f"\nInput: {os.path.basename(log_file)}")
            
            result = run_script("ekf_ins.py", ["--input", log_file])
            script_name = "ekf_ins.py"
            output_pattern = "ekf_ins_"
        else:  # ALGORITHM == "simple"
            print_banner("STEP 2/3: SENSOR FUSION (Simple Integration)")
            print("This will:")
            print("  • Integrate IMU data (accel + gyro)")
            print("  • Apply magnetometer heading correction")
            print("  • Use barometer for altitude")
            print("  • WARNING: High drift expected without EKF")
            print(f"\nInput: {os.path.basename(log_file)}")
            
            result = run_script("dead_reckoning.py", ["--input", log_file])
            script_name = "dead_reckoning.py"
            output_pattern = "dead_reckoning_"
        
        if result is None:
            print(f"\n❌ Pipeline aborted: {script_name} failed")
            return
        
        # Find the generated file
        dead_reckoning_file = find_latest_file("logs", output_pattern)
        if not dead_reckoning_file:
            print(f"\n❌ Pipeline aborted: Could not find {output_pattern} results")
            return
        
        print(f"\n✓ Results created: {os.path.basename(dead_reckoning_file)}")
        time.sleep(2)
    else:
        print_banner("STEP 2/3: SENSOR FUSION (SKIPPED)")
        output_pattern = "ekf_ins_" if ALGORITHM == "ekf" else "dead_reckoning_"
        dead_reckoning_file = find_latest_file("logs", output_pattern)
        if not dead_reckoning_file:
            print(f"\n❌ Pipeline aborted: No {output_pattern} results found")
            return
        print(f"Using existing results: {os.path.basename(dead_reckoning_file)}")
    
    # ─── STEP 3: EVALUATION ───
    if RUN_EVALUATE:
        print_banner("STEP 3/3: EVALUATION")
        print("This will:")
        print("  • Compare estimated vs ground truth GPS")
        print("  • Calculate Haversine distance errors")
        print("  • Generate plots and statistics")
        print(f"\nEstimated: {os.path.basename(dead_reckoning_file)}")
        print(f"Truth:     {os.path.basename(log_file)}")
        
        result = run_script("evaluate.py", [
            "--estimated", dead_reckoning_file,
            "--truth", log_file
        ])
        if result is None:
            print("\n❌ Pipeline aborted: evaluate.py failed")
            return
        
        time.sleep(2)
    else:
        print_banner("STEP 3/3: EVALUATION (SKIPPED)")
    
    # ─── COMPLETION ───
    print_banner("PIPELINE COMPLETE!")
    print("Results saved to: logs/")
    print(f"\nAlgorithm used: {ALGORITHM.upper()}")
    print("\nGenerated files:")
    if RUN_DATA_LOGGER or log_file:
        print(f"  • IMU + GPS log:     {os.path.basename(log_file)}")
    if RUN_DEAD_RECKONING or dead_reckoning_file:
        print(f"  • Sensor fusion:     {os.path.basename(dead_reckoning_file)}")
    if RUN_EVALUATE:
        eval_plot = find_latest_file("logs", "evaluation_plot_")
        if eval_plot:
            print(f"  • Evaluation plot:   {os.path.basename(eval_plot)}")
        eval_errors = find_latest_file("logs", "evaluation_errors_")
        if eval_errors:
            print(f"  • Error analysis:    {os.path.basename(eval_errors)}")
    
    print(f"\nPipeline finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
