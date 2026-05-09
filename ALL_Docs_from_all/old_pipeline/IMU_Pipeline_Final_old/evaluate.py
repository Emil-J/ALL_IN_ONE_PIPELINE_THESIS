"""
Evaluate Dead Reckoning Performance
Compares estimated trajectory against ground truth GPS data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth
    
    Args:
        lat1, lon1: First point in decimal degrees
        lat2, lon2: Second point in decimal degrees
    
    Returns:
        Distance in meters
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in meters
    r = 6371000
    
    return c * r

def evaluate(est_csv_path, truth_csv_path):
    """
    Evaluate dead reckoning performance
    
    Args:
        est_csv_path: Path to dead reckoning estimates CSV
        truth_csv_path: Path to ground truth GPS CSV (from data_logger.py)
    """
    # Load data
    print(f"Loading estimated data from: {est_csv_path}")
    est_df = pd.read_csv(est_csv_path)
    
    # Standardize estimated column names (add _est suffix if not present)
    # dead_reckoning outputs: latitude_est, longitude_est, altitude_est
    # ekf_ins outputs: latitude, longitude, altitude
    if 'latitude_est' not in est_df.columns and 'latitude' in est_df.columns:
        est_df = est_df.rename(columns={
            'latitude': 'latitude_est',
            'longitude': 'longitude_est',
            'altitude': 'altitude_est'
        })
    
    print(f"Loading ground truth from: {truth_csv_path}")
    truth_df = pd.read_csv(truth_csv_path)
    
    # Merge on timestamp - include airspeed and ground velocity if available
    truth_cols = ['timestamp', 'latitude', 'longitude', 'altitude']
    if 'airspeed_true' in truth_df.columns:
        truth_cols.append('airspeed_true')
    if 'ground_velocity' in truth_df.columns:
        truth_cols.append('ground_velocity')
    
    # Extract truth columns and rename to add _truth suffix
    truth_subset = truth_df[truth_cols].copy()
    truth_subset = truth_subset.rename(columns={
        'latitude': 'latitude_truth',
        'longitude': 'longitude_truth',
        'altitude': 'altitude_truth'
    })
    
    # Merge with estimated data
    merged = pd.merge(est_df, truth_subset, on='timestamp')
    
    print(f"\nMatched {len(merged)} data points")
    
    # Calculate Haversine error for each point
    errors = []
    for i in range(len(merged)):
        err = haversine_distance(
            merged['latitude_est'].iloc[i],
            merged['longitude_est'].iloc[i],
            merged['latitude_truth'].iloc[i],
            merged['longitude_truth'].iloc[i]
        )
        errors.append(err)
    
    merged['error_m'] = errors
    
    # ═══════════════════════════════════════════════════════════
    # DETAILED ERROR ANALYSIS
    # ═══════════════════════════════════════════════════════════
    
    # Calculate North/East error components (from truth to estimate)
    R_earth = 6371000.0  # meters
    merged['error_north'] = (merged['latitude_est'] - merged['latitude_truth']) * (np.pi/180) * R_earth
    merged['error_east'] = (merged['longitude_est'] - merged['longitude_truth']) * (np.pi/180) * R_earth * np.cos(np.radians(merged['latitude_truth']))
    
    # Calculate heading error if yaw is available in estimated data
    if 'yaw' in merged.columns:
        # Derive true heading from GPS trajectory (velocity direction)
        if len(merged) > 1:
            # Calculate velocity from position changes
            dt = np.diff(merged['timestamp'].values)
            dlat = np.diff(merged['latitude_truth'].values)
            dlon = np.diff(merged['longitude_truth'].values)
            
            v_north = dlat * (np.pi/180) * R_earth / dt
            v_east = dlon * (np.pi/180) * R_earth * np.cos(np.radians(merged['latitude_truth'].iloc[:-1].values)) / dt
            
            # True heading from velocity
            true_heading = np.degrees(np.arctan2(v_east, v_north))
            true_heading = (true_heading + 360) % 360  # Wrap to [0, 360)
            
            # Estimated heading
            est_heading = merged['yaw'].iloc[:-1].values
            est_heading = (est_heading + 360) % 360  # Wrap to [0, 360)
            
            # Heading error (wrapped to [-180, 180])
            heading_error = est_heading - true_heading
            heading_error = np.array([((e + 180) % 360) - 180 for e in heading_error])
            
            # Add to dataframe (pad last value to match length)
            if len(heading_error) > 0:
                merged['heading_error'] = np.concatenate([heading_error, [heading_error[-1]]])
            else:
                merged['heading_error'] = np.nan
        else:
            merged['heading_error'] = np.nan
    else:
        merged['heading_error'] = np.nan
    
    # Calculate velocity magnitude if available
    if 'vel_n' in merged.columns and 'vel_e' in merged.columns:
        merged['vel_magnitude'] = np.sqrt(merged['vel_n']**2 + merged['vel_e']**2)
        
        # True velocity from GPS (calculate from position changes)
        if len(merged) > 1:
            dt = np.diff(merged['timestamp'].values)
            dlat = np.diff(merged['latitude_truth'].values)
            dlon = np.diff(merged['longitude_truth'].values)
            
            v_north_truth = dlat * (np.pi/180) * R_earth / dt
            v_east_truth = dlon * (np.pi/180) * R_earth * np.cos(np.radians(merged['latitude_truth'].iloc[:-1].values)) / dt
            
            true_vel = np.sqrt(v_north_truth**2 + v_east_truth**2)
            # Pad to match length (use last value for final point)
            if len(true_vel) > 0:
                merged['vel_magnitude_truth'] = np.concatenate([true_vel, [true_vel[-1]]])
                merged['vel_error'] = merged['vel_magnitude'] - merged['vel_magnitude_truth']
            else:
                merged['vel_magnitude_truth'] = np.nan
                merged['vel_error'] = np.nan
        else:
            merged['vel_magnitude_truth'] = np.nan
            merged['vel_error'] = np.nan
    
    # Error growth rate (error per second)
    if len(merged) > 1:
        merged['error_rate'] = np.gradient(merged['error_m'].values, merged['timestamp'].values)
    else:
        merged['error_rate'] = np.nan
    
    # Statistics
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    median_error = np.median(errors)
    
    print(f"\n{'='*60}")
    print(f"SENSOR FUSION EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean error:     {mean_error:.2f} m")
    print(f"Median error:   {median_error:.2f} m")
    print(f"Max error:      {max_error:.2f} m")
    print(f"Min error:      {np.min(errors):.2f} m")
    print(f"Std deviation:  {np.std(errors):.2f} m")
    
    # Percentile analysis
    print(f"\n{'='*60}")
    print(f"PERCENTILE ANALYSIS")
    print(f"{'='*60}")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(errors, p)
        print(f"{p}th percentile: {val:.2f} m")
    
    # Directional error analysis
    print(f"\n{'='*60}")
    print(f"DIRECTIONAL ERROR ANALYSIS")
    print(f"{'='*60}")
    print(f"Mean North error:  {np.mean(merged['error_north']):.2f} m (+ is north)")
    print(f"Mean East error:   {np.mean(merged['error_east']):.2f} m (+ is east)")
    print(f"Std North error:   {np.std(merged['error_north']):.2f} m")
    print(f"Std East error:    {np.std(merged['error_east']):.2f} m")
    print(f"Final North error: {merged['error_north'].iloc[-1]:.2f} m")
    print(f"Final East error:  {merged['error_east'].iloc[-1]:.2f} m")
    
    # Error growth rate
    print(f"\n{'='*60}")
    print(f"ERROR GROWTH ANALYSIS")
    print(f"{'='*60}")
    if 'error_rate' in merged.columns:
        print(f"Mean error rate:   {np.mean(merged['error_rate']):.2f} m/s")
        print(f"Median error rate: {np.median(merged['error_rate']):.2f} m/s")
        print(f"Max error rate:    {np.max(merged['error_rate']):.2f} m/s")
        
        # Error accumulation over time
        duration = merged['timestamp'].iloc[-1] - merged['timestamp'].iloc[0]
        print(f"Total duration:    {duration:.1f} s")
        print(f"Error per minute:  {(mean_error / duration) * 60:.2f} m/min")
    
    # Heading error analysis
    if 'heading_error' in merged.columns and not np.all(np.isnan(merged['heading_error'])):
        print(f"\n{'='*60}")
        print(f"HEADING ERROR ANALYSIS")
        print(f"{'='*60}")
        valid_heading = merged['heading_error'][~np.isnan(merged['heading_error'])]
        if len(valid_heading) > 0:
            print(f"Mean heading error:   {np.mean(valid_heading):.2f}°")
            print(f"Median heading error: {np.median(valid_heading):.2f}°")
            print(f"Std heading error:    {np.std(valid_heading):.2f}°")
            print(f"Max heading error:    {np.max(np.abs(valid_heading)):.2f}°")
    
    # Velocity error analysis
    if 'vel_error' in merged.columns:
        print(f"\n{'='*60}")
        print(f"VELOCITY ERROR ANALYSIS")
        print(f"{'='*60}")
        print(f"Mean velocity error: {np.mean(merged['vel_error']):.2f} m/s")
        print(f"Std velocity error:  {np.std(merged['vel_error']):.2f} m/s")
        print(f"Mean est velocity:   {np.mean(merged['vel_magnitude']):.2f} m/s")
        if 'vel_magnitude_truth' in merged.columns:
            print(f"Mean true velocity:  {np.mean(merged['vel_magnitude_truth']):.2f} m/s")
    
    # Airspeed comparison (if available)
    if 'airspeed_true' in merged.columns:
        print(f"\n{'='*60}")
        print(f"AIRSPEED SENSOR DATA")
        print(f"{'='*60}")
        print(f"Mean airspeed (sensor):  {np.mean(merged['airspeed_true']):.2f} m/s")
        if 'vel_magnitude' in merged.columns:
            airspeed_vs_est = merged['vel_magnitude'] - merged['airspeed_true']
            print(f"Mean est vs airspeed:    {np.mean(airspeed_vs_est):.2f} m/s")
            print(f"Std est vs airspeed:     {np.std(airspeed_vs_est):.2f} m/s")
        if 'ground_velocity' in merged.columns:
            print(f"Mean ground velocity:    {np.mean(merged['ground_velocity']):.2f} m/s")
    
    # Time-specific errors
    print(f"\n{'='*60}")
    print(f"ERROR AT SPECIFIC TIME POINTS")
    print(f"{'='*60}")
    
    time_points = [30, 60, 120]
    for t in time_points:
        idx = (merged['timestamp'] - t).abs().idxmin()
        if merged['timestamp'].iloc[idx] <= merged['timestamp'].max():
            actual_time = merged['timestamp'].iloc[idx]
            error_at_t = merged['error_m'].iloc[idx]
            print(f"Error at t={actual_time:.1f}s (~{t}s): {error_at_t:.2f} m")
    
    # Create plots
    print(f"\n{'='*60}")
    print(f"GENERATING PLOTS")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Plot 1: Error over time
    ax1 = axes[0, 0]
    ax1.plot(merged['timestamp'], merged['error_m'], 'b-', linewidth=1.5)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Position Error (m)', fontsize=12)
    ax1.set_title('Position Error Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.1f}m', linewidth=2)
    ax1.legend()
    
    # Plot 2: 2D trajectory comparison
    ax2 = axes[0, 1]
    ax2.plot(merged['longitude_truth'], merged['latitude_truth'], 
             'g-', label='Ground Truth', linewidth=2, alpha=0.8)
    ax2.plot(merged['longitude_est'], merged['latitude_est'], 
             'r--', label='Estimated', linewidth=2, alpha=0.8)
    ax2.scatter(merged['longitude_truth'].iloc[0], merged['latitude_truth'].iloc[0], 
                c='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(merged['longitude_truth'].iloc[-1], merged['latitude_truth'].iloc[-1], 
                c='blue', s=100, marker='X', label='End (Truth)', zorder=5)
    ax2.scatter(merged['longitude_est'].iloc[-1], merged['latitude_est'].iloc[-1], 
                c='red', s=100, marker='X', label='End (Est)', zorder=5)
    ax2.set_xlabel('Longitude (°)', fontsize=12)
    ax2.set_ylabel('Latitude (°)', fontsize=12)
    ax2.set_title('Trajectory Comparison (Map View)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: North/East error components over time
    ax3 = axes[1, 0]
    ax3.plot(merged['timestamp'], merged['error_north'], 'r-', label='North Error', linewidth=1.5, alpha=0.8)
    ax3.plot(merged['timestamp'], merged['error_east'], 'b-', label='East Error', linewidth=1.5, alpha=0.8)
    ax3.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Error (m)', fontsize=12)
    ax3.set_title('North/East Error Components', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error distribution histogram
    ax4 = axes[1, 1]
    ax4.hist(merged['error_m'], bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax4.axvline(mean_error, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.1f}m')
    ax4.axvline(median_error, color='g', linestyle='--', linewidth=2, label=f'Median: {median_error:.1f}m')
    ax4.set_xlabel('Position Error (m)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Heading error over time (if available)
    ax5 = axes[2, 0]
    if 'heading_error' in merged.columns and not np.all(np.isnan(merged['heading_error'])):
        valid_mask = ~np.isnan(merged['heading_error'])
        ax5.plot(merged['timestamp'][valid_mask], merged['heading_error'][valid_mask], 
                'purple', linewidth=1.5, alpha=0.8)
        ax5.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax5.set_xlabel('Time (s)', fontsize=12)
        ax5.set_ylabel('Heading Error (°)', fontsize=12)
        ax5.set_title('Heading Error Over Time', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Heading error not available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Heading Error Over Time', fontsize=14, fontweight='bold')
    
    # Plot 6: Error rate over time
    ax6 = axes[2, 1]
    if 'error_rate' in merged.columns:
        # Smooth error rate for better visualization
        window = min(10, len(merged) // 10)
        if window > 1:
            error_rate_smooth = pd.Series(merged['error_rate']).rolling(window=window, center=True).mean()
            ax6.plot(merged['timestamp'], error_rate_smooth, 'orange', linewidth=1.5, alpha=0.8)
        else:
            ax6.plot(merged['timestamp'], merged['error_rate'], 'orange', linewidth=1.5, alpha=0.8)
        ax6.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax6.set_xlabel('Time (s)', fontsize=12)
        ax6.set_ylabel('Error Growth Rate (m/s)', fontsize=12)
        ax6.set_title('Error Growth Rate Over Time', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Error rate not calculated', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Error Growth Rate Over Time', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"evaluation_plot_{timestamp_str}.png"
    plot_path = os.path.join("logs", plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_filename}")
    
    # Save error data
    error_csv = f"evaluation_errors_{timestamp_str}.csv"
    error_path = os.path.join("logs", error_csv)
    merged.to_csv(error_path, index=False)
    print(f"✓ Error data saved to: {error_csv}")
    
    plt.show()
    
    return merged

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate sensor fusion / INS performance')
    parser.add_argument('--estimated', type=str, help='Path to estimated position CSV (from ekf_ins.py or dead_reckoning.py)')
    parser.add_argument('--truth', type=str, help='Path to ground truth GPS CSV')
    args = parser.parse_args()
    
    log_dir = "logs"
    
    if args.estimated and args.truth:
        est_path = args.estimated
        truth_path = args.truth
    else:
        # Auto-find most recent files
        if not os.path.exists(log_dir):
            print("Error: logs directory not found.")
            return
        
        # Find most recent estimation file (dead_reckoning or ekf_ins)
        est_files = [f for f in os.listdir(log_dir) 
                     if (f.startswith("dead_reckoning_") or f.startswith("ekf_ins_")) 
                     and f.endswith(".csv")]
        if not est_files:
            print("Error: No estimation results found. Run dead_reckoning.py or ekf_ins.py first.")
            return
        est_files.sort(reverse=True)
        est_path = os.path.join(log_dir, est_files[0])
        
        # Find most recent flight data (ground truth)
        truth_files = [f for f in os.listdir(log_dir) if f.startswith("imu_gps_log_") and f.endswith(".csv")]
        if not truth_files:
            print("Error: No flight data found. Run data_logger.py first.")
            return
        truth_files.sort(reverse=True)
        truth_path = os.path.join(log_dir, truth_files[0])
        
        print(f"Using estimated data: {os.path.basename(est_path)}")
        print(f"Using ground truth: {os.path.basename(truth_path)}")
    
    # Run evaluation
    evaluate(est_path, truth_path)

if __name__ == "__main__":
    main()
