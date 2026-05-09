"""
Evaluation utilities for analyzing localization performance
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .tms_utils import haversine_distance
from .io_utils import save_json, save_csv


def compute_per_frame_metrics(results: List[Dict]) -> pd.DataFrame:
    """
    Compute metrics for each frame
    
    Args:
        results: List of localization results from LocalizationPipeline
    
    Returns:
        DataFrame with per-frame metrics
    """
    rows = []
    
    for res in results:
        row = {
            "frame_path": res.get("query_frame", ""),
            "success": res.get("success", False),
            "num_candidates": res.get("num_candidates", 0),
        }
        
        # IMU metrics
        imu = res.get("imu_prior", {})
        row["imu_lat"] = imu.get("lat", np.nan)
        row["imu_lon"] = imu.get("lon", np.nan)
        row["imu_valid"] = imu.get("valid", False)
        
        # Ground truth
        gt = res.get("ground_truth", {})
        row["gt_lat"] = gt.get("lat", np.nan) if gt else np.nan
        row["gt_lon"] = gt.get("lon", np.nan) if gt else np.nan
        
        # Corrected estimate
        row["corrected_lat"] = res.get("corrected_lat", np.nan)
        row["corrected_lon"] = res.get("corrected_lon", np.nan)
        row["position_refined"] = res.get("position_refined", False)
        
        # Best match info
        best = res.get("best_match", {})
        if best:
            row["best_tile_x"] = best.get("tile_info", {}).get("tile_x", -1)
            row["best_tile_y"] = best.get("tile_info", {}).get("tile_y", -1)
            row["best_score"] = best.get("combined_score", np.nan)
            row["best_num_inliers"] = best.get("dedode_metrics", {}).get("num_inliers", 0)
            row["best_semantic_iou"] = best.get("semantic_metrics", {}).get("iou", np.nan)
        
        # Errors (if GT available)
        errors = res.get("errors", {})
        if errors:
            row["imu_error_m"] = errors.get("imu_error_m", np.nan)
            row["corrected_error_m"] = errors.get("corrected_error_m", np.nan)
            row["improvement_m"] = errors.get("improvement_m", np.nan)
        else:
            # Ensure columns exist even when no GT or errors
            row["imu_error_m"] = np.nan
            row["corrected_error_m"] = np.nan
            row["improvement_m"] = np.nan
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def compute_summary_statistics(metrics_df: pd.DataFrame,
                               thresholds: List[float] = [10, 25, 50, 100, 250, 500]) -> Dict:
    """
    Compute summary statistics over all frames
    
    Args:
        metrics_df: DataFrame from compute_per_frame_metrics()
        thresholds: Distance thresholds in meters for success rate
    
    Returns:
        Dict with summary statistics
    """
    summary = {}
    
    # Overall statistics
    summary["total_frames"] = len(metrics_df)
    summary["successful_localizations"] = metrics_df["success"].sum()
    summary["success_rate"] = float(metrics_df["success"].mean())
    
    # Error statistics (only for frames with GT and successful localization)
    has_errors = "imu_error_m" in metrics_df.columns
    if has_errors:
        valid_errors = metrics_df[
            metrics_df["success"] & 
            ~metrics_df["imu_error_m"].isna()
        ]
    else:
        valid_errors = pd.DataFrame()  # Empty
    
    if len(valid_errors) > 0:
        # IMU errors
        summary["imu_error_mean_m"] = float(valid_errors["imu_error_m"].mean())
        summary["imu_error_median_m"] = float(valid_errors["imu_error_m"].median())
        summary["imu_error_std_m"] = float(valid_errors["imu_error_m"].std())
        summary["imu_error_max_m"] = float(valid_errors["imu_error_m"].max())
        
        # Corrected errors
        summary["corrected_error_mean_m"] = float(valid_errors["corrected_error_m"].mean())
        summary["corrected_error_median_m"] = float(valid_errors["corrected_error_m"].median())
        summary["corrected_error_std_m"] = float(valid_errors["corrected_error_m"].std())
        summary["corrected_error_max_m"] = float(valid_errors["corrected_error_m"].max())
        
        # Improvement
        summary["improvement_mean_m"] = float(valid_errors["improvement_m"].mean())
        summary["improvement_median_m"] = float(valid_errors["improvement_m"].median())
        
        # Success rate at thresholds (for corrected position)
        for thresh in thresholds:
            within_thresh = (valid_errors["corrected_error_m"] <= thresh).sum()
            summary[f"within_{thresh}m_count"] = int(within_thresh)
            summary[f"within_{thresh}m_rate"] = float(within_thresh / len(valid_errors))
    
    # Matching statistics
    if "best_num_inliers" in metrics_df.columns:
        successful = metrics_df[metrics_df["success"]]
        if len(successful) > 0:
            summary["mean_inliers"] = float(successful["best_num_inliers"].mean())
            summary["median_inliers"] = float(successful["best_num_inliers"].median())
    
    return summary


def compute_percentiles(metrics_df: pd.DataFrame,
                       column: str = "corrected_error_m",
                       percentiles: List[float] = [25, 50, 75, 90, 95, 99]) -> Dict:
    """
    Compute percentiles for a metric
    
    Args:
        metrics_df: Metrics DataFrame
        column: Column to compute percentiles for
        percentiles: List of percentile values
    
    Returns:
        Dict mapping percentile -> value
    """
    valid = metrics_df[~metrics_df[column].isna()][column]
    
    if len(valid) == 0:
        return {f"p{p}": np.nan for p in percentiles}
    
    return {
        f"p{p}": float(np.percentile(valid, p))
        for p in percentiles
    }


def analyze_failure_cases(results: List[Dict]) -> Dict:
    """
    Analyze frames where localization failed
    
    Args:
        results: List of localization results
    
    Returns:
        Dict with failure analysis
    """
    failures = [r for r in results if not r.get("success", False)]
    
    analysis = {
        "num_failures": len(failures),
        "failure_rate": len(failures) / len(results) if results else 0.0,
        "failure_reasons": {}
    }
    
    # Count failure reasons
    for fail in failures:
        reason = fail.get("error_message", "Unknown")
        analysis["failure_reasons"][reason] = analysis["failure_reasons"].get(reason, 0) + 1
    
    return analysis


def compare_imu_vs_corrected(metrics_df: pd.DataFrame) -> Dict:
    """
    Compare IMU vs corrected position accuracy
    
    Args:
        metrics_df: Metrics DataFrame
    
    Returns:
        Dict with comparison statistics
    """
    valid = metrics_df[
        ~metrics_df["imu_error_m"].isna() & 
        ~metrics_df["corrected_error_m"].isna()
    ]
    
    if len(valid) == 0:
        return {"error": "No valid comparisons"}
    
    improved = (valid["improvement_m"] > 0).sum()
    degraded = (valid["improvement_m"] < 0).sum()
    unchanged = (valid["improvement_m"] == 0).sum()
    
    return {
        "num_frames": len(valid),
        "improved_count": int(improved),
        "improved_rate": float(improved / len(valid)),
        "degraded_count": int(degraded),
        "degraded_rate": float(degraded / len(valid)),
        "unchanged_count": int(unchanged),
        "mean_improvement_m": float(valid["improvement_m"].mean()),
        "median_improvement_m": float(valid["improvement_m"].median()),
        "improvement_std_m": float(valid["improvement_m"].std())
    }


def generate_evaluation_report(results: List[Dict],
                               output_dir: Path,
                               experiment_name: str = "localization") -> Dict:
    """
    Generate complete evaluation report
    
    Args:
        results: List of localization results
        output_dir: Directory to save report files
        experiment_name: Name for output files
    
    Returns:
        Dict with all evaluation metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute per-frame metrics
    metrics_df = compute_per_frame_metrics(results)
    
    # Save per-frame CSV
    metrics_csv_path = output_dir / f"{experiment_name}_per_frame_metrics.csv"
    save_csv(metrics_df, metrics_csv_path)
    
    # Compute summary statistics
    summary = compute_summary_statistics(metrics_df)
    
    # Compute percentiles
    percentiles = compute_percentiles(metrics_df)
    summary["percentiles"] = percentiles
    
    # Analyze failures
    failure_analysis = analyze_failure_cases(results)
    summary["failure_analysis"] = failure_analysis
    
    # Compare IMU vs corrected
    comparison = compare_imu_vs_corrected(metrics_df)
    summary["imu_vs_corrected"] = comparison
    
    # Save summary JSON
    summary_json_path = output_dir / f"{experiment_name}_summary.json"
    save_json(summary, summary_json_path)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT: {experiment_name}")
    print(f"{'='*60}")
    print(f"\nTotal frames: {summary['total_frames']}")
    print(f"Successful: {summary['successful_localizations']} ({summary['success_rate']*100:.1f}%)")
    
    if "corrected_error_mean_m" in summary:
        print(f"\nCorrected Position Error:")
        print(f"  Mean: {summary['corrected_error_mean_m']:.2f} m")
        print(f"  Median: {summary['corrected_error_median_m']:.2f} m")
        print(f"  Std: {summary['corrected_error_std_m']:.2f} m")
        print(f"  Max: {summary['corrected_error_max_m']:.2f} m")
        
        print(f"\nSuccess rate at thresholds:")
        for thresh in [10, 25, 50, 100, 250, 500]:
            key = f"within_{thresh}m_rate"
            if key in summary:
                print(f"  Within {thresh}m: {summary[key]*100:.1f}%")
    
    if "imu_vs_corrected" in summary and "mean_improvement_m" in summary["imu_vs_corrected"]:
        comp = summary["imu_vs_corrected"]
        print(f"\nIMU vs Corrected:")
        print(f"  Improved: {comp['improved_count']} ({comp['improved_rate']*100:.1f}%)")
        print(f"  Degraded: {comp['degraded_count']} ({comp['degraded_rate']*100:.1f}%)")
        print(f"  Mean improvement: {comp['mean_improvement_m']:.2f} m")
    
    print(f"\nFiles saved:")
    print(f"  - {metrics_csv_path}")
    print(f"  - {summary_json_path}")
    print(f"{'='*60}\n")
    
    return summary
