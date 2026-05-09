"""
Visualization utilities for displaying results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd

from .image_utils import load_image, convert_mask_to_rgb
from .io_utils import ensure_dir


def plot_match_result(query_img: np.ndarray,
                     reference_img: np.ndarray,
                     query_kp: np.ndarray,
                     ref_kp: np.ndarray,
                     matches: np.ndarray,
                     inlier_mask: Optional[np.ndarray] = None,
                     max_matches: int = 50,
                     figsize: Tuple[int, int] = (16, 8)) -> plt.Figure:
    """
    Visualize feature matches between query and reference
    
    Args:
        query_img: Query image
        reference_img: Reference image
        query_kp: Query keypoints (N, 2)
        ref_kp: Reference keypoints (M, 2)
        matches: Match indices (K, 2)
        inlier_mask: Boolean mask of inliers (optional)
        max_matches: Maximum matches to display
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Concatenate images side by side
    h1, w1 = query_img.shape[:2]
    h2, w2 = reference_img.shape[:2]
    h = max(h1, h2)
    
    # Pad images to same height
    canvas1 = np.zeros((h, w1, 3), dtype=np.uint8)
    canvas2 = np.zeros((h, w2, 3), dtype=np.uint8)
    canvas1[:h1, :w1] = query_img
    canvas2[:h2, :w2] = reference_img
    
    combined = np.hstack([canvas1, canvas2])
    ax.imshow(combined)
    
    # Plot matches
    num_matches = min(len(matches), max_matches)
    for i in range(num_matches):
        idx1, idx2 = matches[i]
        pt1 = query_kp[idx1]
        pt2 = ref_kp[idx2] + [w1, 0]  # Offset for concatenated image
        
        # Color based on inlier status
        if inlier_mask is not None and i < len(inlier_mask):
            color = 'green' if inlier_mask[i] else 'red'
            alpha = 0.7 if inlier_mask[i] else 0.3
        else:
            color = 'blue'
            alpha = 0.5
        
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 
               color=color, alpha=alpha, linewidth=1)
        ax.plot(pt1[0], pt1[1], 'o', color=color, markersize=3)
        ax.plot(pt2[0], pt2[1], 'o', color=color, markersize=3)
    
    ax.axis('off')
    ax.set_title(f"Feature Matches (showing {num_matches}/{len(matches)})", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_semantic_comparison(query_mask: np.ndarray,
                            reference_mask: np.ndarray,
                            query_img: Optional[np.ndarray] = None,
                            reference_img: Optional[np.ndarray] = None,
                            color_map: Optional[Dict] = None,
                            iou_score: Optional[float] = None,
                            figsize: Tuple[int, int] = (16, 8)) -> plt.Figure:
    """
    Visualize semantic segmentation comparison
    
    Args:
        query_mask: Query semantic mask
        reference_mask: Reference semantic mask
        query_img: Optional query image
        reference_img: Optional reference image
        color_map: Color map for visualization
        iou_score: IoU score to display
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if color_map is None:
        color_map = {
            0: (4, 4, 255),
            1: (0, 167, 2),
            2: (243, 255, 150),
            3: (193, 105, 53),
            4: (255, 0, 231),
            5: (150, 150, 150)
        }
    
    # Convert masks to RGB
    query_rgb = convert_mask_to_rgb(query_mask, color_map)
    ref_rgb = convert_mask_to_rgb(reference_mask, color_map)
    
    # Create figure
    num_rows = 2 if query_img is not None else 1
    fig, axes = plt.subplots(num_rows, 2, figsize=figsize)
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Row 1: Original images (if provided)
    if query_img is not None:
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title("Query Image", fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(reference_img)
        axes[0, 1].set_title("Reference Image", fontsize=12)
        axes[0, 1].axis('off')
        
        mask_row = 1
    else:
        mask_row = 0
    
    # Masks
    axes[mask_row, 0].imshow(query_rgb)
    axes[mask_row, 0].set_title("Query Segmentation", fontsize=12)
    axes[mask_row, 0].axis('off')
    
    axes[mask_row, 1].imshow(ref_rgb)
    title = "Reference Segmentation"
    if iou_score is not None:
        title += f"\nIoU: {iou_score:.3f}"
    axes[mask_row, 1].set_title(title, fontsize=12)
    axes[mask_row, 1].axis('off')
    
    plt.tight_layout()
    return fig


def plot_localization_result(result: Dict,
                            show_candidates: bool = True,
                            figsize: Tuple[int, int] = (18, 12)) -> plt.Figure:
    """
    Visualize complete localization result for one frame
    
    Args:
        result: Result dict from LocalizationPipeline.localize_frame()
        show_candidates: Show top candidates
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig)
    
    # Load query image
    query_img = load_image(result["query_frame"])
    
    # Query image
    ax_query = fig.add_subplot(gs[0, 0])
    ax_query.imshow(query_img)
    ax_query.set_title("Query Frame", fontsize=12, fontweight='bold')
    ax_query.axis('off')
    
    # IMU prior location
    ax_prior = fig.add_subplot(gs[0, 1])
    imu = result.get("imu_prior", {})
    ax_prior.text(0.5, 0.5, 
                 f"IMU Prior\n"
                 f"Lat: {imu.get('lat', np.nan):.6f}\n"
                 f"Lon: {imu.get('lon', np.nan):.6f}",
                 ha='center', va='center', fontsize=11)
    ax_prior.axis('off')
    
    # Corrected estimate
    ax_est = fig.add_subplot(gs[0, 2])
    if result.get("success", False):
        ax_est.text(0.5, 0.5,
                   f"Corrected Estimate\n"
                   f"Lat: {result.get('corrected_lat', np.nan):.6f}\n"
                   f"Lon: {result.get('corrected_lon', np.nan):.6f}\n"
                   f"Refined: {result.get('position_refined', False)}",
                   ha='center', va='center', fontsize=11, color='green')
    else:
        ax_est.text(0.5, 0.5,
                   f"Localization Failed\n{result.get('error_message', 'Unknown error')}",
                   ha='center', va='center', fontsize=11, color='red')
    ax_est.axis('off')
    
    # Best match
    if result.get("success", False) and result.get("best_match"):
        best = result["best_match"]
        best_img_path = Path(best["tile_info"]["file_path"])
        if best_img_path.exists():
            best_img = load_image(best_img_path)
            
            ax_best = fig.add_subplot(gs[1, 0])
            ax_best.imshow(best_img)
            ax_best.set_title(f"Best Match (Rank 1)\n"
                            f"Tile ({best['tile_info']['tile_x']}, {best['tile_info']['tile_y']})",
                            fontsize=11)
            ax_best.axis('off')
    
    # Metrics
    ax_metrics = fig.add_subplot(gs[1, 1:])
    if result.get("success", False) and result.get("best_match"):
        best = result["best_match"]
        dedode = best.get("dedode_metrics", {})
        semantic = best.get("semantic_metrics", {})
        
        metrics_text = (
            f"Match Metrics:\n"
            f"  Combined Score: {best.get('combined_score', 0):.2f}\n"
            f"  Matches: {dedode.get('num_matches', 0)}\n"
            f"  Inliers: {dedode.get('num_inliers', 0)}\n"
            f"  Inlier Ratio: {dedode.get('inlier_ratio', 0):.3f}\n"
            f"  Reproj Error: {dedode.get('median_reproj_error', np.nan):.2f} px\n"
            f"  Semantic IoU: {semantic.get('iou', np.nan):.3f}\n"
            f"  Semantic Boundary: {semantic.get('boundary', np.nan):.3f}"
        )
        
        if result.get("errors"):
            errors = result["errors"]
            metrics_text += (
                f"\n\nError Metrics (vs Ground Truth):\n"
                f"  IMU Error: {errors['imu_error_m']:.2f} m\n"
                f"  Corrected Error: {errors['corrected_error_m']:.2f} m\n"
                f"  Improvement: {errors['improvement_m']:.2f} m"
            )
        
        ax_metrics.text(0.1, 0.5, metrics_text, ha='left', va='center', 
                       fontsize=10, family='monospace')
    ax_metrics.axis('off')
    
    # Top candidates
    if show_candidates and result.get("candidates"):
        for i, cand in enumerate(result["candidates"][:4]):  # Show top 4
            if i >= 4:
                break
            
            row = 2 + i // 2
            col = i % 2
            
            if row >= 3:
                break
            
            cand_img_path = Path(cand["tile_info"]["file_path"])
            if cand_img_path.exists():
                cand_img = load_image(cand_img_path)
                
                ax_cand = fig.add_subplot(gs[2, col])
                ax_cand.imshow(cand_img)
                ax_cand.set_title(f"Rank {cand.get('rank', i+1)}: Score {cand.get('combined_score', 0):.1f}",
                                fontsize=9)
                ax_cand.axis('off')
    
    plt.tight_layout()
    return fig


def plot_trajectory(metrics_df: pd.DataFrame,
                   plot_type: str = "map",
                   figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Plot trajectory comparison
    
    Args:
        metrics_df: DataFrame from compute_per_frame_metrics()
        plot_type: "map" (lat/lon) or "error_timeline"
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if plot_type == "map":
        # Plot on map
        if "gt_lat" in metrics_df.columns:
            valid_gt = metrics_df[~metrics_df["gt_lat"].isna()]
            ax.plot(valid_gt["gt_lon"], valid_gt["gt_lat"], 
                   'k-', linewidth=2, label='Ground Truth', alpha=0.7)
            ax.plot(valid_gt["gt_lon"], valid_gt["gt_lat"], 
                   'ko', markersize=3)
        
        valid_imu = metrics_df[~metrics_df["imu_lat"].isna()]
        ax.plot(valid_imu["imu_lon"], valid_imu["imu_lat"],
               'b--', linewidth=1.5, label='IMU Prior', alpha=0.6)
        
        valid_corr = metrics_df[metrics_df["success"] & ~metrics_df["corrected_lat"].isna()]
        ax.plot(valid_corr["corrected_lon"], valid_corr["corrected_lat"],
               'g-', linewidth=2, label='Corrected', alpha=0.8)
        ax.plot(valid_corr["corrected_lon"], valid_corr["corrected_lat"],
               'go', markersize=4)
        
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.set_title("Trajectory Comparison", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
    elif plot_type == "error_timeline":
        # Plot error over time
        if "imu_error_m" in metrics_df.columns:
            valid = metrics_df[~metrics_df["imu_error_m"].isna()]
            
            if len(valid) > 0:
                x = np.arange(len(valid))
                ax.plot(x, valid["imu_error_m"], 'b-', label='IMU Error', alpha=0.7)
                ax.plot(x, valid["corrected_error_m"], 'g-', label='Corrected Error', alpha=0.7)
                ax.fill_between(x, valid["imu_error_m"], valid["corrected_error_m"],
                                where=valid["imu_error_m"] > valid["corrected_error_m"],
                                alpha=0.3, color='green', label='Improvement')
                ax.fill_between(x, valid["imu_error_m"], valid["corrected_error_m"],
                                where=valid["imu_error_m"] <= valid["corrected_error_m"],
                                alpha=0.3, color='red', label='Degradation')
            else:
                ax.text(0.5, 0.5, 'No error data available\n(all frames failed or no ground truth)',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No error data available\n(all frames failed or no ground truth)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.set_xlabel("Frame Index", fontsize=12)
        ax.set_ylabel("Error (meters)", fontsize=12)
        ax.set_title("Localization Error Timeline", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_error_distribution(metrics_df: pd.DataFrame,
                           figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """
    Plot error distribution histograms
    
    Args:
        metrics_df: DataFrame from compute_per_frame_metrics()
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if "imu_error_m" not in metrics_df.columns:
        # No error data - create empty figure with message
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, 'No error data available\n(all frames failed or no ground truth)',
               ha='center', va='center', fontsize=14)
        ax.set_title("Error Distribution", fontsize=14, fontweight='bold')
        ax.axis('off')
        return fig
    
    valid = metrics_df[~metrics_df["imu_error_m"].isna()]
    
    if len(valid) == 0:
        # No valid data - create empty figure with message
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, 'No error data available\n(all frames failed or no ground truth)',
               ha='center', va='center', fontsize=14)
        ax.set_title("Error Distribution", fontsize=14, fontweight='bold')
        ax.axis('off')
        return fig
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    valid = metrics_df[~metrics_df["imu_error_m"].isna()]
    
    # IMU error distribution
    axes[0].hist(valid["imu_error_m"], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(valid["imu_error_m"].median(), color='red', linestyle='--', 
                   label=f'Median: {valid["imu_error_m"].median():.1f}m')
    axes[0].set_xlabel("Error (meters)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("IMU Error Distribution", fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Corrected error distribution
    axes[1].hist(valid["corrected_error_m"], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(valid["corrected_error_m"].median(), color='red', linestyle='--',
                   label=f'Median: {valid["corrected_error_m"].median():.1f}m')
    axes[1].set_xlabel("Error (meters)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("Corrected Error Distribution", fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def save_all_visualizations(results: List[Dict],
                           metrics_df: pd.DataFrame,
                           output_dir: Path,
                           save_per_frame: bool = False):
    """
    Generate and save all visualizations
    
    Args:
        results: List of localization results
        metrics_df: Metrics DataFrame
        output_dir: Output directory
        save_per_frame: If True, save individual frame visualizations
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    print("\nGenerating visualizations...")
    
    # Trajectory plot
    fig = plot_trajectory(metrics_df, plot_type="map")
    fig.savefig(output_dir / "trajectory_map.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ trajectory_map.png")
    
    # Error timeline
    fig = plot_trajectory(metrics_df, plot_type="error_timeline")
    fig.savefig(output_dir / "error_timeline.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ error_timeline.png")
    
    # Error distributions
    fig = plot_error_distribution(metrics_df)
    fig.savefig(output_dir / "error_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ error_distribution.png")
    
    # Per-frame visualizations (optional)
    if save_per_frame:
        frame_dir = output_dir / "per_frame"
        ensure_dir(frame_dir)
        
        for i, res in enumerate(results[:10]):  # Save first 10
            if res.get("success", False):
                fig = plot_localization_result(res)
                fig.savefig(frame_dir / f"frame_{i:04d}.png", dpi=100, bbox_inches='tight')
                plt.close(fig)
        
        print(f"  ✓ Saved {min(10, len(results))} per-frame visualizations")
    
    print("Visualizations complete!\n")
