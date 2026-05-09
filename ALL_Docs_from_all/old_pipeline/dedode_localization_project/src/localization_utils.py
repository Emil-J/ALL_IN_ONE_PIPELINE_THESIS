"""
Core localization logic - orchestrates IMU, semantic, and DeDoDe matching

This module implements the frame-by-frame localization pipeline:
1. Get IMU prior for current frame
2. Find candidate tiles within search radius
3. Run DeDoDe matching against candidates
4. Run semantic matching against candidates
5. Combine scores and select best match
6. Estimate corrected position
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

from .tms_utils import find_tiles_within_radius, haversine_distance, tile_bounds
from .dedode_adapter import DeDoDeMatcher, compute_homography, estimate_position_from_homography
from .semantic_adapter import SemanticSegmentationModel
from .semantic_matching_utils import compute_semantic_score
from .matching_utils import score_candidate, rank_candidates, select_best_candidate
from .image_utils import load_image


class LocalizationPipeline:
    """
    Complete frame-by-frame localization pipeline
    
    Combines IMU prior, DeDoDe visual matching, and semantic consistency
    for robust local-area GPS-denied navigation.
    """
    
    def __init__(self,
                 reference_df: pd.DataFrame,
                 dedode_matcher: DeDoDeMatcher,
                 semantic_model: Optional[SemanticSegmentationModel] = None,
                 config: Dict = None):
        """
        Initialize localization pipeline
        
        Args:
            reference_df: DataFrame with reference tile metadata
                Required columns: file_path, tile_x, tile_y, lat, lon, zoom
            dedode_matcher: Initialized DeDoDe matcher
            semantic_model: Initialized semantic segmentation model (optional)
            config: Configuration dict with parameters
        """
        self.reference_df = reference_df
        self.dedode = dedode_matcher
        self.semantic = semantic_model
        self.config = config or {}
        
        # Extract config parameters
        self.use_semantics = self.config.get("USE_SEMANTICS", True) and semantic_model is not None
        self.imu_search_radius = self.config.get("IMU_SEARCH_RADIUS_METERS", 250.0)
        self.max_candidates = self.config.get("MAX_CANDIDATE_TILES", 100)
        self.top_k = self.config.get("TOP_K_MATCHES", 5)
        self.min_inliers = self.config.get("MIN_MATCHES_FOR_HOMOGRAPHY", 8)
        self.ransac_thresh = self.config.get("RANSAC_REPROJ_THRESH", 4.0)
        self.scoring_weights = self.config.get("SCORING_WEIGHTS", {})
        self.semantic_weight = self.config.get("SEMANTIC_WEIGHT_IN_FINAL_SCORE", 3.0)
        self.semantic_filter_classes = self.config.get("SEMANTIC_FILTER_CLASSES", [2, 4, 5])
        self.semantic_score_weights = self.config.get("SEMANTIC_SCORE_WEIGHTS", {"iou": 1.0, "boundary": 0.5})
        self.zoom = self.config.get("TMS_ZOOM_LEVEL", 16)
        
        # Build spatial index for fast candidate lookup
        self._build_spatial_index()
    
    def _build_spatial_index(self):
        """Build spatial index for fast tile lookup"""
        # For now, just ensure we have lat/lon
        if 'lat' not in self.reference_df.columns or 'lon' not in self.reference_df.columns:
            raise ValueError("reference_df must have 'lat' and 'lon' columns")
        
        # Could use KD-tree or R-tree here for large databases
        # For ~300 tiles, linear search is acceptable
        self.spatial_index = self.reference_df
    
    def localize_frame(self,
                      query_frame_path: Path,
                      imu_prior: Dict,
                      ground_truth: Optional[Dict] = None) -> Dict:
        """
        Localize a single query frame
        
        Args:
            query_frame_path: Path to query image
            imu_prior: Dict with IMU estimate:
                - lat, lon: IMU position estimate
                - heading: Heading (optional)
                - confidence: Confidence (optional)
            ground_truth: Optional GT dict with lat, lon for evaluation
        
        Returns:
            Dict with localization results:
                - query_frame: Frame path
                - imu_prior: IMU estimate
                - candidates: List of candidate dicts
                - best_match: Best candidate dict (or None)
                - corrected_lat, corrected_lon: Final position estimate
                - errors: Error metrics vs GT (if provided)
                - success: Boolean indicating if localization succeeded
        """
        result = {
            "query_frame": str(query_frame_path),
            "imu_prior": imu_prior,
            "ground_truth": ground_truth,
            "success": False,
            "error_message": None
        }
        
        # Validate IMU prior
        if not imu_prior.get("valid", False) or np.isnan(imu_prior.get("lat", np.nan)):
            result["error_message"] = "Invalid IMU prior"
            return result
        
        try:
            # Step 1: Find candidate tiles within search radius
            candidates = self._find_candidate_tiles(
                imu_prior["lat"],
                imu_prior["lon"]
            )
            
            if not candidates:
                result["error_message"] = "No candidate tiles found"
                return result
            
            result["num_candidates"] = len(candidates)
            
            # Step 2: Load query image
            query_img = load_image(query_frame_path)
            
            # Step 3: Run semantic segmentation on query (if enabled)
            query_mask = None
            if self.use_semantics:
                query_mask = self.semantic.predict(query_img, use_cache=True)
            
            # Step 4: Match against each candidate
            matched_candidates = []
            
            for cand_info in candidates:
                try:
                    match_result = self._match_candidate(
                        query_img,
                        query_mask,
                        cand_info
                    )
                    if match_result is not None:
                        matched_candidates.append(match_result)
                except Exception as e:
                    warnings.warn(f"Failed to match candidate {cand_info.get('tile_x', '?')},{cand_info.get('tile_y', '?')}: {e}")
                    continue
            
            if not matched_candidates:
                result["error_message"] = "No successful matches"
                return result
            
            # Step 5: Rank candidates
            ranked_candidates = rank_candidates(matched_candidates, self.use_semantics)
            result["candidates"] = ranked_candidates[:self.top_k]
            
            # Step 6: Select best match
            best_match = select_best_candidate(ranked_candidates)
            
            if best_match is None:
                result["error_message"] = "No valid match after filtering"
                return result
            
            result["best_match"] = best_match
            
            # Step 7: Extract position estimate
            # Default to best tile center
            result["corrected_lat"] = best_match["tile_info"]["lat"]
            result["corrected_lon"] = best_match["tile_info"]["lon"]
            
            # Optional: Refine using homography
            if best_match.get("homography") is not None:
                try:
                    tile_bounds_obj = tile_bounds(
                        best_match["tile_info"]["tile_x"],
                        best_match["tile_info"]["tile_y"],
                        self.zoom
                    )
                    
                    refined_lat, refined_lon = estimate_position_from_homography(
                        best_match["homography"],
                        query_img.shape[:2],
                        best_match["tile_info"]["lat"],
                        best_match["tile_info"]["lon"],
                        (tile_bounds_obj.height_meters, tile_bounds_obj.width_meters)
                    )
                    
                    result["corrected_lat"] = refined_lat
                    result["corrected_lon"] = refined_lon
                    result["position_refined"] = True
                except Exception as e:
                    warnings.warn(f"Position refinement failed: {e}")
                    result["position_refined"] = False
            else:
                result["position_refined"] = False
            
            result["success"] = True
            
            # Step 8: Compute errors vs ground truth
            if ground_truth is not None and not np.isnan(ground_truth.get("lat", np.nan)):
                result["errors"] = self._compute_errors(
                    result["corrected_lat"],
                    result["corrected_lon"],
                    imu_prior["lat"],
                    imu_prior["lon"],
                    ground_truth["lat"],
                    ground_truth["lon"]
                )
            
            return result
            
        except Exception as e:
            result["error_message"] = f"Unexpected error: {str(e)}"
            return result
    
    def _find_candidate_tiles(self, lat: float, lon: float) -> List[Dict]:
        """Find candidate tiles within search radius of IMU prior"""
        # Get tiles within radius
        tile_coords = find_tiles_within_radius(
            lat, lon,
            self.imu_search_radius,
            self.zoom
        )
        
        # Look up in reference database
        candidates = []
        for tile_x, tile_y in tile_coords:
            matches = self.reference_df[
                (self.reference_df['tile_x'] == tile_x) &
                (self.reference_df['tile_y'] == tile_y)
            ]
            
            if not matches.empty:
                row = matches.iloc[0]
                candidates.append({
                    "tile_x": int(row['tile_x']),
                    "tile_y": int(row['tile_y']),
                    "lat": float(row['lat']),
                    "lon": float(row['lon']),
                    "file_path": str(row['file_path']),
                    "zoom": int(row.get('zoom', self.zoom))
                })
        
        # Limit to max candidates (keep nearest)
        if len(candidates) > self.max_candidates:
            # Sort by distance from prior
            candidates_with_dist = [
                (c, haversine_distance(lat, lon, c["lat"], c["lon"]))
                for c in candidates
            ]
            candidates_with_dist.sort(key=lambda x: x[1])
            candidates = [c for c, _ in candidates_with_dist[:self.max_candidates]]
        
        return candidates
    
    def _match_candidate(self,
                        query_img: np.ndarray,
                        query_mask: Optional[np.ndarray],
                        cand_info: Dict) -> Optional[Dict]:
        """
        Match query against a single candidate tile
        
        Returns:
            Dict with match results or None if matching failed
        """
        # Load candidate tile
        cand_path = Path(cand_info["file_path"])
        if not cand_path.exists():
            return None
        
        cand_img = load_image(cand_path)
        
        # DeDoDe matching
        query_desc = self.dedode.detect_and_describe(query_img)
        cand_desc = self.dedode.detect_and_describe(cand_img)
        
        matches = self.dedode.match(query_desc, cand_desc)
        
        if len(matches["matches"]) < self.min_inliers:
            return None
        
        # Compute homography
        H, inlier_mask, geo_stats = compute_homography(
            query_desc["keypoints"],
            cand_desc["keypoints"],
            matches["matches"],
            ransac_threshold=self.ransac_thresh,
            min_matches=self.min_inliers
        )
        
        if H is None or geo_stats["num_inliers"] < self.min_inliers:
            return None
        
        # Compute DeDoDe metrics
        dedode_metrics = {
            "num_matches": len(matches["matches"]),
            "num_inliers": geo_stats["num_inliers"],
            "inlier_ratio": geo_stats["inlier_ratio"],
            "median_reproj_error": geo_stats["median_reproj_error"],
            "mean_confidence": float(np.mean(matches["match_confidence"]))
        }
        
        # Semantic matching (if enabled)
        semantic_metrics = {}
        if self.use_semantics and query_mask is not None:
            cand_mask = self.semantic.predict(cand_img, use_cache=True)
            semantic_metrics = compute_semantic_score(
                query_mask,
                cand_mask,
                filter_classes=self.semantic_filter_classes,
                score_weights=self.semantic_score_weights
            )
        
        # Combined score
        combined_score = score_candidate(
            dedode_metrics,
            semantic_metrics,
            self.scoring_weights,
            self.semantic_weight
        )
        
        return {
            "tile_info": cand_info,
            "dedode_metrics": dedode_metrics,
            "semantic_metrics": semantic_metrics,
            "combined_score": combined_score,
            "homography": H
        }
    
    def _compute_errors(self, 
                       corrected_lat: float, corrected_lon: float,
                       imu_lat: float, imu_lon: float,
                       gt_lat: float, gt_lon: float) -> Dict:
        """Compute error metrics"""
        return {
            "imu_error_m": haversine_distance(imu_lat, imu_lon, gt_lat, gt_lon),
            "corrected_error_m": haversine_distance(corrected_lat, corrected_lon, gt_lat, gt_lon),
            "improvement_m": haversine_distance(imu_lat, imu_lon, gt_lat, gt_lon) - 
                            haversine_distance(corrected_lat, corrected_lon, gt_lat, gt_lon)
        }
