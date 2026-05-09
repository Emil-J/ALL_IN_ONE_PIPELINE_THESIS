"""
Matching utilities for combining DeDoDe geometric matching with semantic scoring
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def score_candidate(dedode_metrics: Dict,
                   semantic_metrics: Dict,
                   scoring_weights: Dict,
                   semantic_weight: float = 3.0) -> float:
    """
    Compute combined score for a candidate match
    
    Combines DeDoDe geometric matching scores with semantic consistency scores.
    
    Args:
        dedode_metrics: Dict from DeDoDe matching:
            - num_matches: Total number of feature matches
            - num_inliers: Number of RANSAC inliers
            - inlier_ratio: Ratio of inliers to matches
            - median_reproj_error: Median reprojection error
            - mean_confidence: Mean match confidence
        semantic_metrics: Dict from semantic matching:
            - combined: Combined semantic score
        scoring_weights: Weights for DeDoDe components
        semantic_weight: Weight for semantic score in final combination
    
    Returns:
        Combined score (higher is better)
    """
    # Geometric score components
    geo_score = 0.0
    
    if dedode_metrics.get("num_inliers", 0) > 0:
        geo_score += scoring_weights.get("num_inliers", 1.0) * dedode_metrics["num_inliers"]
        geo_score += scoring_weights.get("num_matches", 0.2) * dedode_metrics.get("num_matches", 0)
        geo_score += scoring_weights.get("inlier_ratio", 10.0) * dedode_metrics.get("inlier_ratio", 0)
        
        # Confidence (if available)
        if "mean_confidence" in dedode_metrics:
            geo_score += scoring_weights.get("median_confidence", 2.0) * dedode_metrics["mean_confidence"]
        
        # Reprojection error (lower is better, so negative weight)
        reproj_error = dedode_metrics.get("median_reproj_error", np.inf)
        if reproj_error < np.inf:
            geo_score += scoring_weights.get("reprojection_error", -0.5) * reproj_error
    
    # Semantic score
    sem_score = semantic_metrics.get("combined", 0.0) * semantic_weight
    
    # Combined score
    final_score = geo_score + sem_score
    
    return float(final_score)


def rank_candidates(candidates: List[Dict],
                   use_semantics: bool = True) -> List[Dict]:
    """
    Rank candidates by combined score
    
    Args:
        candidates: List of candidate dicts with keys:
            - tile_info: Tile metadata
            - dedode_metrics: DeDoDe matching metrics
            - semantic_metrics: Semantic matching metrics (optional)
            - combined_score: Final score
        use_semantics: Whether semantic scores were computed
    
    Returns:
        Sorted list of candidates (best first)
    """
    # Sort by combined_score (descending)
    ranked = sorted(candidates, key=lambda x: x.get("combined_score", -np.inf), reverse=True)
    
    # Add rank field
    for rank, cand in enumerate(ranked, 1):
        cand["rank"] = rank
    
    return ranked


def filter_candidates_by_score(candidates: List[Dict],
                               min_score: Optional[float] = None,
                               min_inliers: int = 8,
                               top_k: Optional[int] = None) -> List[Dict]:
    """
    Filter candidates by score and inlier thresholds
    
    Args:
        candidates: List of candidate dicts
        min_score: Minimum combined score (optional)
        min_inliers: Minimum number of RANSAC inliers
        top_k: Keep only top K candidates (optional)
    
    Returns:
        Filtered and sorted list of candidates
    """
    filtered = []
    
    for cand in candidates:
        # Check inliers
        if cand.get("dedode_metrics", {}).get("num_inliers", 0) < min_inliers:
            continue
        
        # Check score
        if min_score is not None and cand.get("combined_score", -np.inf) < min_score:
            continue
        
        filtered.append(cand)
    
    # Sort by score
    filtered = rank_candidates(filtered)
    
    # Keep top K
    if top_k:
        filtered = filtered[:top_k]
    
    return filtered


def select_best_candidate(candidates: List[Dict],
                         confidence_threshold: Optional[float] = None) -> Optional[Dict]:
    """
    Select best candidate from ranked list
    
    Args:
        candidates: Ranked list of candidates
        confidence_threshold: Optional minimum confidence
    
    Returns:
        Best candidate dict or None if no valid candidates
    """
    if not candidates:
        return None
    
    # Already sorted by rank
    best = candidates[0]
    
    # Check confidence threshold
    if confidence_threshold is not None:
        score = best.get("combined_score", 0)
        if score < confidence_threshold:
            return None
    
    return best


def compute_match_statistics(candidates: List[Dict]) -> Dict:
    """
    Compute statistics over all candidates
    
    Args:
        candidates: List of candidate dicts
    
    Returns:
        Dict with statistics:
            - num_candidates: Number of candidates
            - mean_score: Mean combined score
            - max_score: Maximum score
            - mean_inliers: Mean number of inliers
            - max_inliers: Maximum inliers
    """
    if not candidates:
        return {
            "num_candidates": 0,
            "mean_score": 0.0,
            "max_score": 0.0,
            "mean_inliers": 0.0,
            "max_inliers": 0
        }
    
    scores = [c.get("combined_score", 0) for c in candidates]
    inliers = [c.get("dedode_metrics", {}).get("num_inliers", 0) for c in candidates]
    
    return {
        "num_candidates": len(candidates),
        "mean_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
        "mean_inliers": float(np.mean(inliers)),
        "max_inliers": int(np.max(inliers))
    }


def extract_match_features_for_logging(candidate: Dict) -> Dict:
    """
    Extract key features from candidate for logging
    
    Args candidate: Candidate dict
    
    Returns:
        Flat dict with key metrics for CSV logging
    """
    tile_info = candidate.get("tile_info", {})
    dedode = candidate.get("dedode_metrics", {})
    semantic = candidate.get("semantic_metrics", {})
    
    return {
        "rank": candidate.get("rank", -1),
        "tile_x": tile_info.get("tile_x", -1),
        "tile_y": tile_info.get("tile_y", -1),
        "tile_lat": tile_info.get("lat", np.nan),
        "tile_lon": tile_info.get("lon", np.nan),
        "num_matches": dedode.get("num_matches", 0),
        "num_inliers": dedode.get("num_inliers", 0),
        "inlier_ratio": dedode.get("inlier_ratio", 0.0),
        "reproj_error": dedode.get("median_reproj_error", np.nan),
        "match_confidence": dedode.get("mean_confidence", np.nan),
        "semantic_iou": semantic.get("iou", np.nan),
        "semantic_boundary": semantic.get("boundary", np.nan),
        "semantic_combined": semantic.get("combined", np.nan),
        "combined_score": candidate.get("combined_score", np.nan)
    }
