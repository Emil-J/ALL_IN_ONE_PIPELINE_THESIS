"""
Semantic Matching Utilities

Functions for computing semantic consistency scores between query and reference masks.
Used as an additional cue for robust matching alongside DeDoDe geometric matching.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


def compute_iou(mask1: np.ndarray, mask2: np.ndarray, 
                filter_classes: Optional[List[int]] = None) -> float:
    """
    Compute Intersection over Union (IoU) between two semantic masks
    
    Args:
        mask1: First mask (H, W) with class indices
        mask2: Second mask (H, W) with class indices
        filter_classes: If provided, only compute IoU for these classes
    
    Returns:
        IoU score [0, 1]
    """
    if mask1.shape != mask2.shape:
        # Resize to match
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    if filter_classes:
        # Create binary masks for relevant classes
        binary1 = np.isin(mask1, filter_classes)
        binary2 = np.isin(mask2, filter_classes)
    else:
        # Use all classes
        binary1 = mask1 > 0
        binary2 = mask2 > 0
    
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def compute_class_iou(mask1: np.ndarray, mask2: np.ndarray,
                      filter_classes: Optional[List[int]] = None) -> Dict[int, float]:
    """
    Compute per-class IoU scores
    
    Args:
        mask1, mask2: Semantic masks
        filter_classes: Classes to compute IoU for
    
    Returns:
        Dict mapping class_id -> IoU score
    """
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    classes = filter_classes if filter_classes else np.unique(np.concatenate([mask1, mask2]))
    
    iou_scores = {}
    for class_id in classes:
        binary1 = (mask1 == class_id)
        binary2 = (mask2 == class_id)
        
        intersection = np.logical_and(binary1, binary2).sum()
        union = np.logical_or(binary1, binary2).sum()
        
        if union == 0:
            iou_scores[class_id] = 0.0
        else:
            iou_scores[class_id] = float(intersection) / float(union)
    
    return iou_scores


def compute_boundary_overlap(mask1: np.ndarray, mask2: np.ndarray,
                             filter_classes: Optional[List[int]] = None,
                             kernel_size: int = 3) -> float:
    """
    Compute boundary overlap score between two masks
    
    Extracts boundaries and computes their overlap.
    Useful for matching shapes even with small misalignments.
    
    Args:
        mask1, mask2: Semantic masks
        filter_classes: Classes to extract boundaries from
        kernel_size: Kernel size for boundary extraction
    
    Returns:
        Boundary overlap score [0, 1]
    """
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create binary masks
    if filter_classes:
        binary1 = np.isin(mask1, filter_classes).astype(np.uint8)
        binary2 = np.isin(mask2, filter_classes).astype(np.uint8)
    else:
        binary1 = (mask1 > 0).astype(np.uint8)
        binary2 = (mask2 > 0).astype(np.uint8)
    
    # Extract boundaries using morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    boundary1 = cv2.morphologyEx(binary1, cv2.MORPH_GRADIENT, kernel)
    boundary2 = cv2.morphologyEx(binary2, cv2.MORPH_GRADIENT, kernel)
    
    # Compute overlap
    intersection = np.logical_and(boundary1, boundary2).sum()
    union = np.logical_or(boundary1, boundary2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def compute_dice_coefficient(mask1: np.ndarray, mask2: np.ndarray,
                             filter_classes: Optional[List[int]] = None) -> float:
    """
    Compute Dice coefficient (F1-score equivalent for segmentation)
    
    Args:
        mask1, mask2: Semantic masks
        filter_classes: Classes to compute Dice for
    
    Returns:
        Dice score [0, 1]
    """
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    if filter_classes:
        binary1 = np.isin(mask1, filter_classes)
        binary2 = np.isin(mask2, filter_classes)
    else:
        binary1 = mask1 > 0
        binary2 = mask2 > 0
    
    intersection = np.logical_and(binary1, binary2).sum()
    
    sum_masks = binary1.sum() + binary2.sum()
    if sum_masks == 0:
        return 0.0
    
    return 2.0 * float(intersection) / float(sum_masks)


def compute_semantic_score(query_mask: np.ndarray, 
                          reference_mask: np.ndarray,
                          filter_classes: List[int] = [2, 4, 5],
                          score_weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    Compute combined semantic consistency score
    
    This is the main function for semantic matching scoring.
    
    Args:
        query_mask: Query semantic mask (H, W)
        reference_mask: Reference semantic mask (H, W)
        filter_classes: Classes to use for scoring (default: land, roads, buildings)
        score_weights: Weights for individual score components
            - "iou": Weight for IoU score
            - "boundary": Weight for boundary overlap
            - "dice": Weight for Dice coefficient
    
    Returns:
        Dict with keys:
            - "iou": IoU score
            - "boundary": Boundary overlap score
            - "dice": Dice coefficient
            - "combined": Weighted combined score
            - "per_class_iou": Dict of per-class IoU scores
    """
    if score_weights is None:
        score_weights = {
            "iou": 1.0,
            "boundary": 0.5,
            "dice": 0.0  # Optional, disabled by default
        }
    
    # Compute individual scores
    iou = compute_iou(query_mask, reference_mask, filter_classes)
    boundary = compute_boundary_overlap(query_mask, reference_mask, filter_classes)
    dice = compute_dice_coefficient(query_mask, reference_mask, filter_classes)
    per_class_iou = compute_class_iou(query_mask, reference_mask, filter_classes)
    
    # Compute weighted combined score
    combined = (
        score_weights.get("iou", 1.0) * iou +
        score_weights.get("boundary", 0.5) * boundary +
        score_weights.get("dice", 0.0) * dice
    )
    
    # Normalize by sum of weights
    total_weight = sum(score_weights.values())
    if total_weight > 0:
        combined /= total_weight
    
    return {
        "iou": float(iou),
        "boundary": float(boundary),
        "dice": float(dice),
        "combined": float(combined),
        "per_class_iou": {int(k): float(v) for k, v in per_class_iou.items()}
    }


def compute_histogram_similarity(mask1: np.ndarray, mask2: np.ndarray,
                                 num_classes: int = 6) -> float:
    """
    Compute histogram similarity between masks
    
    Compares class distribution rather than spatial layout.
    Useful as a quick filter before more expensive matching.
    
    Args:
        mask1, mask2: Semantic masks
        num_classes: Total number of classes
    
    Returns:
        Histogram similarity score [0, 1]
    """
    # Compute normalized histograms
    hist1, _ = np.histogram(mask1, bins=num_classes, range=(0, num_classes))
    hist2, _ = np.histogram(mask2, bins=num_classes, range=(0, num_classes))
    
    hist1 = hist1.astype(float) / hist1.sum() if hist1.sum() > 0 else hist1
    hist2 = hist2.astype(float) / hist2.sum() if hist2.sum() > 0 else hist2
    
    # Compute correlation or intersection
    intersection = np.minimum(hist1, hist2).sum()
    
    return float(intersection)


def filter_candidates_by_semantic_precheck(query_mask: np.ndarray,
                                           candidate_masks: List[np.ndarray],
                                           min_histogram_similarity: float = 0.3,
                                           num_classes: int = 6) -> List[int]:
    """
    Fast pre-filtering of candidates based on histogram similarity
    
    Use this before expensive DeDoDe matching to quickly discard
    unlikely candidates.
    
    Args:
        query_mask: Query semantic mask
        candidate_masks: List of candidate masks
        min_histogram_similarity: Minimum similarity threshold
        num_classes: Number of semantic classes
    
    Returns:
        List of indices of candidates that pass the filter
    """
    passed_indices = []
    
    for idx, candidate_mask in enumerate(candidate_masks):
        sim = compute_histogram_similarity(query_mask, candidate_mask, num_classes)
        if sim >= min_histogram_similarity:
            passed_indices.append(idx)
    
    return passed_indices


def compute_spatial_distribution_similarity(mask1: np.ndarray, mask2: np.ndarray,
                                           filter_classes: Optional[List[int]] = None,
                                           grid_size: int = 4) -> float:
    """
    Compute similarity of spatial class distribution
    
    Divides masks into grid and compares class presence in each cell.
    Useful for coarse layout matching.
    
    Args:
        mask1, mask2: Semantic masks
        filter_classes: Classes to consider
        grid_size: Grid size (e.g., 4 = 4x4 grid)
    
    Returns:
        Spatial similarity score [0, 1]
    """
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    h, w = mask1.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    matches = 0
    total = 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Extract cell
            y1, y2 = i * cell_h, (i+1) * cell_h
            x1, x2 = j * cell_w, (j+1) * cell_w
            
            cell1 = mask1[y1:y2, x1:x2]
            cell2 = mask2[y1:y2, x1:x2]
            
            # Get class presence
            if filter_classes:
                present1 = set(np.unique(cell1)) & set(filter_classes)
                present2 = set(np.unique(cell2)) & set(filter_classes)
            else:
                present1 = set(np.unique(cell1))
                present2 = set(np.unique(cell2))
            
            # Compute Jaccard similarity for this cell
            if present1 or present2:
                cell_sim = len(present1 & present2) / len(present1 | present2)
                matches += cell_sim
                total += 1
    
    if total == 0:
        return 0.0
    
    return matches / total
