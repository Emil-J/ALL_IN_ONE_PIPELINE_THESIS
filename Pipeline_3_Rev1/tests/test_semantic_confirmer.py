"""Unit tests for Module 11 — Semantic Confirmer."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from types import SimpleNamespace


def _make_config():
    return SimpleNamespace(
        QUERY_RESIZE_WIDTH=512,
        QUERY_RESIZE_HEIGHT=288,
        SEMANTIC_INPUT_SIZE=512,
        CENTROID_MATCH_DISTANCE_THRESHOLD_PX=50,
        SEMANTIC_CONFIRM_MIN_PAIRS=3,
    )


def _make_semantic_model():
    model = MagicMock()
    # predict returns a 512×512 mask with a few classes
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 1  # forest block
    mask[300:400, 300:400] = 4  # road block
    mask[50:80, 400:450] = 5    # building block
    model.predict.return_value = mask
    return model


@pytest.fixture
def confirmer():
    from src.semantic_confirmer import SemanticConfirmer
    return SemanticConfirmer(_make_semantic_model(), _make_config())


def test_extract_centroids_basic(confirmer):
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 1
    mask[300:400, 300:400] = 4
    centroids = confirmer.extract_centroids(mask)
    assert len(centroids) >= 2
    classes_found = {c["class"] for c in centroids}
    assert 1 in classes_found
    assert 4 in classes_found


def test_extract_centroids_ignores_tiny_fragments(confirmer):
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[0, 0] = 3  # single pixel = area 1 < 10
    centroids = confirmer.extract_centroids(mask)
    # No centroids for class 3 (too small)
    assert all(c["class"] != 3 for c in centroids)


def test_match_centroids_perfect(confirmer):
    """Same centroids should give high match ratio."""
    centroids = [
        {"class": 1, "cx": 150.0, "cy": 150.0, "area": 10000},
        {"class": 4, "cx": 350.0, "cy": 350.0, "area": 10000},
        {"class": 5, "cx": 425.0, "cy": 65.0, "area": 1500},
    ]
    result = confirmer.match_centroids(centroids, centroids)
    assert result["matched_pairs"] == 3
    assert result["match_ratio"] == 1.0
    assert result["confidence"] > 0


def test_match_centroids_none(confirmer):
    """Different classes → no matches."""
    q = [{"class": 1, "cx": 100, "cy": 100, "area": 5000}]
    r = [{"class": 4, "cx": 100, "cy": 100, "area": 5000}]
    result = confirmer.match_centroids(q, r)
    assert result["matched_pairs"] == 0


def test_confirm_returns_all_keys(confirmer):
    query_mask = np.zeros((512, 512), dtype=np.uint8)
    query_mask[100:200, 100:200] = 1
    meta_tile = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    result = confirmer.confirm(query_mask, meta_tile)
    assert "matched_pairs" in result
    assert "match_ratio" in result
    assert "confidence" in result
    assert "query_centroids" in result
    assert "meta_tile_centroids" in result
