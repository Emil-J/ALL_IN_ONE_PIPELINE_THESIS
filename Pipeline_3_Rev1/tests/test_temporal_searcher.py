"""Unit tests for Module 9 — Temporal Searcher."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from pathlib import Path


def _make_config(tmp_path):
    return SimpleNamespace(
        TMS_ZOOM_LEVEL=16,
        TILE_X_MIN=34494, TILE_X_MAX=34508,
        TILE_Y_MIN=45025, TILE_Y_MAX=45042,
        TILE_SIZE_METERS=305.0,
        TMS_TILE_SIZE_PX=512,
        IMU_SEARCH_RADIUS_METERS=350.0,
        MAX_SEARCH_ITERATIONS=200,
        TOP_K_CANDIDATES=5,
        QUERY_RESIZE_WIDTH=512,
        QUERY_RESIZE_HEIGHT=288,
        SEMANTIC_INPUT_SIZE=512,
        MAX_NUM_KEYPOINTS=2048,
        MIN_MATCHES_FOR_HOMOGRAPHY=4,
        RANSAC_REPROJ_THRESH=8.0,
        NUM_PARTICLES=50,
        PROCESS_NOISE_POSITION_M=5.0,
        PROCESS_NOISE_HEADING_DEG=2.0,
        MEASUREMENT_NOISE_POSITION_M=50.0,
        MEASUREMENT_NOISE_HEADING_DEG=10.0,
        RESAMPLE_THRESHOLD=0.5,
        DIVERGENCE_POSITION_THRESHOLD_M=200.0,
        DIVERGENCE_WEIGHT_THRESHOLD=0.01,
        PARTICLE_INIT_SPREAD_HIGH_CONF={"position_meters": 50, "heading_degrees": 10},
        PARTICLE_INIT_SPREAD_MED_CONF={"position_meters": 100, "heading_degrees": 20},
        PARTICLE_INIT_SPREAD_LOW_CONF={"position_meters": 200, "heading_degrees": 30},
        FIRST_PASS_SEARCH_RADIUS_M=300.0,
        SECOND_PASS_NEIGHBOURS=8,
        METATILE_TOP_K=3,
        METATILE_MATCH_THRESHOLD=25,
        METATILE_OUTPUT_DIR=tmp_path / "metatiles",
        SEMANTIC_OUTPUT_DIR=tmp_path / "semantic",
        LOG_OUTPUT_DIR=tmp_path / "logs",
        TRAJECTORY_OUTPUT_DIR=tmp_path / "trajectories",
        CENTROID_MATCH_DISTANCE_THRESHOLD_PX=50,
        SEMANTIC_CONFIRM_MIN_PAIRS=3,
    )


def _make_imu_data():
    return {
        "lat": 55.7, "lon": 9.5, "heading": 180,
        "pos_sigma": 100, "heading_sigma": 5,
        "velocity_mps": 20, "gyro_z_dps": 0,
    }


def _make_matcher(count=50):
    matcher = MagicMock()
    matcher.match.return_value = {
        "keypoints1": np.zeros((100, 2)),
        "keypoints2": np.zeros((100, 2)),
        "matches": np.column_stack([np.arange(count), np.arange(count)]),
        "match_scores": np.ones(count),
        "num_matches": count,
    }
    return matcher


def _make_semantic_model():
    model = MagicMock()
    model.predict.return_value = np.zeros((512, 512), dtype=np.uint8)
    return model


def _make_tile_loader():
    loader = MagicMock()
    loader.exists.return_value = True
    loader.load_aerial.return_value = np.random.randint(
        0, 255, (512, 512, 3), dtype=np.uint8)
    return loader


@pytest.fixture
def searcher(tmp_path):
    from src.temporal_searcher import TemporalSearcher
    cfg = _make_config(tmp_path)
    return TemporalSearcher(
        _make_semantic_model(), _make_matcher(),
        _make_tile_loader(), cfg,
    )


def test_frame_0_is_cold_start(searcher):
    frame = np.random.randint(0, 255, (1079, 1920, 3), dtype=np.uint8)
    with patch("src.best_first_search.find_tiles_within_radius",
               return_value=[(34500, 45030)]):
        result = searcher.process_frame(frame, _make_imu_data(), 0.523)
    assert result["method"] == "cold_start"
    assert "position" in result
    assert "tiles_tested" in result


def test_frame_1_is_temporal(searcher):
    frame = np.random.randint(0, 255, (1079, 1920, 3), dtype=np.uint8)
    # Frame 0
    with patch("src.best_first_search.find_tiles_within_radius",
               return_value=[(34500, 45030)]):
        searcher.process_frame(frame, _make_imu_data(), 0.523)
    # Frame 1
    with patch("src.meta_tile_builder.find_tiles_within_radius",
               return_value=[(34500, 45030), (34501, 45030)]):
        result = searcher.process_frame(frame, _make_imu_data(), 1.113)
    assert result["method"] == "temporal_tracking"
    assert "meta_tile_path" in result
    assert "semantic_confidence" in result


def test_trajectory_saving(searcher, tmp_path):
    frame = np.random.randint(0, 255, (1079, 1920, 3), dtype=np.uint8)
    with patch("src.best_first_search.find_tiles_within_radius",
               return_value=[(34500, 45030)]):
        searcher.process_frame(frame, _make_imu_data(), 0.523)

    filepath = tmp_path / "traj.csv"
    searcher.save_trajectory(filepath)
    assert filepath.exists()
    searcher.close()
