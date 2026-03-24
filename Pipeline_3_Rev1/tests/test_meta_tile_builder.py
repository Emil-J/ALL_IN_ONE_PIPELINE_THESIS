"""Unit tests for Module 10 — Meta-Tile Builder."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
from types import SimpleNamespace


def _make_config(tmp_path):
    return SimpleNamespace(
        TMS_ZOOM_LEVEL=16,
        TILE_X_MIN=34494, TILE_X_MAX=34508,
        TILE_Y_MIN=45025, TILE_Y_MAX=45042,
        FIRST_PASS_SEARCH_RADIUS_M=300.0,
        METATILE_TOP_K=3,
        METATILE_MATCH_THRESHOLD=25,
        TMS_TILE_SIZE_PX=512,
        METATILE_OUTPUT_DIR=tmp_path / "metatiles",
    )


def _make_tile_loader():
    loader = MagicMock()
    loader.exists.return_value = True
    loader.load_aerial.return_value = np.random.randint(
        0, 255, (512, 512, 3), dtype=np.uint8)
    return loader


def _make_matcher(match_count=50):
    matcher = MagicMock()
    matcher.match.return_value = {
        "keypoints1": np.zeros((100, 2)),
        "keypoints2": np.zeros((100, 2)),
        "matches": np.column_stack([np.arange(match_count),
                                     np.arange(match_count)]),
        "match_scores": np.ones(match_count),
        "num_matches": match_count,
    }
    return matcher


@pytest.fixture
def builder(tmp_path):
    from src.meta_tile_builder import MetaTileBuilder
    cfg = _make_config(tmp_path)
    return MetaTileBuilder(_make_matcher(), _make_tile_loader(), cfg)


def test_first_pass_returns_sorted(builder):
    query = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    # Patch find_tiles_within_radius to return a few tiles
    with patch("src.meta_tile_builder.find_tiles_within_radius",
               return_value=[(34500, 45030), (34501, 45030)]):
        results = builder.first_pass(query, 55.7, 9.5, 300)
    assert len(results) == 2
    assert results[0][2] >= results[1][2]


def test_second_pass_at_most_9(builder):
    query = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    results = builder.second_pass(query, 34500, 45030)
    assert len(results) <= 9


def test_build_meta_tile_shape(builder):
    top3 = [(34500, 45030, 80), (34501, 45030, 60), (34500, 45031, 50)]
    meta = builder.build_meta_tile(top3)
    assert meta.dtype == np.uint8
    assert meta.shape[2] == 3
    # Should span 2 cols × 2 rows × 512 px
    assert meta.shape == (2 * 512, 2 * 512, 3)


def test_save_meta_tile_filename(builder, tmp_path):
    builder.cfg.METATILE_OUTPUT_DIR = tmp_path
    meta = np.zeros((512, 512, 3), dtype=np.uint8)
    path = builder.save_meta_tile(meta, query_timestamp=0.523)
    assert path.name == "metatile_0.523.png"
    assert path.exists()


def test_run_returns_complete_dict(builder, tmp_path):
    builder.cfg.METATILE_OUTPUT_DIR = tmp_path / "mt"
    query = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    with patch("src.meta_tile_builder.find_tiles_within_radius",
               return_value=[(34500, 45030), (34501, 45030)]):
        result = builder.run(query, 55.7, 9.5, query_timestamp=0.523)
    assert result is not None
    assert "meta_tile" in result
    assert "meta_tile_path" in result
    assert "top3_tiles" in result
    assert "verification_matches" in result
    assert "verified" in result
    assert "first_pass_candidates" in result


def test_run_returns_none_when_no_tiles(builder):
    query = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    with patch("src.meta_tile_builder.find_tiles_within_radius",
               return_value=[]):
        result = builder.run(query, 55.7, 9.5, query_timestamp=0.523)
    assert result is None
