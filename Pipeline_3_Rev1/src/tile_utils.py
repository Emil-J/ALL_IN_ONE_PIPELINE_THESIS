"""
TMS tile utilities — coordinate conversions, tile loading, path building.
Adapted from dedode_localization_project/src/tms_utils.py.
"""

import math
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
from PIL import Image

EARTH_RADIUS_METERS = 6371000.0
EARTH_CIRCUMFERENCE = 2 * math.pi * EARTH_RADIUS_METERS


# ─── Coordinate conversions ───────────────────────────────────────

def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert (lat, lon) → integer TMS tile (x, y)."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    tile_x = int((lon + 180.0) / 360.0 * n)
    tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return tile_x, tile_y


def latlon_to_tile_float(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    """Convert (lat, lon) → fractional tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    tile_x = (lon + 180.0) / 360.0 * n
    tile_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return tile_x, tile_y


def tile_to_latlon(tile_x: float, tile_y: float, zoom: int) -> Tuple[float, float]:
    """Convert tile (x, y) → center (lat, lon). Accepts float or int."""
    n = 2.0 ** zoom
    lon = tile_x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


@dataclass
class TileBounds:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    @property
    def center_lat(self) -> float:
        return (self.min_lat + self.max_lat) / 2

    @property
    def center_lon(self) -> float:
        return (self.min_lon + self.max_lon) / 2

    @property
    def width_meters(self) -> float:
        return haversine_distance(self.center_lat, self.min_lon,
                                  self.center_lat, self.max_lon)

    @property
    def height_meters(self) -> float:
        return haversine_distance(self.min_lat, self.center_lon,
                                  self.max_lat, self.center_lon)


def tile_bounds(tile_x: int, tile_y: int, zoom: int) -> TileBounds:
    lat_max, lon_min = tile_to_latlon(tile_x, tile_y, zoom)
    lat_min, lon_max = tile_to_latlon(tile_x + 1, tile_y + 1, zoom)
    return TileBounds(min_lat=lat_min, max_lat=lat_max,
                      min_lon=lon_min, max_lon=lon_max)


# ─── Distance ─────────────────────────────────────────────────────

def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """Distance in metres between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_METERS * 2 * math.asin(math.sqrt(a))


# ─── Tile search ──────────────────────────────────────────────────

def find_tiles_within_radius(center_lat: float, center_lon: float,
                             radius_meters: float, zoom: int,
                             x_range: Optional[Tuple[int, int]] = None,
                             y_range: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """
    Return all tiles whose centre is within *radius_meters* of the given point.
    Optionally clamp to the reference map coordinate ranges.
    """
    cx, cy = latlon_to_tile(center_lat, center_lon, zoom)
    bounds = tile_bounds(cx, cy, zoom)
    tw = bounds.width_meters
    th = bounds.height_meters

    dx = int(math.ceil(radius_meters / tw)) + 1
    dy = int(math.ceil(radius_meters / th)) + 1

    candidates = []
    for tdx in range(-dx, dx + 1):
        for tdy in range(-dy, dy + 1):
            tx = cx + tdx
            ty = cy + tdy
            if x_range and not (x_range[0] <= tx <= x_range[1]):
                continue
            if y_range and not (y_range[0] <= ty <= y_range[1]):
                continue
            tlat, tlon = tile_to_latlon(tx + 0.5, ty + 0.5, zoom)
            if haversine_distance(center_lat, center_lon, tlat, tlon) <= radius_meters:
                candidates.append((tx, ty))
    return candidates


# ─── Tile size helpers ────────────────────────────────────────────

def tile_size_meters(zoom: int, latitude: float) -> float:
    """Approximate edge length of a tile in metres at *latitude*."""
    return (EARTH_CIRCUMFERENCE * math.cos(math.radians(latitude))) / (2 ** zoom)


# ─── Path building / loading ──────────────────────────────────────

def build_tile_path(base_dir: Union[str, Path], tile_x: int, tile_y: int,
                    zoom: int, ext: str = "png") -> Path:
    base_dir = Path(base_dir)
    return base_dir / str(zoom) / str(tile_x) / f"{tile_y}.{ext}"


class TileLoader:
    """Convenience wrapper for loading aerial tiles and prediction masks."""

    def __init__(self, aerial_dir: Union[str, Path],
                 prediction_dir: Optional[Union[str, Path]] = None,
                 zoom: int = 16,
                 x_range: Optional[Tuple[int, int]] = None,
                 y_range: Optional[Tuple[int, int]] = None):
        self.aerial_dir = Path(aerial_dir)
        self.prediction_dir = Path(prediction_dir) if prediction_dir else None
        self.zoom = zoom
        self.x_range = x_range
        self.y_range = y_range

    # ── existence ──

    def exists(self, tile_x: int, tile_y: int) -> bool:
        return build_tile_path(self.aerial_dir, tile_x, tile_y, self.zoom).exists()

    # ── loading ──

    def load_aerial(self, tile_x: int, tile_y: int) -> Optional[np.ndarray]:
        p = build_tile_path(self.aerial_dir, tile_x, tile_y, self.zoom)
        if not p.exists():
            return None
        return np.array(Image.open(p).convert("RGB"))

    def load_prediction(self, tile_x: int, tile_y: int) -> Optional[np.ndarray]:
        if self.prediction_dir is None:
            return None
        p = build_tile_path(self.prediction_dir, tile_x, tile_y, self.zoom)
        if not p.exists():
            return None
        return np.array(Image.open(p))

    # ── enumeration ──

    def list_tiles(self) -> List[Tuple[int, int]]:
        """Return all (tile_x, tile_y) pairs on disk."""
        tiles = []
        zoom_dir = self.aerial_dir / str(self.zoom)
        if not zoom_dir.exists():
            return tiles
        for x_dir in sorted(zoom_dir.iterdir()):
            if not x_dir.is_dir():
                continue
            try:
                tx = int(x_dir.name)
            except ValueError:
                continue
            for y_file in sorted(x_dir.iterdir()):
                if y_file.suffix.lower() != ".png":
                    continue
                try:
                    ty = int(y_file.stem)
                except ValueError:
                    continue
                tiles.append((tx, ty))
        return tiles
