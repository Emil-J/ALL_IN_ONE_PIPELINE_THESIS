"""
TMS (Tile Map Service) utilities for coordinate conversions and tile operations
"""

import math
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass


# Earth parameters
EARTH_RADIUS_METERS = 6371000.0  # Mean radius
EARTH_CIRCUMFERENCE = 2 * math.pi * EARTH_RADIUS_METERS


@dataclass
class TileBounds:
    """Geographic bounds of a tile"""
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
        """Approximate width in meters at center latitude"""
        return haversine_distance(
            self.center_lat, self.min_lon,
            self.center_lat, self.max_lon
        )
    
    @property
    def height_meters(self) -> float:
        """Approximate height in meters"""
        return haversine_distance(
            self.min_lat, self.center_lon,
            self.max_lat, self.center_lon
        )


def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """
    Convert lat/lon to TMS tile coordinates
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        zoom: Zoom level
    
    Returns:
        (tile_x, tile_y) tuple
    """
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    
    tile_x = int((lon + 180.0) / 360.0 * n)
    tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    
    return tile_x, tile_y


def tile_to_latlon(tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float]:
    """
    Convert TMS tile coordinates to center lat/lon
    
    Args:
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
    
    Returns:
        (lat, lon) tuple for tile center
    """
    n = 2.0 ** zoom
    
    lon = tile_x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
    lat = math.degrees(lat_rad)
    
    return lat, lon


def tile_bounds(tile_x: int, tile_y: int, zoom: int) -> TileBounds:
    """
    Get geographic bounds of a tile
    
    Args:
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
    
    Returns:
        TileBounds object
    """
    # Top-left corner
    lat_max, lon_min = tile_to_latlon(tile_x, tile_y, zoom)
    
    # Bottom-right corner
    lat_min, lon_max = tile_to_latlon(tile_x + 1, tile_y + 1, zoom)
    
    return TileBounds(
        min_lat=lat_min,
        max_lat=lat_max,
        min_lon=lon_min,
        max_lon=lon_max
    )


def haversine_distance(lat1: float, lon1: float, 
                       lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points
    
    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)
    
    Returns:
        Distance in meters
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return EARTH_RADIUS_METERS * c


def find_tiles_within_radius(center_lat: float, center_lon: float,
                             radius_meters: float, zoom: int) -> List[Tuple[int, int]]:
    """
    Find all tiles within radius of a center point
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_meters: Search radius in meters
        zoom: Zoom level
    
    Returns:
        List of (tile_x, tile_y) tuples
    """
    # Get center tile
    center_x, center_y = latlon_to_tile(center_lat, center_lon, zoom)
    
    # Estimate tile size in meters at this latitude
    bounds = tile_bounds(center_x, center_y, zoom)
    tile_width_meters = bounds.width_meters
    tile_height_meters = bounds.height_meters
    
    # Calculate search range in tiles (add buffer)
    x_range = int(math.ceil(radius_meters / tile_width_meters)) + 1
    y_range = int(math.ceil(radius_meters / tile_height_meters)) + 1
    
    # Collect candidate tiles
    candidates = []
    for dx in range(-x_range, x_range + 1):
        for dy in range(-y_range, y_range + 1):
            tile_x = center_x + dx
            tile_y = center_y + dy
            
            # Get tile center
            tile_lat, tile_lon = tile_to_latlon(tile_x, tile_y, zoom)
            
            # Check if within radius
            dist = haversine_distance(center_lat, center_lon, tile_lat, tile_lon)
            if dist <= radius_meters:
                candidates.append((tile_x, tile_y))
    
    return candidates


def meters_to_degrees_lat(meters: float) -> float:
    """Convert meters to approximate degrees latitude"""
    return meters / (EARTH_CIRCUMFERENCE / 360.0)


def meters_to_degrees_lon(meters: float, at_latitude: float) -> float:
    """Convert meters to approximate degrees longitude at given latitude"""
    circumference_at_lat = EARTH_CIRCUMFERENCE * math.cos(math.radians(at_latitude))
    return meters / (circumference_at_lat / 360.0)


def ned_to_latlon(north: float, east: float, down: float,
                  origin_lat: float, origin_lon: float, origin_alt: float = 0.0) -> Tuple[float, float, float]:
    """
    Convert NED (North-East-Down) coordinates to lat/lon/alt
    
    Args:
        north, east, down: NED displacement in meters
        origin_lat, origin_lon: Origin coordinates in degrees
        origin_alt: Origin altitude in meters
    
    Returns:
        (lat, lon, alt) tuple
    """
    # Latitude
    dlat = meters_to_degrees_lat(north)
    lat = origin_lat + dlat
    
    # Longitude (account for latitude)
    dlon = meters_to_degrees_lon(east, origin_lat)
    lon = origin_lon + dlon
    
    # Altitude (down is negative in NED)
    alt = origin_alt - down
    
    return lat, lon, alt


def latlon_to_ned(lat: float, lon: float, alt: float,
                  origin_lat: float, origin_lon: float, origin_alt: float = 0.0) -> Tuple[float, float, float]:
    """
    Convert lat/lon/alt to NED coordinates relative to origin
    
    Args:
        lat, lon, alt: Position in degrees and meters
        origin_lat, origin_lon, origin_alt: Origin in degrees and meters
    
    Returns:
        (north, east, down) in meters
    """
    # North
    dlat = lat - origin_lat
    north = dlat * (EARTH_CIRCUMFERENCE / 360.0)
    
    # East
    dlon = lon - origin_lon
    avg_lat = (lat + origin_lat) / 2
    east = dlon * (EARTH_CIRCUMFERENCE / 360.0) * math.cos(math.radians(avg_lat))
    
    # Down (negative altitude change)
    down = origin_alt - alt
    
    return north, east, down


def parse_tile_path(tile_path: Union[str, Path], zoom: int) -> Optional[Tuple[int, int]]:
    """
    Parse tile coordinates from standard TMS path structure
    
    Expected formats:
        - .../zoom/x/y.png
        - .../zoom/x/y.jpeg
        - Any structure with zoom/x/y
    
    Args:
        tile_path: Path to tile file
        zoom: Expected zoom level
    
    Returns:
        (tile_x, tile_y) or None if parsing fails
    """
    tile_path = Path(tile_path)
    parts = tile_path.parts
    
    try:
        # Find zoom level in path
        zoom_str = str(zoom)
        if zoom_str in parts:
            zoom_idx = parts.index(zoom_str)
            if zoom_idx + 2 < len(parts):
                tile_x = int(parts[zoom_idx + 1])
                tile_y = int(Path(parts[zoom_idx + 2]).stem)  # Remove extension
                return tile_x, tile_y
    except (ValueError, IndexError):
        pass
    
    return None


def build_tile_path(base_dir: Union[str, Path], 
                    tile_x: int, tile_y: int, zoom: int,
                    extension: str = "png") -> Path:
    """
    Build standard TMS tile path
    
    Args:
        base_dir: Base directory
        tile_x, tile_y: Tile coordinates
        zoom: Zoom level
        extension: File extension (default: "png")
    
    Returns:
        Full path to tile
    """
    base_dir = Path(base_dir)
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    return base_dir / str(zoom) / str(tile_x) / f"{tile_y}{extension}"


def compute_tile_resolution_meters(zoom: int, latitude: float = 0.0) -> float:
    """
    Compute approximate meters per pixel at given zoom and latitude
    
    Args:
        zoom: Zoom level
        latitude: Latitude in degrees (default: equator)
    
    Returns:
        Meters per pixel
    """
    # Standard tile size
    tile_size_pixels = 256
    
    # At equator, tiles cover equal portions of Earth's circumference
    meters_per_tile = EARTH_CIRCUMFERENCE / (2 ** zoom)
    
    # Adjust for latitude
    meters_per_tile *= math.cos(math.radians(latitude))
    
    # Convert to meters per pixel
    meters_per_pixel = meters_per_tile / tile_size_pixels
    
    return meters_per_pixel


def estimate_pixel_offset(lat1: float, lon1: float,
                         lat2: float, lon2: float,
                         zoom: int, tile_size: int = 256) -> Tuple[float, float]:
    """
    Estimate pixel offset between two geographic points within same tile
    
    Args:
        lat1, lon1: First point
        lat2, lon2: Second point
        zoom: Zoom level
        tile_size: Tile size in pixels
    
    Returns:
        (dx, dy) in pixels
    """
    # Convert to tile coordinates (float)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    n = 2.0 ** zoom
    
    x1 = (lon1 + 180.0) / 360.0 * n * tile_size
    y1 = (1.0 - math.asinh(math.tan(lat1_rad)) / math.pi) / 2.0 * n * tile_size
    
    x2 = (lon2 + 180.0) / 360.0 * n * tile_size
    y2 = (1.0 - math.asinh(math.tan(lat2_rad)) / math.pi) / 2.0 * n * tile_size
    
    dx = x2 - x1
    dy = y2 - y1
    
    return dx, dy
