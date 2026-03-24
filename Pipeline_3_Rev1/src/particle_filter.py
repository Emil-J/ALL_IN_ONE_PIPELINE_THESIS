"""
Module 8 — Particle Filter for temporal state tracking.

Maintains a weighted set of pose hypotheses (particles) that are
predicted forward with IMU motion and updated with visual match
measurements.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.tile_utils import (
    latlon_to_tile_float,
    tile_to_latlon,
    tile_size_meters,
)


@dataclass
class Particle:
    x: float          # Tile X coordinate (fractional)
    y: float          # Tile Y coordinate (fractional)
    heading: float    # Heading in degrees [0, 360)
    weight: float     # Particle weight [0, 1]

    def to_latlon(self, zoom: int) -> Tuple[float, float]:
        return tile_to_latlon(self.x, self.y, zoom)

    @classmethod
    def from_latlon(cls, lat: float, lon: float, heading: float,
                    weight: float, zoom: int) -> "Particle":
        tx, ty = latlon_to_tile_float(lat, lon, zoom)
        return cls(x=tx, y=ty, heading=heading % 360, weight=weight)


def _angular_difference(a: float, b: float) -> float:
    """Signed shortest angular difference in degrees (a - b)."""
    d = (a - b) % 360
    return d if d <= 180 else d - 360


class ParticleFilter:
    """
    Bootstrap particle filter for 2-D position + heading tracking.
    """

    def __init__(self,
                 num_particles: int,
                 initial_position: Tuple[float, float],
                 initial_heading: float,
                 initial_spread: Dict[str, float],
                 zoom: int = 16,
                 process_noise_pos_m: float = 5.0,
                 process_noise_hdg_deg: float = 2.0,
                 measurement_noise_pos_m: float = 50.0,
                 measurement_noise_hdg_deg: float = 10.0,
                 resample_threshold: float = 0.5,
                 divergence_pos_thresh_m: float = 200.0,
                 divergence_weight_thresh: float = 0.01):
        self.num_particles = num_particles
        self.zoom = zoom
        self._tile_size_m = tile_size_meters(zoom, initial_position[0])

        # Noise parameters
        self.process_noise_pos_m = process_noise_pos_m
        self.process_noise_hdg_deg = process_noise_hdg_deg
        self.measurement_noise_pos_m = measurement_noise_pos_m
        self.measurement_noise_hdg_deg = measurement_noise_hdg_deg
        self.resample_threshold = resample_threshold
        self.divergence_pos_thresh_m = divergence_pos_thresh_m
        self.divergence_weight_thresh = divergence_weight_thresh

        # Initialise particles
        self.particles = self._init_particles(
            initial_position, initial_heading, initial_spread)

    # ────────────────────────────────────────────────────────────
    # Initialisation
    # ────────────────────────────────────────────────────────────

    def _init_particles(self, position: Tuple[float, float],
                        heading: float,
                        spread: Dict[str, float]) -> List[Particle]:
        lat, lon = position
        cx, cy = latlon_to_tile_float(lat, lon, self.zoom)
        pos_std_tiles = spread["position_meters"] / self._tile_size_m
        hdg_std = spread["heading_degrees"]

        rng = np.random.default_rng()
        particles = []
        w0 = 1.0 / self.num_particles
        for _ in range(self.num_particles):
            px = cx + rng.normal(0, pos_std_tiles)
            py = cy + rng.normal(0, pos_std_tiles)
            ph = (heading + rng.normal(0, hdg_std)) % 360
            particles.append(Particle(x=px, y=py, heading=ph, weight=w0))
        return particles

    # ────────────────────────────────────────────────────────────
    # 8.3  predict
    # ────────────────────────────────────────────────────────────

    def predict(self, dt: float, velocity_mps: float, gyro_z_dps: float):
        """Propagate all particles with IMU motion + process noise."""
        rng = np.random.default_rng()
        noise_pos_tiles = self.process_noise_pos_m / self._tile_size_m
        for p in self.particles:
            hdg_rad = math.radians(p.heading)
            dx_m = velocity_mps * math.sin(hdg_rad) * dt   # East component
            dy_m = -velocity_mps * math.cos(hdg_rad) * dt  # North → tile-Y inverted
            p.x += dx_m / self._tile_size_m + rng.normal(0, noise_pos_tiles)
            p.y += dy_m / self._tile_size_m + rng.normal(0, noise_pos_tiles)
            p.heading = (p.heading + gyro_z_dps * dt
                         + rng.normal(0, self.process_noise_hdg_deg)) % 360

    # ────────────────────────────────────────────────────────────
    # 8.4  update
    # ────────────────────────────────────────────────────────────

    def update(self, measurements: List[Dict]):
        """
        Update particle weights from visual match measurements.

        Each measurement: {'position': (tile_x, tile_y),
                           'heading': degrees, 'score': float}
        """
        if not measurements:
            return

        for p in self.particles:
            best_lk = 1e-30  # floor to avoid log(0)
            for m in measurements:
                mx, my = m["position"]
                dx_m = (p.x - mx) * self._tile_size_m
                dy_m = (p.y - my) * self._tile_size_m
                dist2 = dx_m ** 2 + dy_m ** 2
                spatial_lk = math.exp(-dist2 / (2 * self.measurement_noise_pos_m ** 2))

                hdg_diff = abs(_angular_difference(p.heading, m["heading"]))
                hdg_lk = math.exp(-hdg_diff ** 2 / (2 * self.measurement_noise_hdg_deg ** 2))

                lk = m["score"] * spatial_lk * hdg_lk
                if lk > best_lk:
                    best_lk = lk
            p.weight *= best_lk

        # Normalise
        total = sum(p.weight for p in self.particles)
        if total > 0:
            for p in self.particles:
                p.weight /= total
        else:
            w0 = 1.0 / self.num_particles
            for p in self.particles:
                p.weight = w0

    # ────────────────────────────────────────────────────────────
    # 8.5  resample
    # ────────────────────────────────────────────────────────────

    def resample(self):
        """Systematic low-variance resampling when N_eff is low."""
        n_eff = self._n_eff()
        if n_eff >= self.resample_threshold * self.num_particles:
            return  # particles healthy enough

        weights = np.array([p.weight for p in self.particles])
        cumsum = np.cumsum(weights)
        rng = np.random.default_rng()
        r = rng.uniform(0, 1.0 / self.num_particles)
        positions = r + np.arange(self.num_particles) / self.num_particles

        new_particles = []
        idx = 0
        jitter_tiles = 10.0 / self._tile_size_m
        w0 = 1.0 / self.num_particles
        for pos in positions:
            while idx < len(cumsum) - 1 and cumsum[idx] < pos:
                idx += 1
            src = self.particles[idx]
            new_particles.append(Particle(
                x=src.x + rng.normal(0, jitter_tiles),
                y=src.y + rng.normal(0, jitter_tiles),
                heading=(src.heading + rng.normal(0, 2)) % 360,
                weight=w0,
            ))
        self.particles = new_particles

    # ────────────────────────────────────────────────────────────
    # 8.6  get_estimate
    # ────────────────────────────────────────────────────────────

    def get_estimate(self) -> Tuple[float, float, float]:
        """Weighted mean (tile_x, tile_y, heading_deg)."""
        ws = np.array([p.weight for p in self.particles])
        xs = np.array([p.x for p in self.particles])
        ys = np.array([p.y for p in self.particles])

        mean_x = float(np.average(xs, weights=ws))
        mean_y = float(np.average(ys, weights=ws))

        # Circular weighted mean for heading
        hdg_rad = np.deg2rad([p.heading for p in self.particles])
        mean_sin = float(np.average(np.sin(hdg_rad), weights=ws))
        mean_cos = float(np.average(np.cos(hdg_rad), weights=ws))
        mean_hdg = math.degrees(math.atan2(mean_sin, mean_cos)) % 360

        return mean_x, mean_y, mean_hdg

    # ────────────────────────────────────────────────────────────
    # 8.7  get_uncertainty
    # ────────────────────────────────────────────────────────────

    def get_uncertainty(self) -> Dict:
        """Return position and heading standard deviations."""
        ws = np.array([p.weight for p in self.particles])
        xs = np.array([p.x for p in self.particles]) * self._tile_size_m
        ys = np.array([p.y for p in self.particles]) * self._tile_size_m

        mx = float(np.average(xs, weights=ws))
        my = float(np.average(ys, weights=ws))
        var_x = float(np.average((xs - mx) ** 2, weights=ws))
        var_y = float(np.average((ys - my) ** 2, weights=ws))
        pos_std = math.sqrt(var_x + var_y)

        hdg_rad = np.deg2rad([p.heading for p in self.particles])
        s = float(np.average(np.sin(hdg_rad), weights=ws))
        c = float(np.average(np.cos(hdg_rad), weights=ws))
        R = math.sqrt(s ** 2 + c ** 2)
        hdg_std = math.degrees(math.sqrt(-2 * math.log(max(R, 1e-10))))

        return {
            "position_std_m": pos_std,
            "heading_std_deg": hdg_std,
            "n_eff": self._n_eff(),
        }

    # ────────────────────────────────────────────────────────────
    # 8.8  get_search_region
    # ────────────────────────────────────────────────────────────

    def get_search_region(self) -> Dict:
        """95 % confidence region for guiding temporal search."""
        est = self.get_estimate()
        unc = self.get_uncertainty()

        # 3σ radius in tiles (minimum ≈ 100 m ≈ 0.33 tiles)
        radius_m = max(3 * unc["position_std_m"], 100.0)
        radius_tiles = radius_m / self._tile_size_m

        mean_hdg = est[2]
        hdg_range = max(2 * unc["heading_std_deg"], 10.0)

        return {
            "center": (est[0], est[1]),
            "radius_tiles": radius_tiles,
            "heading_mean": mean_hdg,
            "heading_range": hdg_range,
        }

    # ────────────────────────────────────────────────────────────
    # 8.9  check_divergence
    # ────────────────────────────────────────────────────────────

    def check_divergence(self) -> bool:
        unc = self.get_uncertainty()
        if unc["position_std_m"] > self.divergence_pos_thresh_m:
            return True
        max_w = max(p.weight for p in self.particles)
        if max_w < self.divergence_weight_thresh:
            return True
        return False

    # ── internal ──────────────────────────────────────────────

    def _n_eff(self) -> float:
        ws = np.array([p.weight for p in self.particles])
        return 1.0 / max(float(np.sum(ws ** 2)), 1e-30)
