"""Quick validation script — run all module checks without pytest."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

# ── 1. Particle Filter ─────────────────────────────────────────
print("=== Particle Filter ===")
from src.particle_filter import ParticleFilter, Particle

pf = ParticleFilter(100, (55.7, 9.5), 90,
                    {"position_meters": 50, "heading_degrees": 10}, 16)
assert len(pf.particles) == 100
print("  PASS: init 100 particles")

assert abs(sum(p.weight for p in pf.particles) - 1.0) < 1e-6
print("  PASS: weights sum to 1")

before = pf.particles[0].x
pf.predict(1.0, 10.0, 5.0)
assert pf.particles[0].x != before
print("  PASS: predict moves particles")

est = pf.get_estimate()
assert len(est) == 3
print(f"  PASS: estimate = ({est[0]:.4f}, {est[1]:.4f}, {est[2]:.1f})")

measurements = [{"position": est[:2], "heading": est[2], "score": 150.0}]
pf.update(measurements)
assert abs(sum(p.weight for p in pf.particles) - 1.0) < 0.01
print("  PASS: update normalises weights")

pf.particles[0].weight = 0.99
for i in range(1, 100):
    pf.particles[i].weight = 0.01 / 99
pf.resample()
print("  PASS: resample complete")

region = pf.get_search_region()
assert "center" in region and "radius_tiles" in region
print(f"  PASS: search region radius={region['radius_tiles']:.2f} tiles")

assert pf.check_divergence() is False
print("  PASS: no divergence initially")

p = Particle.from_latlon(55.7, 9.5, 90, 1.0, 16)
lat, lon = p.to_latlon(16)
assert abs(lat - 55.7) < 0.01 and abs(lon - 9.5) < 0.01
print("  PASS: latlon round-trip")

# ── 2. Trajectory Smoother ─────────────────────────────────────
print("\n=== Trajectory Smoother ===")
from src.trajectory_smoother import smooth_trajectory, detect_outliers, fill_gaps

rng = np.random.default_rng(42)
positions = [(55.7 + 0.0001*i + rng.normal(0, 0.00005),
              9.5  + 0.0001*i + rng.normal(0, 0.00005),
              180.0, i*0.46)
             for i in range(20)]

smoothed = smooth_trajectory(positions, method="kalman")
assert len(smoothed) == len(positions)
print("  PASS: kalman smoothing preserves length")

smoothed_ma = smooth_trajectory(positions, method="moving_average")
assert len(smoothed_ma) == len(positions)
print("  PASS: moving_average smoothing preserves length")

outliers = detect_outliers(positions, threshold_meters=100)
print(f"  PASS: outlier detection ({len(outliers)} outliers)")

# Insert a jump
pos_jump = list(positions)
lat0, lon0, h0, t0 = pos_jump[10]
pos_jump[10] = (lat0 + 0.1, lon0 + 0.1, h0, t0)
outliers2 = detect_outliers(pos_jump, threshold_meters=100)
assert 10 in outliers2
print("  PASS: detects injected outlier at index 10")

# Gap filling
gapped = [positions[0], positions[1], positions[-1]]
filled = fill_gaps(gapped, expected_dt=0.46)
assert len(filled) >= len(gapped)
print(f"  PASS: gap filling ({len(gapped)} -> {len(filled)} points)")

# ── 3. Tile Utils ──────────────────────────────────────────────
print("\n=== Tile Utils ===")
from src.tile_utils import (latlon_to_tile, tile_to_latlon,
                            latlon_to_tile_float, haversine_distance,
                            find_tiles_within_radius, TileLoader, tile_size_meters)

tx, ty = latlon_to_tile(55.7, 9.5, 16)
print(f"  PASS: latlon_to_tile(55.7, 9.5, 16) = ({tx}, {ty})")

lat_back, lon_back = tile_to_latlon(tx + 0.5, ty + 0.5, 16)
assert abs(lat_back - 55.7) < 0.1 and abs(lon_back - 9.5) < 0.1
print(f"  PASS: round-trip ({lat_back:.4f}, {lon_back:.4f})")

d = haversine_distance(55.7, 9.5, 55.7, 9.501)
assert d > 0
print(f"  PASS: haversine_distance = {d:.1f} m")

tsm = tile_size_meters(16, 55.7)
print(f"  PASS: tile_size_meters = {tsm:.1f} m")

tiles = find_tiles_within_radius(55.7, 9.5, 500, 16)
print(f"  PASS: find_tiles_within_radius -> {len(tiles)} tiles")

# ── 4. Image Utils ────────────────────────────────────────────
print("\n=== Image Utils ===")
from src.image_utils import preprocess_query_frame, to_grayscale

fake_img = np.random.randint(0, 255, (1079, 1920, 3), dtype=np.uint8)
processed = preprocess_query_frame(fake_img)
assert processed.shape == (512, 512, 3)
print(f"  PASS: preprocess_query_frame -> {processed.shape}")

gray = to_grayscale(fake_img)
assert gray.ndim == 2
print(f"  PASS: to_grayscale -> {gray.shape}")

# ── 5. Config ─────────────────────────────────────────────────
print("\n=== Config ===")
from config import config
config.ensure_output_dirs()
assert config.OUTPUT_DIR.exists()
print(f"  PASS: output dirs created at {config.OUTPUT_DIR}")
print(f"  TILE_SIZE_METERS = {config.TILE_SIZE_METERS:.1f}")
print(f"  NUM_PARTICLES = {config.NUM_PARTICLES}")
print(f"  METATILE_MATCH_THRESHOLD = {config.METATILE_MATCH_THRESHOLD}")

print("\n" + "="*50)
print("ALL VALIDATION TESTS PASSED")
print("="*50)
