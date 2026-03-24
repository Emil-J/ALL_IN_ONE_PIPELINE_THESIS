"""Unit tests for Module 8 — Particle Filter."""

import pytest
import numpy as np
from src.particle_filter import ParticleFilter, Particle


@pytest.fixture
def pf():
    return ParticleFilter(
        num_particles=100,
        initial_position=(55.7, 9.5),
        initial_heading=90,
        initial_spread={"position_meters": 50, "heading_degrees": 10},
        zoom=16,
    )


def test_particle_initialization(pf):
    assert len(pf.particles) == 100
    assert all(0 <= p.heading < 360 for p in pf.particles)
    # Weights should sum to ~1
    total_w = sum(p.weight for p in pf.particles)
    assert abs(total_w - 1.0) < 1e-6


def test_particle_to_latlon():
    p = Particle.from_latlon(55.7, 9.5, heading=90.0, weight=1.0, zoom=16)
    lat, lon = p.to_latlon(16)
    assert abs(lat - 55.7) < 0.01
    assert abs(lon - 9.5) < 0.01


def test_predict_changes_state(pf):
    before_x = [p.x for p in pf.particles]
    pf.predict(dt=1.0, velocity_mps=10.0, gyro_z_dps=5.0)
    after_x = [p.x for p in pf.particles]
    # At least some particles should have moved
    assert any(b != a for b, a in zip(before_x, after_x))


def test_update_normalises_weights(pf):
    estimate = pf.get_estimate()
    measurements = [{
        "position": estimate[:2],
        "heading": estimate[2],
        "score": 150.0,
    }]
    pf.update(measurements)
    total = sum(p.weight for p in pf.particles)
    assert abs(total - 1.0) < 0.01


def test_resample_resets_weights(pf):
    # Force degenerate weights
    pf.particles[0].weight = 0.99
    for i in range(1, 100):
        pf.particles[i].weight = 0.01 / 99
    pf.resample()
    weights = [p.weight for p in pf.particles]
    assert all(abs(w - 0.01) < 0.005 for w in weights)


def test_get_estimate_returns_three_floats(pf):
    est = pf.get_estimate()
    assert len(est) == 3
    assert all(isinstance(v, float) for v in est)


def test_get_uncertainty(pf):
    unc = pf.get_uncertainty()
    assert "position_std_m" in unc
    assert "heading_std_deg" in unc
    assert "n_eff" in unc
    assert unc["position_std_m"] >= 0
    assert unc["n_eff"] > 0


def test_search_region(pf):
    region = pf.get_search_region()
    assert "center" in region
    assert "radius_tiles" in region
    assert "heading_mean" in region
    assert "heading_range" in region
    assert region["radius_tiles"] > 0


def test_check_divergence_false_initially(pf):
    assert pf.check_divergence() is False


def test_check_divergence_after_spreading(pf):
    # Scatter particles widely → trigger divergence
    for i, p in enumerate(pf.particles):
        p.x += np.random.uniform(-5, 5)
        p.y += np.random.uniform(-5, 5)
    assert pf.check_divergence() is True
