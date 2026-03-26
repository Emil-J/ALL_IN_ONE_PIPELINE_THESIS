"""
Module 9 — Temporal Searcher.

Orchestrates the full Pipeline 3 per-frame loop:
  Frame 0: Pipeline 1 cold-start (BestFirstSearcher)
  Frame 1+: IMU predict → particle-guided two-pass search (MetaTileBuilder)
            → particle update → semantic double-confirmation → pose estimate.
"""

import csv
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.tile_utils import TileLoader, tile_to_latlon, tile_size_meters
from src.image_utils import preprocess_query_frame
from src.best_first_search import BestFirstSearcher
from src.particle_filter import ParticleFilter
from src.meta_tile_builder import MetaTileBuilder
from src.semantic_confirmer import SemanticConfirmer
from src.position_estimator import estimate_position

logger = logging.getLogger(__name__)


class TemporalSearcher:
    """
    Top-level frame processor for Pipeline 3.

    Usage:
        searcher = TemporalSearcher(semantic_model, feature_matcher,
                                     tile_loader, config)
        for frame, imu, ts in stream:
            result = searcher.process_frame(frame, imu, ts)
    """

    def __init__(self, semantic_model, feature_matcher,
                 tile_loader: TileLoader, config):
        self.semantic_model = semantic_model
        self.matcher = feature_matcher
        self.tiles = tile_loader
        self.cfg = config

        self.particle_filter: Optional[ParticleFilter] = None
        self.frame_count = 0
        self.last_timestamp: Optional[float] = None
        self.history: List[Dict] = []

        # Sub-modules
        self.meta_tile_builder = MetaTileBuilder(
            feature_matcher, tile_loader, config)
        self.semantic_confirmer = SemanticConfirmer(
            semantic_model, config)

        # JSONL log file handle (opened lazily)
        self._log_path: Optional[Path] = None
        self._log_fh = None

    # ════════════════════════════════════════════════════════════
    # 9.2  Main interface
    # ════════════════════════════════════════════════════════════

    def process_frame(self, query_frame: np.ndarray,
                      imu_data: Dict, timestamp: float) -> Dict:
        """
        Process one frame.

        Args:
            query_frame: Raw frame (1920×1079 or similar).
            imu_data: dict with lat, lon, heading, pos_sigma,
                      heading_sigma, velocity_mps, gyro_z_dps.
            timestamp: Capture time in seconds.

        Returns:
            result dict (fields depend on cold_start vs temporal_tracking).
        """
        if self.frame_count == 0:
            result = self._process_frame_0(query_frame, imu_data, timestamp)
        else:
            result = self._process_frame_N(query_frame, imu_data, timestamp)

        self.frame_count += 1
        self.last_timestamp = timestamp
        result["_timestamp"] = timestamp
        self.history.append(result)
        self._write_log(result, imu_data, timestamp)
        return result

    # ════════════════════════════════════════════════════════════
    # 9.3  Frame 0 — Cold Start (Pipeline 1)
    # ════════════════════════════════════════════════════════════

    def _process_frame_0(self, query_frame: np.ndarray,
                         imu_data: Dict, timestamp: float) -> Dict:
        t0 = time.perf_counter()

        searcher = BestFirstSearcher(self.matcher, self.tiles, self.cfg)
        search_result = searcher.search(
            query_frame, imu_data["lat"], imu_data["lon"])

        score = search_result["score"]
        position = search_result["position"]

        # Determine initial spread based on match quality
        if score >= 150:
            spread = self.cfg.PARTICLE_INIT_SPREAD_HIGH_CONF
        elif score >= 100:
            spread = self.cfg.PARTICLE_INIT_SPREAD_MED_CONF
        else:
            spread = self.cfg.PARTICLE_INIT_SPREAD_LOW_CONF

        # Initialize particle filter
        init_lat = position[0] if position else imu_data["lat"]
        init_lon = position[1] if position else imu_data["lon"]
        init_heading = imu_data["heading"]

        self.particle_filter = ParticleFilter(
            num_particles=self.cfg.NUM_PARTICLES,
            initial_position=(init_lat, init_lon),
            initial_heading=init_heading,
            initial_spread=spread,
            zoom=self.cfg.TMS_ZOOM_LEVEL,
            process_noise_pos_m=self.cfg.PROCESS_NOISE_POSITION_M,
            process_noise_hdg_deg=self.cfg.PROCESS_NOISE_HEADING_DEG,
            measurement_noise_pos_m=self.cfg.MEASUREMENT_NOISE_POSITION_M,
            measurement_noise_hdg_deg=self.cfg.MEASUREMENT_NOISE_HEADING_DEG,
            resample_threshold=self.cfg.RESAMPLE_THRESHOLD,
            divergence_pos_thresh_m=self.cfg.DIVERGENCE_POSITION_THRESHOLD_M,
            divergence_weight_thresh=self.cfg.DIVERGENCE_WEIGHT_THRESHOLD,
        )

        # Branch A — semantic segmentation of query frame
        query_processed = preprocess_query_frame(
            query_frame,
            resize_w=self.cfg.QUERY_RESIZE_WIDTH,
            resize_h=self.cfg.QUERY_RESIZE_HEIGHT,
            target_size=self.cfg.SEMANTIC_INPUT_SIZE,
        )
        query_semantic_map = self.semantic_model.predict(query_processed)

        elapsed = time.perf_counter() - t0

        return {
            "position": position,
            "heading": init_heading,
            "score": score,
            "tiles_tested": search_result["tiles_tested"],
            "search_time": elapsed,
            "method": "cold_start",
            "best_tile": search_result.get("best_tile"),
            "ranked_tiles": search_result.get("ranked_tiles", []),
            "meta_tile_path": None,
            "meta_tile_verified": None,
            "verification_matches": None,
            "semantic_confidence": None,
            "particle_spread": None,
            "n_eff": None,
            "query_semantic_map": query_semantic_map,
        }

    # ════════════════════════════════════════════════════════════
    # 9.4  Frame 1+ — Temporal Tracking
    # ════════════════════════════════════════════════════════════

    def _process_frame_N(self, query_frame: np.ndarray,
                         imu_data: Dict, timestamp: float) -> Dict:
        t0 = time.perf_counter()
        dt = timestamp - self.last_timestamp if self.last_timestamp else 0.5

        # Step 1 — Predict particles with IMU
        self.particle_filter.predict(
            dt, imu_data["velocity_mps"], imu_data["gyro_z_dps"])

        # Step 2 — Get focused search region from particles
        region = self.particle_filter.get_search_region()
        center_lat, center_lon = tile_to_latlon(
            region["center"][0], region["center"][1], self.cfg.TMS_ZOOM_LEVEL)
        search_radius_m = max(
            region["radius_tiles"] * self.cfg.TILE_SIZE_METERS,
            self.cfg.FIRST_PASS_SEARCH_RADIUS_M,
        )

        # Step 3 — Two-pass search via MetaTileBuilder (raw frame for feature matching)
        meta_result = self.meta_tile_builder.run(
            query_frame=query_frame,
            imu_lat=center_lat,
            imu_lon=center_lon,
            query_timestamp=timestamp,
            search_radius_m=search_radius_m,
        )

        # Handle failure: no tiles found
        if meta_result is None:
            unc = self.particle_filter.get_uncertainty()
            elapsed = time.perf_counter() - t0
            return self._imu_fallback_result(
                imu_data, timestamp, elapsed, unc,
                reason="no_first_pass_tiles")

        # Step 4 — Extract measurements for particle update
        if meta_result["verified"]:
            measurements = [
                {"position": (tx, ty), "heading": imu_data["heading"],
                 "score": float(score)}
                for tx, ty, score in meta_result["top3_tiles"]
            ]
        else:
            # Only top-1 as low-confidence measurement
            if meta_result["top3_tiles"]:
                tx, ty, score = meta_result["top3_tiles"][0]
                measurements = [
                    {"position": (tx, ty), "heading": imu_data["heading"],
                     "score": float(score) * 0.3}
                ]
            else:
                measurements = []

        # Step 5 — Update particle filter
        self.particle_filter.update(measurements)
        self.particle_filter.resample()

        # Step 6 — Semantic double-confirmation (preprocess for semantic model only)
        query_processed = preprocess_query_frame(
            query_frame,
            resize_w=self.cfg.QUERY_RESIZE_WIDTH,
            resize_h=self.cfg.QUERY_RESIZE_HEIGHT,
            target_size=self.cfg.SEMANTIC_INPUT_SIZE,
        )
        query_semantic_map = self.semantic_model.predict(query_processed)
        confirm_result = self.semantic_confirmer.confirm(
            query_semantic_map, meta_result["meta_tile"])

        # Step 7 — Get final estimate
        est_x, est_y, est_hdg = self.particle_filter.get_estimate()
        est_lat, est_lon = tile_to_latlon(
            est_x, est_y, self.cfg.TMS_ZOOM_LEVEL)

        # Position estimation via homography (if verified)
        homo_position = None
        if meta_result["verified"] and meta_result.get("match_result"):
            qh, qw = query_frame.shape[:2]
            pos_est = estimate_position(
                meta_result["match_result"],
                meta_result["top3_tiles"],
                query_w=qw,
                query_h=qh,
                tile_px=self.cfg.TMS_TILE_SIZE_PX,
                zoom=self.cfg.TMS_ZOOM_LEVEL,
                ransac_thresh=self.cfg.RANSAC_REPROJ_THRESH,
                min_matches=self.cfg.MIN_MATCHES_FOR_HOMOGRAPHY,
            )
            if pos_est:
                homo_position = (pos_est["lat"], pos_est["lon"])

        # Use homography position if available, else particle estimate
        final_position = homo_position or (est_lat, est_lon)

        unc = self.particle_filter.get_uncertainty()
        elapsed = time.perf_counter() - t0

        # Step 8 — Check divergence
        if self.particle_filter.check_divergence():
            logger.warning("Particle divergence detected at t=%.3f — "
                           "will reinitialise on next frame", timestamp)
            self.frame_count = 0  # triggers cold start on next call

        return {
            "position": final_position,
            "heading": est_hdg,
            "score": meta_result["top3_tiles"][0][2] if meta_result["top3_tiles"] else 0,
            "tiles_tested": meta_result["first_pass_candidates"],
            "search_time": elapsed,
            "method": "temporal_tracking",
            "best_tile": (meta_result["top3_tiles"][0][0],
                          meta_result["top3_tiles"][0][1]) if meta_result["top3_tiles"] else None,
            "ranked_tiles": meta_result["top3_tiles"],
            "meta_tile_path": str(meta_result["meta_tile_path"]),
            "meta_tile_verified": meta_result["verified"],
            "verification_matches": meta_result["verification_matches"],
            "semantic_confidence": confirm_result["confidence"],
            "particle_spread": unc["position_std_m"],
            "n_eff": unc["n_eff"],
            "query_semantic_map": query_semantic_map,
        }

    # ════════════════════════════════════════════════════════════
    # IMU fallback
    # ════════════════════════════════════════════════════════════

    def _imu_fallback_result(self, imu_data: Dict, timestamp: float,
                             elapsed: float, unc: Dict,
                             reason: str) -> Dict:
        return {
            "position": (imu_data["lat"], imu_data["lon"]),
            "heading": imu_data["heading"],
            "score": 0,
            "tiles_tested": 0,
            "search_time": elapsed,
            "method": "imu_fallback",
            "best_tile": None,
            "ranked_tiles": [],
            "meta_tile_path": None,
            "meta_tile_verified": False,
            "verification_matches": 0,
            "semantic_confidence": None,
            "particle_spread": unc.get("position_std_m"),
            "n_eff": unc.get("n_eff"),
            "fallback_reason": reason,
            "query_semantic_map": None,
        }

    # ════════════════════════════════════════════════════════════
    # 9.5  Trajectory access
    # ════════════════════════════════════════════════════════════

    def get_trajectory(self) -> List[Tuple[float, float, float, float]]:
        """Return [(lat, lon, heading, timestamp), ...]"""
        traj = []
        for h in self.history:
            pos = h.get("position")
            if pos is None:
                continue
            hdg = h.get("heading", 0)
            # Recover timestamp from history index + first known timestamp
            traj.append((*pos, hdg, 0.0))
        return traj

    # ════════════════════════════════════════════════════════════
    # 9.6  Save trajectory
    # ════════════════════════════════════════════════════════════

    def save_trajectory(self, filepath: Path):
        """Export trajectory as CSV."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "lat", "lon", "heading", "score", "method",
                "tiles_tested", "meta_tile_verified", "semantic_confidence",
            ])
            for i, h in enumerate(self.history):
                pos = h.get("position") or (None, None)
                writer.writerow([
                    h.get("_timestamp", i),
                    pos[0], pos[1],
                    h.get("heading"),
                    h.get("score"),
                    h.get("method"),
                    h.get("tiles_tested"),
                    h.get("meta_tile_verified"),
                    h.get("semantic_confidence"),
                ])

    # ════════════════════════════════════════════════════════════
    # JSONL logging
    # ════════════════════════════════════════════════════════════

    def _open_log(self, timestamp: float):
        if self._log_fh is not None:
            return
        log_dir = Path(self.cfg.LOG_OUTPUT_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / f"pipeline3_run_{timestamp:.3f}.jsonl"
        self._log_fh = open(self._log_path, "a")

    def _write_log(self, result: Dict, imu_data: Dict, timestamp: float):
        self._open_log(timestamp)

        pos = result.get("position") or (None, None)
        top3 = result.get("ranked_tiles", [])
        best = result.get("best_tile")

        record = {
            "timestamp": timestamp,
            "frame_name": f"frame_{timestamp:.3f}.jpg",
            "mode": result.get("method"),
            "imu_lat": imu_data.get("lat"),
            "imu_lon": imu_data.get("lon"),
            "imu_heading": imu_data.get("heading"),
            "dt": timestamp - self.last_timestamp if self.last_timestamp else None,
            "first_pass_candidates": result.get("tiles_tested"),
            "top1_tile": list(best) if best else None,
            "top3_tiles": [list(t) for t in top3] if top3 else None,
            "verification_matches": result.get("verification_matches"),
            "meta_tile_verified": result.get("meta_tile_verified"),
            "semantic_confidence": result.get("semantic_confidence"),
            "particle_position_std_m": result.get("particle_spread"),
            "particle_heading_std_deg": None,
            "n_eff": result.get("n_eff"),
            "estimated_lat": pos[0],
            "estimated_lon": pos[1],
            "estimated_heading": result.get("heading"),
            "used_gps_feedback": False,
        }
        self._log_fh.write(json.dumps(record) + "\n")
        self._log_fh.flush()

    def close(self):
        """Flush and close JSONL log."""
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
