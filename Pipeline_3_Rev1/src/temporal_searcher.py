"""
Module 9 — Temporal Searcher.

Orchestrates the full Pipeline 3 per-frame loop:
  Frame 0: Pipeline 1 cold-start (BestFirstSearcher)
  Frame 1+: IMU predict → particle-guided two-pass search (MetaTileBuilder)
            → particle update → semantic double-confirmation → pose estimate.
"""

import csv
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from src.tile_utils import (
    TileLoader, tile_to_latlon, tile_size_meters,
    latlon_to_tile_float, haversine_distance,
)
from src.image_utils import preprocess_query_frame
from src.best_first_search import BestFirstSearcher
from src.particle_filter import ParticleFilter
from src.meta_tile_builder import MetaTileBuilder
from src.semantic_confirmer import SemanticConfirmer
from src.visual_measurement import (
    rotate_image, compute_dual_homography, extract_visual_measurements,
)

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
                 tile_loader: TileLoader, config, feature_store=None):
        self.semantic_model = semantic_model
        self.matcher = feature_matcher
        self.tiles = tile_loader
        self.cfg = config
        self.feature_store = feature_store

        self.particle_filter: Optional[ParticleFilter] = None
        self.frame_count = 0
        self.last_timestamp: Optional[float] = None
        self.history: List[Dict] = []

        # Sub-modules
        self.meta_tile_builder = MetaTileBuilder(
            feature_matcher, tile_loader, config,
            feature_store=feature_store)
        self.semantic_confirmer = SemanticConfirmer(
            semantic_model, config)

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
        result["_timestamp"] = timestamp
        if getattr(self.cfg, 'ACCUMULATE_HISTORY', True):
            self.history.append(result)
        self.last_timestamp = timestamp
        return result

    # ════════════════════════════════════════════════════════════
    # 9.3  Frame 0 — Cold Start (Pipeline 1)
    # ════════════════════════════════════════════════════════════

    def _process_frame_0(self, query_frame: np.ndarray,
                         imu_data: Dict, timestamp: float) -> Dict:
        t0 = time.perf_counter()
        _save_timing = getattr(self.cfg, 'SAVE_TIMING_DATA', False)
        _timing: Dict = {}
        _save_trace = getattr(self.cfg, 'SAVE_PIPELINE_TRACE', False)
        _trace_data: Dict = {}

        # ── Phase B1: rotate query by heading for better matching ──
        heading_deg = imu_data.get("heading", 0)
        rotation_angle = -heading_deg
        query_rotated, rot_M_fwd = rotate_image(query_frame, rotation_angle)

        # Resize rotated image to cap performance cost
        query_for_match = self._resize_rotated(query_rotated)

        # BestFirstSearcher exhaustive search on rotated query
        searcher = BestFirstSearcher(self.matcher, self.tiles, self.cfg,
                                     feature_store=self.feature_store)
        _t = time.perf_counter()
        search_result = searcher.search(
            query_for_match, imu_data["lat"], imu_data["lon"])
        if _save_timing:
            _timing['cold_search_ms'] = (time.perf_counter() - _t) * 1000

        score = search_result["score"]
        position = search_result["position"]
        ranked_tiles = search_result.get("ranked_tiles", [])

        if _save_trace:
            _trace_data['rotation_deg'] = rotation_angle
            _trace_data['ranked_tiles'] = ranked_tiles
            _trace_data['query_rotated'] = query_for_match
            _trace_data['match_result'] = search_result.get("match_result")
            best_t = search_result.get("best_tile")
            _trace_data['ref_img'] = (
                self.tiles.load_aerial(best_t[0], best_t[1]) if best_t else None)

        # ── Phase B1: dual homography + visual measurements on best match ──
        homo_position = None
        visual_quality = {"CShape": 0, "inliers": 0, "convex": False}

        _t_homo = time.perf_counter()
        if search_result.get("match_result") and score >= 4:
            mr = search_result["match_result"]
            matches = mr["matches"]
            if len(matches) >= 4:
                src_pts = mr["keypoints1"][matches[:, 0]]
                dst_pts = mr["keypoints2"][matches[:, 1]]
                qh_m, qw_m = query_for_match.shape[:2]

                dual = compute_dual_homography(
                    src_pts, dst_pts, qw_m, qh_m,
                    self.cfg.RANSAC_REPROJ_THRESH)

                if dual["winner"] is not None:
                    winner_branch = dual[dual["winner"]]
                    visual_quality = {
                        "CShape": winner_branch["CShape"],
                        "inliers": winner_branch["inliers"],
                        "convex": winner_branch["convex"],
                    }

                    # Use BFS best tile as single-tile reference for measurements
                    best_tile = search_result.get("best_tile")
                    if best_tile is not None:
                        tiles_for_meas = [(best_tile[0], best_tile[1], score)]
                        pitch_rad = imu_data.get("pitch", 0.0)
                        roll_rad = imu_data.get("roll", 0.0)
                        measurements_dict = extract_visual_measurements(
                            dual["winner_H"], dual["winner_mask"],
                            src_pts, dst_pts, qw_m, qh_m,
                            tiles_for_meas,
                            tile_px=self.cfg.TMS_TILE_SIZE_PX,
                            zoom=self.cfg.TMS_ZOOM_LEVEL,
                            pitch_rad=pitch_rad, roll_rad=roll_rad,
                        )

                        # Cascade: nadir (if near-nadir) > trimmed > inlier > weighted > projected
                        cascade = self._build_cascade(pitch_rad, roll_rad)
                        for mname in cascade:
                            mdata = measurements_dict.get(mname, {})
                            if mdata.get("valid") and mdata["latlon"][0] is not None:
                                homo_position = mdata["latlon"]
                                break

        if _save_timing:
            _timing['homography_ms'] = (time.perf_counter() - _t_homo) * 1000

        # ── Decide PF initialization position using quality gate ──
        cshape = visual_quality["CShape"]
        n_inliers = visual_quality["inliers"]
        gate_pass = (cshape > self.cfg.QUALITY_GATE_CSHAPE
                     and n_inliers > self.cfg.QUALITY_GATE_INLIERS
                     and homo_position is not None)

        if gate_pass:
            init_lat, init_lon = homo_position
            position = homo_position
            spread = self.cfg.PARTICLE_INIT_SPREAD_HIGH_CONF
            logger.info("Cold-start quality gate PASSED (CShape=%.3f, inliers=%d) "
                        "— using visual position", cshape, n_inliers)
        elif score >= 100 and position:
            init_lat, init_lon = position
            spread = self.cfg.PARTICLE_INIT_SPREAD_MED_CONF
        else:
            init_lat, init_lon = imu_data["lat"], imu_data["lon"]
            spread = self.cfg.PARTICLE_INIT_SPREAD_LOW_CONF
            position = (init_lat, init_lon)
            if score > 0:
                logger.info("Cold-start score %d, CShape %.3f — using EKF position",
                            score, cshape)

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
        _t = time.perf_counter()
        query_processed = preprocess_query_frame(
            query_frame,
            resize_w=self.cfg.QUERY_RESIZE_WIDTH,
            resize_h=self.cfg.QUERY_RESIZE_HEIGHT,
            target_size=self.cfg.SEMANTIC_INPUT_SIZE,
        )
        query_semantic_map = self.semantic_model.predict(query_processed)
        if _save_timing:
            _timing['semantic_ms'] = (time.perf_counter() - _t) * 1000

        if _save_trace:
            _trace_data['query_processed'] = query_processed
            _trace_data['query_semantic_map'] = query_semantic_map

        elapsed = time.perf_counter() - t0
        if _save_timing:
            _timing['total_ms'] = elapsed * 1000

        return {
            "position": position,
            "heading": init_heading,
            "score": score,
            "tiles_tested": search_result["tiles_tested"],
            "search_time": elapsed,
            "method": "cold_start",
            "best_tile": search_result.get("best_tile"),
            "ranked_tiles": ranked_tiles,
            "meta_tile_path": None,
            "meta_tile_verified": None,
            "verification_matches": None,
            "semantic_confidence": None,
            "particle_spread": None,
            "n_eff": None,
            "query_semantic_map": query_semantic_map,
            "visual_quality": visual_quality,
            "gate_pass": gate_pass,
            "homo_position": homo_position,
            "pf_position": (init_lat, init_lon),
            "timing": _timing if _save_timing else None,
            "trace_data": _trace_data if _save_trace else None,
        }

    # ════════════════════════════════════════════════════════════
    # 9.4  Frame 1+ — Temporal Tracking
    # ════════════════════════════════════════════════════════════

    def _process_frame_N(self, query_frame: np.ndarray,
                         imu_data: Dict, timestamp: float) -> Dict:
        t0 = time.perf_counter()
        dt = timestamp - self.last_timestamp if self.last_timestamp else 0.5
        _save_timing = getattr(self.cfg, 'SAVE_TIMING_DATA', False)
        _timing: Dict = {}
        _save_trace = getattr(self.cfg, 'SAVE_PIPELINE_TRACE', False)
        _trace_data: Dict = {}
        pf_update_source = "none"
        search_radius_capped = False
        search_radius_m = self.cfg.FIRST_PASS_SEARCH_RADIUS_M

        # Step 1 — Predict particles with IMU
        _t = time.perf_counter()
        self.particle_filter.predict(
            dt, imu_data["velocity_mps"], imu_data["gyro_z_dps"])
        if _save_timing:
            _timing['pf_predict_ms'] = (time.perf_counter() - _t) * 1000

        # Step 2 — Get search region.
        region = self.particle_filter.get_search_region()
        center_lat, center_lon = imu_data["lat"], imu_data["lon"]
        search_radius_m = max(
            region["radius_tiles"] * self.cfg.TILE_SIZE_METERS,
            self.cfg.FIRST_PASS_SEARCH_RADIUS_M,
        )
        _uncapped_radius = search_radius_m
        _max_radius = getattr(self.cfg, 'MAX_TEMPORAL_SEARCH_RADIUS_M', 1500.0)
        if search_radius_m > _max_radius:
            search_radius_m = _max_radius
            search_radius_capped = True
            logger.warning("PF search radius capped at %.0f m (was %.0f m)",
                           search_radius_m, _uncapped_radius)

        # Step 3 — Rotate query by heading for better matching, then two-pass search
        heading_deg = imu_data.get("heading", 0)
        rotation_angle = -heading_deg
        query_rotated, rot_M_fwd = rotate_image(query_frame, rotation_angle)
        query_for_match = self._resize_rotated(query_rotated)

        # Step 3b — Semantic segmentation (before tile search so pre-filter can use it)
        _t = time.perf_counter()
        query_processed = preprocess_query_frame(
            query_frame,
            resize_w=self.cfg.QUERY_RESIZE_WIDTH,
            resize_h=self.cfg.QUERY_RESIZE_HEIGHT,
            target_size=self.cfg.SEMANTIC_INPUT_SIZE,
        )
        query_semantic_map = self.semantic_model.predict(query_processed)
        if _save_timing:
            _timing['semantic_ms'] = (time.perf_counter() - _t) * 1000

        _t = time.perf_counter()
        meta_result = self.meta_tile_builder.run(
            query_frame=query_for_match,
            imu_lat=center_lat,
            imu_lon=center_lon,
            query_timestamp=timestamp,
            search_radius_m=search_radius_m,
            query_semantic_map=query_semantic_map,
        )

        if _save_timing:
            _timing['meta_tile_ms'] = (time.perf_counter() - _t) * 1000

        # Populate trace data — common fields regardless of meta_result outcome
        if _save_trace:
            _trace_data['rotation_deg'] = rotation_angle
            _trace_data['search_radius_m'] = search_radius_m
            pf_lat, pf_lon = tile_to_latlon(
                region["center"][0], region["center"][1], self.cfg.TMS_ZOOM_LEVEL)
            _trace_data['pf_center'] = (pf_lat, pf_lon)
            _trace_data['ekf_center'] = (center_lat, center_lon)
            _trace_data['query_rotated'] = query_for_match
            _trace_data['query_processed'] = query_processed
            _trace_data['query_semantic_map'] = query_semantic_map
            if meta_result is not None:
                _trace_data['first_pass_tiles'] = meta_result.get("first_pass_tiles", [])
                _trace_data['second_pass_tiles'] = meta_result.get("second_pass_tiles", [])
                _trace_data['meta_tile'] = meta_result["meta_tile"]
                _trace_data['match_result'] = meta_result.get("match_result")

        # Handle failure: no tiles found
        if meta_result is None:
            unc = self.particle_filter.get_uncertainty()
            elapsed = time.perf_counter() - t0
            if _save_timing:
                _timing.update({'homography_ms': 0.0, 'pf_update_ms': 0.0,
                                 'total_ms': elapsed * 1000})
            res = self._imu_fallback_result(
                imu_data, timestamp, elapsed, unc,
                reason="no_first_pass_tiles")
            res["pf_update_source"] = pf_update_source
            res["search_radius_m"] = search_radius_m
            res["search_radius_capped"] = search_radius_capped
            if _save_timing:
                res['timing'] = _timing
            if _save_trace:
                res['trace_data'] = _trace_data
            return res

        # Step 4 — Dual homography + visual measurement extraction
        homo_position = None
        homo_tile_pos = None
        visual_quality = {"CShape": 0, "inliers": 0, "convex": False}
        qh_rot, qw_rot = query_for_match.shape[:2]

        _t_homo = time.perf_counter()
        if meta_result.get("match_result"):
            mr = meta_result["match_result"]
            matches = mr["matches"]
            if len(matches) >= 4:
                src_pts = mr["keypoints1"][matches[:, 0]]
                dst_pts = mr["keypoints2"][matches[:, 1]]

                dual = compute_dual_homography(
                    src_pts, dst_pts, qw_rot, qh_rot,
                    self.cfg.RANSAC_REPROJ_THRESH)

                if dual["winner"] is not None:
                    winner_branch = dual[dual["winner"]]
                    visual_quality = {
                        "CShape": winner_branch["CShape"],
                        "inliers": winner_branch["inliers"],
                        "convex": winner_branch["convex"],
                    }

                    # Extract visual measurements
                    pitch_rad = imu_data.get("pitch", 0.0)
                    roll_rad = imu_data.get("roll", 0.0)
                    measurements_dict = extract_visual_measurements(
                        dual["winner_H"], dual["winner_mask"],
                        src_pts, dst_pts, qw_rot, qh_rot,
                        meta_result["top3_tiles"],
                        tile_px=self.cfg.TMS_TILE_SIZE_PX,
                        zoom=self.cfg.TMS_ZOOM_LEVEL,
                        pitch_rad=pitch_rad, roll_rad=roll_rad,
                    )

                    # Select best measurement using attitude-aware cascade
                    cascade = self._build_cascade(pitch_rad, roll_rad)
                    for mname in cascade:
                        mdata = measurements_dict.get(mname, {})
                        if mdata.get("valid") and mdata["latlon"][0] is not None:
                            homo_position = mdata["latlon"]
                            homo_tile_pos = latlon_to_tile_float(
                                homo_position[0], homo_position[1],
                                self.cfg.TMS_ZOOM_LEVEL)
                            break

        if _save_timing:
            _timing['homography_ms'] = (time.perf_counter() - _t_homo) * 1000

        # Step 5 — Extract measurements for particle update
        # Only use homography when it passes the same visual quality gate as the EKF
        # update — prevents bad homography (few inliers, low CShape) from poisoning PF.
        MAX_SCORE = 50.0  # cap for normalization
        homo_innovation_m = None
        max_pf_innovation_m = max(
            150.0,
            3.0 * imu_data.get("pos_sigma", 0.0)
            + imu_data.get("velocity_mps", 0.0) * dt
            + 50.0
        )
        if homo_position is not None:
            homo_innovation_m = haversine_distance(
                homo_position[0], homo_position[1],
                imu_data["lat"], imu_data["lon"]
            )
        visual_rejected_reason = ""
        if (homo_position is not None
                and homo_innovation_m is not None
                and homo_innovation_m > max_pf_innovation_m):
            visual_rejected_reason = "pf_innovation_too_large"
        pf_innovation_ok = (
            homo_innovation_m is not None
            and homo_innovation_m <= max_pf_innovation_m
        )
        visual_gate_ok = (
            visual_quality.get("CShape", 0) > self.cfg.QUALITY_GATE_CSHAPE
            and visual_quality.get("inliers", 0) > self.cfg.QUALITY_GATE_INLIERS
            and homo_position is not None
            and homo_tile_pos is not None
            and pf_innovation_ok
        )
        if visual_gate_ok:
            pf_update_source = "homography"
            inlier_score = min(visual_quality["inliers"], MAX_SCORE) / MAX_SCORE
            measurements = [
                {"position": homo_tile_pos,
                 "heading": imu_data["heading"],
                 "score": inlier_score}
            ]
        elif meta_result["verified"]:
            plausible = []
            for tx, ty, score in meta_result["top3_tiles"]:
                tc_lat, tc_lon = tile_to_latlon(tx + 0.5, ty + 0.5, self.cfg.TMS_ZOOM_LEVEL)
                d = haversine_distance(tc_lat, tc_lon, imu_data["lat"], imu_data["lon"])
                if d <= max_pf_innovation_m:
                    plausible.append((tx, ty, score))
            if plausible:
                pf_update_source = "tile_center"
                measurements = [
                    {"position": (tx + 0.5, ty + 0.5),
                     "heading": imu_data["heading"],
                     "score": min(float(score), MAX_SCORE) / MAX_SCORE * 0.3}
                    for tx, ty, score in plausible
                ]
            else:
                pf_update_source = "none"
                measurements = []
        else:
            if meta_result["top3_tiles"]:
                tx, ty, score = meta_result["top3_tiles"][0]
                tc_lat, tc_lon = tile_to_latlon(tx + 0.5, ty + 0.5, self.cfg.TMS_ZOOM_LEVEL)
                d = haversine_distance(tc_lat, tc_lon, imu_data["lat"], imu_data["lon"])
                if d <= max_pf_innovation_m:
                    pf_update_source = "tile_center"
                    measurements = [
                        {"position": (tx + 0.5, ty + 0.5),
                         "heading": imu_data["heading"],
                         "score": min(float(score), MAX_SCORE) / MAX_SCORE * 0.3}
                    ]
                else:
                    pf_update_source = "none"
                    measurements = []
            else:
                pf_update_source = "none"
                measurements = []

        # Step 6 — Update particle filter
        _t = time.perf_counter()
        self.particle_filter.update(measurements)
        self.particle_filter.resample()

        # Step 7 — Semantic double-confirmation
        # (query_semantic_map already computed in Step 3b above)
        # Use prediction meta-tile when available (faster, no model inference)
        confirm_result = self.semantic_confirmer.confirm(
            query_semantic_map, meta_result["meta_tile"],
            prediction_meta_tile=meta_result.get("prediction_meta_tile"))
        if _save_timing:
            _timing['pf_update_ms'] = (time.perf_counter() - _t) * 1000

        # Step 8 — Get final estimate
        est_x, est_y, est_hdg = self.particle_filter.get_estimate()
        est_lat, est_lon = tile_to_latlon(
            est_x, est_y, self.cfg.TMS_ZOOM_LEVEL)

        # Use homography position if available, else particle estimate
        visual_position = homo_position or (est_lat, est_lon)

        # Step 9 — Quality-gated blending (Phase B1).
        ekf_pos = (imu_data["lat"], imu_data["lon"])
        pf_pos = (est_lat, est_lon)

        cshape = visual_quality["CShape"]
        n_inliers = visual_quality["inliers"]

        if (cshape > self.cfg.QUALITY_GATE_CSHAPE
                and n_inliers > self.cfg.QUALITY_GATE_INLIERS
                and homo_position is not None):
            # High-quality visual: use it directly
            final_position = homo_position
        else:
            # Low-quality or no visual: use PF estimate (preserves visual drift correction)
            final_position = pf_pos

        unc = self.particle_filter.get_uncertainty()
        elapsed = time.perf_counter() - t0
        if _save_timing:
            _timing['total_ms'] = elapsed * 1000

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
            "meta_tile_path": meta_result["meta_tile_path"],
            "meta_tile_verified": meta_result["verified"],
            "verification_matches": meta_result["verification_matches"],
            "semantic_confidence": confirm_result["confidence"],
            "particle_spread": unc["position_std_m"],
            "n_eff": unc["n_eff"],
            "query_semantic_map": query_semantic_map,
            "visual_quality": visual_quality,
            "gate_pass": (cshape > self.cfg.QUALITY_GATE_CSHAPE
                          and n_inliers > self.cfg.QUALITY_GATE_INLIERS
                          and homo_position is not None),
            "homo_position": homo_position,
            "pf_position": pf_pos,
            "pf_update_source": pf_update_source,
            "search_radius_m": search_radius_m,
            "search_radius_capped": search_radius_capped,
            "homo_innovation_m": homo_innovation_m,
            "max_pf_innovation_m": max_pf_innovation_m,
            "visual_rejected_reason": visual_rejected_reason,
            "timing": _timing if _save_timing else None,
            "trace_data": _trace_data if _save_trace else None,
        }

    # ════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════

    def _resize_rotated(self, img: np.ndarray) -> np.ndarray:
        """Resize an image so its longest edge <= MAX_ROTATED_DIMENSION."""
        max_dim = getattr(self.cfg, "MAX_ROTATED_DIMENSION", 1920)
        h, w = img.shape[:2]
        if max(h, w) <= max_dim:
            return img
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _build_cascade(pitch_rad: float, roll_rad: float,
                       threshold: float = 0.087) -> List[str]:
        """Build measurement method cascade.

        nadir_corrected is always first — it shifts the projected nadir
        ground-point for both pitch and roll, and is MOST valuable when
        the aircraft is banking.  The old restriction (only near-nadir)
        was backwards.
        """
        base = ["trimmed_centroid", "inlier_centroid",
                "weighted_centroid", "projected_center"]
        return ["nadir_corrected"] + base

    # ════════════════════════════════════════════════════════════
    # IMU fallback
    # ════════════════════════════════════════════════════════════

    def _imu_fallback_result(self, imu_data: Dict, timestamp: float,
                             elapsed: float, unc: Dict,
                             reason: str) -> Dict:
        # Use particle filter estimate if available (maintains continuity
        # with previous visual corrections) instead of raw EKF position.
        if self.particle_filter is not None:
            est_x, est_y, est_hdg = self.particle_filter.get_estimate()
            fb_lat, fb_lon = tile_to_latlon(
                est_x, est_y, self.cfg.TMS_ZOOM_LEVEL)
            fb_heading = est_hdg
        else:
            fb_lat, fb_lon = imu_data["lat"], imu_data["lon"]
            fb_heading = imu_data["heading"]
        return {
            "position": (fb_lat, fb_lon),
            "heading": fb_heading,
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
