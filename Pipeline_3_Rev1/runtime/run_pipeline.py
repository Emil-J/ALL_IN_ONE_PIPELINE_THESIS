"""
runtime/run_pipeline.py  —  Pure Python frame-by-frame runtime pipeline.

No notebook dependency. Reads from a pre-recorded file source (default) or
a live SimConnect source (--source simconnect). Writes results.csv per frame
and run_meta.json at end. No GT comparison, no visualization.

Usage (file mode):
    python runtime/run_pipeline.py --source file --run-id run_001

Usage (live mode):
    python runtime/run_pipeline.py --source simconnect --run-id live_001

Columns in results.csv (21):
    frame_idx, timestamp, image_name, final_lat, final_lon, heading_deg,
    method, gate_pass, search_time_s, cs_shape, inliers, semantic_conf,
    homo_lat, homo_lon, homo_corrected_lat, homo_corrected_lon,
    meta_tile_verified, ekf_pos_sigma, r_used_sqrt, tiles_tested,
    verification_matches
"""

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2

# ── path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent           # Pipeline_3_Rev1/
REPO = ROOT.parent                 # All_In_One_Pipeline/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from config import config
from src.tile_utils import TileLoader, haversine_distance
from src.image_utils import load_image
from src.geometric_matcher import initialize_matcher
from src.semantic_model import load_semantic_model
from src.temporal_searcher import TemporalSearcher
from src.ekf_ins import (ErrorStateEKF, step_ekf, barometric_altitude)
from src.wmm_declination import get_mag_field
from runtime.simconnect_adapter import FileSource

# ── CSV column order ──────────────────────────────────────────────────────────
RESULT_COLUMNS = [
    "frame_idx", "timestamp", "image_name",
    "final_lat", "final_lon", "heading_deg",
    "altitude_m", "roll_deg", "pitch_deg",
    "vel_n", "vel_e", "vel_d",
    "gps_lat", "gps_lon", "gps_alt_m",
    "method", "gate_pass",
    "search_time_s", "cs_shape", "inliers", "semantic_conf",
    "homo_lat", "homo_lon", "homo_corrected_lat", "homo_corrected_lon",
    "meta_tile_verified", "ekf_pos_sigma", "r_used_sqrt",
    "tiles_tested", "verification_matches",
    "inference_ms",
    "visual_innovation_m", "max_visual_innovation_m", "visual_rejected_reason",
    "pf_update_source", "search_radius_m", "search_radius_capped",
    "visual_quality_pass", "ekf_update_applied",
    "relocalization_candidate", "relocalization_applied",
]

# MAVLink GPS_INPUT (MSG 232) columns — written when SAVE_ANALYSIS_DATA=True
PX4_GPS_COLUMNS = [
    "time_usec", "gps_id", "ignore_flags", "time_week_ms", "time_week",
    "fix_type", "lat", "lon", "alt", "hdop", "vdop",
    "vn", "ve", "vd", "speed_accuracy",
    "horiz_accuracy", "vert_accuracy", "satellites_visible", "yaw",
]

# Per-frame analysis extras not in results.csv — written when SAVE_ANALYSIS_DATA=True
EXTRAS_COLUMNS = [
    "frame_idx", "timestamp", "n_eff", "particle_spread",
    "homo_offset_north_m", "homo_offset_east_m",
]

# Per-component timing columns — written when SAVE_TIMING_DATA=True
TIMING_COLUMNS = [
    "frame_idx", "timestamp",
    "frame_capture_ts", "gps_estimate_ts",
    "cold_search_ms", "pf_predict_ms", "semantic_ms", "meta_tile_ms",
    "homography_ms", "pf_update_ms", "total_ms",
]


def _json_default(obj):
    """JSON serializer for numpy scalars/arrays and NaN floats."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert (H,W) uint8 class mask to (H,W,3) RGB using config COLOR_MAP."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in config.COLOR_MAP.items():
        rgb[mask == cls_id] = color
    return rgb


def _draw_match_viz(img1: np.ndarray, img2: np.ndarray,
                    match_result: dict, max_matches: int = 80) -> np.ndarray:
    """Draw SP+LG keypoint matches between two RGB images. Returns BGR ndarray."""
    if match_result is None:
        return None
    matches = match_result.get("matches", [])
    if len(matches) == 0:
        return None
    kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 3)
           for p in match_result["keypoints1"]]
    kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 3)
           for p in match_result["keypoints2"]]
    dms = [cv2.DMatch(int(m[0]), int(m[1]), 0) for m in matches[:max_matches]]
    i1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR) if img1.ndim == 3 else img1
    i2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR) if img2.ndim == 3 else img2
    return cv2.drawMatches(i1, kp1, i2, kp2, dms, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def _save_trace_images(trace_dir: Path, query_frame: np.ndarray, result: dict):
    """Save per-frame pipeline trace images into trace_dir."""
    td = result.get("trace_data") or {}

    cv2.imwrite(str(trace_dir / "query.jpg"),
                cv2.cvtColor(query_frame, cv2.COLOR_RGB2BGR))

    qr = td.get("query_rotated")
    if qr is not None:
        cv2.imwrite(str(trace_dir / "query_rotated.jpg"),
                    cv2.cvtColor(qr, cv2.COLOR_RGB2BGR))

    qsm = td.get("query_semantic_map")
    if qsm is not None:
        cv2.imwrite(str(trace_dir / "semantic_mask.png"),
                    cv2.cvtColor(_mask_to_rgb(qsm), cv2.COLOR_RGB2BGR))

    ref_img = td.get("meta_tile")
    if ref_img is None:
        ref_img = td.get("ref_img")
    if ref_img is not None:
        cv2.imwrite(str(trace_dir / "reference_tile.png"),
                    cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))

    mr = td.get("match_result")
    viz_query = qr if qr is not None else query_frame
    if mr is not None and ref_img is not None:
        viz = _draw_match_viz(viz_query, ref_img, mr)
        if viz is not None:
            cv2.imwrite(str(trace_dir / "matches.png"), viz)


def _build_trace_json(frame_idx: int, ts: float, result: dict,
                      result_row: dict) -> dict:
    """Build JSON-serializable trace dict from result dicts (no image arrays)."""
    td = result.get("trace_data") or {}
    eb = result.get("_ekf_before") or {}
    ea = result.get("_ekf_after") or {}

    _ekf_keys = ("latitude", "longitude", "yaw", "altitude",
                 "vel_n", "vel_e", "vel_d")

    fp = [{"tx": int(t[0]), "ty": int(t[1]), "n_matches": int(t[2])}
          for t in td.get("first_pass_tiles", [])]
    sp = [{"tx": int(t[0]), "ty": int(t[1])}
          for t in td.get("second_pass_tiles", [])]
    ranked = [{"tx": int(t[0]), "ty": int(t[1]), "score": int(t[2])}
              for t in td.get("ranked_tiles", [])]
    top3 = result.get("ranked_tiles", [])

    return {
        "frame_idx": frame_idx,
        "timestamp": round(ts, 4),
        "method": result.get("method", ""),
        "gate_pass": bool(result.get("gate_pass", False)),
        "rotation_deg": round(float(td.get("rotation_deg", 0)), 2),
        "search_radius_m": (round(float(td["search_radius_m"]), 1)
                            if td.get("search_radius_m") else None),
        "ekf_before": {k: (round(float(eb[k]), 6) if eb.get(k) is not None else None)
                       for k in _ekf_keys},
        "ekf_after":  {k: (round(float(ea[k]), 6) if ea.get(k) is not None else None)
                       for k in _ekf_keys},
        "pf_center":  (list(td["pf_center"])  if td.get("pf_center")  else None),
        "ekf_center": (list(td["ekf_center"]) if td.get("ekf_center") else None),
        "pf_state": {
            "n_eff":     result.get("n_eff"),
            "spread_m":  result.get("particle_spread"),
        },
        "first_pass_tiles": fp,
        "second_pass_tiles": sp,
        "ranked_tiles": ranked,
        "meta_tile_info": {
            "n_tiles": len(top3),
            "tiles": [[int(t[0]), int(t[1])] for t in top3],
            "verification_matches": result.get("verification_matches"),
            "verified": bool(result.get("meta_tile_verified") or False),
        },
        "homography": {
            "cs_shape":            result_row.get("cs_shape"),
            "inliers":             result_row.get("inliers"),
            "homo_lat":            result_row.get("homo_lat"),
            "homo_lon":            result_row.get("homo_lon"),
            "homo_corrected_lat":  result_row.get("homo_corrected_lat"),
            "homo_corrected_lon":  result_row.get("homo_corrected_lon"),
        },
        "semantic": {
            "conf": result.get("semantic_confidence"),
        },
        "ekf_pos_sigma_m": result_row.get("ekf_pos_sigma"),
        "r_used_sqrt":     result_row.get("r_used_sqrt"),
        "visual_innovation_m":      result_row.get("visual_innovation_m"),
        "max_visual_innovation_m":  result_row.get("max_visual_innovation_m"),
        "visual_rejected_reason":   result_row.get("visual_rejected_reason", ""),
        "pf_update_source":         result_row.get("pf_update_source", ""),
        "search_radius_capped":     bool(result_row.get("search_radius_capped", 0)),
        "visual_quality_pass":      bool(result_row.get("visual_quality_pass", 0)),
        "ekf_update_applied":       bool(result_row.get("ekf_update_applied", 0)),
        "relocalization_candidate": bool(result_row.get("relocalization_candidate", 0)),
        "relocalization_applied":   bool(result_row.get("relocalization_applied", 0)),
    }


def _parse_args():
    p = argparse.ArgumentParser(description="Pipeline 3 runtime")
    p.add_argument("--source", choices=["file", "simconnect"], default="file")
    p.add_argument("--imu-csv",    type=Path, default=None,
                   help="Override IMU CSV path (file mode)")
    p.add_argument("--frames-dir", type=Path, default=None,
                   help="Override frames directory (file mode)")
    p.add_argument("--run-id",     default=None,
                   help="Run identifier (default: timestamp-based)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Root for output dirs (default: config.RUNS_OUTPUT_DIR)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many frames")
    p.add_argument("--start-row",  type=int, default=0,
                   help="IMU CSV row to start from (file mode)")
    p.add_argument("--debug",      action="store_true",
                   help="Enable meta-tile PNG saves and history accumulation")
    return p.parse_args()


def _build_run_dir(args) -> Path:
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = "run_" + time.strftime("%Y%m%d_%H%M%S")
    base = args.output_dir or config.RUNS_OUTPUT_DIR
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def _set_deployment_flags(debug: bool):
    config.DEBUG_SAVE_METATILES = debug
    config.ACCUMULATE_HISTORY   = debug


def _init_models():
    semantic_model = load_semantic_model(
        config.SEMANTIC_MODEL_PATH, config.DEVICE)
    matcher = initialize_matcher(config.DEVICE, config.MAX_NUM_KEYPOINTS)
    tile_loader = TileLoader(
        config.REFERENCE_TILES_DIR,
        config.REFERENCE_PRED_DIR,
        zoom=config.TMS_ZOOM_LEVEL,
        x_range=(config.TILE_X_MIN, config.TILE_X_MAX),
        y_range=(config.TILE_Y_MIN, config.TILE_Y_MAX),
    )
    feature_store = None
    if (hasattr(config, "REFERENCE_FEATURES_PATH")
            and config.REFERENCE_FEATURES_PATH.exists()):
        from Dataset_Preprocessing.feature_store import FeatureStoreLoader
        feature_store = FeatureStoreLoader(
            config.REFERENCE_FEATURES_PATH, device=config.DEVICE)
        feature_store.open()
    return semantic_model, matcher, tile_loader, feature_store

def _safe_int(value, default: int = 0) -> int:
    """Safely convert value to int. Returns default for None/NaN/bad values."""
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_meta_quality(result: dict) -> tuple[bool, int, int]:
    """
    Extract meta-tile verification values from a TemporalSearcher result.

    Returns:
        meta_verified: bool
        tiles_tested: int
        verification_matches: int

    This prevents live/file mode from crashing when relocalization logic expects
    verification values that may be missing or stored under different keys.
    """
    meta_info = result.get("meta_tile_info") or {}

    meta_verified_raw = result.get("meta_tile_verified")
    if meta_verified_raw is None:
        meta_verified_raw = meta_info.get("verified", False)

    tiles_tested_raw = result.get("tiles_tested")
    if tiles_tested_raw is None:
        tiles_tested_raw = meta_info.get("tiles_tested", 0)

    ver_matches_raw = result.get("verification_matches")
    if ver_matches_raw is None:
        ver_matches_raw = meta_info.get("verification_matches", 0)

    meta_verified = bool(meta_verified_raw)
    tiles_tested = _safe_int(tiles_tested_raw, 0)
    ver_matches = _safe_int(ver_matches_raw, 0)

    return meta_verified, tiles_tested, ver_matches

def _init_ekf(raw_df: pd.DataFrame, start_row: int):
    """Warm up a live EKF through start_row."""
    lat0      = raw_df["latitude"].iloc[0]
    lon0      = raw_df["longitude"].iloc[0]
    alt0      = barometric_altitude(raw_df["barometer_pressure"].iloc[0])
    heading0  = np.degrees(raw_df["heading_magnetic"].iloc[0])
    airspeed0 = (raw_df["airspeed_true"].iloc[0]
                 if "airspeed_true" in raw_df.columns else None)
    mag_dec_deg, mag_inc_deg = get_mag_field(lat0, lon0, alt0)
    print(f"[EKF] WMM2025 dec={mag_dec_deg:.2f}°  inc={mag_inc_deg:.2f}°")
    ekf = ErrorStateEKF(lat0, lon0, alt0, heading0, airspeed0,
                        mag_dec_deg=mag_dec_deg, mag_inc_deg=mag_inc_deg)
    prev_ts = None
    for i in range(start_row + 1):
        row_dict = raw_df.iloc[i].to_dict()
        step_ekf(ekf, row_dict, prev_ts)
        prev_ts = row_dict["timestamp"]
    return ekf, prev_ts


def _process_one_frame(frame_idx, csv_idx, ts, frame_path,
                       raw_df, ekf, prev_ts_ekf, searcher,
                       frame_capture_ts=None,
                       recovery_state=None,
                       prev_frame_ts=None) -> tuple:
    """
    Returns (query_frame, row_dict, final_lat, final_lon, prev_ts_ekf_new,
             result_row_dict, result_dict, gps_estimate_ts).
    Mutates ekf in place.
    """
    ekf_state_before = ekf.get_state()   # snapshot before IMU predict step
    row_dict = raw_df.iloc[csv_idx].to_dict()
    step_ekf(ekf, row_dict, prev_ts_ekf)
    prev_ts_ekf_new = row_dict["timestamp"]

    ekf_state = ekf.get_state()
    ekf_lat   = ekf_state["latitude"]
    ekf_lon   = ekf_state["longitude"]
    ekf_yaw   = ekf_state["yaw"]
    vel       = math.sqrt(ekf_state["vel_n"] ** 2 + ekf_state["vel_e"] ** 2)
    bank_rad  = abs(row_dict.get("bank", 0.0))

    imu_data = {
        "lat":           ekf_lat,
        "lon":           ekf_lon,
        "heading":       ekf_yaw,
        "pos_sigma":     math.sqrt(max(ekf.P[8, 8], ekf.P[9, 9])),
        "heading_sigma": 15.0,
        "velocity_mps":  vel,
        "gyro_z_dps":    row_dict.get("gyro_y", 0.0) * (180.0 / math.pi),
        "pitch":         row_dict.get("pitch", 0.0),
        "roll":          row_dict.get("bank", 0.0),
    }

    if frame_capture_ts is None:
        frame_capture_ts = time.perf_counter()
    query_frame = load_image(frame_path)
    result      = searcher.process_frame(query_frame, imu_data, timestamp=ts)

    gate_pass     = result.get("gate_pass", False)
    homo_pos      = result.get("homo_position")
    vq            = result.get("visual_quality", {})
    cs            = vq.get("CShape", 0.0)
    ni            = vq.get("inliers", 0)
    sem_conf      = result.get("semantic_confidence") or 0.5
    meta_verified, tiles_tested, ver_matches = _extract_meta_quality(result)
    search_time   = result.get("search_time", 0.0)

    result["meta_tile_verified"] = meta_verified
    result["tiles_tested"] = tiles_tested
    result["verification_matches"] = ver_matches

    # Snapshot visual quality gate result BEFORE innovation gate can override it.
    # True = visual localisation itself succeeded (CShape + inliers + homo_position);
    # does NOT reflect EKF innovation acceptance.
    visual_quality_pass = gate_pass

    # Lookahead correction — scale by cos(bank) because during a banked turn
    # the camera's forward component decreases, reducing the ground footprint offset.
    homo_lat_raw = homo_lon_raw = None
    homo_corr_lat = homo_corr_lon = None
    visual_innovation_m = None
    max_innovation_m = None
    visual_rejected_reason = ""
    if homo_pos is not None:
        homo_lat_raw, homo_lon_raw = homo_pos
        if config.LOOKAHEAD_M > 0:
            h_rad = math.radians(ekf_yaw)
            effective_lookahead = config.LOOKAHEAD_M * math.cos(bank_rad)
            corr_north = -effective_lookahead * math.cos(h_rad)
            corr_east  = -effective_lookahead * math.sin(h_rad)
            homo_corr_lat = (homo_lat_raw
                             + corr_north / 111320.0)
            homo_corr_lon = (homo_lon_raw
                             + corr_east / (111320.0
                                            * math.cos(math.radians(homo_lat_raw))))
            homo_pos = (homo_corr_lat, homo_corr_lon)

        # Innovation gate: reject if corrected position is implausibly far from EKF.
        # dt clamped to [0.5, 4.0] s — prevents both over-rejection (stale 1 s floor)
        # and over-permissiveness (pauses or stalls producing very large dt).
        visual_innovation_m = haversine_distance(
            homo_pos[0], homo_pos[1], ekf_lat, ekf_lon)
        pos_sigma_now = math.sqrt(max(ekf.P[8, 8], ekf.P[9, 9]))
        dt_gate = min(max((ts - prev_frame_ts) if prev_frame_ts is not None else 1.0,
                          0.5), 4.0)
        max_innovation_m = max(150.0, 3.0 * pos_sigma_now + vel * dt_gate + 50.0)
        if gate_pass and visual_innovation_m > max_innovation_m:
            gate_pass = False
            visual_rejected_reason = "innovation_too_large"

    # ── Relocalization: recover EKF after streak of strong innovation rejections ──
    # When the EKF has drifted far enough that its covariance no longer reflects the
    # true uncertainty, visually strong frames are rejected even though the homography
    # is correct.  After CONSECUTIVE_THRESHOLD such frames with coherent positions,
    # inflate P and apply a recovery update so the EKF can converge again.
    relocalization_candidate = False
    relocalization_applied = False
    if recovery_state is not None:
        cfg_r = searcher.cfg
        if (not gate_pass
                and visual_rejected_reason == "innovation_too_large"
                and homo_pos is not None):
            relocalization_candidate = (
                cs >= cfg_r.RELOCALIZATION_CSHAPE_MIN
                and ni >= cfg_r.RELOCALIZATION_INLIERS_MIN
                and meta_verified
                and ver_matches >= cfg_r.RELOCALIZATION_VERIFICATION_MIN
            )
            if relocalization_candidate:
                recovery_state["consecutive"] += 1
                recovery_state["positions"].append(homo_pos)
                if len(recovery_state["positions"]) > 5:
                    recovery_state["positions"].pop(0)
            else:
                recovery_state["consecutive"] = 0
                recovery_state["positions"].clear()

            if (relocalization_candidate
                    and recovery_state["consecutive"] >= cfg_r.RELOCALIZATION_CONSECUTIVE_THRESHOLD
                    and len(recovery_state["positions"]) >= 3):
                recent = recovery_state["positions"][-3:]
                dt_hop = min(max((ts - prev_frame_ts) if prev_frame_ts is not None else 1.0,
                                 0.5), 4.0)
                coherent = all(
                    haversine_distance(recent[k][0], recent[k][1],
                                       recent[k + 1][0], recent[k + 1][1])
                    <= cfg_r.RELOCALIZATION_COHERENCE_HOP_FACTOR * vel * dt_hop
                    for k in range(len(recent) - 1)
                )
                if coherent:
                    ekf.P[8, 8] = max(ekf.P[8, 8],
                                      cfg_r.RELOCALIZATION_PRIOR_STD_M ** 2)
                    ekf.P[9, 9] = max(ekf.P[9, 9],
                                      cfg_r.RELOCALIZATION_PRIOR_STD_M ** 2)
                    ekf.update_position(homo_pos[0], homo_pos[1],
                                        R_pos_m2=cfg_r.RELOCALIZATION_R_M ** 2)
                    gate_pass = True
                    visual_rejected_reason = "relocalization_applied"
                    relocalization_applied = True
                    recovery_state["consecutive"] = 0
                    recovery_state["positions"].clear()
                    searcher.frame_count = 0   # force PF cold-start on next frame
        elif gate_pass:
            recovery_state["consecutive"] = 0
            recovery_state["positions"].clear()

    result["gate_pass"] = gate_pass
    result["visual_rejected_reason"] = visual_rejected_reason
    result["visual_innovation_m"] = visual_innovation_m
    result["max_visual_innovation_m"] = max_innovation_m
    result["relocalization_candidate"] = relocalization_candidate
    result["relocalization_applied"] = relocalization_applied

    method_str = result.get("method", "")
    r_used = None
    ekf_update_applied = False
    if gate_pass and homo_pos is not None:
        if relocalization_applied:
            # EKF already updated above with RELOCALIZATION_R_M; record for logging.
            r_used = searcher.cfg.RELOCALIZATION_R_M ** 2
            ekf_update_applied = True
        else:
            if method_str == "cold_start":
                r_used = config.R_COLD_START
            else:
                r_used = config.R_HIGH if (cs > 0.5 and ni > 100) else config.R_MED
            if bank_rad > config.TURN_ROLL_THRESHOLD_RAD:
                r_used *= config.TURN_R_MULTIPLIER
            if not meta_verified:
                r_used *= 2.0
            r_used *= max(0.5, 2.0 - 1.5 * sem_conf)
            ekf.update_position(homo_pos[0], homo_pos[1], R_pos_m2=r_used)
            ekf_update_applied = True

    final = ekf.get_state()
    final_lat, final_lon = final["latitude"], final["longitude"]
    pos_sigma = math.sqrt(max(ekf.P[8, 8], ekf.P[9, 9]))
    gps_estimate_ts = time.perf_counter()
    result["_ekf_before"] = ekf_state_before
    result["_ekf_after"]  = final

    gps_alt_ft = row_dict.get("altitude")
    result_row = {
        "frame_idx":          frame_idx,
        "timestamp":          ts,
        "image_name":         Path(frame_path).stem,
        "final_lat":          final_lat,
        "final_lon":          final_lon,
        "heading_deg":        ekf_yaw,
        "altitude_m":         round(final["altitude"], 2),
        "roll_deg":           round(final["roll"], 3),
        "pitch_deg":          round(final["pitch"], 3),
        "vel_n":              round(final["vel_n"], 3),
        "vel_e":              round(final["vel_e"], 3),
        "vel_d":              round(final["vel_d"], 3),
        "gps_lat":            row_dict.get("latitude"),
        "gps_lon":            row_dict.get("longitude"),
        "gps_alt_m":          round(gps_alt_ft * 0.3048, 2) if gps_alt_ft is not None else None,
        "method":             result.get("method", ""),
        "gate_pass":          int(gate_pass),
        "search_time_s":      round(search_time, 4),
        "cs_shape":           round(cs, 4),
        "inliers":            ni,
        "semantic_conf":      round(sem_conf, 4),
        "homo_lat":           homo_lat_raw,
        "homo_lon":           homo_lon_raw,
        "homo_corrected_lat": homo_corr_lat,
        "homo_corrected_lon": homo_corr_lon,
        "meta_tile_verified": int(meta_verified),
        "ekf_pos_sigma":      round(pos_sigma, 2),
        "r_used_sqrt":        round(math.sqrt(r_used), 2) if r_used else None,
        "tiles_tested":       tiles_tested,
        "verification_matches": ver_matches,
        "inference_ms":       None,   # file mode has no real-time capture latency
        "visual_innovation_m": round(visual_innovation_m, 1) if visual_innovation_m is not None else None,
        "max_visual_innovation_m": round(max_innovation_m, 1) if max_innovation_m is not None else None,
        "visual_rejected_reason": visual_rejected_reason,
        "pf_update_source":   result.get("pf_update_source", ""),
        "search_radius_m":    round(result.get("search_radius_m") or 0.0, 1) if result.get("search_radius_m") is not None else None,
        "search_radius_capped": int(bool(result.get("search_radius_capped", False))),
        "visual_quality_pass":      int(visual_quality_pass),
        "ekf_update_applied":       int(ekf_update_applied),
        "relocalization_candidate": int(relocalization_candidate),
        "relocalization_applied":   int(relocalization_applied),
    }
    return (query_frame, row_dict, final_lat, final_lon,
            prev_ts_ekf_new, result_row, result, gps_estimate_ts)


def run_file_mode(args, run_dir: Path, run_id: str):
    imu_csv    = args.imu_csv    or config.IMU_CSV_PATH
    frames_dir = args.frames_dir or config.QUERY_FRAMES_DIR
    start_row  = args.start_row

    # FileSource handles alignment and provides raw_df for EKF stepping.
    # No batch EKF (preprocess_imu_csv) needed — that only exists for GT comparison.
    src    = FileSource(imu_csv, frames_dir)
    raw_df = src.raw_df
    aligned = list(src.iter_aligned(start_row, args.max_frames))

    print(f"[run_pipeline] run_id={run_id}  source=file  frames={len(aligned)}")
    print(f"[run_pipeline] output: {run_dir}")

    ekf, prev_ts_ekf = _init_ekf(raw_df, start_row)
    semantic_model, matcher, tile_loader, feature_store = _init_models()
    searcher = TemporalSearcher(semantic_model, matcher, tile_loader, config,
                                feature_store=feature_store)
    searcher.frame_count = 0
    searcher.particle_filter = None

    csv_path   = run_dir / "results.csv"
    flight_dir = run_dir / "flight_data"
    t0         = time.perf_counter()
    gate_count = 0
    timing_rows = []
    recovery_state = {"consecutive": 0, "positions": []}
    prev_frame_ts  = None

    # Open optional extra output files
    _px4_f = _extras_f = _px4_w = _extras_w = None
    if config.SAVE_ANALYSIS_DATA:
        _px4_f   = open(run_dir / "px4_gps_input.csv",  "w", newline="", encoding="utf-8")
        _extras_f = open(run_dir / "analysis_extras.csv", "w", newline="", encoding="utf-8")
        _px4_w   = csv.DictWriter(_px4_f,   fieldnames=PX4_GPS_COLUMNS)
        _extras_w = csv.DictWriter(_extras_f, fieldnames=EXTRAS_COLUMNS)
        _px4_w.writeheader()
        _extras_w.writeheader()

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()

            for i, (csv_idx, _row_dict, ts, frame_path) in enumerate(aligned):
                frame_capture_ts = time.perf_counter()
                (query_frame, row_dict, final_lat, final_lon,
                 prev_ts_ekf, result_row, result, gps_estimate_ts) = _process_one_frame(
                    i, csv_idx, ts, frame_path,
                    raw_df, ekf, prev_ts_ekf, searcher,
                    frame_capture_ts=frame_capture_ts,
                    recovery_state=recovery_state,
                    prev_frame_ts=prev_frame_ts)
                prev_frame_ts = ts

                writer.writerow(result_row)
                if result_row["gate_pass"]:
                    gate_count += 1

                # ── Optional: save query frame JPEG ────────────────────────
                if config.SAVE_QUERY_FRAMES:
                    flight_dir.mkdir(exist_ok=True)
                    cv2.imwrite(
                        str(flight_dir / f"frame_{i:04d}.jpg"),
                        cv2.cvtColor(query_frame, cv2.COLOR_RGB2BGR))

                # ── Optional: save raw IMU row JSON ────────────────────────
                if config.SAVE_IMU_ROWS:
                    flight_dir.mkdir(exist_ok=True)
                    (flight_dir / f"frame_{i:04d}_imu.json").write_text(
                        json.dumps({
                            k: (None if isinstance(v, float) and math.isnan(v) else v)
                            for k, v in row_dict.items()
                        }), encoding="utf-8")

                # ── Optional: PX4 GPS_INPUT + analysis extras ──────────────
                if config.SAVE_ANALYSIS_DATA:
                    ekf_state = ekf.get_state()
                    pos_sigma = result_row["ekf_pos_sigma"]
                    _px4_w.writerow({
                        "time_usec":          int(ts * 1e6),
                        "gps_id":             0,
                        "ignore_flags":       0x0006,
                        "time_week_ms":       0,
                        "time_week":          0,
                        "fix_type":           3,
                        "lat":                int(final_lat * 1e7),
                        "lon":                int(final_lon * 1e7),
                        "alt":                round(ekf_state["altitude"], 2),
                        "hdop":               0,
                        "vdop":               0,
                        "vn":                 round(ekf_state["vel_n"], 3),
                        "ve":                 round(ekf_state["vel_e"], 3),
                        "vd":                 round(ekf_state["vel_d"], 3),
                        "speed_accuracy":     0.5,
                        "horiz_accuracy":     pos_sigma,
                        "vert_accuracy":      round(2.0 * pos_sigma, 2),
                        "satellites_visible": 0,
                        "yaw":                int(result_row["heading_deg"] * 100),
                    })
                    homo_cl = result_row.get("homo_corrected_lat")
                    homo_clon = result_row.get("homo_corrected_lon")
                    gps_lat = result_row.get("gps_lat")
                    gps_lon = result_row.get("gps_lon")
                    off_n = off_e = None
                    if (homo_cl is not None and gps_lat is not None
                            and not (isinstance(homo_cl, float) and math.isnan(homo_cl))):
                        off_n = round((homo_cl - gps_lat) * 111320.0, 1)
                        off_e = round((homo_clon - gps_lon) * 111320.0
                                      * math.cos(math.radians(gps_lat)), 1)
                    _extras_w.writerow({
                        "frame_idx":          i,
                        "timestamp":          ts,
                        "n_eff":              result.get("n_eff"),
                        "particle_spread":    result.get("particle_spread"),
                        "homo_offset_north_m": off_n,
                        "homo_offset_east_m":  off_e,
                    })

                # ── Optional: save full pipeline trace ────────────────────
                if config.SAVE_PIPELINE_TRACE:
                    trace_dir = run_dir / "pipeline_data" / f"frame_{i:04d}"
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    _save_trace_images(trace_dir, query_frame, result)
                    (trace_dir / "imu.json").write_text(
                        json.dumps({
                            k: (None if isinstance(v, float) and math.isnan(v) else v)
                            for k, v in row_dict.items()
                        }), encoding="utf-8")
                    (trace_dir / "trace.json").write_text(
                        json.dumps(_build_trace_json(i, ts, result, result_row),
                                   indent=2, default=_json_default),
                        encoding="utf-8")

                # ── Optional: collect timing data ──────────────────────────
                if config.SAVE_TIMING_DATA:
                    t_dict = result.get("timing") or {}
                    timing_rows.append({
                        "frame_idx":        i,
                        "timestamp":        ts,
                        "frame_capture_ts": frame_capture_ts,
                        "gps_estimate_ts":  gps_estimate_ts,
                        "cold_search_ms":   t_dict.get("cold_search_ms", 0.0),
                        "pf_predict_ms":    t_dict.get("pf_predict_ms", 0.0),
                        "semantic_ms":      t_dict.get("semantic_ms", 0.0),
                        "meta_tile_ms":     t_dict.get("meta_tile_ms", 0.0),
                        "homography_ms":    t_dict.get("homography_ms", 0.0),
                        "pf_update_ms":     t_dict.get("pf_update_ms", 0.0),
                        "total_ms":         t_dict.get("total_ms", 0.0),
                    })

                if (i + 1) % 10 == 0:
                    f.flush()

                elapsed = time.perf_counter() - t0
                fps = (i + 1) / elapsed
                print(f"  F{i:4d} | {result_row['image_name']:<26s} | "
                      f"gate={'PASS' if result_row['gate_pass'] else 'fail'} | "
                      f"({final_lat:.6f}, {final_lon:.6f}) | "
                      f"{elapsed:.1f}s ({fps:.2f}fps)")
    finally:
        if _px4_f:
            _px4_f.close()
        if _extras_f:
            _extras_f.close()

    if config.SAVE_TIMING_DATA and timing_rows:
        import pandas as pd
        pd.DataFrame(timing_rows, columns=TIMING_COLUMNS).to_csv(
            run_dir / "timing_data.csv", index=False)
        print(f"[run_pipeline] timing_data: {run_dir / 'timing_data.csv'}")

    total = time.perf_counter() - t0
    _save_meta(run_dir, run_id, "file", len(aligned), gate_count, total,
               str(imu_csv), str(frames_dir))
    print(f"\n[run_pipeline] Done — {len(aligned)} frames in {total:.1f}s "
          f"({total / max(len(aligned), 1):.2f}s/frame)")
    print(f"[run_pipeline] results: {csv_path}")


def run_simconnect_mode(args, run_dir: Path, run_id: str):
    from runtime.simconnect_adapter import SimConnectLiveSource

    print(f"[run_pipeline] run_id={run_id}  source=simconnect")
    print(f"[run_pipeline] output: {run_dir}")

    semantic_model, matcher, tile_loader, feature_store = _init_models()
    searcher = TemporalSearcher(semantic_model, matcher, tile_loader, config,
                                feature_store=feature_store)
    searcher.frame_count = 0
    searcher.particle_filter = None

    source = SimConnectLiveSource()
    source.connect()

    # Bootstrap EKF from first good GPS sample
    print("[run_pipeline] Waiting for first valid SimConnect sample...")
    while True:
        row = source.get_latest_row()
        if row and row.get("latitude") and abs(row["latitude"]) > 1.0:
            break
        time.sleep(0.05)

    lat0     = row["latitude"]
    lon0     = row["longitude"]
    alt0     = barometric_altitude(row.get("barometer_pressure") or 1013.25)
    heading0 = math.degrees(row.get("heading_magnetic") or 0.0)
    mag_dec_deg, mag_inc_deg = get_mag_field(lat0, lon0, alt0)
    print(f"[run_pipeline] WMM2025 dec={mag_dec_deg:.2f}°  inc={mag_inc_deg:.2f}°")
    ekf      = ErrorStateEKF(lat0, lon0, alt0, heading0, None,
                             mag_dec_deg=mag_dec_deg, mag_inc_deg=mag_inc_deg)
    prev_ts  = row.get("timestamp", time.time())
    print(f"[run_pipeline] EKF bootstrapped: ({lat0:.6f}, {lon0:.6f})  yaw={heading0:.1f}°")

    csv_path    = run_dir / "results.csv"
    flight_dir  = run_dir / "flight_data"
    max_frames  = args.max_frames
    frame_idx   = 0
    gate_count  = 0
    t0          = time.perf_counter()
    last_frame_id = None
    last_imu_ts   = None   # guards against repeated step_ekf on same row
    timing_rows   = []
    recovery_state = {"consecutive": 0, "positions": []}
    prev_frame_ts  = None

    # Open optional extra output files
    _px4_f = _extras_f = _px4_w = _extras_w = None
    if config.SAVE_ANALYSIS_DATA:
        _px4_f   = open(run_dir / "px4_gps_input.csv",  "w", newline="", encoding="utf-8")
        _extras_f = open(run_dir / "analysis_extras.csv", "w", newline="", encoding="utf-8")
        _px4_w   = csv.DictWriter(_px4_f,   fieldnames=PX4_GPS_COLUMNS)
        _extras_w = csv.DictWriter(_extras_f, fieldnames=EXTRAS_COLUMNS)
        _px4_w.writeheader()
        _extras_w.writeheader()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()

        try:
            while max_frames is None or frame_idx < max_frames:
                # ── EKF predict at background-thread IMU rate ──────────────
                # Only step when a genuinely new IMU row arrives; calling
                # step_ekf with the same row repeatedly drives P → 0 because
                # sensor updates (baro, accel/mag, airspeed) run each call.
                row = source.get_latest_row()
                if row:
                    row_ts = row.get("timestamp")
                    if row_ts is not None and row_ts != last_imu_ts:
                        step_ekf(ekf, row, prev_ts)
                        prev_ts = row_ts
                        last_imu_ts = row_ts

                # ── Visual processing when a new frame arrives ─────────────
                frame_img, frame_id, frame_capture_ts = source.get_latest_frame()
                if frame_img is None or frame_id == last_frame_id:
                    time.sleep(0.005)
                    continue
                last_frame_id  = frame_id
                _frame_cap_ts  = frame_capture_ts  # perf_counter at capture time

                ekf_state = ekf.get_state()
                ekf_lat   = ekf_state["latitude"]
                ekf_lon   = ekf_state["longitude"]
                ekf_yaw   = ekf_state["yaw"]
                vel       = math.sqrt(ekf_state["vel_n"] ** 2
                                      + ekf_state["vel_e"] ** 2)

                imu_data = {
                    "lat":           ekf_lat,
                    "lon":           ekf_lon,
                    "heading":       ekf_yaw,
                    "pos_sigma":     math.sqrt(max(ekf.P[8, 8], ekf.P[9, 9])),
                    "heading_sigma": 15.0,
                    "velocity_mps":  vel,
                    "gyro_z_dps":    row.get("gyro_y", 0.0) * (180.0 / math.pi)
                                     if row else 0.0,
                    "pitch":         row.get("pitch", 0.0) if row else 0.0,
                    "roll":          row.get("bank", 0.0) if row else 0.0,
                }

                ts = row.get("timestamp", time.time()) if row else time.time()
                ekf_state_before = ekf_state  # snapshot before visual update
                result = searcher.process_frame(frame_img, imu_data, timestamp=ts)

                gate_pass     = result.get("gate_pass", False)
                homo_pos      = result.get("homo_position")
                vq            = result.get("visual_quality", {})
                cs            = vq.get("CShape", 0.0)
                ni            = vq.get("inliers", 0)
                sem_conf      = result.get("semantic_confidence") or 0.5
                meta_verified, tiles_tested, ver_matches = _extract_meta_quality(result)

                result["meta_tile_verified"] = meta_verified
                result["tiles_tested"] = tiles_tested
                result["verification_matches"] = ver_matches

                # Snapshot visual quality gate result BEFORE innovation gate can override it.
                visual_quality_pass = gate_pass

                homo_lat_raw = homo_lon_raw = homo_corr_lat = homo_corr_lon = None
                r_used = None
                visual_innovation_m = None
                max_innovation_m = None
                visual_rejected_reason = ""
                bank_rad = abs(imu_data["roll"])

                if homo_pos is not None:
                    homo_lat_raw, homo_lon_raw = homo_pos
                    h_rad = math.radians(ekf_yaw)
                    effective_lookahead = config.LOOKAHEAD_M * math.cos(bank_rad)
                    corr_north = -effective_lookahead * math.cos(h_rad)
                    corr_east  = -effective_lookahead * math.sin(h_rad)
                    homo_corr_lat = (homo_lat_raw
                                     + corr_north / 111320.0)
                    homo_corr_lon = (homo_lon_raw
                                     + corr_east / (111320.0
                                                    * math.cos(math.radians(homo_lat_raw))))
                    homo_pos = (homo_corr_lat, homo_corr_lon)

                    # Innovation gate: reject if corrected position is implausibly far from EKF.
                    # dt clamped to [0.5, 4.0] s — prevents both over-rejection and
                    # over-permissiveness from pauses or stalls.
                    visual_innovation_m = haversine_distance(
                        homo_pos[0], homo_pos[1], ekf_lat, ekf_lon)
                    pos_sigma_now = math.sqrt(max(ekf.P[8, 8], ekf.P[9, 9]))
                    dt_gate = min(max((ts - prev_frame_ts) if prev_frame_ts is not None else 1.0,
                                      0.5), 4.0)
                    max_innovation_m = max(150.0, 3.0 * pos_sigma_now + vel * dt_gate + 50.0)
                    if gate_pass and visual_innovation_m > max_innovation_m:
                        gate_pass = False
                        visual_rejected_reason = "innovation_too_large"

                # ── Relocalization ────────────────────────────────────────────────
                relocalization_candidate = False
                relocalization_applied = False
                cfg_r = config
                if (not gate_pass
                        and visual_rejected_reason == "innovation_too_large"
                        and homo_pos is not None):
                    relocalization_candidate = (
                        cs >= cfg_r.RELOCALIZATION_CSHAPE_MIN
                        and ni >= cfg_r.RELOCALIZATION_INLIERS_MIN
                        and meta_verified
                        and ver_matches >= cfg_r.RELOCALIZATION_VERIFICATION_MIN
                    )
                    if relocalization_candidate:
                        recovery_state["consecutive"] += 1
                        recovery_state["positions"].append(homo_pos)
                        if len(recovery_state["positions"]) > 5:
                            recovery_state["positions"].pop(0)
                    else:
                        recovery_state["consecutive"] = 0
                        recovery_state["positions"].clear()

                    if (relocalization_candidate
                            and recovery_state["consecutive"] >= cfg_r.RELOCALIZATION_CONSECUTIVE_THRESHOLD
                            and len(recovery_state["positions"]) >= 3):
                        recent = recovery_state["positions"][-3:]
                        dt_hop = min(max((ts - prev_frame_ts) if prev_frame_ts is not None else 1.0,
                                         0.5), 4.0)
                        coherent = all(
                            haversine_distance(recent[k][0], recent[k][1],
                                               recent[k + 1][0], recent[k + 1][1])
                            <= cfg_r.RELOCALIZATION_COHERENCE_HOP_FACTOR * vel * dt_hop
                            for k in range(len(recent) - 1)
                        )
                        if coherent:
                            ekf.P[8, 8] = max(ekf.P[8, 8],
                                              cfg_r.RELOCALIZATION_PRIOR_STD_M ** 2)
                            ekf.P[9, 9] = max(ekf.P[9, 9],
                                              cfg_r.RELOCALIZATION_PRIOR_STD_M ** 2)
                            ekf.update_position(homo_pos[0], homo_pos[1],
                                                R_pos_m2=cfg_r.RELOCALIZATION_R_M ** 2)
                            gate_pass = True
                            visual_rejected_reason = "relocalization_applied"
                            relocalization_applied = True
                            recovery_state["consecutive"] = 0
                            recovery_state["positions"].clear()
                            searcher.frame_count = 0   # force PF cold-start on next frame
                elif gate_pass:
                    recovery_state["consecutive"] = 0
                    recovery_state["positions"].clear()

                prev_frame_ts = ts

                result["gate_pass"] = gate_pass
                result["visual_rejected_reason"] = visual_rejected_reason
                result["visual_innovation_m"] = visual_innovation_m
                result["max_visual_innovation_m"] = max_innovation_m
                result["relocalization_candidate"] = relocalization_candidate
                result["relocalization_applied"] = relocalization_applied

                sc_method = result.get("method", "")
                ekf_update_applied = False
                if gate_pass and homo_pos is not None:
                    if relocalization_applied:
                        r_used = cfg_r.RELOCALIZATION_R_M ** 2
                        ekf_update_applied = True
                        gate_count += 1
                    else:
                        if sc_method == "cold_start":
                            r_used = config.R_COLD_START
                        else:
                            r_used = config.R_HIGH if (cs > 0.5 and ni > 100) else config.R_MED
                        if bank_rad > config.TURN_ROLL_THRESHOLD_RAD:
                            r_used *= config.TURN_R_MULTIPLIER
                        if not meta_verified:
                            r_used *= 2.0
                        r_used *= max(0.5, 2.0 - 1.5 * sem_conf)
                        ekf.update_position(homo_pos[0], homo_pos[1],
                                            R_pos_m2=r_used)
                        ekf_update_applied = True
                        gate_count += 1

                final = ekf.get_state()
                pos_sigma = math.sqrt(max(ekf.P[8, 8], ekf.P[9, 9]))
                _gps_est_ts = time.perf_counter()
                result["_ekf_before"] = ekf_state_before
                result["_ekf_after"]  = final

                gps_alt_ft = row.get("altitude") if row else None
                result_row = {
                    "frame_idx":          frame_idx,
                    "timestamp":          ts,
                    "image_name":         f"live_{frame_id}",
                    "final_lat":          final["latitude"],
                    "final_lon":          final["longitude"],
                    "heading_deg":        ekf_yaw,
                    "altitude_m":         round(final["altitude"], 2),
                    "roll_deg":           round(final["roll"], 3),
                    "pitch_deg":          round(final["pitch"], 3),
                    "vel_n":              round(final["vel_n"], 3),
                    "vel_e":              round(final["vel_e"], 3),
                    "vel_d":              round(final["vel_d"], 3),
                    "gps_lat":            row.get("latitude") if row else None,
                    "gps_lon":            row.get("longitude") if row else None,
                    "gps_alt_m":          round(gps_alt_ft * 0.3048, 2) if gps_alt_ft is not None else None,
                    "method":             result.get("method", ""),
                    "gate_pass":          int(gate_pass),
                    "search_time_s":      round(result.get("search_time", 0.0), 4),
                    "cs_shape":           round(cs, 4),
                    "inliers":            ni,
                    "semantic_conf":      round(sem_conf, 4),
                    "homo_lat":           homo_lat_raw,
                    "homo_lon":           homo_lon_raw,
                    "homo_corrected_lat": homo_corr_lat,
                    "homo_corrected_lon": homo_corr_lon,
                    "meta_tile_verified": int(meta_verified),
                    "ekf_pos_sigma":      round(pos_sigma, 2),
                    "r_used_sqrt":        round(math.sqrt(r_used), 2) if r_used else None,
                    "tiles_tested":       tiles_tested,
                    "verification_matches": ver_matches,
                    "inference_ms":       round(
                        (time.perf_counter() - frame_capture_ts) * 1000.0, 1),
                    "visual_innovation_m": round(visual_innovation_m, 1) if visual_innovation_m is not None else None,
                    "max_visual_innovation_m": round(max_innovation_m, 1) if max_innovation_m is not None else None,
                    "visual_rejected_reason": visual_rejected_reason,
                    "pf_update_source":   result.get("pf_update_source", ""),
                    "search_radius_m":    round(result.get("search_radius_m") or 0.0, 1) if result.get("search_radius_m") is not None else None,
                    "search_radius_capped": int(bool(result.get("search_radius_capped", False))),
                    "visual_quality_pass":      int(visual_quality_pass),
                    "ekf_update_applied":       int(ekf_update_applied),
                    "relocalization_candidate": int(relocalization_candidate),
                    "relocalization_applied":   int(relocalization_applied),
                }
                writer.writerow(result_row)

                # ── Optional: save query frame JPEG ────────────────────────
                if config.SAVE_QUERY_FRAMES:
                    flight_dir.mkdir(exist_ok=True)
                    cv2.imwrite(
                        str(flight_dir / f"frame_{frame_idx:04d}.jpg"),
                        cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))

                # ── Optional: save raw IMU row JSON ────────────────────────
                if config.SAVE_IMU_ROWS and row:
                    flight_dir.mkdir(exist_ok=True)
                    (flight_dir / f"frame_{frame_idx:04d}_imu.json").write_text(
                        json.dumps({
                            k: (None if isinstance(v, float) and math.isnan(v) else v)
                            for k, v in row.items()
                        }), encoding="utf-8")

                # ── Optional: PX4 GPS_INPUT + analysis extras ──────────────
                if config.SAVE_ANALYSIS_DATA:
                    _px4_w.writerow({
                        "time_usec":          int(ts * 1e6),
                        "gps_id":             0,
                        "ignore_flags":       0x0006,
                        "time_week_ms":       0,
                        "time_week":          0,
                        "fix_type":           3,
                        "lat":                int(final["latitude"] * 1e7),
                        "lon":                int(final["longitude"] * 1e7),
                        "alt":                round(final["altitude"], 2),
                        "hdop":               0,
                        "vdop":               0,
                        "vn":                 round(final["vel_n"], 3),
                        "ve":                 round(final["vel_e"], 3),
                        "vd":                 round(final["vel_d"], 3),
                        "speed_accuracy":     0.5,
                        "horiz_accuracy":     round(pos_sigma, 2),
                        "vert_accuracy":      round(2.0 * pos_sigma, 2),
                        "satellites_visible": 0,
                        "yaw":                int(ekf_yaw * 100),
                    })
                    off_n = off_e = None
                    if homo_corr_lat is not None:
                        sim_lat = row.get("latitude") if row else None
                        sim_lon = row.get("longitude") if row else None
                        if sim_lat is not None and abs(float(sim_lat)) > 1.0:
                            off_n = round((homo_corr_lat - float(sim_lat)) * 111320.0, 1)
                            off_e = round(
                                (homo_corr_lon - float(sim_lon)) * 111320.0
                                * math.cos(math.radians(float(sim_lat))), 1)
                    _extras_w.writerow({
                        "frame_idx":           frame_idx,
                        "timestamp":           ts,
                        "n_eff":               result.get("n_eff"),
                        "particle_spread":     result.get("particle_spread"),
                        "homo_offset_north_m": off_n,
                        "homo_offset_east_m":  off_e,
                    })

                # ── Optional: collect timing data ──────────────────────────
                if config.SAVE_TIMING_DATA:
                    t_dict = result.get("timing") or {}
                    timing_rows.append({
                        "frame_idx":        frame_idx,
                        "timestamp":        ts,
                        "frame_capture_ts": _frame_cap_ts,
                        "gps_estimate_ts":  _gps_est_ts,
                        "cold_search_ms":   t_dict.get("cold_search_ms", 0.0),
                        "pf_predict_ms":    t_dict.get("pf_predict_ms", 0.0),
                        "semantic_ms":      t_dict.get("semantic_ms", 0.0),
                        "meta_tile_ms":     t_dict.get("meta_tile_ms", 0.0),
                        "homography_ms":    t_dict.get("homography_ms", 0.0),
                        "pf_update_ms":     t_dict.get("pf_update_ms", 0.0),
                        "total_ms":         t_dict.get("total_ms", 0.0),
                    })

                # ── Optional: save full pipeline trace ────────────────────
                if config.SAVE_PIPELINE_TRACE:
                    trace_dir = run_dir / "pipeline_data" / f"frame_{frame_idx:04d}"
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    _save_trace_images(trace_dir, frame_img, result)
                    if row:
                        (trace_dir / "imu.json").write_text(
                            json.dumps({
                                k: (None if isinstance(v, float) and math.isnan(v) else v)
                                for k, v in row.items()
                            }), encoding="utf-8")
                    (trace_dir / "trace.json").write_text(
                        json.dumps(_build_trace_json(frame_idx, ts, result, result_row),
                                   indent=2, default=_json_default),
                        encoding="utf-8")

                if (frame_idx + 1) % 10 == 0:
                    f.flush()

                elapsed = time.perf_counter() - t0
                print(f"  F{frame_idx:4d} | gate={'PASS' if gate_pass else 'fail'} | "
                      f"({final['latitude']:.6f}, {final['longitude']:.6f}) | "
                      f"{elapsed:.1f}s")
                frame_idx += 1

        except KeyboardInterrupt:
            print("\n[run_pipeline] Interrupted by user")
        finally:
            source.close()
            if _px4_f:
                _px4_f.close()
            if _extras_f:
                _extras_f.close()

    if config.SAVE_TIMING_DATA and timing_rows:
        import pandas as pd
        pd.DataFrame(timing_rows, columns=TIMING_COLUMNS).to_csv(
            run_dir / "timing_data.csv", index=False)
        print(f"[run_pipeline] timing_data: {run_dir / 'timing_data.csv'}")

    total = time.perf_counter() - t0
    _save_meta(run_dir, run_id, "simconnect", frame_idx, gate_count, total,
               None, None)
    print(f"\n[run_pipeline] Done — {frame_idx} frames in {total:.1f}s")
    print(f"[run_pipeline] results: {csv_path}")


def _save_meta(run_dir, run_id, source, n_frames, gate_count, elapsed,
               imu_csv, frames_dir):
    meta = {
        "run_id":      run_id,
        "source":      source,
        "n_frames":    n_frames,
        "gate_count":  gate_count,
        "elapsed_s":   round(elapsed, 2),
        "fps":         round(n_frames / max(elapsed, 1e-6), 3),
        "imu_csv":     imu_csv,
        "frames_dir":  frames_dir,
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")


def main():
    args = _parse_args()
    config.ensure_output_dirs()
    _set_deployment_flags(args.debug)

    run_dir, run_id = _build_run_dir(args)

    if args.source == "file":
        run_file_mode(args, run_dir, run_id)
    else:
        run_simconnect_mode(args, run_dir, run_id)


if __name__ == "__main__":
    main()
